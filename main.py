"""
Main script for training and evaluating neural network models for jet energy calibration.

This script handles both training new models and plotting results from trained models.
It uses PyTorch Lightning for training and includes functionality for:

- Loading and preprocessing data using custom DataModule classes
- Training models with early stopping and checkpointing
- Making predictions on test sets
- Generating various diagnostic plots
- Handling both single-target and multi-target (energy and mass) regression

Command line arguments:
    type: Either "train" or "plot"
        - train: Train a new model using parameters from a config file
        - plot: Generate plots for an existing trained model
    path: 
        - If type=train: Path to YAML parameter config file
        - If type=plot: Path to directory containing trained model

The parameter config file should contain:
    - data_params: Configuration for data loading and preprocessing
    - model_params: Model architecture and training parameters
    - run_name: Name for the training run
    - epochs: Number of training epochs
    - save_predictions: Whether to save model predictions
    - n_samples: Number of samples for uncertainty estimation

Key components:
    - Uses PyTorch Lightning for training
    - Supports different model architectures (MLP, GMM)
    - Includes comprehensive logging
    - Generates various diagnostic plots
    - Handles data preprocessing and uncertainty quantification
"""

import os
import sys
import torch
import yaml
from datetime import datetime
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging

import math

from Source.dataset import DataModule_Single
from Source.model import *
from Source.util import PrintingCallback
from Source.plots import Plotter, plot_pred_correlation, plot_pred_jet_correlation, plot_pred_jet_correlation_marginalized

# Configure PyTorch and logging settings
# torch.set_float32_matmul_precision('medium')
logging.getLogger("lightning_fabric.plugins.environments.slurm").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)


def main():
    """
    Main function that handles the training and plotting workflows.
    Sets up logging, loads data, creates/loads models, and generates plots.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('type')  # train or plot
    parser.add_argument('path')  # config file path or results directory
    args = parser.parse_args()

    # Handle training setup
    if args.type == "train":
        # Load parameters from config file
        with open(args.path, 'r') as f:
            params = yaml.safe_load(f)

        # Create results directory and save parameters
        dir_path = os.path.dirname(os.path.realpath(__file__))
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + params["run_name"]
        run_dir = os.path.join(dir_path, "results", run_name)
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "params.yaml"), 'w') as f:
            yaml.dump(params, f)

        log_file = os.path.join(run_dir, "train_log.txt")

    # Handle plotting setup
    elif args.type == "plot":
        run_dir = args.path
        with open(os.path.join(run_dir, "params.yaml"), 'r') as f:
            params = yaml.safe_load(f)

        log_file = os.path.join(run_dir, "plot_log.txt")

    else:
        raise NotImplementedError(f"type {args.type} not recognised")
    
    if params.get("use_tf32", True):
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Set up exception handling
    def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
        logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = log_uncaught_exceptions
    logging.info("\n")
    logging.info(f"Starting {args.type} run with run_dir {run_dir}")

    logging.info(f"{torch.cuda.device_count() } GPUs available.")

    # Initialize data module based on configuration
    logging.info(f"Loading data from {params['data_params']['data_folder']}")
    data_module_type = params['data_params'].get("data_module_type", "single")
    if data_module_type == "single":
        data_module = DataModule_Single(params["data_params"])
    elif data_module_type == "full":
        raise NotImplementedError()
        # data_module = DataModule_Full(params["data_params"])
    else:
        raise ValueError(f"data_module_type {data_module_type} not recognised")

    # Configure PyTorch Lightning trainer
    logging.info(f"Creating trainer")
    printing_callback = PrintingCallback()
    ckpt_callback: pl.callbacks.ModelCheckpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        max_epochs=params.get("epochs", 100),
        accelerator="gpu" if torch.cuda.device_count() > 0 else "auto", 
        devices=1, 
        log_every_n_steps=100,
        enable_progress_bar=False,
        callbacks=[
            ckpt_callback,
            printing_callback,
            lr_callback,
            # pl.callbacks.EarlyStopping(
            #     monitor="val_loss",
            #     patience=50,
            #     verbose=True,
            #     mode="min"
            # )
        ],
        check_val_every_n_epoch=1,
        logger=pl.loggers.TensorBoardLogger(save_dir=run_dir) if args.type == "train" else False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        detect_anomaly=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    # Training workflow
    if args.type == "train":
        # Create and train model
        logging.info(f"Creating model {params['model_params'].get('model', 'MLP_MSE_Regression')}")
        model: pl.LightningModule = eval(params["model_params"]["model"])(
            input_dim=data_module.input_dim, 
            target_dim=data_module.target_dim, 
            parameters=params["model_params"]
        )

        if params.get("compile", False):
            # I think we need this to use TF32
            opt_model: pl.LightningModule = torch.compile(model)
        else:
            opt_model: pl.LightningModule = model

        opt_model.train()

        logging.info(f"Training model for {params.get('epochs', 100)} epochs")
        trainer.fit(opt_model, data_module)
        # opt_model.load_from_checkpoint(ckpt_callback.best_model_path)
        logging.info(f"Loading model checkpoint from {ckpt_callback.best_model_path}")
        model = eval(params["model_params"]["model"]).load_from_checkpoint(ckpt_callback.best_model_path)
        if params.get("compile", False):
            model = torch.compile(model)

        model.eval()

        # Generate predictions on test set
        logging.info(f"Making predictions on test set with {params.get("n_samples", 10)} samples")
        if params["model_params"]["model"] in ["MLP_CFM"]:
            n_samples = params.get("n_samples", 10)
            test_predictions = []
            for i_sample in range(n_samples):
                logging.info(f"Sampling step {i_sample}/{n_samples}")
                test_predictions += trainer.predict(model, data_module.test_dataloader())
        else:
            test_predictions = trainer.predict(model, data_module.test_dataloader())
        keys = test_predictions[0].keys()
        test_predictions = {key: torch.cat([pred[key] for pred in test_predictions], dim=0) for key in keys}
        if params["model_params"]["model"] in ["MLP_CFM"]:
            test_predictions["samples"] = test_predictions["samples"].reshape(n_samples, -1, len(params["data_params"]["target_dims"]))
            test_predictions["log_likelihoods"] = test_predictions["log_likelihoods"].reshape(n_samples, -1, len(params["data_params"]["target_dims"]))
        test_predictions["preprocess_mean"] = data_module.target_preprocessor.mean
        test_predictions["preprocess_std"] = data_module.target_preprocessor.std

        if params["save_predictions"]:
            logging.info(f"Saving predictions")
            torch.save(test_predictions, os.path.join(run_dir, "test_predictions.pt"))

    # Plotting workflow
    elif args.type == "plot":
        # Load trained model from checkpoint
        checkpoint_dir = os.path.join(run_dir, "lightning_logs/version_0/checkpoints")
        checkpoints = os.listdir(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
        logging.info(f"Loading model checkpoint from {checkpoint_path}")
        model = eval(params["model_params"]["model"]).load_from_checkpoint(checkpoint_path)
        if params.get("compile", False):
            model = torch.compile(model)

        model.eval()

        # Load or generate predictions
        logging.info(f"Loading predictions")
        try:
            test_predictions_loaded = torch.load(os.path.join(run_dir, "test_predictions.pt"), weights_only=False)
            test_predictions = {key: test_predictions_loaded[key] for key in test_predictions_loaded.keys()}
        except FileNotFoundError:
            logging.info(f"No predictions found in {run_dir}, making predictions on test set")
            if params["model_params"]["model"] in ["MLP_CFM"]:
                n_samples = params.get("n_samples", 10)
                test_predictions = []
                for i_sample in range(n_samples):
                    logging.info(f"Sampling step {i_sample}/{n_samples}")
                    test_predictions += trainer.predict(model, data_module.test_dataloader())
            else:
                test_predictions = trainer.predict(model, data_module.test_dataloader())
            keys = test_predictions[0].keys()
            test_predictions = {key: torch.cat([pred[key] for pred in test_predictions], dim=0) for key in keys}
            if params["model_params"]["model"] in ["MLP_CFM"]:
                test_predictions["samples"] = test_predictions["samples"].reshape(n_samples, -1, len(params["data_params"]["target_dims"]))
                test_predictions["log_likelihoods"] = test_predictions["log_likelihoods"].reshape(n_samples, -1)    
            test_predictions["preprocess_mean"] = data_module.target_preprocessor.mean
            test_predictions["preprocess_std"] = data_module.target_preprocessor.std

            if params["save_predictions"]:
                logging.info(f"Saving predictions")
                torch.save(test_predictions, os.path.join(run_dir, "test_predictions.pt"))

    # Generate samples for uncertainty estimation
    logging.info(f"Sampling")
    if params["model_params"]["model"] == "MLP_GMM_Regression":
        if len(params["data_params"]["target_dims"]) == 1:
            # Single target sampling
            samples, log_likelihoods = model.sample(
                test_predictions["mu"], 
                test_predictions["sigma"], 
                test_predictions["weights"], 
                n_samples=params.get("n_samples", 10)
            )
            samples = samples*test_predictions["preprocess_std"] + test_predictions["preprocess_mean"]
            log_likelihoods = log_likelihoods/test_predictions["preprocess_std"]
        else:
            # Multi-target sampling (energy and mass)
            samples_E, log_likelihoods_E = model.sample(
                test_predictions["mu_E"], 
                test_predictions["sigma_E"], 
                test_predictions["weights_E"], 
                n_samples=params.get("n_samples", 10)
            )
            samples_E = samples_E*test_predictions["preprocess_std"][0] + test_predictions["preprocess_mean"][0]
            log_likelihoods_E = log_likelihoods_E/test_predictions["preprocess_std"][0]
            
            samples_m, log_likelihoods_m = model.sample(
                test_predictions["mu_m"], 
                test_predictions["sigma_m"], 
                test_predictions["weights_m"], 
                n_samples=params.get("n_samples", 10)
            )
            samples_m = samples_m*test_predictions["preprocess_std"][1] + test_predictions["preprocess_mean"][1]
            log_likelihoods_m = log_likelihoods_m/test_predictions["preprocess_std"][1]
            
            samples = torch.stack([samples_E, samples_m], dim=1)
            log_likelihoods = torch.stack([log_likelihoods_E, log_likelihoods_m], dim=1)
    elif params["model_params"]["model"] == "MLP_Multivariate_GMM_Regression":
        samples, log_likelihoods = model.sample(
            test_predictions["mu"], 
            test_predictions["sigma_inv"], 
            test_predictions["weights"], 
            n_samples=params.get("n_samples", 10)
        )
        print(f"samples: {samples.shape}")
        samples[..., 0] = samples[..., 0] * test_predictions["preprocess_std"][0] + test_predictions["preprocess_mean"][0]
        samples[..., 1] = samples[..., 1] * test_predictions["preprocess_std"][1] + test_predictions["preprocess_mean"][1]
        log_likelihoods = log_likelihoods/math.prod(test_predictions["preprocess_std"])
        log_likelihoods = log_likelihoods.unsqueeze(-1).expand(-1, -1, 2)

        samples = samples.permute(0, 2, 1)
        log_likelihoods = log_likelihoods.permute(0, 2, 1)
    elif params["model_params"]["model"] in ["MLP_CFM"]:
        samples = test_predictions["samples"]
        log_likelihoods = test_predictions["log_likelihoods"]
        # print(f"samples: {samples.shape}")
        # print(f"log_likelihoods: {log_likelihoods.shape}")
        samples[..., 0] = samples[..., 0] * test_predictions["preprocess_std"][0] + test_predictions["preprocess_mean"][0]
        samples[..., 1] = samples[..., 1] * test_predictions["preprocess_std"][1] + test_predictions["preprocess_mean"][1]
        log_likelihoods = log_likelihoods/math.prod(test_predictions["preprocess_std"])
        # log_likelihoods = log_likelihoods.unsqueeze(-1).expand(-1, -1, 2)

        samples = samples.permute(1, 2, 0) # Batch, target, samples
        log_likelihoods = log_likelihoods.permute(1, 0).unsqueeze(1).expand(-1, 2, -1) # Batch, target, samples
        print(f"samples: {samples.shape}")
        print(f"log_likelihoods: {log_likelihoods.shape}")
    else:
        # Similar sampling logic for non-GMM models
        if len(params["data_params"]["target_dims"]) == 1:
            samples, log_likelihoods = model.sample(
                test_predictions["mu"], 
                test_predictions["sigma"], 
                n_samples=params.get("n_samples", 10)
            )
            samples = samples*test_predictions["preprocess_std"] + test_predictions["preprocess_mean"]
            log_likelihoods = log_likelihoods/test_predictions["preprocess_std"]
        else:
            samples_E, log_likelihoods_E = model.sample(
                test_predictions["mu_E"], 
                test_predictions["sigma_E"], 
                n_samples=params.get("n_samples", 10)
            )
            samples_E = samples_E*test_predictions["preprocess_std"][0] + test_predictions["preprocess_mean"][0]
            log_likelihoods_E = log_likelihoods_E/test_predictions["preprocess_std"][0]
            
            samples_m, log_likelihoods_m = model.sample(
                test_predictions["mu_m"], 
                test_predictions["sigma_m"], 
                n_samples=params.get("n_samples", 10)
            )
            samples_m = samples_m*test_predictions["preprocess_std"][1] + test_predictions["preprocess_mean"][1]
            log_likelihoods_m = log_likelihoods_m/test_predictions["preprocess_std"][1]
            
            samples = torch.stack([samples_E, samples_m], dim=1)
            log_likelihoods = torch.stack([log_likelihoods_E, log_likelihoods_m], dim=1)

    logging.info(f"There are {samples.shape} samples.")

    with torch.inference_mode():
        # Generate plots
        logging.info(f"Making Plots")

        # Prepare data for plotting
        test_inputs = data_module.test_dataset.tensors[0]
        test_inputs = data_module.input_preprocessor.preprocess_backward(test_inputs)
        test_targets = data_module.test_dataset.tensors[1]
        test_targets = data_module.target_preprocessor.preprocess_backward(test_targets)

        train_inputs = data_module.train_dataset.tensors[0]
        train_inputs = data_module.input_preprocessor.preprocess_backward(train_inputs)
        train_targets = data_module.train_dataset.tensors[1]
        train_targets = data_module.target_preprocessor.preprocess_backward(train_targets)

        logging.info("Plotting joint plots")

        os.makedirs(os.path.join(run_dir, "plots_joint"), exist_ok=True)
        plot_pred_correlation(
            name="target_correlations.pdf",
            targets=test_targets,
            samples=samples,
            log_likelihoods=log_likelihoods,
            plot_dir=os.path.join(run_dir, "plots_joint"),
            train_targets=train_targets
        )
        plot_pred_jet_correlation(
            name="jet_correlations.pdf",
            targets=test_targets,
            samples=samples,
            input_data=test_inputs,
            log_likelihoods=log_likelihoods,
            plot_dir=os.path.join(run_dir, "plots_joint"),
            train_targets=train_targets,
            train_input_data=train_inputs
        )
        # plot_pred_jet_correlation_marginalized(
        #     name="jet_correlations_marginalized.pdf",
        #     targets=test_targets,
        #     samples=samples,
        #     input_data=test_inputs,
        #     log_likelihoods=log_likelihoods,
        #     plot_dir=os.path.join(run_dir, "plots_joint"),
        #     train_targets=train_targets,
        #     train_input_data=train_inputs
        # )

        # Generate plots for each target dimension
        for target_dim in params["data_params"]["target_dims"]:
            variable = ["E", "m"][target_dim]
            logging.info(f"Plotting variable {variable}")
            os.makedirs(os.path.join(run_dir, f"plots_{variable}"), exist_ok=True)
            plot_dir = os.path.join(run_dir, f"plots_{variable}")
            
            # Initialize plotter with appropriate data
            if len(params["data_params"]["target_dims"]) == 1:
                plotter = Plotter(
                    plot_dir, 
                    params,
                    test_predictions, 
                    test_inputs, 
                    test_targets, 
                    samples, 
                    log_likelihoods,
                    variable,
                    train_inputs=train_inputs,
                    train_targets=train_targets,
                    skip_dims=params["data_params"].get("skip_dims", [])
                )
            else:
                plotter = Plotter(
                    plot_dir, 
                    params, 
                    {key: test_predictions[key] for key in test_predictions.keys() if variable in key}, 
                    test_inputs, 
                    test_targets[:, target_dim], 
                    samples[:, target_dim], 
                    log_likelihoods[:, target_dim],
                    variable,
                    train_inputs=train_inputs,
                    train_targets=train_targets[:, target_dim],
                    additional_test_inputs=data_module.additional_input_test,
                    skip_dims=params["data_params"].get("skip_dims", [])
                )

            # Generate various plots
            # if args.type == "train":
            #     plotter.plot_loss_history("loss.pdf", model.train_epoch_losses, model.val_epoch_losses)

            # plotter.r_predictions_histogram(f"{variable}_r_predictions_histogram.pdf")
            # plotter.E_M_predictions_histogram(f"{variable}_predictions_histogram.pdf")
            # plotter.r_2d_histogram(f"{variable}_r_2d_histogram.pdf")
            # plotter.E_M_2d_histogram(f"{variable}_2d_histogram.pdf")
            # plotter.E_M_predictions_input_histogram(f"{variable}_pred_inputs_histogram.pdf")
            # plotter.pred_inputs_histogram(f"{variable}_r_pred_inputs_histogram.pdf")
            # plotter.r_rel_over_input(f"{variable}_r_relativ_over_input.pdf")
            # plotter.pred_inputs_histogram_marginalized(f"{variable}_pred_inputs_histogram_marginalized.pdf")

            # Additional plots for GMM models
            # if params["model_params"]["model"] == "MLP_GMM_Regression":
            #     plotter.plot_GMM_weights("GMM_weights.pdf")

    logging.info(f"Done")

if __name__ == "__main__":
    main()
