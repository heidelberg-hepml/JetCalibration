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
from Source.plots import Plotter, plot_pred_correlation

# Configure PyTorch and logging settings
# torch.set_float32_matmul_precision('medium')
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
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

    # Training workflow
    if args.type == "train":
        # Configure PyTorch Lightning trainer
        logging.info(f"Creating trainer")
        printing_callback = PrintingCallback()
        ckpt_callback: pl.callbacks.ModelCheckpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
        trainer = pl.Trainer(
            max_epochs=params.get("epochs", 100),
            accelerator="gpu", 
            devices=1, 
            log_every_n_steps=100,
            enable_progress_bar=False,
            callbacks=[
                ckpt_callback,
                printing_callback,
                # pl.callbacks.EarlyStopping(
                #     monitor="val_loss",
                #     patience=50,
                #     verbose=True,
                #     mode="min"
                # )
            ],
            check_val_every_n_epoch=1,
            logger=pl.loggers.TensorBoardLogger(save_dir=run_dir),
            enable_model_summary=False,
            num_sanity_val_steps=0,
            detect_anomaly=False,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        )

        # Create and train model
        logging.info(f"Creating model {params['model_params'].get('model', 'MLP_MSE_Regression')}")
        model: pl.LightningModule = eval(params["model_params"]["model"])(
            input_dim=data_module.input_dim, 
            target_dim=data_module.target_dim, 
            parameters=params["model_params"]
        )

        # I think we need this to use TF32
        # opt_model: pl.LightningModule = torch.compile(model)
        opt_model: pl.LightningModule = model

        opt_model.train()

        logging.info(f"Training model for {params.get('epochs', 100)} epochs")
        trainer.fit(opt_model, data_module)
        # opt_model.load_from_checkpoint(ckpt_callback.best_model_path)
        model = eval(params["model_params"]["model"]).load_from_checkpoint(ckpt_callback.best_model_path)

        model.eval()

        # Generate predictions on test set
        logging.info(f"Making predictions on test set")
        test_predictions = trainer.predict(model, data_module.test_dataloader())
        keys = test_predictions[0].keys()
        test_predictions = {key: torch.cat([pred[key] for pred in test_predictions], dim=0) for key in keys}
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
        model = eval(params["model_params"]["model"]).load_from_checkpoint(checkpoint_path)

        model.eval()

        # Load or generate predictions
        logging.info(f"Loading predictions")
        try:
            test_predictions_loaded = torch.load(os.path.join(run_dir, "test_predictions.pt"), weights_only=False)
            test_predictions = {key: test_predictions_loaded[key] for key in test_predictions_loaded.keys()}
        except FileNotFoundError:
            logging.info(f"No predictions found in {run_dir}, making predictions on test set")
            test_predictions = trainer.predict(model, data_module.test_dataloader())
            keys = test_predictions[0].keys()
            test_predictions = {key: torch.cat([pred[key] for pred in test_predictions], dim=0) for key in keys}
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

    # Generate plots
    logging.info(f"Making Plots")

    # Prepare data for plotting
    test_inputs = data_module.test_dataset.tensors[0]
    test_inputs = data_module.input_preprocessor.preprocess_backward(test_inputs)
    test_targets = data_module.test_dataset.tensors[1]
    test_targets = data_module.target_preprocessor.preprocess_backward(test_targets)

    os.makedirs(os.path.join(run_dir, "plots_joint"), exist_ok=True)
    plot_pred_correlation(
        name="target_correlations.pdf",
        samples=samples,
        log_likelihoods=log_likelihoods,
        plot_dir=os.path.join(run_dir, "plots_joint")
    )

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
                variable
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
                variable
            )

        # Generate various plots
        if args.type == "train":
            plotter.plot_loss_history("loss.pdf", model.train_epoch_losses, model.val_epoch_losses)

        plotter.r_predictions_histogram(f"{variable}_r_predictions_histogram.pdf")
        plotter.E_M_predictions_histogram(f"{variable}_predictions_histogram.pdf")
        plotter.r_2d_histogram(f"{variable}_r_2d_histogram.pdf")
        plotter.E_M_2d_histogram(f"{variable}_2d_histogram.pdf")
        plotter.pred_inputs_histogram(f"{variable}_pred_inputs_histogram.pdf")
        # plotter.pred_inputs_histogram_marginalized(f"{variable}_pred_inputs_histogram_marginalized.pdf")

        # Additional plots for GMM models
        if params["model_params"]["model"] == "MLP_GMM_Regression":
            plotter.plot_GMM_weights("GMM_weights.pdf")

    logging.info(f"Done")

if __name__ == "__main__":
    main()
