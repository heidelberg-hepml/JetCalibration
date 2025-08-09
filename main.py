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

import os, sys, shutil
import torch
import yaml
from datetime import datetime
import pytorch_lightning as pl
import numpy as np
import argparse
import logging
from pathlib import Path
import gc

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import roc_curve, auc

import math

from Source.dataset import DataModule_Single, Classifier_DataModule, SampleEvaluation
from Source.model import *
from Source.util import PrintingCallback, init_weights
from Source.plots import Plotter, plot_pred_correlation, plot_pred_jet_correlation, plot_pred_jet_correlation_marginalized, compute_range

# Configure PyTorch and logging settings
# torch.set_float32_matmul_precision('medium')
logging.getLogger("lightning_fabric.plugins.environments.slurm").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

def get_params(args):
    # Handle training setup
    if args.type == "train":
        # Load parameters from config file
        with open(args.path, 'r') as f:
            params = yaml.safe_load(f)
    # Handle plotting setup
    elif args.type in ["plot", "classifier", "plot_classifier"]:
        run_dir = args.path
        with open(os.path.join(run_dir, "params.yaml"), 'r') as f:
            params = yaml.safe_load(f)
    else:
        raise NotImplementedError(f"type {args.type} not recognised")    

    return params

def setup_run_dir(args, params) -> str:
    # Handle training setup
    if args.type == "train":
        # Create results directory and save parameters
        dir_path = os.path.dirname(os.path.realpath(__file__))
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + params["run_name"]
        run_dir = os.path.join(dir_path, "results", run_name)
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "params.yaml"), 'w') as f:
            yaml.dump(params, f)
    # Handle plotting setup
    elif args.type in ["plot", "classifier", "plot_classifier"]:
        run_dir = args.path
    else:
        raise NotImplementedError(f"type {args.type} not recognised")
    
    return run_dir

def setup_logging(run_type, run_dir):
    if run_type == "train":
        log_file = os.path.join(run_dir, "train_log.txt")
    elif run_type == "plot":
        log_file = os.path.join(run_dir, "plot_log.txt")
    elif run_type == "classifier":
        log_file = os.path.join(run_dir, "classifier_log.txt")
    elif run_type == "plot_classifier":
        log_file = os.path.join(run_dir, "plot_classifier.txt")
    else:
        raise NotImplementedError(f"type {run_type} not recognised")
    
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
    logging.info(f"Starting {run_type} run with run_dir {run_dir}")
    logging.info(f"{torch.cuda.device_count() } GPUs available.")

def setup_tf32(params):
    if params.get("use_tf32", True):
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        logging.info("TF32 enabled")
    else:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        logging.info("TF32 disabled")

def get_data_module(params) -> DataModule_Single:
    # Initialize data module based on configuration
    logging.info(f"Loading data from {params['data_params']['data_folder']}")
    # data_module_type = params['data_params'].get("data_module_type", "single")
    # if data_module_type == "single":
    data_module = DataModule_Single(params["data_params"])
    # elif data_module_type == "full":
    #     raise NotImplementedError()
    #     # data_module = DataModule_Full(params["data_params"])
    # else:
    #     raise ValueError(f"data_module_type {data_module_type} not recognised")
    return data_module

def get_trainer(run_type, params, run_dir) -> tuple[pl.Trainer, pl.callbacks.ModelCheckpoint]:
    logging.info(f"Creating trainer")
    printing_callback = PrintingCallback()
    ckpt_callback: pl.callbacks.ModelCheckpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        # monitor="val_loss_epoch",
        mode="min",
        save_top_k=1
    )
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
        logger=pl.loggers.TensorBoardLogger(save_dir=run_dir) if run_type == "train" else False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        detect_anomaly=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
    return trainer, ckpt_callback

def get_model(run_type, params, data_module: DataModule_Single, run_dir) -> tuple[pl.LightningModule, pl.LightningModule]:
    lr_warmup_epochs = params["model_params"].get("lr_warmup_epochs", 0)
    lr_warmup_steps = lr_warmup_epochs * data_module.train_batches_per_epoch

    if run_type == "train":
        logging.info(f"Creating model {params['model_params'].get('model', 'MLP_MSE_Regression')}")
        model: pl.LightningModule = eval(params["model_params"]["model"])(
            input_dim=data_module.input_dim, 
            target_dim=data_module.target_dim, 
            parameters=params["model_params"],
            lr_warmup_steps=lr_warmup_steps
        )

        if params.get("compile", False):
            # I think we need this to use TF32
            opt_model: pl.LightningModule = torch.compile(model)
        else:
            opt_model: pl.LightningModule = model
    elif run_type in ["plot", "classifier", "plot_classifier"]:
        # Load trained model from checkpoint
        checkpoint_dir = os.path.join(run_dir, "lightning_logs/version_0/checkpoints")
        checkpoints = os.listdir(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
        logging.info(f"Loading model checkpoint from {checkpoint_path}")
        model = eval(params["model_params"]["model"]).load_from_checkpoint(checkpoint_path)
        if params.get("compile", False):
            model = torch.compile(model)
        opt_model = None

    return model, opt_model

def train(opt_model, trainer, data_module, ckpt_callback, params, run_dir) -> pl.LightningModule:
    opt_model.train()

    logging.info(f"Training model for {params.get('epochs', 100)} epochs")
    trainer.fit(opt_model, data_module)

    train_loss = opt_model.train_epoch_losses
    val_loss = opt_model.val_epoch_losses

    for e in range(len(val_loss)):
        logging.info(f"Epoch {e}: Training loss={train_loss[e]:.10f}, Validation loss={val_loss[e]:.10f}")

    # opt_model.load_from_checkpoint(ckpt_callback.best_model_path)
    logging.info(f"Loading model checkpoint from {ckpt_callback.best_model_path}")
    model = eval(params["model_params"]["model"]).load_from_checkpoint(ckpt_callback.best_model_path)
    if params.get("compile", False):
        model = torch.compile(model)

    ckpt = len(model.train_epoch_losses) - 1

    plt.figure()
    plt.plot(train_loss, label="Training loss", color="blue")
    plt.plot(val_loss, label="Validation loss", color="orange")
    plt.scatter(ckpt, val_loss[ckpt], label=f"best val loss", color="red", marker="o")
    plt.hlines(val_loss[ckpt], [0], [len(opt_model.val_epoch_losses) - 1], color="red", linestyles='dashed', alpha=0.5)
    plt.title(f"{val_loss[ckpt]:.10f} val loss at epoch {ckpt}")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss history")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss.pdf"), format='pdf', bbox_inches="tight")
    plt.close()

    logging.info(f"Full loss curve is saved at {run_dir}")

    logging.info(f"Best validation loss: {val_loss[ckpt]:.10f} at epoch {ckpt}")

    return model

def make_predictions(model, params, trainer, data_module, run_dir) -> dict:
    model.eval()

    # Generate predictions on test set
    logging.info(f"Making predictions on test set with {params.get("n_samples", 10)} samples")
    if params["model_params"]["model"] in ["MLP_CFM"]:
        n_samples = params.get("n_samples", 10)
        test_predictions = []
        test_data_loader = data_module.test_dataloader()
        for i_sample in range(n_samples):
            logging.info(f"Sampling step {i_sample+1}/{n_samples}")
            test_predictions += trainer.predict(model, test_data_loader)
            torch.cuda.empty_cache()
            gc.collect()
    else:
        with torch.inference_mode():
            test_predictions = trainer.predict(model, data_module.test_dataloader())
    with torch.inference_mode():
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

    return test_predictions

def try_load_predictions(model, run_dir, params, trainer, data_module) -> dict:
    model.eval()

    # Load or generate predictions
    logging.info(f"Loading predictions")
    try:
        test_predictions_loaded = torch.load(os.path.join(run_dir, "test_predictions.pt"), weights_only=False)
        test_predictions = {key: test_predictions_loaded[key] for key in test_predictions_loaded.keys()}
    except FileNotFoundError:
        logging.info(f"No predictions found in {run_dir}, making predictions on test set")
        test_predictions = make_predictions(model, params, trainer, data_module, run_dir)

    return test_predictions

@torch.inference_mode()
def sample(test_predictions, params, model) -> tuple[torch.FloatTensor, torch.FloatTensor]:
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
            log_likelihoods = log_likelihoods # /test_predictions["preprocess_std"]
        else:
            # Multi-target sampling (energy and mass)
            samples_E, log_likelihoods_E = model.sample(
                test_predictions["mu_E"], 
                test_predictions["sigma_E"], 
                test_predictions["weights_E"], 
                n_samples=params.get("n_samples", 10)
            )
            samples_E = samples_E*test_predictions["preprocess_std"][0] + test_predictions["preprocess_mean"][0]
            log_likelihoods_E = log_likelihoods_E # /test_predictions["preprocess_std"][0]
            
            samples_m, log_likelihoods_m = model.sample(
                test_predictions["mu_m"], 
                test_predictions["sigma_m"], 
                test_predictions["weights_m"], 
                n_samples=params.get("n_samples", 10)
            )
            samples_m = samples_m*test_predictions["preprocess_std"][1] + test_predictions["preprocess_mean"][1]
            log_likelihoods_m = log_likelihoods_m # /test_predictions["preprocess_std"][1]
            
            samples = torch.stack([samples_E, samples_m], dim=1)
            # log_likelihoods = torch.stack([log_likelihoods_E, log_likelihoods_m], dim=1)
            log_likelihoods = log_likelihoods_E + log_likelihoods_m
    elif params["model_params"]["model"] == "MLP_Multivariate_GMM_Regression":
        samples, log_likelihoods = model.sample(
            test_predictions["mu"], 
            test_predictions["sigma_inv"], 
            test_predictions["weights"], 
            n_samples=params.get("n_samples", 10)
        )
        samples[..., 0] = samples[..., 0] * test_predictions["preprocess_std"][0] + test_predictions["preprocess_mean"][0]
        samples[..., 1] = samples[..., 1] * test_predictions["preprocess_std"][1] + test_predictions["preprocess_mean"][1]
        log_likelihoods = log_likelihoods # /math.prod(test_predictions["preprocess_std"])
        log_likelihoods = log_likelihoods #.unsqueeze(-1).expand(-1, -1, 2)

        samples = samples.permute(0, 2, 1)
        # log_likelihoods = log_likelihoods.permute(0, 2, 1)
    elif params["model_params"]["model"] in ["MLP_CFM"]:
        samples = test_predictions["samples"]
        log_likelihoods = test_predictions["log_likelihoods"]
        # print(f"samples: {samples.shape}")
        # print(f"log_likelihoods: {log_likelihoods.shape}")
        samples[..., 0] = samples[..., 0] * test_predictions["preprocess_std"][0] + test_predictions["preprocess_mean"][0]
        samples[..., 1] = samples[..., 1] * test_predictions["preprocess_std"][1] + test_predictions["preprocess_mean"][1]
        log_likelihoods = log_likelihoods # /math.prod(test_predictions["preprocess_std"])

        samples = samples.permute(1, 2, 0) # Batch, target, samples
        log_likelihoods = log_likelihoods.permute(1, 0) # .unsqueeze(1).expand(-1, 2, -1) # size, target, samples
    else:
        # Similar sampling logic for non-GMM models
        if len(params["data_params"]["target_dims"]) == 1:
            samples, log_likelihoods = model.sample(
                test_predictions["mu"], 
                test_predictions["sigma"], 
                n_samples=params.get("n_samples", 10)
            )
            samples = samples*test_predictions["preprocess_std"] + test_predictions["preprocess_mean"]
            log_likelihoods = log_likelihoods # /test_predictions["preprocess_std"]
        else:
            samples_E, log_likelihoods_E = model.sample(
                test_predictions["mu_E"], 
                test_predictions["sigma_E"], 
                n_samples=params.get("n_samples", 10)
            )
            samples_E = samples_E*test_predictions["preprocess_std"][0] + test_predictions["preprocess_mean"][0]
            log_likelihoods_E = log_likelihoods_E # /test_predictions["preprocess_std"][0]
            
            samples_m, log_likelihoods_m = model.sample(
                test_predictions["mu_m"], 
                test_predictions["sigma_m"], 
                n_samples=params.get("n_samples", 10)
            )
            samples_m = samples_m*test_predictions["preprocess_std"][1] + test_predictions["preprocess_mean"][1]
            log_likelihoods_m = log_likelihoods_m/test_predictions["preprocess_std"][1]
            
            samples = torch.stack([samples_E, samples_m], dim=1)
            # log_likelihoods = torch.stack([log_likelihoods_E, log_likelihoods_m], dim=1)
            log_likelihoods = log_likelihoods_E + log_likelihoods_m

    logging.info(f"samples: {samples.shape}")
    logging.info(f"log_likelihoods: {log_likelihoods.shape}")
    return samples, log_likelihoods

@torch.inference_mode()
def plot(eval_samples: SampleEvaluation, test_predictions, data_module, run_dir, params, run_type, model):
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
    # plot_pred_correlation(
    #     name="target_correlations.pdf",
    #     targets=test_targets,
    #     samples=eval_samples,
    #     plot_dir=os.path.join(run_dir, "plots_joint"),
    #     train_targets=train_targets
    # )
    # plot_pred_jet_correlation(
    #     name="jet_correlations.pdf",
    #     targets=test_targets,
    #     samples=eval_samples,
    #     input_data=test_inputs,
    #     plot_dir=os.path.join(run_dir, "plots_joint"),
    #     train_targets=train_targets,
    #     train_input_data=train_inputs
    # )
    # plot_pred_jet_correlation_marginalized(
    #     name="jet_correlations_marginalized.pdf",
    #     targets=test_targets,
    #     samples=eval_samples,
    #     input_data=test_inputs,
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
        # if len(params["data_params"]["target_dims"]) == 1:
        #     plotter = Plotter(
        #         plot_dir, 
        #         params,
        #         test_predictions, 
        #         test_inputs, 
        #         test_targets, 
        #         samples, 
        #         log_likelihoods,
        #         variable,
        #         train_inputs=train_inputs,
        #         train_targets=train_targets,
        #         skip_dims=params["data_params"].get("skip_dims", [])
        #     )
        # else:
        plotter = Plotter(
            plot_dir, 
            params, 
            {key: test_predictions[key] for key in test_predictions.keys() if variable in key}, 
            test_inputs, 
            test_targets[:, target_dim], 
            eval_samples,
            variable,
            target_dim=target_dim,
            train_inputs=train_inputs,
            train_targets=train_targets[:, target_dim],
            additional_test_inputs=data_module.additional_input_test,
            skip_dims=params["data_params"].get("skip_dims", []),
        )

        # Generate various plots
        # if run_type == "train":
        plotter.plot_loss_history("loss.pdf", model.train_epoch_losses, model.val_epoch_losses)

        plotter.r_predictions_histogram(f"{variable}_r_predictions_histogram.pdf")
        plotter.E_M_predictions_histogram(f"{variable}_predictions_histogram.pdf")
        # plotter.r_2d_histogram(f"{variable}_r_2d_histogram.pdf")
        # plotter.E_M_2d_histogram(f"{variable}_2d_histogram.pdf")
        # plotter.E_M_predictions_input_histogram(f"{variable}_pred_inputs_histogram.pdf")
        # plotter.pred_inputs_histogram(f"{variable}_r_pred_inputs_histogram.pdf")
        # plotter.r_rel_over_input(f"{variable}_r_relativ_over_input.pdf")
        # plotter.pred_inputs_histogram_marginalized(f"{variable}_pred_inputs_histogram_marginalized.pdf")

        # Additional plots for GMM models
        # if params["model_params"]["model"] == "MLP_GMM_Regression":
        #     plotter.plot_GMM_weights("GMM_weights.pdf")

def get_classifier_trainer(run_type, classifier_params, run_dir, dir_name) -> tuple[pl.Trainer, pl.callbacks.ModelCheckpoint]:
    logging.info(f"Creating classifier trainer")
    
    trainer_dir = os.path.join(run_dir, "classifier", dir_name)
    if run_type == "classifier":
        if os.path.isdir(trainer_dir):
            shutil.rmtree(trainer_dir, ignore_errors=True)

    printing_callback = PrintingCallback()
    ckpt_callback: pl.callbacks.ModelCheckpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        # monitor="val_loss_epoch",
        mode="min",
        save_top_k=1,
        dirpath=os.path.join(run_dir, "classifier", dir_name, "checkpoints"),
        filename='{epoch}-{step}-{val_loss:.10f}'
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        max_epochs=classifier_params.get("epochs", 100),
        accelerator="gpu" if torch.cuda.device_count() > 0 else "auto", 
        devices=1, 
        log_every_n_steps=100,
        enable_progress_bar=False,
        callbacks=[
            ckpt_callback,
            printing_callback,
            lr_callback,
            pl.callbacks.EarlyStopping(
                # monitor="val_loss_epoch",
                monitor="val_loss",
                patience=30,
                verbose=True,
                mode="min"
            )
        ],
        check_val_every_n_epoch=1,
        logger=pl.loggers.TensorBoardLogger(save_dir=os.path.join(run_dir, "classifier", dir_name)) if run_type == "classifier" else False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        detect_anomaly=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
    return trainer, ckpt_callback



def get_classifier(run_type, classifier_params, data_module, run_dir, cond_type) -> tuple[pl.LightningModule, pl.LightningModule]:
    lr_warmup_epochs = classifier_params["model_params"].get("lr_warmup_epochs", 0)
    lr_warmup_steps = lr_warmup_epochs * data_module.train_batches_per_epoch

    if run_type == "classifier":
        logging.info(f"Creating classifier {classifier_params['model_params'].get('model', 'MLP')}")
        if cond_type.lower() == "FullPhaseSpace".lower():
            logging.info("Classifier sees the whole phase space")
            model: pl.LightningModule = eval(classifier_params["model_params"]["model"])(
                input_dim=data_module.input_dim + 2, 
                target_dim=1, 
                parameters=classifier_params["model_params"],
                lr_warmup_steps=lr_warmup_steps
            )
        elif cond_type.lower() == "NoPhaseSpace".lower():
            logging.info("Classifier does not see the phase space")
            parameters=classifier_params["model_params"]
            parameters["drop"] = 0.5
            parameters["hidden_dim"] = parameters["hidden_dim"] // 2
            model: pl.LightningModule = eval(classifier_params["model_params"]["model"])(
                input_dim=2, 
                target_dim=1, 
                parameters=parameters,
                lr_warmup_steps=lr_warmup_steps
            )
        else:
            raise NotImplementedError(f"Conditioning type {cond_type} is not implemented")
        
        model.apply(lambda m: init_weights(m, classifier_params["model_params"].get("init_scale", 0.1)))

        if classifier_params.get("compile", False):
            # I think we need this to use TF32
            opt_model: pl.LightningModule = torch.compile(model)
        else:
            opt_model: pl.LightningModule = model
    elif run_type == "plot_classifier":
        # Load trained model from checkpoint
        checkpoint_dir = os.path.join(run_dir, "classifier", run_dir, "checkpoints")
        checkpoints = os.listdir(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
        logging.info(f"Loading model checkpoint from {checkpoint_path}")
        model = eval(classifier_params["model_params"]["model"]).load_from_checkpoint(checkpoint_path)
        if classifier_params.get("compile", False):
            model = torch.compile(model)
        opt_model = None

    return model, opt_model

def train_classifier(opt_model, trainer, data_module: DataModule_Single, ckpt_callback, params, run_dir) -> pl.LightningModule:
    opt_model.train()

    logging.info(f"Training classifier for {params.get('epochs', 100)} epochs")
    trainer.fit(opt_model, data_module)
    # opt_model.load_from_checkpoint(ckpt_callback.best_model_path)
    logging.info(f"Loading classifier checkpoint from {ckpt_callback.best_model_path}")
    model = eval(params["classifier"]["model_params"]["model"]).load_from_checkpoint(ckpt_callback.best_model_path)
    if params.get("compile", False):
        model = torch.compile(model)
    return model

@ torch.inference_mode()
def get_classifier_data_module(data_module, samples: SampleEvaluation, classifier_params, target_type, sample_type, cond_type) -> pl.LightningDataModule:
    # target_type: response, jet
    # sample_type: mc, mean, mode
    # cond_type: FullPhaseSpace, NoPhaseSpace

    if target_type.lower() == "Response".lower():
        if sample_type.lower() == "MC".lower():
            sample_targets = torch.from_numpy(samples.log_response_samples)
        elif sample_type.lower() == "mean".lower():
            sample_targets = torch.from_numpy(samples.log_response_mean).unsqueeze(-1)
        elif sample_type.lower() == "mode".lower():
            sample_targets = torch.from_numpy(samples.log_response_mode).unsqueeze(-1)
        else:
            raise NotImplementedError(f"Classifier sample type {sample_type} is not implemented for target type {target_type}.")
        sample_targets = ((sample_targets.permute(0, 2, 1) - data_module.target_preprocessor.mean) / data_module.target_preprocessor.std).permute(0, 2, 1)

        test_targets = data_module.test_dataset.tensors[1]

    elif target_type.lower() == "Jet".lower():
        test_inputs = data_module.test_dataset.tensors[0]
        test_inputs = data_module.input_preprocessor.preprocess_backward(test_inputs)[:, :2]

        if sample_type.lower() == "MC".lower():
            sample_responses = torch.from_numpy(samples.response_samples)
            _test_inputs = test_inputs.unsqueeze(-1).expand(-1, -1, sample_responses.size(-1))
            sample_targets = _test_inputs / sample_responses
        elif sample_type.lower() == "mean".lower():
            sample_targets = (test_inputs / torch.from_numpy(samples.response_mean)).unsqueeze(-1)
        elif sample_type.lower() == "mode".lower():
            sample_targets = (test_inputs / torch.from_numpy(samples.response_mode)).unsqueeze(-1)
        elif sample_type.lower() == "jetmean".lower():
            sample_targets = (test_inputs / torch.from_numpy(samples.response_jet_mean)).unsqueeze(-1)
        elif sample_type.lower() == "jetmode".lower():
            # print(f"test_inputs: {test_inputs.shape}")
            # print(f"jet mode samples: {samples.response_jet_mode.shape}")
            sample_targets = (test_inputs / torch.from_numpy(samples.response_jet_mode)).unsqueeze(-1)
        else:
            raise NotImplementedError(f"Classifier sample type {sample_type} is not implemented for target type {target_type}.")
        sample_targets = torch.log10(sample_targets)
        sample_targets = ((sample_targets.permute(0, 2, 1) - data_module.input_preprocessor.mean[:2]) / data_module.input_preprocessor.std[:2]).permute(0, 2, 1)
        
        test_targets = data_module.test_dataset.tensors[1]
        test_targets = data_module.target_preprocessor.preprocess_backward(test_targets)
        test_targets = 10.**test_targets
        test_targets = test_inputs / test_targets
        test_targets = torch.log10(test_targets)
        test_targets = (test_targets - data_module.input_preprocessor.mean[:2]) / data_module.input_preprocessor.std[:2]
    else:
        raise NotImplementedError(f"Classifier target_type {target_type} is not implemented.")
    
    inputs = data_module.test_dataset.tensors[0]

    # logging.info(f"inputs: {inputs[0:5]}")
    # logging.info(f"data_targets: {test_targets[0:5]}")
    # logging.info(f"samples: {sample_targets[0:5]}")
    
    classifier_data_module = Classifier_DataModule(
        data_params=classifier_params["data_params"],
        inputs=inputs,
        data_targets=test_targets,
        samples=sample_targets,
        cond_type=cond_type
    )

    return classifier_data_module

def make_classifier_predictions(model, params, trainer, data_module, run_dir, dir_name) -> dict:
    model.eval()

    # Generate predictions on test set
    logging.info(f"Making predictions on test set")
    test_predictions = trainer.predict(model, data_module.test_dataloader())
    keys = test_predictions[0].keys()
    test_predictions = {key: torch.cat([pred[key] for pred in test_predictions], dim=0) for key in keys}

    torch.save(
        {
            "weight": test_predictions["weight"],
            "class": test_predictions["class"]
        },
        os.path.join(run_dir, "classifier", dir_name, "weights.pt")
    )

    # if params["save_predictions"]:
    #     logging.info(f"Saving predictions")
    #     torch.save(test_predictions, os.path.join(run_dir, "classifier", dir_name, "test_predictions.pt"))

    return test_predictions

def try_load_classifier_predictions(model, run_dir, dir_name, params, trainer, data_module) -> dict:
    model.eval()

    # Load or generate predictions
    logging.info(f"Loading predictions")
    try:
        test_predictions_loaded = torch.load(os.path.join(run_dir, "classifier", dir_name, "test_predictions.pt"), weights_only=False)
        test_predictions = {key: test_predictions_loaded[key] for key in test_predictions_loaded.keys()}
    except FileNotFoundError:
        logging.info(f"No predictions found in {run_dir}, making predictions on test set")
        test_predictions = make_classifier_predictions(model, params, trainer, data_module, run_dir, dir_name)

    return test_predictions

@torch.inference_mode()
def plot_classifier(run_type, model, predictions, run_dir, dir_name, params):
    logging.info(f"Making classifier plot")

    plot_dir = os.path.join(run_dir, "classifier_plots", dir_name)
    Path(plot_dir).mkdir(exist_ok=True, parents=True)

    if run_type == "classifier":
        plt.figure()
        plt.plot(model.train_epoch_losses, label="Training loss")
        plt.plot(model.val_epoch_losses, label="Validation loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss history")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "loss.pdf"), format='pdf', bbox_inches="tight")
        plt.close()

    weights = predictions["weight"]
    classes = predictions["class"]

    data_predictions = weights[classes == 1]
    gen_predictions = weights[classes == 0]

    logging.info(f"Predicted weights: {weights.shape}, data prediction: {data_predictions.shape}, generated predictions: {gen_predictions.shape}")

    data_predictions = data_predictions.numpy()
    gen_predictions = gen_predictions.numpy()

    pred_range = compute_range([data_predictions, gen_predictions], quantile=0.00001)

    with PdfPages(os.path.join(plot_dir, "score.pdf")) as pdf:
        nbins = 100
        fig1, ax = plt.subplots()
        ax.hist(data_predictions, bins=nbins, range=pred_range, density=True, histtype='step', label="Data")
        ax.hist(gen_predictions, bins=nbins, range=pred_range, density=True, histtype='step', label="Generated")
        ax.legend()
        ax.set_xlabel("Score", loc="left")
        ax.set_ylabel('Density')

        plt.tight_layout()
        pdf.savefig(fig1)

        nbins = 100
        fig2, ax = plt.subplots()
        ax.hist(data_predictions, bins=nbins, range=pred_range, density=True, histtype='step', label="Data")
        ax.hist(gen_predictions, bins=nbins, range=pred_range, density=True, histtype='step', label="Generated")
        ax.legend()
        ax.set_xlabel("Score", loc="left")
        ax.set_ylabel('Density')
        ax.set_yscale('log')

        plt.tight_layout()
        pdf.savefig(fig2)

        plt.close(fig1)
        plt.close(fig2)

    data_weights = data_predictions / (1. - data_predictions + 1.e-5)
    gen_weights = gen_predictions / (1. - gen_predictions + 1.e-5)

    weight_range = compute_range([data_weights, gen_weights], quantile=0.00001)

    with PdfPages(os.path.join(plot_dir, "weight.pdf")) as pdf:
        nbins = 100
        bins = np.linspace(weight_range[0], weight_range[1], nbins+1)
        fig1, ax = plt.subplots()
        ax.hist(data_weights, bins=bins, range=weight_range, density=True, histtype='step', label="Data")
        ax.hist(gen_weights, bins=bins, range=weight_range, density=True, histtype='step', label="Generated")
        ax.legend()
        ax.set_xlabel("Weight", loc="left")
        ax.set_ylabel('Density')
            
        plt.tight_layout()
        pdf.savefig(fig1)

        nbins = 100
        bins = np.logspace(np.log10(weight_range[0]), np.log10(weight_range[1]), nbins+1)
        fig2, ax = plt.subplots()
        ax.hist(data_weights, bins=bins, range=weight_range, density=True, histtype='step', label="Data")
        ax.hist(gen_weights, bins=bins, range=weight_range, density=True, histtype='step', label="Generated")
        ax.legend()
        ax.set_xlabel("Weight", loc="left")
        ax.set_ylabel('Density')
        ax.set_yscale('log')
        # ax.set_xscale('log')
            
        plt.tight_layout()
        pdf.savefig(fig2)

        plt.close(fig1)
        plt.close(fig2)

    weights_np = weights.numpy()
    classes_np = classes.numpy()

    fpr, tpr, _ = roc_curve(classes_np, weights_np)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f"ROC curve (AUC = {roc_auc:.10f})")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "roc.pdf"))

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

    run_type = args.type

    params = get_params(args)
    run_dir = setup_run_dir(args, params)
    setup_logging(run_type, run_dir)
    setup_tf32(params)

    if params["model_params"]["model"] in ["MLP_CFM"]:
        params["data_params"]["loader_params"]["cfm"] = True
    
    data_module = get_data_module(params)
    trainer, ckpt_callback = get_trainer(run_type, params, run_dir)
    model, opt_model = get_model(run_type, params, data_module, run_dir)

    if run_type == "train":
        model = train(opt_model, trainer, data_module, ckpt_callback, params, run_dir)
        test_predictions = make_predictions(model, params, trainer, data_module, run_dir)
    else:
        test_predictions = try_load_predictions(model, run_dir, params, trainer, data_module)

    samples, log_likelihoods = sample(test_predictions, params, model)
    eval_samples = SampleEvaluation(samples.numpy(), log_likelihoods.numpy())

    if run_type in ["train", "plot"]:
        plot(eval_samples, test_predictions, data_module, run_dir, params, run_type, model)

    if run_type in ["classifier", "plot_classifier"]:
        for target_type in ["Response", "Jet"]:
        # for target_type in ["Jet"]:
            # for sample_type in ["MC", "Mean", "Mode"]:
            for sample_type in ["MC"]:
            # for sample_type in ["MC", "JetMode"]:
            # for sample_type in ["JetMode"]:
                for cond_type in ["FullPhaseSpace", "NoPhaseSpace"]:
                    logging.info(f"Using classifier with target type: {target_type}, sample type: {sample_type} and conditioning type: {cond_type}.")
                    classifier_data_module = get_classifier_data_module(
                        data_module,
                        eval_samples,
                        params["classifier"],
                        target_type,
                        sample_type,
                        cond_type
                    )
                    dir_name = f"{target_type}_{sample_type}_{cond_type}"
                    Path(os.path.join(run_dir, "classifier", dir_name)).mkdir(exist_ok=True, parents=True)
                    classifier_trainer, classifier_ckpt_callback = get_classifier_trainer(run_type, params["classifier"], run_dir, dir_name)
                    classifier, opt_classifier = get_classifier(run_type, params["classifier"], data_module, run_dir, cond_type)
                    if run_type == "classifier":
                        classifier = train(
                            opt_classifier,
                            classifier_trainer,
                            classifier_data_module,
                            classifier_ckpt_callback,
                            params["classifier"],
                            os.path.join(run_dir, "classifier", dir_name)
                            )
                        classifier_test_predictions = make_classifier_predictions(classifier, params["classifier"], classifier_trainer, classifier_data_module, run_dir, dir_name)
                    elif run_type == "plot_classifier":
                        classifier_test_predictions = try_load_classifier_predictions(classifier, run_dir, dir_name, params["classifier"], classifier_trainer, classifier_data_module)
                
                    plot_classifier(run_type, classifier, classifier_test_predictions, run_dir, dir_name, params["classifier"])


    logging.info(f"Done")

if __name__ == "__main__":
    main()
