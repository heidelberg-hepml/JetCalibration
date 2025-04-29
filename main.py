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

from Source.dataset import DataModule_Single, DataModule_Full
from Source.model import *
from Source.util import PrintingCallback
from Source.plots import Plotter


torch.set_float32_matmul_precision('medium')
logging.getLogger("lightning_fabric.plugins.environments.slurm").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)


def main():

    parser = argparse.ArgumentParser()
    # type is either train or plot
    parser.add_argument('type')
    # if type is train, path is the path to the parameter card
    # if type is plot, path is the path to the saved run directory
    parser.add_argument('path')
    args = parser.parse_args()


    if args.type == "train":
        # read in the parameters
        with open(args.path, 'r') as f:
            params = yaml.safe_load(f)

        # create a results dir and save parameters to it
        dir_path = os.path.dirname(os.path.realpath(__file__))
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + params["run_name"]
        run_dir = os.path.join(dir_path, "results", run_name)
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "params.yaml"), 'w') as f:
            yaml.dump(params, f)

        # define log file path
        log_file = os.path.join(run_dir, "train_log.txt")

    elif args.type == "plot":
        # read in saved run directory and parameters
        run_dir = args.path
        with open(os.path.join(run_dir, "params.yaml"), 'r') as f:
            params = yaml.safe_load(f)

        # define log file path
        log_file = os.path.join(run_dir, "plot_log.txt")

    else:
        raise NotImplementedError(f"type {args.type} not recognised")
    

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # Log to a file
            logging.StreamHandler()          # Log to the console
        ]
    )
    def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
        logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = log_uncaught_exceptions
    logging.info("\n")
    logging.info(f"Starting {args.type} run with run_dir {run_dir}")

    # load data
    logging.info(f"Loading data from {params['data_params']['data_folder']}")
    data_module_type = params['data_params'].get("data_module_type", "single")
    if data_module_type == "single":
        data_module = DataModule_Single(params["data_params"])
    elif data_module_type == "full":
        data_module = DataModule_Full(params["data_params"])
    else:
        raise ValueError(f"data_module_type {data_module_type} not recognised")

    logging.info(f"Creating trainer")
    printing_callback = PrintingCallback()
    trainer = pl.Trainer(
        max_epochs=params.get("epochs", 100),
        accelerator="gpu", 
        devices=1, 
        log_every_n_steps=1,
        enable_progress_bar=False,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
            printing_callback,
            pl.callbacks.EarlyStopping(
                monitor="val_loss",   # metric name to monitor (logged in validation_epoch_end, for example)
                patience=50,           # how many epochs with no improvement before stopping
                verbose=True,         # prints a message when stopping
                mode="min"            # "min" if lower metric is better; "max" if higher is better
            )
        ],
        logger=pl.loggers.TensorBoardLogger(save_dir=run_dir),
        enable_model_summary=False,
        num_sanity_val_steps=0,
        detect_anomaly=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )


    if args.type == "train":
        logging.info(f"Creating model {params['model_params'].get('model', 'MLP_MSE_Regression')}")
        model = eval(params["model_params"]["model"])(input_dim=data_module.input_dim, target_dim=data_module.target_dim, parameters=params["model_params"])

        logging.info(f"Training model for {params.get('epochs', 100)} epochs")
        trainer.fit(model, data_module)

        logging.info(f"Making predictions on test set")
        test_predictions = trainer.predict(model, data_module.test_dataloader())
        keys = test_predictions[0].keys()
        test_predictions = {key: torch.cat([pred[key] for pred in test_predictions], dim=0) for key in keys}

        if params["save_predictions"]:
            logging.info(f"Saving predictions")
            test_predictions["preprocess_mean"] = data_module.target_preprocessor.mean
            test_predictions["preprocess_std"] = data_module.target_preprocessor.std
            torch.save(test_predictions, os.path.join(run_dir, "test_predictions.pt"))

    # if plotting, plot the model
    elif args.type == "plot":
        checkpoint_dir = os.path.join(run_dir, "lightning_logs/version_0/checkpoints")
        checkpoints = os.listdir(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
        model = eval(params["model_params"]["model"]).load_from_checkpoint(checkpoint_path)

        logging.info(f"Loading predictions")
        try:
            test_predictions_loaded = torch.load(os.path.join(run_dir, "test_predictions.pt"), weights_only=False)
            test_predictions = {key: test_predictions_loaded[key] for key in test_predictions_loaded.keys()}
        except FileNotFoundError:
            logging.info(f"No predictions found in {run_dir}, making predictions on test set")
            test_predictions = trainer.predict(model, data_module.test_dataloader())
            keys = test_predictions[0].keys()
            test_predictions = {key: torch.cat([pred[key] for pred in test_predictions], dim=0) for key in keys}

            if params["save_predictions"]:
                logging.info(f"Saving predictions")
                test_predictions["preprocess_mean"] = data_module.target_preprocessor.mean
                test_predictions["preprocess_std"] = data_module.target_preprocessor.std
                torch.save(test_predictions, os.path.join(run_dir, "test_predictions.pt"))

    logging.info(f"Sampling")
    if params["model_params"]["model"] == "MLP_GMM_Regression":
        if len(params["data_params"]["target_dims"]) == 1:
            samples, log_likelihoods = model.sample(test_predictions["mu"], test_predictions["sigma"], test_predictions["weights"], n_samples=params.get("n_samples", 10))
            samples = samples*test_predictions["preprocess_std"] + test_predictions["preprocess_mean"]
            log_likelihoods = log_likelihoods/test_predictions["preprocess_std"]
        else:
            samples_E, log_likelihoods_E = model.sample(test_predictions["mu_E"], test_predictions["sigma_E"], test_predictions["weights_E"], n_samples=params.get("n_samples", 10))
            samples_E = samples_E*test_predictions["preprocess_std"][0] + test_predictions["preprocess_mean"][0]
            log_likelihoods_E = log_likelihoods_E/test_predictions["preprocess_std"][0]
            samples_m, log_likelihoods_m = model.sample(test_predictions["mu_m"], test_predictions["sigma_m"], test_predictions["weights_m"], n_samples=params.get("n_samples", 10))
            samples_m = samples_m*test_predictions["preprocess_std"][1] + test_predictions["preprocess_mean"][1]
            log_likelihoods_m = log_likelihoods_m/test_predictions["preprocess_std"][1]
            samples = torch.stack([samples_E, samples_m], dim=1)
            log_likelihoods = torch.stack([log_likelihoods_E, log_likelihoods_m], dim=1)
    else:
        if len(params["data_params"]["target_dims"]) == 1:
            samples, log_likelihoods = model.sample(test_predictions["mu"], test_predictions["sigma"], n_samples=params.get("n_samples", 10))
            samples = samples*test_predictions["preprocess_std"] + test_predictions["preprocess_mean"]
            log_likelihoods = log_likelihoods/test_predictions["preprocess_std"]
        else:
            samples_E, log_likelihoods_E = model.sample(test_predictions["mu_E"], test_predictions["sigma_E"], n_samples=params.get("n_samples", 10))
            samples_E = samples_E*test_predictions["preprocess_std"][0] + test_predictions["preprocess_mean"][0]
            log_likelihoods_E = log_likelihoods_E/test_predictions["preprocess_std"][0]
            samples_m, log_likelihoods_m = model.sample(test_predictions["mu_m"], test_predictions["sigma_m"], n_samples=params.get("n_samples", 10))
            samples_m = samples_m*test_predictions["preprocess_std"][1] + test_predictions["preprocess_mean"][1]
            log_likelihoods_m = log_likelihoods_m/test_predictions["preprocess_std"][1]
            samples = torch.stack([samples_E, samples_m], dim=1)
            log_likelihoods = torch.stack([log_likelihoods_E, log_likelihoods_m], dim=1)

    logging.info(f"Making Plots")

    test_inputs = data_module.test_dataset.tensors[0]
    test_inputs = data_module.input_preprocessor.preprocess_backward(test_inputs)
    test_targets = data_module.test_dataset.tensors[1]
    test_targets = data_module.target_preprocessor.preprocess_backward(test_targets)

    logging.info(samples.shape)
    logging.info(samples.mean(dim=0))
    logging.info(samples.std(dim=0))
    logging.info(test_targets.shape)
    logging.info(test_targets.mean(dim=0))
    logging.info(test_targets.std(dim=0))

    #raise ValueError("Stop here")

    for target_dim in params["data_params"]["target_dims"]:
        variable = ["E", "m"][target_dim]
        logging.info(f"Plotting variable {variable}")
        os.makedirs(os.path.join(run_dir, f"plots_{variable}"), exist_ok=True)
        plot_dir = os.path.join(run_dir, f"plots_{variable}")
        if len(params["data_params"]["target_dims"]) == 1:
            plotter = Plotter(plot_dir, 
                              params, 
                              test_predictions, 
                              test_inputs, 
                              test_targets, 
                              samples, 
                              log_likelihoods,
                              variable)
        else:
            plotter = Plotter(plot_dir, 
                              params, 
                              {key: test_predictions[key] for key in test_predictions.keys() if variable in key}, 
                              test_inputs, 
                              test_targets[:, target_dim], 
                              samples[:, target_dim], 
                              log_likelihoods[:, target_dim],
                              variable)

        if args.type == "train":
            plotter.plot_loss_history("loss.pdf", model.train_epoch_losses, model.val_epoch_losses)
            #plotter.plot_inputs_histogram("inputs_histogram.pdf")

        plotter.r_predictions_histogram(f"{variable}_r_predictions_histogram.pdf")
        plotter.E_M_predictions_histogram(f"{variable}_predictions_histogram.pdf")

        plotter.r_2d_histogram(f"{variable}_r_2d_histogram.pdf")
        #continue
        plotter.E_M_2d_histogram(f"{variable}_2d_histogram.pdf")
        #plotter.plot_standard_deviations(f"{variable}_standard_deviations.pdf")
        plotter.pred_inputs_histogram(f"{variable}_pred_inputs_histogram.pdf")
        plotter.pred_inputs_histogram_marginalized(f"{variable}_pred_inputs_histogram_marginalized.pdf")

        if params["model_params"]["model"] == "MLP_GMM_Regression":
            plotter.plot_GMM_weights("GMM_weights.pdf")

    logging.info(f"Done")

if __name__ == "__main__":
    main()
