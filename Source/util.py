"""
This module provides callbacks for logging training progress.

The PrintingCallback class logs training metrics and timing information during model training.
It inherits from PyTorch Lightning's Callback class to hook into the training lifecycle.
"""

from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule
import time
import logging


class PrintingCallback(Callback):
    """
    Callback that logs training progress metrics and timing information.
    
    Logs the start/end of training, epoch timing, and loss metrics at the end of each epoch.
    
    Attributes:
        start_time (float): Time when current epoch started
        training_end_time (float): Time when training phase of epoch ended  
        training_losses (list): History of training losses
        validation_losses (list): History of validation losses
    """

    def __init__(self):
        """Initialize timing and loss tracking attributes."""
        self.start_time = None
        self.training_end_time = None 
        self.training_losses = []
        self.validation_losses = []

    def on_train_start(self, trainer, pl_module):
        """Log when training starts."""
        logging.info("Training is starting")

    def on_train_end(self, trainer, pl_module):
        """Log when training ends."""
        logging.info("Training is ending")

    def on_train_epoch_start(self, trainer, pl_module):
        """Record start time of epoch."""
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        """Record end time of training phase."""
        self.training_end_time = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Log epoch timing and loss metrics.
        
        Args:
            trainer: PyTorch Lightning trainer instance
            pl_module: The Lightning Module being trained
            
        Logs epoch number, duration, training loss and validation loss.
        Skips logging for epoch 0.
        """
        # if trainer.current_epoch == 0:
        #     return
        end_time = time.time()
        logging.info(f"Finished epoch {trainer.current_epoch} in {round((end_time - self.start_time), 1)} seconds. Training loss: {trainer.callback_metrics['train_loss']:.5f}, Validation loss: {trainer.callback_metrics['val_loss']:.5f}")
        # logging.info(f"Finished epoch {trainer.current_epoch} in {round((end_time - self.start_time), 1)} seconds.")
