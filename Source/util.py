"""
This module provides callbacks for logging training progress.

The PrintingCallback class logs training metrics and timing information during model training.
It inherits from PyTorch Lightning's Callback class to hook into the training lifecycle.
"""

from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule
import time
import logging

import torch.nn as nn

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
        logging.info(f"Finished epoch {trainer.current_epoch} in {round((end_time - self.start_time), 1)} seconds") #. Training loss: {pl_module.train_epoch_losses[-1]:.10f}, Validation loss: {pl_module.val_epoch_losses[-1]:.10f}")
        # try:
        #   logging.info(f"Finished epoch {trainer.current_epoch} in {round((end_time - self.start_time), 1)} seconds. Training loss: {trainer.callback_metrics['train_loss']:.10f}, Validation loss: {trainer.callback_metrics['val_loss']:.10f}")
        # except:
        #    logging.info(f"Finished epoch {trainer.current_epoch} in {round((end_time - self.start_time), 1)} seconds.")
        # logging.info(f"Finished epoch {trainer.current_epoch} in {round((end_time - self.start_time), 1)} seconds.")

def init_weights(module, init_scale):
  if isinstance(module, nn.Linear):
    # nn.init.normal_(module.weight, mean=0.0, std=init_scale)
    nn.init.xavier_normal_(module.weight, gain=init_scale)
    if module.bias is not None:
      nn.init.zeros_(module.bias)
  elif isinstance(module, nn.Embedding):
    nn.init.normal_(module.weight, mean=0.0, std=init_scale)
  elif isinstance(module, nn.MultiheadAttention):
    module._reset_parameters()
  elif (
    isinstance(module, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm))
    or "LayerNorm" in module.__class__.__name__
    or "RMSNorm" in module.__class__.__name__
  ):
    if hasattr(module, "weight") and module.weight is not None:
        module.weight.data.fill_(1.0)
    if hasattr(module, "bias") and module.bias is not None:
        module.bias.data.zero_()
