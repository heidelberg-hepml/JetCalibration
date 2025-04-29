from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule
import time
import logging


class PrintingCallback(Callback):
    def __init__(self):
        self.start_time = None
        self.training_end_time = None
        self.training_losses = []
        self.validation_losses = []

    def on_train_start(self, trainer, pl_module):
        logging.info("Training is starting")

    def on_train_end(self, trainer, pl_module):
        logging.info("Training is ending")

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        self.training_end_time = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            return
        end_time = time.time()
        logging.info(f"Finished epoch {trainer.current_epoch} in {round((end_time - self.start_time), 1)} seconds. Training loss: {trainer.callback_metrics['train_loss']:.5f}, Validation loss: {trainer.callback_metrics['val_loss']:.5f}")

