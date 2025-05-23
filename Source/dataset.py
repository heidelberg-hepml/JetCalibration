"""
PyTorch Lightning data modules and preprocessors for jet calibration data.

This module contains classes for loading and preprocessing jet calibration data:

- DataModule_Single: Loads data from multiple .npy files
- InputPreprocessor: Handles preprocessing of input features
- TargetPreprocessor: Handles preprocessing of target variables

The data modules handle:
- Loading data from files
- Splitting into train/val/test sets
- Preprocessing features and targets
- Creating PyTorch DataLoaders
"""

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import numpy as np
import os
from torch.utils.data import random_split, TensorDataset, Dataset
import logging


class DataModule_Single(pl.LightningDataModule):
    """
    Data module for loading jet calibration data from multiple .npy files.
    
    Handles loading data, preprocessing, and creating train/val/test splits.

    Args:
        data_params (dict): Dictionary containing data configuration parameters:
            - data_folder: Path to folder containing .npy data files
            - n_files_train: Number of files to use for training
            - n_files_test: Number of files to use for testing
            - n_data: Optional limit on number of samples to use
            - target_dims: Indices of target variables
            - input_preprocessor: Config for input preprocessing
            - target_preprocessor: Config for target preprocessing
            - loader_params: Parameters for DataLoader
    """
    def __init__(self, data_params):
        super().__init__()
        self.data_params = data_params
        self.loader_params = data_params["loader_params"]

        data_folder = data_params["data_folder"]
        n_files_train = data_params.get("n_files_train", 1)
        n_files_test = data_params.get("n_files_test", 1)
        n_data = data_params.get("n_data", None)

        # Get sorted list of .npy files
        files = os.listdir(data_folder)
        files = [file for file in files if file.endswith(".npy") and "full_data" not in file]
        files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        data_paths = [os.path.join(data_folder, file) for file in files]

        train_files = data_paths[:n_files_train]
        test_files = data_paths[-n_files_test:]

        # Load training data
        targets_train = []
        inputs_train = []
        for i in range(n_files_train):
            data_i = np.load(train_files[i])

            # Remove samples with NaN or inf values
            nan_mask = np.isnan(data_i).any(axis=1)
            inf_mask = np.isinf(data_i).any(axis=1)
            data_i = data_i[~nan_mask & ~inf_mask]

            targets_train.append(data_i[:, data_params["target_dims"]])
            inputs_train.append(data_i[:, 2:])
            
        targets_train = np.concatenate(targets_train, axis=0)
        inputs_train = np.concatenate(inputs_train, axis=0)

        # Load test data
        targets_test = []
        inputs_test = []
        for i in range(n_files_test):
            data_i = np.load(test_files[i])

            nan_mask = np.isnan(data_i).any(axis=1)
            inf_mask = np.isinf(data_i).any(axis=1)
            data_i = data_i[~nan_mask & ~inf_mask]

            targets_test.append(data_i[:, data_params["target_dims"]])
            inputs_test.append(data_i[:, 2:])

        targets_test = np.concatenate(targets_test, axis=0)
        inputs_test = np.concatenate(inputs_test, axis=0)

        # Convert to PyTorch tensors
        targets_train = torch.from_numpy(targets_train)
        inputs_train = torch.from_numpy(inputs_train)
        targets_test = torch.from_numpy(targets_test)
        inputs_test = torch.from_numpy(inputs_test)
        
        # Initialize preprocessors
        self.input_preprocessor = InputPreprocessor(data_params["input_preprocessor"])
        self.target_preprocessor = TargetPreprocessor(data_params["target_preprocessor"])

        # Preprocess data
        inputs_train = self.input_preprocessor.preprocess_forward(inputs_train)
        targets_train = self.target_preprocessor.preprocess_forward(targets_train)

        inputs_test = self.input_preprocessor.preprocess_forward(inputs_test)
        targets_test = self.target_preprocessor.preprocess_forward(targets_test)

        self.input_dim = inputs_train.shape[1]
        self.target_dim = len(data_params["target_dims"])

        # Create train/val split
        train_val_split = int(0.9 * len(inputs_train))
        logging.info(f"Train size: {train_val_split:,}, Val size: {len(inputs_train) - train_val_split:,}, Test size: {len(inputs_test):,}")

        self.train_dataset = TensorDataset(inputs_train[:train_val_split], targets_train[:train_val_split])
        self.val_dataset = TensorDataset(inputs_train[train_val_split:], targets_train[train_val_split:])
        self.test_dataset = TensorDataset(inputs_test, targets_test)
        # self.test_dataset = self.train_dataset

    def train_dataloader(self):
        """Creates DataLoader for training data"""
        batch_size = self.loader_params["batch_size"]
        num_workers = self.loader_params["num_workers"]
        pin_memory = self.loader_params["pin_memory"]
        shuffle = self.loader_params["shuffle"]
        persistent_workers = self.loader_params["persistent_workers"]
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

    def val_dataloader(self):
        """Creates DataLoader for validation data"""
        batch_size = self.loader_params["batch_size"]
        num_workers = self.loader_params["num_workers"]
        pin_memory = self.loader_params["pin_memory"]
        persistent_workers = self.loader_params["persistent_workers"]
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    
    def test_dataloader(self):
        """Creates DataLoader for test data"""
        batch_size = self.loader_params["batch_size"]
        num_workers = self.loader_params["num_workers"]
        pin_memory = self.loader_params["pin_memory"]
        persistent_workers = self.loader_params["persistent_workers"]
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    

class TargetPreprocessor():
    """
    Preprocessor for target variables.
    
    Standardizes target variables by subtracting mean and dividing by standard deviation.

    Args:
        params (dict): Configuration parameters (currently unused)
    """
    def __init__(self, params):
        self.params = params
        self.mean = None
        self.std = None

    def preprocess_forward(self, data_in):
        """Standardize data by subtracting mean and dividing by std"""
        if self.mean is None:
            self.mean = data_in.mean(0)
            self.std = data_in.std(0)
        return  (data_in - self.mean) / self.std

    def preprocess_backward(self, data_in):
        """Reverse standardization by multiplying by std and adding mean"""
        if self.mean is None or self.std is None:
            raise ValueError("Mean and std must be set before calling preprocess_backward")
        return data_in * self.std + self.mean
        
        
class InputPreprocessor():
    """
    Preprocessor for input features.
    
    Applies log transformation to specified dimensions then standardizes all features.

    Args:
        params (dict): Configuration parameters:
            - log_dims: List of indices for dimensions to log transform
    """
    def __init__(self, params):
        self.params = params
        self.mean = None
        self.std = None

    def preprocess_forward(self, data_in):
        """Apply log transform and standardization"""
        data = data_in.clone()

        # Apply log transform to specified dimensions
        log_dims = self.params["log_dims"]
        data[:, log_dims] = torch.log10(data[:, log_dims]+1.)

        # Compute mean/std if not already set
        if self.mean is None:
            self.mean = data.mean(0)
            self.std = data.std(0)

        # Standardize
        data = (data - self.mean) / self.std

        return data.squeeze()   
        
    def preprocess_backward(self, data_in):
        """Reverse standardization and log transform"""
        data = data_in.clone()

        if self.mean is None or self.std is None:
            raise ValueError("Mean and std must be set before calling preprocess_backward")
            
        # Reverse standardization
        data = data * self.std + self.mean

        # Reverse log transform
        log_dims = self.params["log_dims"]
        data[:, log_dims] = 10**(data[:, log_dims])-1.

        return data.squeeze()
