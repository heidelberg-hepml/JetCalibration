from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import numpy as np
import os
from torch.utils.data import random_split, TensorDataset, Dataset
import logging


class DataModule_Single(pl.LightningDataModule):
    def __init__(self, data_params):
        super().__init__()
        self.data_params = data_params
        self.loader_params = data_params["loader_params"]

        data_folder = data_params["data_folder"]
        n_files_train = data_params.get("n_files_train", 1)
        n_files_test = data_params.get("n_files_test", 1)
        n_data = data_params.get("n_data", None)

        files = os.listdir(data_folder)
        files = [file for file in files if file.endswith(".npy") and "full_data" not in file]
        files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        data_paths = [os.path.join(data_folder, file) for file in files]

        train_files = data_paths[:n_files_train]
        test_files = data_paths[-n_files_test:]

        targets_train = []
        inputs_train = []
        for i in range(n_files_train):
            data_i = np.load(train_files[i])

            nan_mask = np.isnan(data_i).any(axis=1)
            inf_mask = np.isinf(data_i).any(axis=1)
            data_i = data_i[~nan_mask & ~inf_mask]

            targets_train.append(data_i[:, data_params["target_dims"]])
            inputs_train.append(data_i[:, 2:])
            
        targets_train = np.concatenate(targets_train, axis=0)
        inputs_train = np.concatenate(inputs_train, axis=0)

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

        targets_train = torch.from_numpy(targets_train)
        inputs_train = torch.from_numpy(inputs_train)
        targets_test = torch.from_numpy(targets_test)
        inputs_test = torch.from_numpy(inputs_test)
        
        self.input_preprocessor = InputPreprocessor(data_params["input_preprocessor"])
        self.target_preprocessor = TargetPreprocessor(data_params["target_preprocessor"])

        inputs_train = self.input_preprocessor.preprocess_forward(inputs_train)
        targets_train = self.target_preprocessor.preprocess_forward(targets_train)

        inputs_test = self.input_preprocessor.preprocess_forward(inputs_test)
        targets_test = self.target_preprocessor.preprocess_forward(targets_test)

        self.input_dim = inputs_train.shape[1]
        self.target_dim = len(data_params["target_dims"])

        train_val_split = int(0.9 * len(inputs_train))
        logging.info(f"Train size: {train_val_split}, Val size: {len(inputs_train) - train_val_split}, Test size: {len(inputs_test)}")

        self.train_dataset = TensorDataset(inputs_train[:train_val_split], targets_train[:train_val_split])
        self.val_dataset = TensorDataset(inputs_train[train_val_split:], targets_train[train_val_split:])
        self.test_dataset = TensorDataset(inputs_test, targets_test)


    def train_dataloader(self):
        batch_size = self.loader_params["batch_size"]
        num_workers = self.loader_params["num_workers"]
        pin_memory = self.loader_params["pin_memory"]
        shuffle = self.loader_params["shuffle"]
        persistent_workers = self.loader_params["persistent_workers"]
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

    def val_dataloader(self):
        batch_size = self.loader_params["batch_size"]
        num_workers = self.loader_params["num_workers"]
        pin_memory = self.loader_params["pin_memory"]
        persistent_workers = self.loader_params["persistent_workers"]
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    
    def test_dataloader(self):  
        batch_size = self.loader_params["batch_size"]
        num_workers = self.loader_params["num_workers"]
        pin_memory = self.loader_params["pin_memory"]
        persistent_workers = self.loader_params["persistent_workers"]
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    

class TargetPreprocessor():
    def __init__(self, params):
        self.params = params

        self.mean = None
        self.std = None

    def preprocess_forward(self, data_in):
        if self.mean is None:
            self.mean = data_in.mean(0)
            self.std = data_in.std(0)
        return  (data_in - self.mean) / self.std

    def preprocess_backward(self, data_in):
        if self.mean is None or self.std is None:
            raise ValueError("Mean and std must be set before calling preprocess_backward")
        return data_in * self.std + self.mean
        
        
class InputPreprocessor():
    def __init__(self, params):
        self.params = params

        self.mean = None
        self.std = None

    def preprocess_forward(self, data_in):
        data = data_in.clone()

        log_dims = self.params["log_dims"]
        data[:, log_dims] = torch.log10(data[:, log_dims]+1.)

        if self.mean is None:
            self.mean = data.mean(0)
            self.std = data.std(0)

        data = (data - self.mean) / self.std

        return data.squeeze()   
        
    def preprocess_backward(self, data_in):
        data = data_in.clone()

        if self.mean is None or self.std is None:
            raise ValueError("Mean and std must be set before calling preprocess_backward")
        data = data * self.std + self.mean

        log_dims = self.params["log_dims"]
        data[:, log_dims] = 10**(data[:, log_dims])-1.

        return data.squeeze()


class DataModule_Full(pl.LightningDataModule):
    def __init__(self, data_params):
        super().__init__()
        self.data_params = data_params

        data_folder = data_params["data_folder"]
        data_file = os.path.join(data_folder, "full_data_preprocessed.npy")
        #data_file = os.path.join(data_folder, "Ak10Jet_9.npy")
        n_data = data_params.get("n_data", None)
        if n_data is not None:
            data = np.load(data_file)[:n_data]
        else:
            data = np.load(data_file)

        
        loaded_params = np.load("/remote/gpu07/huetsch/JetCalibration/data_v2/full_data_mean_std.npz")
        mean = loaded_params["mean"]
        std = loaded_params["std"]
        self.input_preprocessor = InputPreprocessor(data_params["input_preprocessor"])
        self.target_preprocessor = TargetPreprocessor(data_params["target_preprocessor"])
        self.input_preprocessor.mean = torch.from_numpy(mean[2:])
        self.input_preprocessor.std = torch.from_numpy(std[2:])
        self.target_preprocessor.mean = torch.from_numpy(mean[self.data_params["target_dims"]])
        self.target_preprocessor.std = torch.from_numpy(std[self.data_params["target_dims"]])

        self.input_dim = 21
        self.target_dim = len(data_params["target_dims"])

        val_split = data_params["val_split"]
        test_split = data_params["test_split"]
        train_split = 1 - val_split - test_split

        train_size = int(train_split * len(data))
        val_size = int(val_split * len(data))
        test_size = int(test_split * len(data))

        logging.info(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

        self.train_dataset = TensorDataset(torch.from_numpy(data[:train_size, 2:]), torch.from_numpy(data[:train_size, self.data_params["target_dims"]]))
        self.val_dataset = TensorDataset(torch.from_numpy(data[train_size:train_size+val_size, 2:]), torch.from_numpy(data[train_size:train_size+val_size, self.data_params["target_dims"]]))
        self.test_dataset = TensorDataset(torch.from_numpy(data[train_size+val_size:, 2:]), torch.from_numpy(data[train_size+val_size:, self.data_params["target_dims"]]))

    def train_dataloader(self):
        batch_size = self.data_params["loader_params"]["batch_size"]
        num_workers = self.data_params["loader_params"]["num_workers"]
        pin_memory = self.data_params["loader_params"]["pin_memory"]
        shuffle = self.data_params["loader_params"]["shuffle"]
        persistent_workers = self.data_params["loader_params"]["persistent_workers"]
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

    def val_dataloader(self):
        batch_size = self.data_params["loader_params"]["batch_size"]
        num_workers = self.data_params["loader_params"]["num_workers"]
        pin_memory = self.data_params["loader_params"]["pin_memory"]
        persistent_workers = self.data_params["loader_params"]["persistent_workers"]
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    
    def test_dataloader(self):  
        batch_size = self.data_params["loader_params"]["batch_size"]
        num_workers = self.data_params["loader_params"]["num_workers"]
        pin_memory = self.data_params["loader_params"]["pin_memory"]
        persistent_workers = self.data_params["loader_params"]["persistent_workers"]
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    
