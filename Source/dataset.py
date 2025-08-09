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
import awkward as ak
import uproot as up

from collections.abc import Iterator

import random

import ot

KEYS_TO_KEEP = [
    'ak10_E',
    'ak10_mass',
    'ak10_rap',
    'ak10_groomMRatio',
    'ak10_Width',
    'ak10_Split12',
    'ak10_Split23',
    'ak10_C2',
    'ak10_D2',
    'ak10_Tau21',
    'ak10_Tau32',
    'ak10_Qw',
    'ak10_EMFracCaloBased',
    'ak10_EM3FracCaloBased',
    'ak10_Tile0FracCaloBased',
    'ak10_EffNClustsCaloBased',
    'ak10_NeutralEFrac',
    'ak10_ChargePTFrac',
    'ak10_ChargeMFrac',
    'averageMu',
    'NPV',    
]

ADDITIONAL_KEYS_TO_KEEP = [
    'ak10_true_pt',
    'ak10_true_rap'
]

@torch.no_grad()
def _cfm_collate_fct(examples):
    x, y = torch.utils.data.default_collate(examples)

    noise = torch.randn_like(y)
    t = torch.rand(y.shape[0], 1, device=y.device)

    y_t = (1 - t) * noise + t * y
    y_t_dot = y - noise

    state = torch.cat([t, y_t, x], dim=-1)
        
    return state, y_t_dot

@torch.no_grad()
def _cfm_ot_collate_fct(examples):
    x, y = torch.utils.data.default_collate(examples)

    noise = torch.randn_like(y)
    t = torch.rand(y.shape[0], 1,device=y.device)

    batch_size = x.shape[0]

    cost = torch.cdist(noise, y)
    a, b = ot.unif(batch_size), ot.unif(batch_size)
    p = ot.emd(a, b, cost.cpu().numpy())
    if not np.all(np.isfinite(p)):
        logging.error(
            f"""ERROR: p is not finite"
            p: {p}
            Cost mean {cost.mean()}, cost max {cost.max()}
            noise: {noise}
            y: {y}
            """)
    if np.abs(p.sum()) < 1e-8:
        logging.warning("Numerical errors in OT plan, reverting to uniform plan.")
        p = np.ones_like(p) / p.size

    _p = p.flatten()
    _p = _p / _p.sum()
    choices = np.random.choice(
        p.shape[0] * p.shape[1], p=_p, size=batch_size, replace=True
    )
    i, j = np.divmod(choices, p.shape[1])
    
    noise_ot = noise[i]
    x_ot= x[j]
    y_ot = y[j]

    y_t = (1 - t) * noise_ot + t * y_ot
    y_t_dot = y_ot - noise
        
    state = torch.cat([t, y_t, x], dim=-1)

    return state, y_t_dot

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

        # Get sorted list of .npy files
        files = os.listdir(data_folder)
        files = [file for file in files if file.endswith(".npy") and "full_data" not in file]
        files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        data_paths = [os.path.join(data_folder, file) for file in files]

        assert n_files_train + n_files_test <= len(data_paths), f"You chose to many files. Total number of files: {len(data_paths)}"

        train_files = data_paths[:n_files_train]
        test_files = data_paths[-n_files_test:]

        logging.info(f"Train files: {len(train_files)}, test files: {len(test_files)}")

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
        additional_input_test = []
        for i in range(n_files_test):
            # data_i = np.load(test_files[i])

            # nan_mask = np.isnan(data_i).any(axis=1)
            # inf_mask = np.isinf(data_i).any(axis=1)
            # data_i = data_i[~nan_mask & ~inf_mask]

            # targets_test.append(data_i[:, data_params["target_dims"]])
            # inputs_test.append(data_i[:, 2:])

            test_file: str = test_files[i]
            test_file = test_file.replace(".npy", ".root")
            test_file = up.open(test_file)

            filekeys = test_file.keys()
            filekeys.sort(key=lambda k: int(k.split(";")[-1]), reverse=True)
            latest_key = filekeys[0]

            tree = test_file[latest_key]

            true_E = ak.to_numpy(tree["ak10_true_E"].array())/1000
            rec_E = ak.to_numpy(tree["ak10_E"].array())
            E_ratio = rec_E / true_E

            true_m = ak.to_numpy(tree["ak10_true_mass"].array())/1000
            rec_m = ak.to_numpy(tree["ak10_mass"].array())
            m_ratio = rec_m / true_m

            true_pt = ak.to_numpy(tree["ak10_true_pt"].array())
            rec_pt = ak.to_numpy(tree["ak10_pt"].array())

            true_pt_cut = true_pt > 100
            rec_pt_cut = rec_pt > 100
            true_mass_cut = true_m > 50
            rec_mass_cut = rec_m > 50
            full_cut = true_pt_cut & rec_pt_cut & true_mass_cut & rec_mass_cut

            inputs = np.array([ak.to_numpy(tree[key].array()) for key in KEYS_TO_KEEP])
            data_i = np.concatenate([E_ratio[:, None], m_ratio[:, None], inputs.T], axis=1)
            data_i = data_i[full_cut]
            data_i[:, 0] = np.log10(data_i[:, 0])
            data_i[:, 1] = np.log10(data_i[:, 1])
            additional_input_i = np.array([ak.to_numpy(tree[key].array()) for key in ADDITIONAL_KEYS_TO_KEEP]).T
            for i, key in enumerate(ADDITIONAL_KEYS_TO_KEEP):
                if key == "ak10_true_pt":
                    additional_input_i[i] /= 1000

            additional_input_i = additional_input_i[full_cut]

            nan_mask = np.isnan(data_i).any(axis=1)
            inf_mask = np.isinf(data_i).any(axis=1)
            data_i = data_i[~nan_mask & ~inf_mask]
            additional_input_i = additional_input_i[~nan_mask & ~inf_mask]

            targets_test.append(data_i[:, data_params["target_dims"]])
            inputs_test.append(data_i[:, 2:])
            additional_input_test.append(additional_input_i)

        targets_test = np.concatenate(targets_test, axis=0)
        inputs_test = np.concatenate(inputs_test, axis=0)
        self.additional_input_test = np.concatenate(additional_input_test, axis=0)

        # Convert to PyTorch tensors
        targets_train = torch.from_numpy(targets_train)
        inputs_train = torch.from_numpy(inputs_train)
        targets_test = torch.from_numpy(targets_test)
        inputs_test = torch.from_numpy(inputs_test)
        
        # Initialize preprocessors
        self.input_preprocessor = InputPreprocessor(data_params["input_preprocessor"], skip_dims=data_params.get("skip_dims", []))
        self.target_preprocessor = TargetPreprocessor(data_params["target_preprocessor"])

        _input_dim = inputs_train.shape[1]
        keep_dims = torch.full((_input_dim,), True, dtype=torch.bool)
        keep_dims[data_params.get("skip_dims", [])] = False
        inputs_train = inputs_train[:, keep_dims]
        inputs_test = inputs_test[:, keep_dims]

        self.input_dim = inputs_train.shape[1]
        self.target_dim = len(data_params["target_dims"])

        # Preprocess data
        inputs_train = self.input_preprocessor.preprocess_forward(inputs_train)
        targets_train = self.target_preprocessor.preprocess_forward(targets_train)

        inputs_test = self.input_preprocessor.preprocess_forward(inputs_test)
        targets_test = self.target_preprocessor.preprocess_forward(targets_test)

        # Create train/val split
        train_val_split = int(0.9 * len(inputs_train))
        logging.info(f"Train size: {train_val_split:,}, Val size: {len(inputs_train) - train_val_split:,}, Test size: {len(inputs_test):,}")

        self.train_dataset = TensorDataset(inputs_train[:train_val_split], targets_train[:train_val_split])
        self.val_dataset = TensorDataset(inputs_train[train_val_split:], targets_train[train_val_split:])
        self.test_dataset = TensorDataset(inputs_test, targets_test)

        self.train_batches_per_epoch = len(self.train_dataset) // self.loader_params["batch_size"]

    def train_dataloader(self):
        """Creates DataLoader for training data"""
        batch_size = self.loader_params["batch_size"]
        num_workers = self.loader_params["num_workers"]
        pin_memory = self.loader_params["pin_memory"]
        shuffle = self.loader_params["shuffle"]
        persistent_workers = self.loader_params["persistent_workers"]

        collate_fct = None
        cfm_ot = self.loader_params.get("cfm_ot", False)
        if cfm_ot:
            collate_fct = _cfm_ot_collate_fct
        else:
            cfm = self.loader_params.get("cfm", False)
            if cfm:
                collate_fct = _cfm_collate_fct

        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=True,
            collate_fn=collate_fct
        )

    def val_dataloader(self):
        """Creates DataLoader for validation data"""
        batch_size = self.loader_params["batch_size"]
        num_workers = self.loader_params.get("val_num_worker", self.loader_params["num_workers"])
        pin_memory = self.loader_params["pin_memory"]
        persistent_workers = self.loader_params["persistent_workers"]

        collate_fct = None
        cfm_ot = self.loader_params.get("cfm_ot", False)
        if cfm_ot:
            collate_fct = _cfm_ot_collate_fct
        else:
            cfm = self.loader_params.get("cfm", False)
            if cfm:
                collate_fct = _cfm_collate_fct

        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=True,
            collate_fn=collate_fct
        )
    
    def test_dataloader(self):
        """Creates DataLoader for test data"""
        batch_size = self.loader_params.get("test_batch_size", self.loader_params["batch_size"])
        # num_workers = self.loader_params["num_workers"]
        num_workers = 1
        pin_memory = self.loader_params["pin_memory"]
        persistent_workers = self.loader_params["persistent_workers"]
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False,
            prefetch_factor=1
        )

class DataSampleDataset(Dataset):
    def __init__(self, inputs, data_targets, samples_target, cond_type):
        self.inputs = inputs
        self.data_targets = data_targets
        self.samples_target = samples_target

        self._n_samples = samples_target.shape[-1] - 1

        self.cond_type = cond_type
        if cond_type.lower() == "FullPhaseSpace".lower():
            self.concat_inputs = True
        elif cond_type.lower() == "NoPhaseSpace".lower():
            self.concat_inputs = False
        else:
            raise NotImplementedError(f"Conditioning type {cond_type} is not implemented")


    def __len__(self):
        return 2 * self.inputs.shape[0]
    
    def __getitem__(self, idx):
        if idx < self.inputs.shape[0]:
            if self.concat_inputs:
                return torch.cat([self.data_targets[idx], self.inputs[idx]]), torch.tensor([1.], dtype=torch.float32)
            else:
                return self.data_targets[idx], torch.tensor([1.], dtype=torch.float32)
        else:
            if self.concat_inputs:
                return torch.cat([self.samples_target[idx - self.inputs.shape[0], :, random.randint(0, self._n_samples)], self.inputs[idx - self.inputs.shape[0]]]), torch.tensor([0.], dtype=torch.float32)
            else:
                return self.samples_target[idx - self.inputs.shape[0], :, random.randint(0, self._n_samples)], torch.tensor([0.], dtype=torch.float32)

class Classifier_Sampler(torch.utils.data.Sampler[int]):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return 2 * self.size
    
    def __iter__(self) -> Iterator[int]:
        ind1 = torch.randperm(self.size)
        ind2 = torch.randperm(self.size) + self.size

        for i in range(self.size):
            yield ind1[i]
            yield ind2[i]

class Classifier_DataModule(pl.LightningDataModule):
    def __init__(self, data_params, inputs, data_targets, samples, cond_type):
        super().__init__()
        self.data_params = data_params
        self.loader_params = data_params["loader_params"]

        self.cond_type = cond_type

        # test_dataset:
        #    inputs: [size, phasespace]
        #    targets: [size, targets]
        # samples:  [size, targets, samples]

        n_train = int(inputs.shape[0] * data_params["split_frac"][0])
        n_test = int(inputs.shape[0] * data_params["split_frac"][2])
        n_val = inputs.shape[0] - n_train - n_test

        # perm = torch.randperm(inputs.shape[0])
        # inputs = inputs[perm]
        # data_targets = data_targets[perm]
        # samples = samples[perm]

        train_inputs = inputs[:n_train]
        train_data_targets = data_targets[:n_train]
        train_samples = samples[:n_train]

        eval_inputs = inputs[n_train:n_train+n_val]
        eval_data_targets = data_targets[n_train:n_train+n_val]
        eval_samples = samples[n_train:n_train+n_val]

        test_inputs = inputs[n_train+n_val:]
        test_data_targets = data_targets[n_train+n_val:]
        test_samples = samples[n_train+n_val:]

        n_samples = samples.shape[-1]
        
        if cond_type.lower() == "FullPhaseSpace".lower():
            test_data_inputs = torch.cat([test_data_targets, test_inputs], dim=1)
            test_data_class = torch.ones(test_data_inputs.shape[0], 1)

            test_gen_inputs = torch.cat([test_samples, test_inputs.unsqueeze(-1).expand(-1, -1, n_samples)], dim=1).permute(0, 2, 1).reshape(-1, test_data_inputs.shape[1])
            test_gen_class = torch.zeros(test_gen_inputs.shape[0], 1)

            test_x = torch.cat([test_data_inputs, test_gen_inputs], dim=0)
            test_classes = torch.cat([test_data_class, test_gen_class], dim=0)
        elif cond_type.lower() == "NoPhaseSpace".lower():
            test_data_inputs = test_data_targets
            test_data_class = torch.ones(test_data_inputs.shape[0], 1)

            test_gen_inputs = test_samples.permute(0, 2, 1).reshape(-1, test_data_inputs.shape[1])
            test_gen_class = torch.zeros(test_gen_inputs.shape[0], 1)

            test_x = torch.cat([test_data_inputs, test_gen_inputs], dim=0)
            test_classes = torch.cat([test_data_class, test_gen_class], dim=0)
        else:
            raise NotImplementedError(f"Conditioning type {cond_type} is not implemented")

        self.train_dataset = DataSampleDataset(train_inputs, train_data_targets, train_samples, cond_type)
        self.val_dataset = DataSampleDataset(eval_inputs, eval_data_targets, eval_samples, cond_type)
        self.test_dataset = TensorDataset(test_x, test_classes)

        self.train_batches_per_epoch = len(self.train_dataset) // self.loader_params["batch_size"]

        logging.info(f"Classifier true data: Train size: {train_data_targets.shape[0]:,}, Val size: {eval_data_targets.shape[0]:,}, Test size: {test_data_inputs.shape[0]:,}")

    def train_dataloader(self):
        """Creates DataLoader for training data"""
        batch_size = self.loader_params["batch_size"]
        num_workers = self.loader_params["num_workers"]
        pin_memory = self.loader_params["pin_memory"]
        # shuffle = self.loader_params["shuffle"]
        persistent_workers = self.loader_params["persistent_workers"]

        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=Classifier_Sampler(len(self.train_dataset) // 2),
            # shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=True
        )

    def val_dataloader(self):
        """Creates DataLoader for validation data"""
        batch_size = self.loader_params["batch_size"]
        num_workers = self.loader_params["num_workers"]
        pin_memory = self.loader_params["pin_memory"]
        persistent_workers = self.loader_params["persistent_workers"]

        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            sampler=Classifier_Sampler(len(self.val_dataset) // 2),
            # shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=True
        )
    
    def test_dataloader(self):
        """Creates DataLoader for test data"""
        batch_size = self.loader_params.get("test_batch_size", self.loader_params["batch_size"])
        num_workers = self.loader_params["num_workers"]
        # num_workers = 1
        pin_memory = self.loader_params["pin_memory"]
        persistent_workers = self.loader_params["persistent_workers"]
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False
        )


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
    def __init__(self, params, skip_dims=[]):
        self.params = params
        self.mean = None
        self.std = None

        self.skip_dims = skip_dims
        self.skip_dims.sort()

        self.log_dims = [log_dim for log_dim in self.params.get("log_dims", []) if log_dim not in self.skip_dims]
        logging.info(f"Log dims before correcting for skip dims: {self.log_dims}")
        logging.info(f"Skip dims are: {self.skip_dims}")

        # correct for skipped dims
        offsets = [0 for _ in self.log_dims]
        for i in range(len(self.skip_dims)):
            for j in range(len(self.log_dims)):
                if self.log_dims[j] > self.skip_dims[i]:
                    offsets[j] += 1
        self.log_dims = [log_dim - offset for log_dim, offset in zip(self.log_dims, offsets)]
        logging.info(f"Log dims after correcting for skip dims: {self.log_dims}")

    def preprocess_forward(self, data_in):
        """Apply log transform and standardization"""

        data = data_in.clone()

        # Apply log transform to specified dimensions
        log_dims = self.log_dims
        data[:, log_dims] = torch.log10(data[:, log_dims] + 1.)

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
        log_dims = self.log_dims
        data[:, log_dims] = 10**(data[:, log_dims]) - 1.

        return data.squeeze()

def _get_log_response_mode(log_responses: np.ndarray, log_likelihoods: np.ndarray):
    return log_responses[np.arange(log_responses.shape[0]), :, np.argmax(log_likelihoods, axis=1)]

def _get_response_mode(responses: np.ndarray, log_likelihoods: np.ndarray):
    log_likelihoods = log_likelihoods - np.sum(np.log(responses), axis=1)

    return responses[np.arange(responses.shape[0]), :, np.argmax(log_likelihoods, axis=1)]

def _get_response_jet_mean(responses: np.ndarray):
    inv_responses = 1. / responses
    mean_inv_response = np.mean(inv_responses, axis=-1)
    return 1. / mean_inv_response

def _get_response_jet_mode(responses: np.ndarray, log_likelihoods: np.ndarray):
    log_likelihoods = log_likelihoods + np.sum(np.log(responses), axis=1)

    return responses[np.arange(responses.shape[0]), :, np.argmax(log_likelihoods, axis=1)]

class SampleEvaluation():
    def __init__(
            self,
            log_response_samples: np.ndarray,  # (size, targets, n_samples), targets: log r_E, log r_m
            log_likelihoods: np.ndarray        # (size, n_samples)
    ):
        logging.info("Evaluating samples")

        self.log_response_samples = log_response_samples
        self.log_response_mean = np.mean(self.log_response_samples, axis=-1)
        self.log_response_median = np.median(self.log_response_samples, axis=-1)
        self.log_response_mode = _get_log_response_mode(self.log_response_samples, log_likelihoods)

        logging.info("Done evaluating log responses")

        self.response_samples = 10.**log_response_samples
        self.response_mean = np.mean(self.response_samples, axis=-1)
        self.response_median = np.median(self.response_samples, axis=-1)
        self.response_mode = _get_response_mode(self.response_samples, log_likelihoods)

        self.response_jet_mean = _get_response_jet_mean(self.response_samples)
        self.response_jet_mode = _get_response_jet_mode(self.response_samples, log_likelihoods)

        logging.info("Done evaluating responses")




