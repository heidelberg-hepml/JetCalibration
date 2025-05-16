"""
PyTorch Lightning models for jet calibration regression tasks.

This module contains model classes for different regression approaches:

- BaseModel: Abstract base class with common functionality
- MLP_MSE_Regression: Simple MLP with MSE loss
- MLP_Heteroscedastic_Regression: MLP predicting mean and variance 
- MLP_GMM_Regression: MLP predicting Gaussian mixture model parameters

The models handle:
- Network architecture setup
- Loss function computation
- Training and validation steps
- Prediction and sampling methods
- Checkpoint saving/loading
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from .network import MLP
import time
import logging
import torch.distributions as dist


class BaseModel(pl.LightningModule):
    """
    Base model class implementing common functionality.

    Handles training loop, optimization, checkpointing etc.

    Args:
        input_dim (int): Dimension of input features
        target_dim (int): Dimension of target variables 
        parameters (dict): Model configuration parameters
    """
    def __init__(self, input_dim, target_dim, parameters):
        super().__init__()
        self.params = parameters
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.save_hyperparameters()

        # Track losses over epochs
        self.train_epoch_losses = []
        self.val_epoch_losses = []

        # Track losses within epochs
        self.train_loss = []
        self.val_loss = []

    def forward(self, x):
        """Forward pass through model"""
        return self.model(x)
    
    def configure_optimizers(self):
        """Setup optimizer and learning rate scheduler"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.params.get('learning_rate', 0.001),
            weight_decay=self.params.get('weight_decay', 0.)
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        lr_sheduler_factor = self.params.get('lr_sheduler_factor', 0.1)
        if lr_sheduler_factor == 1.0:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 1.)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.params.get('lr_sheduler_factor', 0.1),
                patience=self.params.get('lr_sheduler_patience', 10)
            )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def training_step(self, batch, batch_idx):
        """Single training step"""
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred.squeeze(), y.squeeze())
        self.train_loss.append(loss.detach())
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss 

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        """Single validation step"""
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred.squeeze(), y.squeeze())
        self.val_loss.append(loss.detach())
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss 

    @torch.inference_mode()
    def on_train_epoch_end(self):
        """Compute and store average training loss for epoch"""
        epoch_loss = torch.stack(self.train_loss).mean()
        self.train_epoch_losses.append(epoch_loss.item())
        self.log("train_loss",  self.train_epoch_losses[-1], on_step=False, on_epoch=True)
        self.train_loss = []

    @torch.inference_mode()
    def on_validation_epoch_end(self):
        """Compute and store average validation loss for epoch"""
        epoch_loss = torch.stack(self.val_loss).mean()
        self.val_epoch_losses.append(epoch_loss.item())
        self.log("val_loss", self.val_epoch_losses[-1], on_step=False, on_epoch=True)
        self.val_loss = []

    def on_save_checkpoint(self, checkpoint):
        """Save custom attributes to checkpoint"""
        checkpoint["train_epoch_losses"] = self.train_epoch_losses
        checkpoint["val_epoch_losses"] = self.val_epoch_losses

    def on_load_checkpoint(self, checkpoint):
        """Load custom attributes from checkpoint"""
        self.train_epoch_losses = checkpoint["train_epoch_losses"]
        self.val_epoch_losses = checkpoint["val_epoch_losses"]


class MLP_MSE_Regression(BaseModel):
    """
    Simple MLP model with MSE loss.

    Basic regression model predicting single target value.

    Args:
        input_dim (int): Input feature dimension
        target_dim (int): Target variable dimension
        parameters (dict): Model configuration parameters
    """
    def __init__(self, input_dim, target_dim, parameters):
        super().__init__(input_dim, target_dim, parameters)
        self.model = MLP(input_dim, target_dim, self.params['hidden_dim'], self.params['num_layers'])
        self.loss_fn = nn.MSELoss()
    
    def predict_step(self, batch, batch_idx):
        """Generate predictions for a batch"""
        x, y = batch
        y_pred = self(x)
        return {'mu': y_pred.squeeze()}

class MLP_Heteroscedastic_Regression(BaseModel):
    """
    MLP model predicting both mean and variance.

    Implements heteroscedastic regression by outputting both predicted value
    and estimated uncertainty.

    Args:
        input_dim (int): Input feature dimension
        target_dim (int): Target variable dimension
        parameters (dict): Model configuration parameters
    """
    def __init__(self, input_dim, target_dim, parameters):
        super().__init__(input_dim, target_dim, parameters)
        self.model = MLP(input_dim, 2*target_dim, self.params['hidden_dim'], self.params['num_layers'])
        self.gaussian_nlll = nn.GaussianNLLLoss(reduction='mean', full=True)

    def loss_fn(self, y_pred, y):
        """Compute negative log likelihood loss"""
        mu_pred = y_pred[:, :self.target_dim]
        log_var_pred = torch.clamp(y_pred[:, self.target_dim:], min=-10, max=10)
        var_pred = torch.exp(log_var_pred)
        return self.gaussian_nlll(mu_pred, y, var_pred)
    
    def predict_step(self, batch, batch_idx):
        """Generate predictions with uncertainties"""
        x, y = batch
        y_pred = self(x)
        mu_pred = y_pred[:, :self.target_dim]
        log_var_pred = torch.clamp(y_pred[:, self.target_dim:], min=-10, max=10)
        var_pred = torch.exp(log_var_pred)

        if self.target_dim == 1:
            return {'mu': mu_pred, 'sigma': torch.sqrt(var_pred)}
        
        else:
            return {'mu_E': mu_pred[:, 0], 'sigma_E': torch.sqrt(var_pred[:, 0]), 
                   'mu_m': mu_pred[:, 1], 'sigma_m': torch.sqrt(var_pred[:, 1])}
    
    def sample(self, mu_pred, sigma_pred, n_samples=1):
        """Draw samples from predicted distribution"""
        batch_size = mu_pred.shape[0]

        mu_pred = mu_pred.unsqueeze(1).expand(batch_size, n_samples)
        sigma_pred = sigma_pred.unsqueeze(1).expand(batch_size, n_samples)  

        normal_dist = dist.Normal(mu_pred, sigma_pred)  
        samples = normal_dist.sample()
        log_likelihoods = normal_dist.log_prob(samples)
        
        return samples, log_likelihoods

class MLP_MultivariatHeteroscedastic_Regression(BaseModel):
    """
    MLP model predicting both mean and variance for a multivariat distribution.

    Implements heteroscedastic regression by outputting both predicted value
    and estimated uncertainty.

    Args:
        input_dim (int): Input feature dimension
        target_dim (int): Target variable dimension
        parameters (dict): Model configuration parameters
    """
    def __init__(self, input_dim, target_dim, parameters):
        super().__init__(input_dim, target_dim, parameters)

        self._index_diag = torch.arange(0, target_dim)
        self._index_off_diag_0, self._index_off_diag_1 = torch.tril_indices(target_dim, target_dim, -1)

        self.register_buffer("sigma_scale", (1. / torch.sqrt(0.25 * torch.arange(1, target_dim + 1, dtype=torch.float32))).unsqueeze(-1))

        output_dim = target_dim + (target_dim + 1) * target_dim // 2
        self.model = MLP(input_dim, output_dim, self.params['hidden_dim'], self.params['num_layers'])

    def loss_fn(self, y_pred, y):
        """Compute negative log likelihood loss"""
        mu_pred = y[...,:self.target_dim].clone()

        sigmas_diag = y[...,self.target_dim:2*self.target_dim].clone()
        sigmas_off_diag = y[...,2*self.target_dim:].clone()

        sigmas_diag = torch.nn.functional.softplus(sigmas_diag)

        _sigmas = torch.zeros(*mu_pred.shape[:-1], self.target_dim, self.target_dim, dtype=sigmas_diag.dtype, device=sigmas_diag.device)
        _sigmas[..., self._index_diag, self._index_diag] = sigmas_diag
        _sigmas[..., self._index_off_diag_0, self._index_off_diag_1] = sigmas_off_diag
        _sigmas = _sigmas * self.sigma_scale

        _diag = torch.diagonal(_sigmas, 0, -2, -1)
        # _logdet = torch.nn.functional.softplus(torch.prod(_diag, dim=-1), beta=1.0e4) # diag is 1 / (sigma_1 * sigma_2 * ...) with sigma_i < 1
        _logdet = torch.prod(_diag, dim=-1)
        _logdet = torch.log(_logdet)

        sigmas = torch.matmul(_sigmas, _sigmas.transpose(-2, -1))

        _diff = mu_pred - y
        _diff2 = torch.matmul(sigmas, _diff.unsqueeze(-1)).squeeze(-1)

        loss = 0.5 * torch.sum(_diff * _diff2, dim=-1) - _logdet
        loss = torch.mean(loss)

        return loss

    def predict_step(self, batch, batch_idx):
        """Generate predictions with uncertainties"""
        x, y = batch
        y_pred = self(x)
        
        mu_pred = y_pred[:, :self.target_dim]
        
        sigmas_diag = y[...,self.target_dim:2*self.target_dim].clone()
        sigmas_off_diag = y[...,2*self.target_dim:].clone()

        sigmas_diag = torch.nn.functional.softplus(sigmas_diag)

        _sigmas = torch.zeros(*mu_pred.shape[:-1], self.target_dim, self.target_dim, dtype=sigmas_diag.dtype, device=sigmas_diag.device)
        _sigmas[..., self._index_diag, self._index_diag] = sigmas_diag
        _sigmas[..., self._index_off_diag_0, self._index_off_diag_1] = sigmas_off_diag
        _sigmas = _sigmas * self.sigma_scale

        sigmas = torch.matmul(_sigmas, _sigmas.transpose(-2, -1))
            
        return { 'mu': mu_pred, 'sigma_inv': sigmas } 
    
    def sample(self, mu_pred, sigma_pred, n_samples=1):
        """Draw samples from predicted distribution"""
        batch_size = mu_pred.shape[0]

        multivariat_normal_dist = dist.MultivariateNormal(loc=mu_pred, precision_matrix=sigma_pred)

        samples = multivariat_normal_dist.sample((n_samples,))
        log_likelihoods = multivariat_normal_dist.log_prob(samples)
        
        return samples.permute(1, 2, 0), log_likelihoods.permute(1, 0)

class MLP_GMM_Regression(BaseModel):
    """
    MLP model predicting Gaussian mixture model parameters.

    Implements regression using a mixture of Gaussians to capture
    multi-modal distributions.

    Args:
        input_dim (int): Input feature dimension
        target_dim (int): Target variable dimension
        parameters (dict): Model configuration parameters including n_Gaussians
    """
    def __init__(self, input_dim, target_dim, parameters):
        super().__init__(input_dim, target_dim, parameters)
        self.n_Gaussians = parameters['n_Gaussians']
        self.model = MLP(input_dim, self.n_Gaussians*3*target_dim, self.params['hidden_dim'], self.params['num_layers'])
        self.gaussian_nlll = nn.GaussianNLLLoss(reduction='none', full=True)

    def loss_fn(self, y_pred, y):
        """Compute negative log likelihood loss for mixture model"""
        if self.target_dim == 1:
            mu_pred = y_pred[:, :self.n_Gaussians]

            log_var_pred = y_pred[:, self.n_Gaussians:2*self.n_Gaussians]
            log_var_pred = torch.clamp(log_var_pred, min=-10, max=10)
            var_pred = torch.exp(log_var_pred)

            weights_pred = y_pred[:, 2*self.n_Gaussians:]
            weights_pred = torch.softmax(weights_pred, dim=1)

            log_likelihood_per_Gaussian = (-1.)*self.gaussian_nlll(mu_pred, y.unsqueeze(1), var_pred) + torch.log(weights_pred+1e-12)
            return (-1.)*torch.logsumexp(log_likelihood_per_Gaussian, dim=1).mean()
    
        else:
            # Handle energy and mass predictions separately
            preds_E = y_pred[:, :self.n_Gaussians*3]
            preds_m = y_pred[:, self.n_Gaussians*3:]

            # Energy component
            mu_E = preds_E[:, :self.n_Gaussians]
            log_var_E = preds_E[:, self.n_Gaussians:2*self.n_Gaussians]
            log_var_E = torch.clamp(log_var_E, min=-10, max=10)
            var_E = torch.exp(log_var_E)
            weights_E = torch.softmax(preds_E[:, 2*self.n_Gaussians:], dim=1)
            log_likelihood_per_Gaussian_E = (-1.)*self.gaussian_nlll(mu_E, y[:, 0].unsqueeze(1), var_E) + torch.log(weights_E+1e-12)
            loss_E = (-1.)*torch.logsumexp(log_likelihood_per_Gaussian_E, dim=1).mean()

            # Mass component
            mu_m = preds_m[:, :self.n_Gaussians]
            log_var_m = preds_m[:, self.n_Gaussians:2*self.n_Gaussians]
            log_var_m = torch.clamp(log_var_m, min=-10, max=10)
            var_m = torch.exp(log_var_m)
            weights_m = torch.softmax(preds_m[:, 2*self.n_Gaussians:], dim=1)
            log_likelihood_per_Gaussian_m = (-1.)*self.gaussian_nlll(mu_m, y[:, 1].unsqueeze(1), var_m) + torch.log(weights_m+1e-12)
            loss_m = (-1.)*torch.logsumexp(log_likelihood_per_Gaussian_m, dim=1).mean()
            return (loss_E + loss_m) / 2
        
    def predict_step(self, batch, batch_idx):
        """Generate predictions for mixture model components"""
        x, y = batch
        y_pred = self(x)
        if self.target_dim == 1:
            mu_pred = y_pred[:, :self.n_Gaussians]

            log_var_pred = torch.clamp(y_pred[:, self.n_Gaussians:2*self.n_Gaussians], min=-10, max=10)
            var_pred = torch.exp(log_var_pred)

            weights_pred = torch.softmax(y_pred[:, 2*self.n_Gaussians:], dim=1)
            return {'mu': mu_pred, 'sigma': torch.sqrt(var_pred), 'weights': weights_pred}
        
        else:
            # Split predictions for energy and mass
            preds_E = y_pred[:, :self.n_Gaussians*3]
            preds_m = y_pred[:, self.n_Gaussians*3:]

            # Process energy predictions
            mu_E = preds_E[:, :self.n_Gaussians]
            log_var_E = preds_E[:, self.n_Gaussians:2*self.n_Gaussians]
            log_var_E = torch.clamp(log_var_E, min=-10, max=10)
            var_E = torch.exp(log_var_E)
            weights_E = torch.softmax(preds_E[:, 2*self.n_Gaussians:], dim=1)

            # Process mass predictions
            mu_m = preds_m[:, :self.n_Gaussians]
            log_var_m = preds_m[:, self.n_Gaussians:2*self.n_Gaussians]
            log_var_m = torch.clamp(log_var_m, min=-10, max=10)
            var_m = torch.exp(log_var_m)
            weights_m = torch.softmax(preds_m[:, 2*self.n_Gaussians:], dim=1)

            return {'mu_E': mu_E, 'sigma_E': torch.sqrt(var_E), 'weights_E': weights_E,
                   'mu_m': mu_m, 'sigma_m': torch.sqrt(var_m), 'weights_m': weights_m}
            
    def sample(self, mu_pred, sigma_pred, weights_pred, n_samples=1):
        """Draw samples from mixture model distribution"""
        batch_size, num_components = weights_pred.shape  # Shape (batch, num_components)

        # Step 1: Expand inputs to allow multiple samples
        mu_pred = mu_pred.unsqueeze(1).expand(batch_size, n_samples, num_components)  # (batch, n_samples, num_components)
        sigma_pred = sigma_pred.unsqueeze(1).expand(batch_size, n_samples, num_components)  
        weights_pred = weights_pred.unsqueeze(1).expand(batch_size, n_samples, num_components)

        # Step 2: Sample component indices per batch element
        categorical_dist = dist.Categorical(weights_pred)  # Batch-wise categorical distribution
        gaussian_choices = categorical_dist.sample()  # (batch, n_samples)

        # Step 3: Gather selected means and variances
        selected_means = torch.gather(mu_pred, 2, gaussian_choices.unsqueeze(-1)).squeeze(-1)  # (batch, n_samples)
        selected_sigmas = torch.gather(sigma_pred, 2, gaussian_choices.unsqueeze(-1)).squeeze(-1)  # (batch, n_samples)

        # Step 4: Sample from selected Gaussian distributions
        normal_dist = dist.Normal(selected_means, selected_sigmas)  
        samples = normal_dist.sample()  # (batch, n_samples)

        # Step 5: Compute log likelihoods
        log_categorical_probs = categorical_dist.log_prob(gaussian_choices)  # (batch, n_samples)
        log_gaussian_likelihoods = normal_dist.log_prob(samples)  # (batch, n_samples)
        log_likelihoods = log_categorical_probs + log_gaussian_likelihoods  # (batch, n_samples)

        return samples, log_likelihoods

class MLP_Multivariate_GMM_Regression(BaseModel):
    """
    MLP model predicting multivariat Gaussian mixture model parameters.

    Implements regression using a mixture of Gaussians to capture
    multi-modal distributions.

    Args:
        input_dim (int): Input feature dimension
        target_dim (int): Target variable dimension
        parameters (dict): Model configuration parameters including n_Gaussians
    """
    def __init__(self, input_dim, target_dim, parameters):
        super().__init__(input_dim, target_dim, parameters)
        self.n_Gaussians = parameters['n_Gaussians']

        self._index_diag = torch.arange(0, target_dim)
        self._index_off_diag_0, self._index_off_diag_1 = torch.tril_indices(target_dim, target_dim, -1)

        self.register_buffer("sigma_scale", (1. / torch.sqrt(torch.arange(1, target_dim + 1, dtype=torch.float32))).unsqueeze(-1))

        self.output_dim_per_mode = target_dim + (target_dim + 1) * target_dim // 2
        # print(f"self.output_dim_per_mode: {self.output_dim_per_mode}")
        self.model = MLP(
            input_dim=input_dim,
            output_dim=(self.output_dim_per_mode + 1) * self.n_Gaussians,
            hidden_dim=self.params['hidden_dim'],
            num_layers=self.params['num_layers'],
            drop=parameters.get("drop", 0.)
        )
        
    def loss_fn(self, y_pred: torch.FloatTensor, y: torch.FloatTensor):
        """Compute negative log likelihood loss for mixture model"""

        # print(f"y_pred: {y_pred.shape}")

        weights = y_pred[...,:self.n_Gaussians].clone()
        y_pred = y_pred[...,self.n_Gaussians:].clone()

        y_pred = y_pred.reshape(-1, self.n_Gaussians, self.output_dim_per_mode)
        y = y.unsqueeze(-2).expand(-1, self.n_Gaussians, -1)

        mu_pred = y_pred[...,:self.target_dim].clone()

        sigmas_diag = y_pred[...,self.target_dim:2*self.target_dim].clone()
        sigmas_off_diag = y_pred[...,2*self.target_dim:].clone()

        # sigmas_diag = torch.nn.functional.softplus(sigmas_diag)
        sigmas_diag = sigmas_diag.clamp(min=-10., max=10.)
        _logdet = torch.sum(sigmas_diag, dim=-1)
        sigmas_diag = torch.exp(sigmas_diag)

        _sigmas = torch.zeros(*mu_pred.shape[:-1], self.target_dim, self.target_dim, dtype=sigmas_diag.dtype, device=sigmas_diag.device)
        _sigmas[..., self._index_diag, self._index_diag] = sigmas_diag
        _sigmas[..., self._index_off_diag_0, self._index_off_diag_1] = sigmas_off_diag
        _sigmas = _sigmas * self.sigma_scale

        # _diag = torch.diagonal(_sigmas, 0, -2, -1)
        # _logdet = torch.prod(_diag, dim=-1)
        # _logdet = torch.log(_logdet)

        sigmas = torch.matmul(_sigmas, _sigmas.transpose(-2, -1))

        _diff = mu_pred - y
        _diff2 = torch.matmul(sigmas, _diff.unsqueeze(-1)).squeeze(-1)

        _log_prob_mix = torch.log_softmax(weights, dim=-1)
        _log_prob_gaus = -0.5 * torch.sum(_diff * _diff2, dim=-1) + _logdet
        
        loss = -torch.logsumexp(_log_prob_mix + _log_prob_gaus, dim=-1)
        loss = torch.mean(loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        """Generate predictions for mixture model components"""
        x, y = batch
        y_pred = self(x)
        
        weights = y_pred[...,:self.n_Gaussians].clone()
        weights = torch.nn.functional.softmax(weights, dim=-1)

        y_pred = y_pred[...,self.n_Gaussians:].clone()

        y_pred = y_pred.reshape(-1, self.n_Gaussians, self.output_dim_per_mode)

        mu_pred = y_pred[...,:self.target_dim].clone()

        sigmas_diag = y_pred[...,self.target_dim:2*self.target_dim].clone()
        sigmas_off_diag = y_pred[...,2*self.target_dim:].clone()

        # sigmas_diag = torch.nn.functional.softplus(sigmas_diag)
        sigmas_diag = sigmas_diag.clamp(min=-10., max=10.)
        sigmas_diag = torch.exp(sigmas_diag)

        _sigmas = torch.zeros(*mu_pred.shape[:-1], self.target_dim, self.target_dim, dtype=sigmas_diag.dtype, device=sigmas_diag.device)
        _sigmas[..., self._index_diag, self._index_diag] = sigmas_diag
        _sigmas[..., self._index_off_diag_0, self._index_off_diag_1] = sigmas_off_diag
        _sigmas = _sigmas * self.sigma_scale

        sigmas = torch.matmul(_sigmas, _sigmas.transpose(-2, -1))

        return {
            "mu": mu_pred,
            "sigma_inv": sigmas,
            "weights": weights
        }
            
    def sample(self, mu_pred, sigma_pred, weights_pred, n_samples=1):
        """Draw samples from mixture model distribution"""
        batch_size, num_components = weights_pred.shape  # Shape (batch, num_components)

        # Step 2: Sample component indices per batch element
        categorical_dist = dist.Categorical(weights_pred)  # Batch-wise categorical distribution
        _gaussian_choices = categorical_dist.sample((n_samples,))  # (n_samples, batch)
        gaussian_choices = _gaussian_choices.reshape(-1)
        batch_indices = torch.arange(batch_size).unsqueeze(0).expand(n_samples, -1).reshape(-1)

        # Step 3: Gather selected means and variances
        selected_means = mu_pred[batch_indices, gaussian_choices].reshape(n_samples, batch_size, -1).permute(1,0,2)  # (batch, n_samples)
        selected_sigmas = sigma_pred[batch_indices, gaussian_choices].reshape(n_samples, batch_size, *sigma_pred.shape[2:]).permute(1,0,2,3)  # (batch, n_samples, target_dim, target_dim)

        # Step 4: Sample from selected Gaussian distributions
        normal_dist = dist.MultivariateNormal(selected_means, precision_matrix=selected_sigmas)  
        samples = normal_dist.sample()  # (batch, n_samples)

        # Step 5: Compute log likelihoods
        log_categorical_probs = categorical_dist.log_prob(_gaussian_choices).permute(1, 0)  # (batch, n_samples)
        log_gaussian_likelihoods = normal_dist.log_prob(samples)  # (batch, n_samples)
        log_likelihoods = log_categorical_probs + log_gaussian_likelihoods  # (batch, n_samples)

        return samples, log_likelihoods


class MLP_CFM(BaseModel):

    def __init__(self, input_dim, target_dim, parameters):
        super().__init__(input_dim, target_dim, parameters)
        self.model = MLP(input_dim+target_dim+1, input_dim, self.params['hidden_dim'], self.params['num_layers'])
        self.loss_fn = nn.MSELoss()


    def training_step(self, batch, batch_idx):
        """Single training step"""
        x, y = batch
        noise = torch.randn_like(y)
        t = torch.rand(y.shape[0], 1,device=y.device)
        y_t = (1 - t) * noise + t * y
        y_t_dot = y - noise
        v_pred = self(torch.cat([t, y_t, x], dim=-1))
        loss = self.loss_fn(v_pred.squeeze(), y_t_dot.squeeze())
        self.train_loss.append(loss.item())
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        """Single validation step"""
        x, y = batch
        noise = torch.randn_like(y)
        t = torch.rand(y.shape[0], 1,device=y.device)
        y_t = (1 - t) * noise + t * y
        y_t_dot = y - noise
        v_pred = self(torch.cat([t, y_t, x], dim=-1))
        loss = self.loss_fn(v_pred.squeeze(), y_t_dot.squeeze())
        self.val_loss.append(loss.item())
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss 

        
    def predict_step(self, batch, batch_idx):
        x, y = batch

        batch_size = x.shape[0]
        dtype = x.dtype
        device = x.device

        def net_wrapper(t, y_t):
            t = t * torch.ones_like(y_t[:, [0]], dtype=dtype, device=device)
            v = self(torch.cat([t, y_t, x], dim=-1))
            return v
        
        noise = torch.randn_like(y)
        y_t = odeint(func=net_wrapper, 
                     y0=noise,
                     t=torch.tensor([0, 1], device=device, dtype=dtype),
                     rtol=1e-5,
                     atol=1e-7,
                     method='dopri5')
        return y_t[-1]
