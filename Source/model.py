import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from .network import MLP
import time
import logging
import torch.distributions as dist


class BaseModel(pl.LightningModule):
    def __init__(self, input_dim, target_dim, parameters):
        super().__init__()
        self.params = parameters
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.save_hyperparameters()

        self.train_epoch_losses = []
        self.val_epoch_losses = []

        self.train_loss = []
        self.val_loss = []

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.get('learning_rate', 0.001))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
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
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred.squeeze(), y.squeeze())
        self.train_loss.append(loss.item())
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred.squeeze(), y.squeeze())
        self.val_loss.append(loss.item())
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss 

    def on_train_epoch_end(self):
        epoch_loss = torch.tensor(self.train_loss).mean()
        self.train_epoch_losses.append(epoch_loss.item())
        self.train_loss = []

    def on_validation_epoch_end(self):
        epoch_loss = torch.tensor(self.val_loss).mean()
        self.val_epoch_losses.append(epoch_loss.item())
        self.val_loss = []

    def on_save_checkpoint(self, checkpoint):
        """Called by Lightning to serialize model + any extra attributes you add here."""
        checkpoint["train_epoch_losses"] = self.train_epoch_losses
        checkpoint["val_epoch_losses"] = self.val_epoch_losses

    def on_load_checkpoint(self, checkpoint):
        """Called by Lightning to deserialize model + restore any custom attributes."""
        self.train_epoch_losses = checkpoint["train_epoch_losses"]
        self.val_epoch_losses = checkpoint["val_epoch_losses"]



class MLP_MSE_Regression(BaseModel):
    def __init__(self, input_dim, target_dim, parameters):
        super().__init__(input_dim, target_dim, parameters)
        self.model = MLP(input_dim, target_dim, self.params['hidden_dim'], self.params['num_layers'])
        self.loss_fn = nn.MSELoss()
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        return {'mu': y_pred.squeeze()}
    


class MLP_Heteroscedastic_Regression(BaseModel):
    def __init__(self, input_dim, target_dim, parameters):
        super().__init__(input_dim, target_dim, parameters)
        self.model = MLP(input_dim, 2*target_dim, self.params['hidden_dim'], self.params['num_layers'])
        self.gaussian_nlll = nn.GaussianNLLLoss(reduction='mean', full=True)

    def loss_fn(self, y_pred, y):
        mu_pred = y_pred[:, :self.target_dim]
        log_var_pred = torch.clamp(y_pred[:, self.target_dim:], min=-10, max=10)
        var_pred = torch.exp(log_var_pred)
        return self.gaussian_nlll(mu_pred, y, var_pred)
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        mu_pred = y_pred[:, :self.target_dim]
        log_var_pred = torch.clamp(y_pred[:, self.target_dim:], min=-10, max=10)
        var_pred = torch.exp(log_var_pred)

        if self.target_dim == 1:
            return {'mu': mu_pred, 'sigma': torch.sqrt(var_pred)}
        
        else:
            return {'mu_E': mu_pred[:, 0], 'sigma_E': torch.sqrt(var_pred[:, 0]), 'mu_m': mu_pred[:, 1], 'sigma_m': torch.sqrt(var_pred[:, 1])}
    

    def sample(self, mu_pred, sigma_pred, n_samples=1):

        #original_shape = mu_pred.shape
        batch_size = mu_pred.shape[0]
        #if len(original_shape) == 2:
        #    n_dim = original_shape[1]
        #else:
        #    n_dim = 1

        mu_pred = mu_pred.unsqueeze(1).expand(batch_size, n_samples)
        sigma_pred = sigma_pred.unsqueeze(1).expand(batch_size, n_samples)  

        normal_dist = dist.Normal(mu_pred, sigma_pred)  
        samples = normal_dist.sample()
        log_likelihoods = normal_dist.log_prob(samples)
        
        return samples, log_likelihoods


class MLP_GMM_Regression(BaseModel):
    def __init__(self, input_dim, target_dim, parameters):
        super().__init__(input_dim, target_dim, parameters)
        self.n_Gaussians = parameters['n_Gaussians']
        self.model = MLP(input_dim, self.n_Gaussians*3*target_dim, self.params['hidden_dim'], self.params['num_layers'])
        self.gaussian_nlll = nn.GaussianNLLLoss(reduction='none', full=True)

    def loss_fn(self, y_pred, y):
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
            preds_E = y_pred[:, :self.n_Gaussians*3]
            preds_m = y_pred[:, self.n_Gaussians*3:]

            mu_E = preds_E[:, :self.n_Gaussians]
            log_var_E = preds_E[:, self.n_Gaussians:2*self.n_Gaussians]
            log_var_E = torch.clamp(log_var_E, min=-10, max=10)
            var_E = torch.exp(log_var_E)
            weights_E = torch.softmax(preds_E[:, 2*self.n_Gaussians:], dim=1)
            log_likelihood_per_Gaussian_E = (-1.)*self.gaussian_nlll(mu_E, y[:, 0].unsqueeze(1), var_E) + torch.log(weights_E+1e-12)
            loss_E = (-1.)*torch.logsumexp(log_likelihood_per_Gaussian_E, dim=1).mean()

            mu_m = preds_m[:, :self.n_Gaussians]
            log_var_m = preds_m[:, self.n_Gaussians:2*self.n_Gaussians]
            log_var_m = torch.clamp(log_var_m, min=-10, max=10)
            var_m = torch.exp(log_var_m)
            weights_m = torch.softmax(preds_m[:, 2*self.n_Gaussians:], dim=1)
            log_likelihood_per_Gaussian_m = (-1.)*self.gaussian_nlll(mu_m, y[:, 1].unsqueeze(1), var_m) + torch.log(weights_m+1e-12)
            loss_m = (-1.)*torch.logsumexp(log_likelihood_per_Gaussian_m, dim=1).mean()
            return (loss_E + loss_m) / 2
        
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        if self.target_dim == 1:
            mu_pred = y_pred[:, :self.n_Gaussians]

            log_var_pred = torch.clamp(y_pred[:, self.n_Gaussians:2*self.n_Gaussians], min=-10, max=10)
            var_pred = torch.exp(log_var_pred)

            weights_pred = torch.softmax(y_pred[:, 2*self.n_Gaussians:], dim=1)
            return {'mu': mu_pred, 'sigma': torch.sqrt(var_pred), 'weights': weights_pred}
        
        else:
            preds_E = y_pred[:, :self.n_Gaussians*3]
            preds_m = y_pred[:, self.n_Gaussians*3:]

            mu_E = preds_E[:, :self.n_Gaussians]
            log_var_E = preds_E[:, self.n_Gaussians:2*self.n_Gaussians]
            log_var_E = torch.clamp(log_var_E, min=-10, max=10)
            var_E = torch.exp(log_var_E)
            weights_E = torch.softmax(preds_E[:, 2*self.n_Gaussians:], dim=1)

            mu_m = preds_m[:, :self.n_Gaussians]
            log_var_m = preds_m[:, self.n_Gaussians:2*self.n_Gaussians]
            log_var_m = torch.clamp(log_var_m, min=-10, max=10)
            var_m = torch.exp(log_var_m)
            weights_m = torch.softmax(preds_m[:, 2*self.n_Gaussians:], dim=1)

            return {'mu_E': mu_E, 'sigma_E': torch.sqrt(var_E), 'weights_E': weights_E, 'mu_m': mu_m, 'sigma_m': torch.sqrt(var_m), 'weights_m': weights_m}
            
    def sample(self, mu_pred, sigma_pred, weights_pred, n_samples=1):

        batch_size, num_components = weights_pred.shape  # Shape (batch, num_components)

        # Step 1: Expand inputs to allow multiple samples
        mu_pred = mu_pred.unsqueeze(1).expand(batch_size, n_samples, num_components)  # (batch, n_samples, num_components)
        sigma_pred = sigma_pred.unsqueeze(1).expand(batch_size, n_samples, num_components)  
        weights_pred = weights_pred.unsqueeze(1).expand(batch_size, n_samples, num_components)

        # Step 2: Sample `n_samples` component indices per batch element
        categorical_dist = dist.Categorical(weights_pred)  # Batch-wise categorical distribution
        gaussian_choices = categorical_dist.sample()  # (batch, n_samples)

        # Gather selected means and variances based on sampled component indices
        selected_means = torch.gather(mu_pred, 2, gaussian_choices.unsqueeze(-1)).squeeze(-1)  # (batch, n_samples)
        selected_sigmas = torch.gather(sigma_pred, 2, gaussian_choices.unsqueeze(-1)).squeeze(-1)  # (batch, n_samples)

        # Step 3: Sample from the selected Gaussian distributions
        normal_dist = dist.Normal(selected_means, selected_sigmas)  
        samples = normal_dist.sample()  # (batch, n_samples)

        # Step 4: Compute Log Likelihoods
        log_categorical_probs = categorical_dist.log_prob(gaussian_choices)  # (batch, n_samples)
        log_gaussian_likelihoods = normal_dist.log_prob(samples)  # (batch, n_samples)
        log_likelihoods = log_categorical_probs + log_gaussian_likelihoods  # (batch, n_samples)

        return samples, log_likelihoods

        
