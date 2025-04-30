
import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils.calculate_metrics import calculate_metrics
from utils.logging_helpers import plot_tensors_hist
from utils.logging_helpers import plot_index
import wandb


class Linear_NIR(pl.LightningModule):
    def __init__(self,config):
        super().__init__()
        print("Creating Baseline Linear NIR Model")
        self.config = config
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        rgb, nir = batch["rgb"],batch["nir"]  # rgb: [B, 3, H, W], nir: [B, 1, H, W]
        rgb = rgb.view(-1, 3).float()
        nir = nir.view(-1, 1).float()
        pred = self(rgb)
        loss = nn.functional.mse_loss(pred, nir)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        rgb, nir = batch["rgb"],batch["nir"]
        nir_pred = self(rgb)  # [B, 1, H, W]
        
        # calculate metrics
        if self.logger and hasattr(self.logger, 'experiment'):   
            metrics = calculate_metrics(pred=torch.clone(nir_pred).cpu(),target=torch.clone(nir).cpu(),phase="val")
            self.log_dict(metrics,on_step=False,on_epoch=True,sync_dist=True)
            del metrics

        # only perform image logging for n pics, not all in val loader
        if batch_idx<self.config.custom_configs.Logging.num_val_images:
            if self.logger and hasattr(self.logger, 'experiment'):
                
                # log standard image visualizations, deep copy to avoid graph problems
                val_img = plot_tensors_hist(rgb, torch.clone(nir), torch.clone(nir_pred),title="Validation")
                ndvi_img = plot_index(rgb, torch.clone(nir), torch.clone(nir_pred),title="Validation")
                self.logger.experiment.log({"Images/Val NIR":  wandb.Image(val_img)}) # log val image
                del val_img
                
                if self.config.custom_configs.Logging.log_ndvi: # plot NDVI image
                    ndvi_img = plot_index(rgb, torch.clone(nir), torch.clone(nir_pred),title="Validation")
                    self.logger.experiment.log({"Images/Val NDVI":  wandb.Image(ndvi_img)}) # log val image
                    del ndvi_img
                
                # Log Input and Prediction Value Statistics
                if self.config.custom_configs.Logging.log_input_stats:
                    self.log_dict({"val_stats/min_pred":torch.min(nir_pred).item(), # log stats
                                "val_stats/max_pred":torch.max(nir_pred).item(),
                                "val_stats/mean_pred":torch.mean(nir_pred).item()},
                                on_epoch=True,sync_dist=True)
                    self.log_dict({"val_stats/min_input":torch.min(nir).item(), # log stats
                                "val_stats/max_input":torch.max(nir).item(),
                                "val_stats/mean_input":torch.mean(nir).item()},
                                on_epoch=True,sync_dist=True)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.base_configs.learning_rate)




class MLP_NIR(pl.LightningModule):
    def __init__(self,config):
        super().__init__()
        print("Creating Baseline MLP NIR Model")
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.mlp(x)

    def training_step(self, batch, batch_idx):
        rgb, nir = batch["rgb"],batch["nir"]  # rgb: [B, 3, H, W], nir: [B, 1, H, W]
        rgb = rgb.view(-1, 3).float()
        nir = nir.view(-1, 1).float()
        pred = self(rgb)
        loss = nn.functional.mse_loss(pred, nir)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        rgb, nir = batch["rgb"],batch["nir"]
        nir_pred = self(rgb)  # [B, 1, H, W]
        
        # calculate metrics
        if self.logger and hasattr(self.logger, 'experiment'):   
            metrics = calculate_metrics(pred=torch.clone(nir_pred).cpu(),target=torch.clone(nir).cpu(),phase="val")
            self.log_dict(metrics,on_step=False,on_epoch=True,sync_dist=True)
            del metrics
        
        # only perform image logging for n pics, not all in val loader
        if batch_idx<self.config.custom_configs.Logging.num_val_images:
            if self.logger and hasattr(self.logger, 'experiment'):
                
                # log standard image visualizations, deep copy to avoid graph problems
                val_img = plot_tensors_hist(rgb, torch.clone(nir), torch.clone(nir_pred),title="Validation")
                ndvi_img = plot_index(rgb, torch.clone(nir), torch.clone(nir_pred),title="Validation")
                self.logger.experiment.log({"Images/Val NIR":  wandb.Image(val_img)}) # log val image
                del val_img
                
                if self.config.custom_configs.Logging.log_ndvi: # plot NDVI image
                    ndvi_img = plot_index(rgb, torch.clone(nir), torch.clone(nir_pred),title="Validation")
                    self.logger.experiment.log({"Images/Val NDVI":  wandb.Image(ndvi_img)}) # log val image
                    del ndvi_img
                
                # Log Input and Prediction Value Statistics
                if self.config.custom_configs.Logging.log_input_stats:
                    self.log_dict({"val_stats/min_pred":torch.min(nir_pred).item(), # log stats
                                "val_stats/max_pred":torch.max(nir_pred).item(),
                                "val_stats/mean_pred":torch.mean(nir_pred).item()},
                                on_epoch=True,sync_dist=True)
                    self.log_dict({"val_stats/min_input":torch.min(nir).item(), # log stats
                                "val_stats/max_input":torch.max(nir).item(),
                                "val_stats/mean_input":torch.mean(nir).item()},
                                on_epoch=True,sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.base_configs.learning_rate)
    
    
class CNN_NIR(pl.LightningModule):
    def __init__(self,config):
        super().__init__()
        print("Creating Baseline CNN NIR Model")
        self.config = config
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        rgb, nir = batch["rgb"],batch["nir"]  # rgb: [B, 3, H, W], nir: [B, 1, H, W]
        pred = self(rgb)
        loss = nn.functional.mse_loss(pred, nir)
        
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        rgb, nir = batch["rgb"],batch["nir"]
        nir_pred = self(rgb)  # [B, 1, H, W]

        # calculate metrics
        if self.logger and hasattr(self.logger, 'experiment'):   
            metrics = calculate_metrics(pred=torch.clone(nir_pred).cpu(),target=torch.clone(nir).cpu(),phase="val")
            self.log_dict(metrics,on_step=False,on_epoch=True,sync_dist=True)
            del metrics
        
        # only perform image logging for n pics, not all in val loader
        if batch_idx<self.config.custom_configs.Logging.num_val_images:
            if self.logger and hasattr(self.logger, 'experiment'):
                
                # log standard image visualizations, deep copy to avoid graph problems
                val_img = plot_tensors_hist(rgb, torch.clone(nir), torch.clone(nir_pred),title="Validation")
                ndvi_img = plot_index(rgb, torch.clone(nir), torch.clone(nir_pred),title="Validation")
                self.logger.experiment.log({"Images/Val NIR":  wandb.Image(val_img)}) # log val image
                del val_img
                
                if self.config.custom_configs.Logging.log_ndvi: # plot NDVI image
                    ndvi_img = plot_index(rgb, torch.clone(nir), torch.clone(nir_pred),title="Validation")
                    self.logger.experiment.log({"Images/Val NDVI":  wandb.Image(ndvi_img)}) # log val image
                    del ndvi_img
                
                # Log Input and Prediction Value Statistics
                if self.config.custom_configs.Logging.log_input_stats:
                    self.log_dict({"val_stats/min_pred":torch.min(nir_pred).item(), # log stats
                                "val_stats/max_pred":torch.max(nir_pred).item(),
                                "val_stats/mean_pred":torch.mean(nir_pred).item()},
                                on_epoch=True,sync_dist=True)
                    self.log_dict({"val_stats/min_input":torch.min(nir).item(), # log stats
                                "val_stats/max_input":torch.max(nir).item(),
                                "val_stats/mean_input":torch.mean(nir).item()},
                                on_epoch=True,sync_dist=True)
                    
                    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.base_configs.learning_rate)
    
    
if __name__ == "__main__":

    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/config_baselines.yaml")
    model = CNN_NIR(config)
    
    # Get Data
    from data.select_dataset import dataset_selector
    pl_datamodule = dataset_selector(config)
    batch = next(iter(pl_datamodule.train_dataloader()))
    
    model.training_step(batch,0)
    
    
    