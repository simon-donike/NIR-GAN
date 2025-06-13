import torch
import torch.nn as nn
from transformers import ViTModel
import pytorch_lightning as pl
import torch.nn.functional as F
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau


from utils.calculate_metrics import calculate_metrics
from utils.logging_helpers import plot_tensors_hist
from utils.logging_helpers import plot_index


class NIRGenerator(nn.Module):
    def __init__(self, config, location_dim=256, pretrained_model="google/vit-base-patch16-224-in21k"):
        super().__init__()
        
        # Get Device
        if config.custom_configs.Training.accelerator == "gpu":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # create SatClip
        from model.satclip.satclip_wrapper import SatClIP_wrapper
        self.satclip_model = SatClIP_wrapper(device=self.device)
        self.satclip_model = self.satclip_model.eval()
        
        # Create ViT Model
        self.transformer = ViTModel.from_pretrained(pretrained_model)
        self.hidden_size = self.transformer.config.hidden_size
        self.patch_size = self.transformer.config.patch_size
        self.image_size = self.transformer.config.image_size

        # Create Token Fusion
        self.token_fusion = nn.Linear(location_dim, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, 1)

    def forward(self, rgb, location):
        
        # Get location embedding
        location_emb = self.satclip_model.predict(location)
        
        # Get Vit Outputs
        outputs = self.transformer(pixel_values=rgb)
        x = outputs.last_hidden_state[:, 1:, :]  # remove CLS token

        # Add location embedding to the token embeddings
        loc_proj = self.token_fusion(location_emb).unsqueeze(1)
        x = x + loc_proj

        # Create NIR prediction
        patch_preds = self.output_proj(x)  # (B, num_patches, 1)
        h = w = self.image_size // self.patch_size
        nir = patch_preds.view(rgb.size(0), 1, h, w)
        nir = nn.functional.interpolate(nir, scale_factor=self.patch_size, mode='bilinear')

        return nir


# PL Model
class NIRLitModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = NIRGenerator(config)
        self.config = config
        self.lr = config.base_configs.learning_rate

    def forward(self, rgb, location):
        return self.model(rgb, location)

    def training_step(self, batch, batch_idx):
        rgb, location, nir_gt = batch
        nir_pred = self(rgb, location)
        loss = F.l1_loss(nir_pred, nir_gt)
        # Calculate and Log Train Metrics  - only every 10th batch on Gen step
        if batch_idx % 10 == 0 and self.logger and hasattr(self.logger, 'experiment'):
            metrics = calculate_metrics(pred=torch.clone(nir_pred).cpu(),target=torch.clone(nir_gt).cpu(),phase="train")
            self.log_dict(metrics,on_step=True,sync_dist=True)
            del metrics
        return loss
        

    def validation_step(self, batch, batch_idx):
        rgb, location, nir_gt = batch
        nir_pred = self(rgb, location)
        loss = F.l1_loss(nir_pred, nir_gt)
        self.log("val_loss", loss)
        
        # only perform image logging for n pics, not all in val loader
        if batch_idx<self.config.custom_configs.Logging.num_val_images:
            if self.logger and hasattr(self.logger, 'experiment'):
                
                # log standard image visualizations, deep copy to avoid graph problems
                val_img = plot_tensors_hist(rgb, torch.clone(nir_gt), torch.clone(nir_pred),title="Validation")
                ndvi_img = plot_index(rgb, torch.clone(nir_gt), torch.clone(nir_pred),title="Validation")
                self.logger.experiment.log({"Images/Val NIR":  wandb.Image(val_img)}) # log val image
                del val_img
                
                if self.config.custom_configs.Logging.log_ndvi: # plot NDVI image
                    ndvi_img = plot_index(rgb, torch.clone(nir_gt), torch.clone(nir_pred),title="Validation")
                    self.logger.experiment.log({"Images/Val NDVI":  wandb.Image(ndvi_img)}) # log val image
                    del ndvi_img
                
                # Log Input and Prediction Value Statistics
                if self.config.custom_configs.Logging.log_input_stats:
                    self.log_dict({"val_stats/min_pred":torch.min(nir_pred).item(), # log stats
                                "val_stats/max_pred":torch.max(nir_pred).item(),
                                "val_stats/mean_pred":torch.mean(nir_pred).item()},
                                on_epoch=True,sync_dist=True)
                    self.log_dict({"val_stats/min_input":torch.min(nir_gt).item(), # log stats
                                "val_stats/max_input":torch.max(nir_gt).item(),
                                "val_stats/mean_input":torch.mean(nir_gt).item()},
                                on_epoch=True,sync_dist=True)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sched = ReduceLROnPlateau(optim, mode='min',
                                  patience=self.config.Schedulers.patience,
                                  factor=self.config.Schedulers.factor)

        return {
                        "optimizer": optim,
                        "lr_scheduler": {
                            "scheduler": sched,
                            "monitor": "val_loss",  # name of the metric you log
                            "interval": "epoch",
                            "frequency": 1
                        }
                    } 


if __name__ == "__main__":
    # Example usage
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/config_vit.yaml")
    model = NIRGenerator(config)
    rgb = torch.randn(2, 3, 224, 224)  # Example RGB input
    location = torch.randn(2, 2)  # Example location embedding
    nir_output = model(rgb, location)
    