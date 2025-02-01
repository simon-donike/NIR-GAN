import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from omegaconf import OmegaConf
import wandb
from utils.normalise_s2 import normalize_rgb
from utils.normalise_s2 import normalize_nir
from utils.calculate_metrics import calculate_metrics
from utils.logging_helpers import plot_tensors
from utils.logging_helpers import plot_ndvi
from model.pix2pix_model import Pix2PixModel

class Px2Px_PL(pl.LightningModule):
    def __init__(self, config_dict="configs/config_px2px.yaml"):
        super().__init__()
        # Load configuration
        if isinstance(config_dict, str):
            self.config = OmegaConf.load(config_dict)
        else:
            self.config = config_dict
            
        # load Px2Px configs
        self.model = Pix2PixModel(self.config)

        # set setings
        self.normalize = self.config.Data.normalize
        

    def forward(self, input):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pred = self.model.netG(input)  # G(A)
        return pred

    @torch.no_grad()
    def predict_step(self, rgb):
        #if self.training:
        #    self.model = self.model.eval()
        #assert not self.training, "Model should be in eval mode for predictions"
        if self.normalize:
            from utils.normalise_s2 import normalize_rgb, normalize_nir
            rgb_norm = normalize_rgb(rgb, stage="norm")
            nir_pred = self.model.netG(rgb_norm)
            return normalize_nir(nir_pred, stage="denorm")
        else:
            nir_pred = self.model.netG(rgb)
            return nir_pred

    def training_step(self, batch, batch_idx, optimizer_idx):
        rgb, nir = self.extract_batch(batch)
        if self.normalize:
            rgb = normalize_rgb(rgb, stage="norm")
            nir = normalize_nir(nir, stage="norm")

        # TODO: make this be ran only once per training step
        pred = self.model.netG(rgb)

        # 1. Backward_G
        if optimizer_idx == 1:
            # 1.1 emulate backward_D
            # 1.1.1 Fake
            fake_AB = torch.cat((rgb, pred), 1)
            pred_fake = self.model.netD(fake_AB.detach())
            loss_D_fake = self.model.criterionGAN(pred_fake, False)
            # 1.1.2 Real
            real_AB = torch.cat((rgb, nir), 1)
            pred_real = self.model.netD(real_AB)
            loss_D_real = self.model.criterionGAN(pred_real, True)
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            self.log("generator_loss", loss_D)
            return loss_D

        # Discriminator Step
        if optimizer_idx == 0:
            fake_AB = torch.cat((rgb, pred), 1)
            pred_fake = self.model.netD(fake_AB)
            loss_G_GAN = self.model.criterionGAN(pred_fake, True)
            loss_G_L1 = self.model.criterionL1(pred, nir) * self.config.base_configs.lambda_L1
            loss_G = loss_G_GAN + loss_G_L1
            self.log("discriminator_loss", loss_G)
            return loss_G
        
    @torch.no_grad()
    def validation_step(self,batch,batch_idx):
        
        """ 1. Extract and Predict """
        rgb,nir = self.extract_batch(batch)
                        
        # Predict, returns denormed NIR
        nir_pred = self.predict_step(rgb)

        """ 2. Log Generator Metrics """
        # log image metrics        
        metrics = calculate_metrics(pred=torch.clone(nir_pred).cpu(),target=torch.clone(nir).cpu(),phase="val")
        self.log_dict(metrics,on_step=False,on_epoch=True,sync_dist=True)

        # only perform image logging for n pics, not all 200
        if batch_idx<self.config.custom_configs.Logging.num_val_images:
            if self.logger and hasattr(self.logger, 'experiment'):
                # log Stadard image visualizations, deep copy to avoid graph problems
                val_img = plot_tensors(rgb, torch.clone(nir), torch.clone(nir_pred),title="Validation")
                ndvi_img = plot_ndvi(rgb, torch.clone(nir), torch.clone(nir_pred),title="Validation")
                self.logger.experiment.log({"Images/Val NIR":  wandb.Image(val_img)}) # log val image
                self.logger.experiment.log({"Images/Val NDVI":  wandb.Image(ndvi_img)}) # log val image
                self.log_dict({"pred_stats/min":torch.min(torch.clone(nir_pred)).item(), # log stats
                            "pred_stats/max":torch.max(torch.clone(nir_pred)).item(),
                            "pred_stats/mean":torch.min(torch.clone(nir_pred)).item()},
                            on_epoch=True,sync_dist=True)

    def configure_optimizers(self):
        optim_g = self.model.optimizer_G
        optim_d = self.model.optimizer_D
        sched_g = ReduceLROnPlateau(optim_g, mode='min', patience=self.config.Schedulers.patience_g)
        sched_d = ReduceLROnPlateau(optim_d, mode='min', patience=self.config.Schedulers.patience_d)
        return ([optim_d, optim_g], 
                [{'scheduler': sched_d, 'monitor': self.config.Schedulers.metric, 'interval': 'epoch'},
                 {'scheduler': sched_g, 'monitor': self.config.Schedulers.metric, 'interval': 'epoch'}])
    
    def extract_batch(self, batch):
        rgb = batch["rgb"]
        nir = batch["nir"]
        return rgb, nir
    
    
if __name__ == "__main__":
    config = OmegaConf.load("configs/config_px2px.yaml")
    m = Px2Px_PL(config)

    # try out simple forward
    rgb = torch.rand(5,3,512,512)
    nir = torch.rand(5,1,512,512)
    pred= m.predict_step(rgb)

    m = m.cpu()
    # try out training step
    batch = {"rgb":rgb, "nir":nir}
    m.training_step(batch, batch_idx=0, optimizer_idx=1)
    m.validation_step(batch, batch_idx=0)




