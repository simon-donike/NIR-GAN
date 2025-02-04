import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from omegaconf import OmegaConf
import wandb
from utils.calculate_metrics import calculate_metrics
from utils.logging_helpers import plot_tensors_hist
from utils.logging_helpers import plot_ndvi
from model.pix2pix_model import Pix2PixModel
from model.base_model import BaseModel
from model import networks


class Px2Px_PL(pl.LightningModule):
    def __init__(self, opt):
        super(Px2Px_PL, self).__init__()  # Initialize the base class first
        self.opt = opt.base_configs
        self.config = opt
        self.isTrain = self.opt.isTrain

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.netG, self.opt.norm,
                                      not self.opt.no_dropout, self.opt.init_type, self.opt.init_gain)

        self.netD = networks.define_D(self.opt.input_nc + self.opt.output_nc, self.opt.ndf, self.opt.netD,
                                          self.opt.n_layers_D, self.opt.norm, self.opt.init_type, self.opt.init_gain)
        self.criterionGAN = networks.GANLoss(self.opt.gan_mode)
        self.criterionL1 = torch.nn.L1Loss()

    def forward(self, input):
        pred = self.netG(input)
        return pred
    
    def on_train_batch_start(self, batch, batch_idx):
        # Reset cache at the start of each batch
        # Tis holds the pred so that we can do it only once per optimizer
        self.pred_cache = None

    @torch.no_grad()
    def predict_step(self, rgb):
        assert self.training == False, "Model is in training mode, set to eval mode before predicting"
        nir_pred = self.forward(rgb)
        return nir_pred

    def training_step(self, batch, batch_idx, optimizer_idx):
        assert self.training == True, "Model is in eval mode, set to training mode before training"
        rgb, nir = self.extract_batch(batch)

        # check for OPtimization regarding multiple pred steps in 1 training step
        # TODO: get optimization to work with cache, keep gradients intact
        use_optimization = False
        if use_optimization:
            # If Cache is empty, run prediction and save cache
            if self.pred_cache is None:
                pred = self.forward(rgb)
                self.pred_cache = pred
            # if something is cached, read cache
            else:
                pred = self.pred_cache
                self.cache = None
        else:
            pred = self.forward(rgb)

        if optimizer_idx == 0 and batch_idx % 10 == 0:
            metrics = calculate_metrics(pred=torch.clone(pred).cpu(),target=torch.clone(nir).cpu(),phase="train")
            self.log_dict(metrics,on_step=True,sync_dist=True)

        # Discriminator Step
        if optimizer_idx == 0:
            # 1.1.1 Fake
            fake_AB = torch.cat((rgb, pred), 1)
            pred_fake = self.netD(fake_AB)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # 1.1.2 Real
            real_AB = torch.cat((rgb, nir), 1)
            pred_real = self.netD(real_AB)
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            self.log("model_loss/discriminator_loss", loss_D)
            return loss_D

        # Generator Step
        if optimizer_idx == 1:
            fake_AB = torch.cat((rgb, pred), 1)
            pred_fake = self.netD(fake_AB)
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            loss_G_L1 = self.criterionL1(pred, nir) * self.config.base_configs.lambda_L1
            loss_G = loss_G_GAN + loss_G_L1
            self.log("model_loss/generator_GAN_loss", loss_G_GAN)
            self.log("model_loss/generator_L1", loss_G_L1)
            self.log("model_loss/generator_total_loss", loss_G)
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
                val_img = plot_tensors_hist(rgb, torch.clone(nir), torch.clone(nir_pred),title="Validation")
                ndvi_img = plot_ndvi(rgb, torch.clone(nir), torch.clone(nir_pred),title="Validation")
                self.logger.experiment.log({"Images/Val NIR":  wandb.Image(val_img)}) # log val image
                self.logger.experiment.log({"Images/Val NDVI":  wandb.Image(ndvi_img)}) # log val image
                self.log_dict({"val_stats/min_pred":torch.min(nir_pred).item(), # log stats
                            "val_stats/max_pred":torch.max(nir_pred).item(),
                            "val_stats/mean_pred":torch.mean(nir_pred).item()},
                            on_epoch=True,sync_dist=True)
                self.log_dict({"val_stats/min_input":torch.min(nir).item(), # log stats
                            "val_stats/max_input":torch.max(nir).item(),
                            "val_stats/mean_input":torch.mean(nir).item()},
                            on_epoch=True,sync_dist=True)
                
    def extract_batch(self, batch):
        rgb = batch["rgb"]
        nir = batch["nir"]
        return rgb, nir

    def configure_optimizers(self):
        optim_g = torch.optim.Adam(self.netG.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        optim_d = torch.optim.Adam(self.netD.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        sched_g = ReduceLROnPlateau(optim_g, mode='min', patience=self.config.Schedulers.patience_g)
        sched_d = ReduceLROnPlateau(optim_d, mode='min', patience=self.config.Schedulers.patience_d)
        return ([optim_d, optim_g], 
                [{'scheduler': sched_d, 'monitor': self.config.Schedulers.metric, 'interval': 'epoch'},
                 {'scheduler': sched_g, 'monitor': self.config.Schedulers.metric, 'interval': 'epoch'}])
    
    

# Testing
if __name__ == "__main__":
    config = OmegaConf.load("configs/config_px2px.yaml")
    m = Px2Px_PL(config)

    # try out simple forward
    rgb = torch.rand(5,3,512,512)
    nir = torch.rand(5,1,512,512)
    m = m.cpu()

    m = m.eval()
    pred= m.predict_step(rgb)
    m = m.train()

    # try out training step
    batch = {"rgb":rgb, "nir":nir}
    m.training_step(batch, batch_idx=0, optimizer_idx=1)
    m.validation_step(batch, batch_idx=0)
    



