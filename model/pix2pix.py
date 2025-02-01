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
from model.base_model import BaseModel
from model import networks


class Px2Px_PL(pl.LightningModule):
    """
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
    """
        
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
        #if self.training:
        #    self.model = self.model.eval()
        #assert not self.training, "Model should be in eval mode for predictions"
        if self.config.Data.normalize:
            from utils.normalise_s2 import normalize_rgb, normalize_nir
            rgb_norm = normalize_rgb(rgb, stage="norm")
            nir_pred = self.netG(rgb_norm)
            return normalize_nir(nir_pred, stage="denorm")
        else:
            nir_pred = self.netG(rgb)
            return nir_pred

    def training_step(self, batch, batch_idx, optimizer_idx):
        rgb, nir = self.extract_batch(batch)
        if self.config.Data.normalize:
            rgb = normalize_rgb(rgb, stage="norm")
            nir = normalize_nir(nir, stage="norm")

        # check for OPtimization regarding multiple pred steps in 1 training step
        # TODO: get optimization to work with cache, keep gradients intact
        use_optimization = False
        if use_optimization:
            # If Cache is empty, run prediction and save cache
            if self.pred_cache is None:
                pred = self.netG(rgb)
                self.pred_cache = pred
                print("Cache is empty, running prediction")
            else:
                pred = self.pred_cache
                print("Cache is not empty, using cache")
        else:
            pred = self.netG(rgb)


        # 1. Backward_G
        # TODO: This seems the wring way around. Veryfiy order
        # TODO: Log metrics such as L1 and stuff at every step
        if optimizer_idx == 1:
            # 1.1 emulate backward_D
            # 1.1.1 Fake
            fake_AB = torch.cat((rgb, pred), 1)
            pred_fake = self.netD(fake_AB)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # 1.1.2 Real
            real_AB = torch.cat((rgb, nir), 1)
            pred_real = self.netD(real_AB)
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            self.log("generator_loss", loss_D)
            return loss_D

        # Discriminator Step
        if optimizer_idx == 0:
            fake_AB = torch.cat((rgb, pred), 1)
            pred_fake = self.netD(fake_AB)
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            loss_G_L1 = self.criterionL1(pred, nir) * self.config.base_configs.lambda_L1
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



