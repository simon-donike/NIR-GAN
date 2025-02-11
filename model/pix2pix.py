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
from utils.losses import ssim_loss
from utils.losses import hist_loss



class Px2Px_PL(pl.LightningModule):
    def __init__(self, opt):
        super(Px2Px_PL, self).__init__()
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
        
    def on_train_start(self):
        # runs before training, used to set buffer shapes for multi GPU training optimization
        # get batch to see shapes
        try:
            sample_batch = next(iter(self.trainer.datamodule.train_dataloader()))
        except AttributeError:
            print("No datamodule found, using test batch")
            sample_batch = torch.zeros(5,1,512,512)
        _, nir = self.extract_batch(sample_batch) 
        # set Cache - Important: Needs to be correct shape!
        self.register_buffer("pred_cache", torch.zeros_like(nir))
        
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
        # handle padding
        # trained model expectes 512 + 10 pads in each direction. Results might be worse without padding.
        if rgb.shape[-1]<= 512+2*self.opt.pad:
            rgb = torch.nn.functional.pad(rgb,(self.opt.pad,self.opt.pad,self.opt.pad,self.opt.pad),mode="reflect")
        if rgb.shape[-2]<= 512+2*self.opt.pad:
            pass # TODO: implement cropping of middle if image is bigger
        nir_pred = self.forward(rgb)
        return nir_pred

    def training_step(self, batch, batch_idx, optimizer_idx):
        assert self.training == True, "Model is in eval mode, set to training mode before training"
        rgb, nir = self.extract_batch(batch)

        # Checking if using optimization - (optim only available when training with PL):
        # 1. Optimization enabled in Settings
        # 2. Buffer exists and is registered as 'pred_cache'
        using_optimization = self.config.base_configs.use_training_pred_optimization and "pred_cache" in dict(self.named_buffers())
        # contiion to perform registering of prediction in order to do it once per training step
        if using_optimization:
            if optimizer_idx == 0 or torch.all(self.pred_cache == 0):
                pred = self.forward(rgb)
                print("Step0: Buffer: ",self.pred_cache.mean())
                self.pred_cache.copy_(pred)
                print("Step0: Writing to buffer",self.pred_cache.mean())
            if optimizer_idx == 1:
                print("Step1: Reading from buffer:",self.pred_cache.mean())
                pred = self.pred_cache
        # if not using optimization, just predict it again for optimizer 1
        else:
            pred = self.forward(rgb)

        if optimizer_idx == 0 and batch_idx % 10 == 0:
            metrics = calculate_metrics(pred=torch.clone(pred).cpu(),target=torch.clone(nir).cpu(),phase="train")
            self.log_dict(metrics,on_step=True,sync_dist=True)

        # Discriminator Step
        if optimizer_idx == 0:
            # 1.1.1 Fake
            fake_AB = torch.cat((rgb, pred), 1)
            pred_fake = self.netD(fake_AB.detach())
            self.log("model_loss/discriminator_predFake", pred_fake.mean())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # 1.1.2 Real
            real_AB = torch.cat((rgb, nir), 1)
            pred_real = self.netD(real_AB)
            self.log("model_loss/discriminator_predReal", pred_real.mean())
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D = (loss_D_fake + loss_D_real)
            self.log("model_loss/discriminator_real", loss_D_real)
            self.log("model_loss/discriminator_fake", loss_D_fake)
            self.log("model_loss/discriminator_loss", loss_D)
            return loss_D

        # Generator Step
        if optimizer_idx == 1:
            fake_AB = torch.cat((rgb, pred), 1)
            pred_fake = self.netD(fake_AB)
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            self.log("model_loss/generator_GAN_loss", loss_G_GAN)
            loss_G_L1 = self.criterionL1(pred, nir)
            self.log("model_loss/generator_L1", loss_G_L1)
            loss_G_ssim = ssim_loss(pred, nir)
            self.log("model_loss/generator_ssim", loss_G_ssim)
            loss_G_hist = hist_loss(pred, nir)
            self.log("model_loss/generator_hist", loss_G_hist)
            # weight losses
            loss_G_GAN_weighted = loss_G_GAN * self.config.base_configs.lambda_GAN
            loss_G_L1_weighted = loss_G_L1 * self.config.base_configs.lambda_L1
            loss_G_ssim_weighted = loss_G_ssim * self.config.base_configs.lambda_ssim
            loss_G_hist_weighted = loss_G_hist * self.config.base_configs.lambda_hist
            # final weighting
            loss_G = loss_G_GAN_weighted + loss_G_L1_weighted + loss_G_ssim_weighted + loss_G_hist_weighted
            self.log("model_loss/generator_total_loss", loss_G)
            
            # if optimizing with registered buffer, oveerwrite buffer with empty tensor
            if using_optimization:
                self.pred_cache.detach_().zero_()            
            return loss_G
        
    @torch.no_grad()
    def validation_step(self,batch,batch_idx):
        
        """ 1. Extract and Predict """
        rgb,nir = self.extract_batch(batch)
                        
        # Predict, returns denormed NIR
        nir_pred = self.predict_step(rgb)

        """ 2. Log Generator Metrics """
        # log image metrics     
        if self.logger and hasattr(self.logger, 'experiment'):   
            metrics = calculate_metrics(pred=torch.clone(nir_pred).cpu(),target=torch.clone(nir).cpu(),phase="val")
            self.log_dict(metrics,on_step=False,on_epoch=True,sync_dist=True)

        # only perform image logging for n pics, not all in val loader
        if batch_idx<self.config.custom_configs.Logging.num_val_images:
            if self.logger and hasattr(self.logger, 'experiment'):
                # log Stadard image visualizations, deep copy to avoid graph problems
                val_img = plot_tensors_hist(rgb, torch.clone(nir), torch.clone(nir_pred),title="Validation")
                ndvi_img = plot_ndvi(rgb, torch.clone(nir), torch.clone(nir_pred),title="Validation")
                self.logger.experiment.log({"Images/Val NIR":  wandb.Image(val_img)}) # log val image
                if False:
                    ndvi_img = plot_ndvi(rgb, torch.clone(nir), torch.clone(nir_pred),title="Validation")
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
    m.on_train_start()

    # try out simple forward
    rgb = torch.rand(5,3,512,512)
    nir = torch.rand(5,1,512,512)
    m = m.cpu()
    
    # try out training step
    def simulate_1_step():
        # try out simple forward
        rgb = torch.rand(5,3,512,512)
        nir = torch.rand(5,1,512,512)
        batch = {"rgb":rgb, "nir":nir}
        l0=m.training_step(batch, batch_idx=0, optimizer_idx=0)
        l1= m.training_step(batch, batch_idx=0, optimizer_idx=1)

    from datetime import datetime
    from tqdm import tqdm
    start = datetime.now()
    for i in tqdm(range(10)):
        simulate_1_step()
    print(datetime.now()-start)
    
    #m.validation_step(batch, batch_idx=0)
    



