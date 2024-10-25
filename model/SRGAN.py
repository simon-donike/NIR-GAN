# Package Imports
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import numpy as np
import time
from omegaconf import OmegaConf
import wandb
from einops import rearrange
import random

# local imports
from utils.calculate_metrics import calculate_metrics
from utils.logging_helpers import plot_tensors
from utils.normalise_s2 import normalise_s2
from utils.dataloader_utils import histogram as histogram_match


#############################################################################################################
# Build PL MODEL


class SRGAN_model(pl.LightningModule):

    def __init__(self, config_dict="configs/config_NIR.yaml"):
        super(SRGAN_model, self).__init__()

        # get config file
        if type(config_dict)==str:
            self.config = OmegaConf.load(config_dict)
        else:
            self.config = config_dict

        """ IMPORT MODELS """
        # if MISR is wanted, instantiate fusion net
        if self.config.SR_type=="MISR":
            from model.fusion import RecursiveNet,RecursiveNet_pl
            self.fusion = RecursiveNet_pl()
                # load pretrained weights
            if self.config.Model.load_fusion_checkpoint:
                self.fusion = RecursiveNet_pl.load_from_checkpoint(self.config.Model.fusion_ckpt_path, strict=False)

        # Generator
        from model.model_blocks import Generator
        self.generator = Generator(large_kernel_size=self.config.Generator.large_kernel_size,
                        small_kernel_size=self.config.Generator.small_kernel_size,
                        n_channels=self.config.Generator.n_channels,
                        n_blocks=self.config.Generator.n_blocks,
                        scaling_factor=self.config.Generator.scaling_factor,
                        bands_in=self.config.Generator.bands_in,
                        bands_out=self.config.Generator.bands_out)
        
        # Discriminator
        from model.model_blocks import Discriminator
        self.discriminator = Discriminator(kernel_size=self.config.Discriminator.kernel_size,
                            n_channels=self.config.Discriminator.n_channels,
                            n_blocks=self.config.Discriminator.n_blocks,
                            fc_size=self.config.Discriminator.fc_size,
                            bands_in=self.config.Discriminator.bands_in)
        
        # VGG for encoding
        from model.model_blocks import TruncatedVGG19
        self.truncated_vgg19 = TruncatedVGG19(i=self.config.TruncatedVGG.i, j=self.config.TruncatedVGG.j)
        # freeze VGG19
        for param in self.truncated_vgg19.parameters():
            param.requires_grad = False

        # set up Losses
        self.content_loss_criterion = torch.nn.MSELoss()
        self.adversarial_loss_criterion = torch.nn.BCEWithLogitsLoss()


    def forward(self,lr_imgs):
        # if MISR, perform Fusion first
        if self.config.SR_type=="MISR":
            lr_imgs = self.fusion(lr_imgs)
        # perform generative Step
        sr_imgs = self.generator(lr_imgs)
        return(sr_imgs)
    

    @torch.no_grad()
    def predict_step(self,rgb,normalize=True):
        """
        This function is for the prediction in the Deployment stage, therefore
        the normalization and denormalization needs to happen here.
        Input:
            - unnormalized lLR imgs
        Output:
            - normalized SR images
        Info:
            - This function currently only performs SISR SR
        """
        # assert model in eval mode
        assert self.training==False
        
        # move to GPU if possible
        rgb = rgb.to(self.device)
        # normalize images
        if normalize:
            rgb = normalise_s2(rgb,stage="norm")
        # preform SR
        with torch.no_grad():
            nir_pred = self.generator(rgb)
        # denormalize images
        if normalize:
            nir_pred = normalise_s2(nir_pred,stage="denorm")

        return nir_pred


    def training_step(self,batch,batch_idx,optimizer_idx):
        # access data
        rgb,nir = self.extract_batch(batch,overwrite_norm=False)

        nir = normalise_s2(nir,stage="norm")
        # generate NIR images, log losses immediately
        nir_pred = self.forward(rgb)

        nir_denorm = normalise_s2(torch.clone(nir),stage="denorm")
        nir_pred_denorm = normalise_s2(torch.clone(nir_pred),stage="denorm")

        metrics = calculate_metrics(nir_denorm,nir_pred_denorm,phase="train")
        #for key, value in metrics.items():
        #    self.log(f'{key}', value)
        # log dict
        self.log_dict(metrics,on_step=True,on_epoch=True,sync_dist=True)
        
        # Discriminator Step & Loss
        if optimizer_idx==0:
            # run discriminator and get loss between pred labels and true labels
            nir_discriminated = self.discriminator(nir)
            pred_discriminated = self.discriminator(nir_pred)
            adversarial_loss = self.adversarial_loss_criterion(pred_discriminated, torch.ones_like(pred_discriminated))

            # Binary Cross-Entropy loss
            adversarial_loss = self.adversarial_loss_criterion(pred_discriminated,
                                                        torch.zeros_like(pred_discriminated)) +self.adversarial_loss_criterion(nir_discriminated,
                                                        torch.ones_like(nir_discriminated))
            # logg Discriminator loss
            self.log("discriminator/adverserial_loss",adversarial_loss)

            # return weighted discriminator loss
            return adversarial_loss

        # Generator Step & Loss
        if optimizer_idx==1:
            
            """ 1. Get VGG space loss """

            # calc loss
            pred_imgs_in_vgg_space = self.truncated_vgg19(nir_pred.repeat(1, 3, 1, 1))
            nir_imgs_in_vgg_space = self.truncated_vgg19(nir.repeat(1, 3, 1, 1)).detach()  # detached because they're constant, targets
            
            # Calculate the Perceptual loss between VGG encoded images to receive content loss
            content_loss = self.content_loss_criterion(pred_imgs_in_vgg_space, nir_imgs_in_vgg_space)
            
            """ 2. Get Discriminator Opinion and loss """
            # run discriminator and get loss between pred labels and true labels
            pred_discriminated = self.discriminator(nir_pred)
            adversarial_loss = self.adversarial_loss_criterion(pred_discriminated, torch.ones_like(pred_discriminated))
            
            """ 3. Weight the losses"""
            perceptual_loss = content_loss + self.config.Losses.adv_loss_beta * adversarial_loss

            """ 4. Log Generator Loss """
            self.log("generator/discr_gan_loss",perceptual_loss)
            self.log("generator/adversarial_loss",adversarial_loss)
            
            # return Generator loss
            return perceptual_loss
        
    @torch.no_grad()
    def validation_step(self,batch,batch_idx):
        
        """ 1. Extract and Predict """
        rgb,nir = self.extract_batch(batch)
                
        # Predict
        nir_pred = self.predict_step(rgb,normalize=True)

        """ 2. Log Generator Metrics """
        # log image metrics
        metrics_nir_img = torch.clone(nir)  # deep copy to avoid graph problems
        metrics_pred_img = torch.clone(nir_pred) # deep copy to avoid graph problems
        metrics = calculate_metrics(metrics_pred_img.cpu(),metrics_nir_img.cpu(),phase="val")
        self.log_dict(metrics,on_step=True,on_epoch=True,sync_dist=True)

        # only perform image logging for n pics, not all 200
        if batch_idx<self.config.Logging.num_val_images:
            # log Stadard image visualizations, deep copy to avoid graph problems
            val_img = plot_tensors(rgb, nir, nir_pred,title="Validation",
                                   stretch=None, #self.config.Data.dataset_type,
                                   config=self.config)
            self.logger.experiment.log({"Images/Val SR":  wandb.Image(val_img)}) # log val image
            self.log_dict({"pred_stats/min":torch.min(nir_pred).item(), # log stats
                           "pred_stats/max":torch.max(nir_pred).item(),
                           "pred_stats/mean":torch.min(nir_pred).item()},
                           on_step=True,on_epoch=True,sync_dist=True)
           
        """ 3. Log Discriminator metrics """
        # run discriminator and get loss between pred labels and true labels
        nir_discriminated = self.discriminator(nir)
        pred_discriminated = self.discriminator(nir_pred)
        adversarial_loss = self.adversarial_loss_criterion(pred_discriminated, torch.ones_like(nir_discriminated))

        # Binary Cross-Entropy loss
        adversarial_loss = self.adversarial_loss_criterion(pred_discriminated,
                                                        torch.zeros_like(pred_discriminated)) + self.adversarial_loss_criterion(nir_discriminated,
                                                                                                                                torch.ones_like(nir_discriminated))
        self.log("validation/DISC_adversarial_loss",adversarial_loss,sync_dist=True)


    @torch.no_grad()
    def on_validation_epoch_end(self):
        pass

    def extract_batch(self,batch,overwrite_norm=False):
        rgb = batch["rgb"]
        nir = batch["nir"]
        return rgb,nir

    def configure_optimizers(self):

        # configure Generator SISR/MISR optimizers
        optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.generator.parameters()),lr=self.config.Optimizers.optim_g_lr)

        # configure Discriminator optimizers
        optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.discriminator.parameters()),lr=self.config.Optimizers.optim_d_lr)

        # configure schedulers
        scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=self.config.Schedulers.factor_g, patience=self.config.Schedulers.patience_g, verbose=self.config.Schedulers.verbose)
        scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', factor=self.config.Schedulers.factor_d, patience=self.config.Schedulers.patience_d, verbose=self.config.Schedulers.verbose)

        # return schedulers and optimizers
        return [
                    [optimizer_d, optimizer_g],
                    [{'scheduler': scheduler_d, 'monitor': self.config.Schedulers.metric, 'reduce_on_plateau': True, 'interval': self.config.Schedulers.interval, 'frequency': 1},
                     {'scheduler': scheduler_g, 'monitor': self.config.Schedulers.metric, 'reduce_on_plateau': True, 'interval': self.config.Schedulers.interval, 'frequency': 1}],
                ]

