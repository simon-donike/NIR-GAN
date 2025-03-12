import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from omegaconf import OmegaConf
import wandb,os
from utils.calculate_metrics import calculate_metrics
from utils.logging_helpers import plot_tensors_hist
from utils.logging_helpers import plot_index
#from model.pix2pix_model import Pix2PixModel
#from model.base_model import BaseModel
from model import networks
from utils.losses import ssim_loss
from utils.losses import emd_loss as hist_loss
from validation_utils.time_series_validation import calculate_and_plot_timeline
import gc


class Px2Px_PL(pl.LightningModule):
    def __init__(self, opt):
        super(Px2Px_PL, self).__init__()
        self.opt = opt.base_configs
        self.config = opt
        self.isTrain = self.opt.isTrain

        # TODO: THIS GETS CALLED TWICE; TRACE IT BACK!!!!!!
        # Choose Generator based on SatCLIP settings
        # (a) Concatenate SatCLIP to input
        if self.config.satclip.use_satclip == True and self.config.satclip.satclip_style=="concat":
            print("Creating SatCLIP Concatenation Generator.")
            self.netG = networks.define_G(self.opt.input_nc+1, self.opt.output_nc, self.opt.ngf, self.opt.netG, self.opt.norm,
                                          not self.opt.no_dropout, self.opt.init_type, self.opt.init_gain)
        # (b) Inject SatCLIP to model
        elif self.config.satclip.use_satclip == True and self.config.satclip.satclip_style=="inject":
            print(f"Creating SatCLIP Injection Generator with injection style: '{self.config.satclip.satclip_inject_style}'.")
            from model.generator_inject import define_G_inject
            self.netG = define_G_inject(input_nc = self.opt.input_nc,
                                        output_nc = self.opt.output_nc,
                                        inject_style = self.config.satclip.satclip_inject_style,
                                        post_correction = self.config.satclip.post_correction,
                                        ngf = self.opt.ngf,
                                        netG = self.opt.netG,
                                        norm = self.opt.norm,
                                        use_dropout= not self.opt.no_dropout,
                                        init_type = self.opt.init_type,
                                        init_gain = self.opt.init_gain)
            
        # (c) No SatCLIP - standard model
        else:
            print("Creating Standard Pix2Pix Generator.")
            self.netG = networks.define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.netG, self.opt.norm,
                                          not self.opt.no_dropout, self.opt.init_type, self.opt.init_gain)
            

        # Get Discriminator
        self.netD = networks.define_D(self.opt.input_nc + self.opt.output_nc, self.opt.ndf, self.opt.netD,
                                          self.opt.n_layers_D, self.opt.norm, self.opt.init_type, self.opt.init_gain)
        self.criterionGAN = networks.GANLoss(self.opt.gan_mode)
        self.criterionL1 = torch.nn.L1Loss()
        
        # define remote sensing losses if enabled
        if self.config.base_configs.lambda_rs_losses>0.0:
            from utils.remote_sensing_indices import RemoteSensingIndices
            crit = self.config.base_configs.rs_losses_criterium
            self.rs_losses = RemoteSensingIndices(mode="loss",criterion=crit)
            
        # define weather SatClip to be used or not
        if "satclip" in self.config:
            self.satclip = self.config.satclip.use_satclip
            if self.satclip:
                # Using Wrapper
                from model.satclip.satclip_wrapper import SatClIP_wrapper
                self.satclip_model = SatClIP_wrapper(device=self.device)
                self.satclip_model = self.satclip_model.eval()
                """
                # straight loading
                from model.satclip.load import get_satclip
                satclip_path = self.config.satclip.satclip_path
                self.satclip_model = get_satclip(satclip_path)
                """
            else:
                self.satclip = False
        else:
            self.satclip = False

        
    def forward(self, input,embeds=None):
        if self.satclip==False:
            pred = self.netG(input)
        else: #self.satclip==True:
            if self.config.satclip.satclip_style=="concat":
                pred = self.netG(input)
            elif self.config.satclip.satclip_style=="inject":
                pred = self.netG(input,embeds)
            else:
                raise NotImplementedError("SatClip Style not recognized")
        return pred
    
    def on_train_batch_start(self, batch, batch_idx):
        # Reset cache at the start of each batch
        # Tis holds the pred so that we can do it only once per optimizer
        #self.pred_cache = None
        pass
            
    def clean_checkpoint(self,checkpoint_path,unexpected_keys=[]):
        # Removes Unexpected keys from checkpoint
        # Use this right adfter instanciating the model when facing this issue
        checkpoint = torch.load(checkpoint_path)        
        # Remove the unexpected key
        unexpected_keys = unexpected_keys
        for k in unexpected_keys:
            if k in checkpoint['state_dict']:
                del checkpoint['state_dict'][k]
        # Save the modified checkpoint or return it
        torch.save(checkpoint, checkpoint_path)
        print("Removed unexpected keys from checkpoint: ",unexpected_keys)
        return checkpoint_path


    @torch.no_grad()
    def predict_step(self, rgb,coords=None):
        assert self.training == False, "Model is in training mode, set to eval mode before predicting"        
        
        """ 1. Extract and Predict """
        if self.satclip==False:
            batch = {"rgb":rgb,"nir":torch.Tensor([0])}
            rgb,_ = self.extract_batch(batch) # only gets rgb and nir
            nir_pred = self.forward(rgb) # pred on only rgb
        else: # self.satclip==True
            if self.config.satclip.satclip_style=="concat":
                batch = {"rgb":rgb,"nir":torch.Tensor([0]),"coords":coords}
                rgb,_ = self.extract_batch(batch) # handles embedding extraction
                nir_pred = self.forward(rgb) # pred on rgb+embeds 
            elif self.config.satclip.satclip_style=="inject":
                batch = {"rgb":rgb,"nir":torch.Tensor([0]),"coords":coords}
                rgb,_,embeds = self.extract_batch(batch)
                nir_pred = self.forward(rgb,embeds) # pred on rgb+embeds 
            else:
                raise NotImplementedError("SatClip Style not recognized, choose 'concat' or 'inject'")

        # Return Prediction        
        return nir_pred

    def training_step(self, batch, batch_idx, optimizer_idx):
        assert self.training == True, "Model is in eval mode, set to training mode before training"
        
        # Extract batch and batch Info
        if self.satclip==False:
            rgb,nir = self.extract_batch(batch) # only gets rgb and nir
        else: # self.satclip==True
            if self.config.satclip.satclip_style=="concat":
                rgb,nir = self.extract_batch(batch) # handles embedding extraction
            if self.config.satclip.satclip_style=="inject":
                rgb,nir,embeds = self.extract_batch(batch)
        
        if self.satclip and self.config.satclip.satclip_style=="inject":
            pred = self.forward(rgb,embeds) 
        else:
            pred = self.forward(rgb)

        # Calculate and Log Train Metrics  - only every 10th batch on Gen step
        if optimizer_idx == 0 and batch_idx % 10 == 0 and self.logger and hasattr(self.logger, 'experiment'):
            metrics = calculate_metrics(pred=torch.clone(pred).cpu(),target=torch.clone(nir).cpu(),phase="train")
            self.log_dict(metrics,on_step=True,sync_dist=True)
            del metrics
            
            # if model has scale parameter, log it
            if hasattr(self.netG, "scale_param"):
                self.log("scale_param", self.netG.scale_param.item())
            if hasattr(self.netG, "post_correction_param"):
                self.log("post_correction_param", self.netG.post_correction_param.item())
                
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
            del pred_fake, pred_real, fake_AB, real_AB
            torch.cuda.empty_cache()
            return loss_D

        # Generator Step
        if optimizer_idx == 1:
            fake_AB = torch.cat((rgb, pred), 1)
            pred_fake = self.netD(fake_AB)
            
            # Calculate Standard Losses
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            self.log("model_loss/generator_GAN_loss", loss_G_GAN)
            loss_G_L1 = self.criterionL1(pred, nir)
            self.log("model_loss/generator_L1", loss_G_L1)

            # Weight Standard Losses
            loss_G_GAN_weighted = loss_G_GAN * self.config.base_configs.lambda_GAN
            loss_G_L1_weighted = loss_G_L1 * self.config.base_configs.lambda_L1
            # sum standard losses
            loss_G = loss_G_GAN_weighted + loss_G_L1_weighted

            # CALCULATE FANCY LOSSES
            # Calculate, Weight and Log SSIM Losses
            if self.config.base_configs.lambda_ssim>0.0:
                loss_G_ssim = ssim_loss(pred, nir)
                self.log("model_loss/generator_ssim", loss_G_ssim) 
                loss_G_ssim_weighted = loss_G_ssim * self.config.base_configs.lambda_ssim
                loss_G = loss_G + loss_G_ssim_weighted

            if self.config.base_configs.lambda_hist>0.0:
                loss_G_hist = hist_loss(pred, nir)
                self.log("model_loss/generator_hist", loss_G_hist)
                loss_G_hist_weighted = loss_G_hist * self.config.base_configs.lambda_hist
                loss_G = loss_G + loss_G_hist_weighted
            
            # Calculate, Weight and Log RS Losses
            if self.config.base_configs.lambda_rs_losses>0.0:
                losses_rs_indices = self.rs_losses.get_and_weight_losses(rgb,nir,pred,
                                                                         loss_config=dict(self.config.base_configs.internal_rs_loss_weights))
                self.log("model_loss/indices_loss_weighted", losses_rs_indices)
                losses_rs_indices_weighted = losses_rs_indices * self.config.base_configs.lambda_rs_losses
                loss_G = loss_G + losses_rs_indices_weighted
            
            # log total weighted loss
            self.log("model_loss/generator_total_loss", loss_G) # log final loss   
            del fake_AB, pred_fake
            torch.cuda.empty_cache()
            return loss_G
        
    @torch.no_grad()
    def validation_step(self,batch,batch_idx):
        
        """ 1. Extract and Predict """
        if self.satclip==False:
            rgb,nir = self.extract_batch(batch) # only gets rgb and nir
            nir_pred = self.predict_step(rgb) # pred on only rgb
        else: # self.satclip==True
            if self.config.satclip.satclip_style=="concat":
                rgb,nir = self.extract_batch(batch) # handles embedding extraction
                nir_pred = self.predict_step(rgb) # pred on rgb+embeds 
            if self.config.satclip.satclip_style=="inject":
                rgb,nir,embeds = self.extract_batch(batch)
                nir_pred = self.predict_step(rgb,embeds) # pred on rgb+embeds 
            
        
        # Extract Embeddings if SatClip is enabled
        rgb = rgb[:,:3,:,:] # get first 3 chanels, depending on settings channel 4 might be embeddings

        """ 2. Log Generator Metrics """
        # log image metrics     
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
                
                # get and log RS indices losses
                if self.config.base_configs.lambda_rs_losses>0.0:
                    indices_dict = self.rs_losses.get_and_weight_losses(rgb,nir,nir_pred,mode="logging_dict")
                    self.log_dict(indices_dict,on_epoch=True,sync_dist=True)
                    del indices_dict

                    
    def on_validation_epoch_end(self):
        # if first epoch, save config to experiment path
        try:
            if self.current_epoch==0:
                out_path = os.path.join(self.dir_save_checkpoints,"config.yaml")
                OmegaConf.save(self.config, out_path)
        except:
            pass
            
        # Time Series Logging
        if self.logger and hasattr(self.logger, 'experiment'):
            # predict time series and get plot for WandB
            pil_image_bavaria = calculate_and_plot_timeline(model = self,
                                                    device=self.device,
                                                    root_dir="validation_utils/time_series_bavaria/*.tif",
                                                    size_input=self.config.Data.S2_100k.image_size,
                                                    mean_patch_size=4)
            self.logger.experiment.log({"Images/Timeline Bavaria":  wandb.Image(pil_image_bavaria)}) # log plot
            del pil_image_bavaria

            pil_image_texas = calculate_and_plot_timeline(model = self,
                                                    device=self.device,
                                                    root_dir="validation_utils/time_series_texas/*.tif",
                                                    size_input=self.config.Data.S2_100k.image_size,
                                                    mean_patch_size=4)
            self.logger.experiment.log({"Images/Timeline Texas":  wandb.Image(pil_image_texas)}) # log plot
            del pil_image_texas

            pil_image_michigan = calculate_and_plot_timeline(model = self,
                                                    device=self.device,
                                                    root_dir="validation_utils/time_series_michigan/*.tif",
                                                    size_input=self.config.Data.S2_100k.image_size,
                                                    mean_patch_size=4)
            self.logger.experiment.log({"Images/Timeline Michigan":  wandb.Image(pil_image_michigan)}) # log plot
            del pil_image_michigan

            pil_image_california = calculate_and_plot_timeline(model = self,
                                                    device=self.device,
                                                    root_dir="validation_utils/time_series_california/*.tif",
                                                    size_input=self.config.Data.S2_100k.image_size,
                                                    mean_patch_size=4)
            self.logger.experiment.log({"Images/Timeline California":  wandb.Image(pil_image_california)}) # log plot
            del pil_image_california

            pil_image_texas_cropcircles = calculate_and_plot_timeline(model = self,
                                                    device=self.device,
                                                    root_dir="validation_utils/time_series_texas_cropcircles/*.tif",
                                                    size_input=self.config.Data.S2_100k.image_size,
                                                    mean_patch_size=4)
            self.logger.experiment.log({"Images/Timeline Tx_CropCircles":  wandb.Image(pil_image_texas_cropcircles)}) # log plot
            del pil_image_texas_cropcircles

            pil_image_brazil = calculate_and_plot_timeline(model = self,
                                                    device=self.device,
                                                    root_dir="validation_utils/time_series_brazil/*.tif",
                                                    size_input=self.config.Data.S2_100k.image_size,
                                                    mean_patch_size=4)
            self.logger.experiment.log({"Images/Timeline Brazil":  wandb.Image(pil_image_brazil)}) # log plot
            del pil_image_brazil

            pil_image_iowa = calculate_and_plot_timeline(model = self,
                                                    device=self.device,
                                                    root_dir="validation_utils/time_series_iowa/*.tif",
                                                    size_input=self.config.Data.S2_100k.image_size,
                                                    mean_patch_size=4)
            self.logger.experiment.log({"Images/Timeline Iowa":  wandb.Image(pil_image_iowa)}) # log plot
            del pil_image_iowa

        else: # save to local if there is no logger being used
            pass
            #pil_image_bavaria.save("validation_utils/timeline_bavaria.png")
            #pil_image_texas.save("validation_utils/timeline_texas.png")
            #pil_image_michigan.save("validation_utils/timeline_michigan.png")

        # delete images to avoid memory issues
        torch.cuda.empty_cache()
        gc.collect()
                
    def extract_batch(self, batch):
        """
        Handles the extraction and return of input and output.
        If wanted, also handles encoding of location information and appending to input tensor.
        Either:
            - No Coords
            - Concatenation of Coord Embeddings to RGB input
            - Injection of Coord Embeddings to model

        Args:
            batch (dict): with keys "rgb" and "nir" containing the respective tensors, optionally "coords" with location information

        Returns:
            _type_: input and target of model
        """
        rgb = batch["rgb"]
        nir = batch["nir"]
        
        # return if no location encoding wanted
        if not self.satclip:
            return rgb, nir
        else: # if self.satclip==True
            coords = batch["coords"] # extract coordinates

            # If Concatenation, predict and reshape embeddings to apppend to RTG
            if self.config.satclip.satclip_style=="concat":
                rgb = self.satclip_get_concat(coords,rgb)
                return(rgb,nir)
            
            # If Injection, predict embeds and return all
            elif self.config.satclip.satclip_style=="inject":
                embeds = self.satclip_get_inject(coords)
                return rgb,nir,embeds
            
            # if none of the satclip styles, raise error
            else:
                raise NotImplementedError("SatClip Style not recognized, choose 'concat' or 'inject'")            
        
            
    def satclip_get_concat(self,coords,rgb):
        # Predict Embeddings
        with torch.no_grad():
            coords_embeds = self.satclip_model.predict(coords.double()).float()
        # Manipulate Embeddigs to be concatenated to RGB
        coords_embeds = coords_embeds.view(rgb.shape[0], 1, 1, 256) # add 1 dimension
        coords_embeds = coords_embeds.expand(rgb.shape[0], 1, 256, 256) # expand to height dimension
        coords_embeds = torch.nn.functional.interpolate(coords_embeds, size=(rgb.shape[-1],rgb.shape[-2]), mode='bicubic') # interpolate to image size
        coords_embeds = coords_embeds * self.config.satclip.scaling_factor # scale satclip numbers to closer match input distribution
        rgb = torch.cat((rgb, coords_embeds), dim=1) # stack location embeddings to rgb conditioning
        return(rgb)
    
    def satclip_get_inject(self,coords):
        # get and return predictions
        with torch.no_grad():
            coords_embeds = self.satclip_model.predict(coords.double()).float()
        return(coords_embeds)


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
    """
    # Testing for SatCLIP inclusion
    rgb = torch.rand(5,3,512,512)
    nir = torch.rand(5,1,512,512)
    coords = torch.rand(5,2)
    batch = {"rgb":rgb, "nir":nir, "coords":coords}
    l0=m.training_step(batch, batch_idx=0, optimizer_idx=0)
    l1= m.training_step(batch, batch_idx=0, optimizer_idx=1)
    """
    # Test Model
    config = OmegaConf.load("configs/config_px2px.yaml")
    m = Px2Px_PL(config)
    
    # Test Predict Step
    m = m.eval()
    pred = m.predict_step(torch.rand(5,3,512,512),torch.rand(5,256))
    print("Pred works: ",pred.shape)
    
    # Test Train Step
    m = m.train()
    rgb = torch.rand(5,3,512,512)
    nir = torch.rand(5,1,512,512)
    coords = torch.rand(5,256)
    batch = {"rgb":rgb, "nir":nir, "coords":coords}
    l0=m.training_step(batch, batch_idx=0, optimizer_idx=0)
    l1= m.training_step(batch, batch_idx=0, optimizer_idx=1)
    print("Train works: ",l0.item()," - ",l1.item())
    
    # test validation step
    m = m.eval()
    m.validation_step(batch,0)
    m = m.train()
    
    
    



