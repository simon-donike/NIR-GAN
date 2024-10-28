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

class Px2Px(pl.LightningModule):
    def __init__(self, config_dict="configs/config_NIR.yaml"):
        super(Px2Px, self).__init__()

        # Load configuration
        if isinstance(config_dict, str):
            self.config = OmegaConf.load(config_dict)
        else:
            self.config = config_dict

        # Import and initialize Generator and Discriminator
        from model.pix2pix_modules import Generator128 as Generator
        from model.pix2pix_modules import Discriminator128 as Discriminator

        # Initialize generator and discriminator with specified parameters
        self.generator = Generator(input_dim=3, num_filter=64, output_dim=1)
        self.discriminator = Discriminator(input_dim=1, num_filter=64,output_dim=1)

        # Initialize weights as per the previous script
        self.generator.normal_weight_init(mean=0.0, std=0.02)
        self.discriminator.normal_weight_init(mean=0.0, std=0.02)

        # BCE loss
        self.L1_loss = torch.nn.MSELoss()
        self.BCE_loss = torch.nn.BCELoss()
        

    def forward(self, lr_imgs):
        return self.generator(lr_imgs)

    @torch.no_grad()
    def predict_step(self, rgb, normalize=True):
        assert not self.training, "Model should be in eval mode for predictions"
        rgb = rgb.to(self.device)
        if normalize:
            from utils.normalise_s2 import normalize_rgb, normalize_nir
            rgb_norm = normalize_rgb(rgb, stage="norm")
            nir_pred = self.generator(rgb_norm)
            return normalize_nir(nir_pred, stage="denorm")
        else:
            return self.generator(rgb)

    def training_step(self, batch, batch_idx, optimizer_idx):
        rgb, nir = self.extract_batch(batch)
        rgb_norm = normalize_rgb(rgb, stage="norm")
        nir_pred = self.generator(rgb_norm)

        # Generator Step
        if optimizer_idx == 1:
            content_loss = self.L1_loss(nir_pred, normalize_nir(nir, stage="norm"))
            fake_disc = self.discriminator(nir_pred).squeeze()
            adv_loss = self.BCE_loss(fake_disc, torch.ones_like(fake_disc))
            loss = content_loss + self.config.Losses.adv_loss_beta * adv_loss
            self.log("generator_loss", loss)
            return loss

        # Discriminator Step
        if optimizer_idx == 0:
            fake_disc = self.discriminator(nir_pred).squeeze()
            real_dic = self.discriminator(nir).squeeze()
            fake_loss = self.BCE_loss(fake_disc,torch.zeros_like(fake_disc))
            real_loss = self.BCE_loss(real_dic,torch.ones_like(real_dic))
            disc_loss = (real_loss + fake_loss) / 2
            return disc_loss
        
    @torch.no_grad()
    def validation_step(self,batch,batch_idx):
        
        """ 1. Extract and Predict """
        rgb,nir = self.extract_batch(batch)
                        
        # Predict, returns denormed NIR
        nir_pred = self.predict_step(rgb,normalize=True)

        """ 2. Log Generator Metrics """
        # log image metrics        
        metrics = calculate_metrics(pred=torch.clone(nir_pred).cpu(),target=torch.clone(nir).cpu(),phase="val")
        self.log_dict(metrics,on_step=False,on_epoch=True,sync_dist=True)

        # only perform image logging for n pics, not all 200
        if batch_idx<self.config.Logging.num_val_images:
            # log Stadard image visualizations, deep copy to avoid graph problems
            val_img = plot_tensors(rgb, torch.clone(nir), torch.clone(nir_pred),title="Validation")
            ndvi_img = plot_ndvi(rgb, torch.clone(nir), torch.clone(nir_pred),title="Validation")
            self.logger.experiment.log({"Images/Val NIR":  wandb.Image(val_img)}) # log val image
            self.logger.experiment.log({"Images/Val NDVI":  wandb.Image(ndvi_img)}) # log val image
            self.log_dict({"pred_stats/min":torch.min(torch.clone(nir_pred)).item(), # log stats
                           "pred_stats/max":torch.max(torch.clone(nir_pred)).item(),
                           "pred_stats/mean":torch.min(torch.clone(nir_pred)).item()},
                           on_epoch=True,sync_dist=True)
           
        """ 3. Log Discriminator metrics """
        # run discriminator and get loss between pred labels and true labels
        nir_discriminated = self.discriminator(nir)
        pred_discriminated = self.discriminator(nir_pred)
        adversarial_loss = self.BCE_loss(pred_discriminated, torch.ones_like(nir_discriminated))

        # Binary Cross-Entropy loss
        adversarial_loss = self.BCE_loss(pred_discriminated,
                                            torch.zeros_like(pred_discriminated)) + self.BCE_loss(nir_discriminated,
                                            torch.ones_like(nir_discriminated))
        self.log("val/Disc_adversarial_loss",adversarial_loss,sync_dist=True)


    def configure_optimizers(self):
        #parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
        #parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
        optim_g = torch.optim.Adam(self.generator.parameters(),
                                   lr=self.config.Optimizers.optim_g_lr,
                                   betas=(0.5, 0.999))
        optim_d = torch.optim.Adam(self.discriminator.parameters(),
                                   lr=self.config.Optimizers.optim_d_lr,
                                   betas=(0.5, 0.999))
        sched_g = ReduceLROnPlateau(optim_g, mode='min', patience=self.config.Schedulers.patience_g)
        sched_d = ReduceLROnPlateau(optim_d, mode='min', patience=self.config.Schedulers.patience_d)
        return ([optim_d, optim_g], 
                [{'scheduler': sched_d, 'monitor': self.config.Schedulers.metric, 'interval': 'epoch'},
                 {'scheduler': sched_g, 'monitor': self.config.Schedulers.metric, 'interval': 'epoch'}])
    
    def extract_batch(self, batch):
        return batch["rgb"], batch["nir"]
    
    
if __name__ == "__main__":
    config = OmegaConf.load("configs/config_px.yaml")
    model = Px2Px().cuda()
    model.forward(torch.rand(1,3,128,128).cuda()).shape
    
    batch = {"rgb": torch.rand(1,3,128,128).cuda(), "nir": torch.rand(1,1,128,128).cuda()}
    model.training_step(batch,1,optimizer_idx=1)
    
    model.discriminator.forward(torch.rand(1,1,256,256).cuda()).squeeze()
    
