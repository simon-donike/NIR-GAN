# Package Imports
import torch
from pytorch_lightning import Trainer
from omegaconf import OmegaConf
import wandb
import os, datetime
from multiprocessing import freeze_support

# local imports
from model.SRGAN import SRGAN_model

#os.environ["WANDB_MODE"] = "online"

# Set GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Run Main Function
if __name__ == '__main__':
    # enable if running on Windows
    #freeze_support()

    # General
    torch.set_float32_matmul_precision('medium')
    # load config
    config = OmegaConf.load("configs/config_NIR.yaml")

    #############################################################################################################
    " LOAD MODEL "
    #############################################################################################################
    # load rpetrained or instanciate new
    model = SRGAN_model(config).to(device)

    # set reload checkpoint settings for trainer
    resume_from_checkpoint=None
    if config.Model.load_checkpoint==True:
        resume_from_checkpoint=config.Model.ckpt_path

    #############################################################################################################
    """ GET DATA """
    #############################################################################################################
    # create dataloaders via dataset_selector -> config -> class selection -> convert to pl_module
    from utils.S2_dataset import S2_datamodule
    from utils.SEN2NAIP_v4 import SEN2NAIP_datamodule
    pl_datamodule = SEN2NAIP_datamodule(config)

    #############################################################################################################
    """ Configure Trainer """
    #############################################################################################################
    # set up logging
    from pytorch_lightning.loggers import WandbLogger
    wandb_project = "GAN_NIR" #"testing"
    wandb_logger = WandbLogger(project=wandb_project,entity="simon-donike")

    from pytorch_lightning import loggers as pl_loggers
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.normpath("logs/"))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.normpath('logs/tmp'))

    from pytorch_lightning.callbacks import ModelCheckpoint
    dir_save_checkpoints = os.path.join(tb_logger.save_dir,wandb_project,
                                                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print("Experiment Path:",dir_save_checkpoints)

    checkpoint_callback = ModelCheckpoint(dirpath=dir_save_checkpoints,
                                            monitor='val/L1',
                                        mode='min',
                                        save_last=True,
                                        save_top_k=2)

    from pytorch_lightning.callbacks import LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # callback to set up early stopping
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    early_stop_callback = EarlyStopping(monitor="val/L1", min_delta=0.00, patience=1000, verbose=True,
                                    mode="min",check_finite=True) # patience in epochs


    #############################################################################################################
    """ Start Training """
    #############################################################################################################
    
    trainer = Trainer(accelerator='cuda',
                    devices=[0,1,2,3],
                    strategy="ddp",
                    check_val_every_n_epoch=1,
                    val_check_interval=1.,
                    limit_val_batches=5,
                    max_epochs=99999,
                    resume_from_checkpoint=resume_from_checkpoint,
                    logger=[ 
                                wandb_logger,
                            ],
                    callbacks=[ checkpoint_callback,
                                early_stop_callback,
                                lr_monitor
                                ])


    trainer.fit(model, datamodule=pl_datamodule)
    wandb.finish()
    writer.close()

