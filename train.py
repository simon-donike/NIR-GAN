# Package Imports
import torch
from pytorch_lightning import Trainer
from omegaconf import OmegaConf
import wandb
import os, datetime
#from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import time
import argparse
from utils.other_utils import str2bool

# local imports
from model.pix2pix import Px2Px_PL

# Run Main Function
if __name__ == '__main__':
    # get CL arguments
    parser = argparse.ArgumentParser(description='Training script for NIR-GAN.')
    parser.add_argument('--satclip', required=False,default=True,
                        help='Enable satclip (default: True)')
    args = parser.parse_args()
    args.satclip = str2bool(args.satclip)
    print("Satclip:",args.satclip)

    # General
    torch.set_float32_matmul_precision('medium')

    # load config depending on setting
    if args.satclip:
        config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml") # SatCLIP
    elif not args.satclip:
        config = OmegaConf.load("configs/config_px2px.yaml") # Standard
    else:
        raise ValueError("Invalid Argument for Satclip")

    #############################################################################################################
    " LOAD MODEL "
    #############################################################################################################
    model = Px2Px_PL(config)

    # set reload checkpoint settings for trainer
    if config.custom_configs.Model.load_weights_only==True:
        ckpt = config.custom_configs.Model.weights_path
        ckpt = torch.load(ckpt)
        model.load_state_dict(ckpt['state_dict'],strict=False)
        print("Loaded (only) Weights from:",config.custom_configs.Model.weights_path)

    resume_from_checkpoint=None
    if config.custom_configs.Model.load_checkpoint==True:
        resume_from_checkpoint=config.custom_configs.Model.ckpt_path
        #resume_from_checkpoint = model.clean_checkpoint(resume_from_checkpoint,["pred_cache"]) # clean state dict manually
        print("Resuming from checkpoint PL-style:",resume_from_checkpoint)

    #############################################################################################################
    """ GET DATA """
    #############################################################################################################
    # create dataloaders via dataset_selector -> config -> class selection -> convert to pl_module
    from data.select_dataset import dataset_selector
    pl_datamodule = dataset_selector(config)

    #############################################################################################################
    """ Configure Trainer """
    #############################################################################################################
    # set up logging
    from pytorch_lightning.loggers import WandbLogger
    wandb_project = config.custom_configs.Logging.wandb_project
    wandb_logger = WandbLogger(project=wandb_project)

    from pytorch_lightning import loggers as pl_loggers
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.normpath("logs/"))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.normpath('logs/tmp'))

    from pytorch_lightning.callbacks import ModelCheckpoint
    dir_save_checkpoints = os.path.join(tb_logger.save_dir,wandb_project,
                                                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print("Experiment Path:",dir_save_checkpoints)
    model.dir_save_checkpoints = dir_save_checkpoints # save to model

    checkpoint_callback = ModelCheckpoint(dirpath=dir_save_checkpoints,
                                            monitor=config.Schedulers.metric,
                                        mode='min',
                                        save_last=True,
                                        save_top_k=1)

    from pytorch_lightning.callbacks import LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # callback to set up early stopping
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    early_stop_callback = EarlyStopping(monitor=config.Schedulers.metric, min_delta=0.00, patience=1000, verbose=True,
                                    mode="min",check_finite=True) # patience in epochs


    #############################################################################################################
    """ Start Training """
    #############################################################################################################
    # extract training devices
    trainer = Trainer(accelerator=config.custom_configs.Training.accelerator,
                    devices=config.custom_configs.Training.devices,
                    strategy=config.custom_configs.Training.strategy, 
                    check_val_every_n_epoch=1,
                    #val_check_interval=0.25,
                    limit_val_batches=25,
                    max_epochs=99999,
                    resume_from_checkpoint=resume_from_checkpoint,
                    logger=[ 
                                wandb_logger,
                            ],
                    callbacks=[ checkpoint_callback,
                                #early_stop_callback,
                                lr_monitor
                                ])


    trainer.fit(model, datamodule=pl_datamodule)
    wandb.finish()
    writer.close()

