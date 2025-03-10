# Package Imports
import torch
from pytorch_lightning import Trainer
from omegaconf import OmegaConf
import wandb
import os, datetime
#from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import time
from pytorch_lightning.loggers import WandbLogger


# Only Run on one GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set up multiprocessing safely
#import torch.multiprocessing as mp
#mp.set_start_method('spawn', force=True)

# local imports
from model.pix2pix import Px2Px_PL


def train_model(config=None):
    with wandb.init(config=config):
        config = wandb.config  # Override with sweep hyperparams
        # If we're doing SatCLIP - this determines the config file itself
        if config.SatCLIP == True:
            # Load config and update hyperparameters from W&B sweep
            yaml_config = OmegaConf.load("configs/config_px2px_SatCLIP.yaml")        
            yaml_config.base_configs.lambda_rs_losses = config.lambda_rs_losses
            yaml_config.base_configs.lambda_hist = config.lambda_histogram_loss
        elif config.SatCLIP == False:
            # Load config and update hyperparameters from W&B sweep
            yaml_config = OmegaConf.load("configs/config_px2px.yaml")        
            yaml_config.base_configs.lambda_rs_losses = config.lambda_rs_losses
            yaml_config.base_configs.lambda_hist = config.lambda_histogram_loss
        else:
            raise ValueError("SatCLIP must be True or False")
        
        print(f"Performing Run with:\n - SatCLIP: {config.SatCLIP}\n - lambda_rs_losses {config.lambda_rs_losses}\n - lambda_histogram_loss {config.lambda_histogram_loss}")


        #############################################################################################################
        """ LOAD MODEL """
        #############################################################################################################
        model = Px2Px_PL(yaml_config)
        
        # Load weights or checkpoints if specified
        if yaml_config.custom_configs.Model.load_weights_only:
            ckpt = yaml_config.custom_configs.Model.weights_path
            ckpt = torch.load(ckpt)
            model.load_state_dict(ckpt['state_dict'], strict=False)
            print("Loaded (only) Weights from:", yaml_config.custom_configs.Model.weights_path)

        resume_from_checkpoint = None
        if yaml_config.custom_configs.Model.load_checkpoint:
            resume_from_checkpoint = yaml_config.custom_configs.Model.ckpt_path
            print("Resuming from checkpoint PL-style:", resume_from_checkpoint)

        #############################################################################################################
        """ GET DATA """
        #############################################################################################################
        from data.select_dataset import dataset_selector
        pl_datamodule = dataset_selector(yaml_config)

        #############################################################################################################
        """ Configure Trainer """
        #############################################################################################################
        wandb_logger = WandbLogger()

        from pytorch_lightning.callbacks import LearningRateMonitor
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        #############################################################################################################
        """ Start Training """
        #############################################################################################################
        trainer = Trainer(
            accelerator='cuda',
            devices=[2,3],
            strategy="ddp",
            check_val_every_n_epoch=1,
            limit_val_batches=25,
            max_steps=75000,
            resume_from_checkpoint=resume_from_checkpoint,
            logger=[wandb_logger],
            callbacks=[#checkpoint_callback,
                       lr_monitor
                       ]
        )

        trainer.fit(model, datamodule=pl_datamodule)
        wandb.finish()
        
        
if __name__ == "__main__":    
    # define sweep config
    wandb_project_name = "NIRGAN_Sweep_noSatCLIP"
    sweep_rsLoss_values = [0.00, 0.01, 0.1,0.5, 1.0]
    sweep_histLoss_values = [0.00, 0.01, 0.1, 1.0]
    sweep_SatCLip_values = [False]
    
    run_count = len(sweep_rsLoss_values)*len(sweep_histLoss_values)*len(sweep_SatCLip_values)
    print(f"Running {run_count} experiments.")

    sweep_config = {
        "method": "bayes",  # or "grid" or "bayesian"
        "metric": {"name": "train/L1", "goal": "minimize"},
        "max_trials": 50,  # Caps the total runs
        "parameters": {
            "lambda_rs_losses": {
                "values": sweep_rsLoss_values
            },
            "SatCLIP": {
                "values": sweep_SatCLip_values
            },
            "lambda_histogram_loss": {  # Added histogram loss parameter
                "values": sweep_histLoss_values
        }
        }
    }
    
    # Start Sweeps
    sweep_id = wandb.sweep(sweep_config, project=wandb_project_name)
    wandb.agent(sweep_id, train_model, count=run_count)  # Runs 10 experiments
    
    
