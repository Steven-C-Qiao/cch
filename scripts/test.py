import os
import torch
import argparse

from loguru import logger

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

import sys
sys.path.append('.')

from core.configs.cch_cfg import get_cch_cfg_defaults
from core.models.trainer_4ddress import CCHTrainer
from core.data.full_dataset import FullDataModule


def run_test(exp_dir, dev=False, resume_path=None, load_path=None, plot=False):
    seed_everything(42)
    
    # Get config
    cfg = get_cch_cfg_defaults()

    vis_save_dir = os.path.join(exp_dir, 'vis')
    if not os.path.exists(vis_save_dir):
        os.makedirs(vis_save_dir)

    model = CCHTrainer(
        cfg=cfg,
        dev=dev,
        vis_save_dir=vis_save_dir,
        plot=plot
    )

    if load_path is not None:
        logger.info(f"Loading checkpoint: {load_path}")
        ckpt = torch.load(load_path, weights_only=False, map_location='cpu')
        model_state = model.state_dict()
        pretrained_state = ckpt['state_dict']
        
        # Print shape mismatches
        mismatched = {k: (v.shape, model_state[k].shape) 
                     for k, v in pretrained_state.items() 
                     if k in model_state and v.shape != model_state[k].shape}
        if mismatched:
            logger.info("Shape mismatches found:")
            for k, (pretrained_shape, model_shape) in mismatched.items():
                logger.info(f"{k}: checkpoint shape {pretrained_shape}, model shape {model_shape}")
        
        filtered_state = {k: v for k, v in pretrained_state.items() 
                         if k in model_state and v.shape == model_state[k].shape}
        logger.info(f"Loading {len(filtered_state)}/{len(pretrained_state)} keys from checkpoint")
        model.load_state_dict(filtered_state, strict=True)
        # logger.log_hyperparams(ckpt['hyper_parameters'])
    model.eval()

    datamodule = FullDataModule(cfg)


    tensorboard_logger = TensorBoardLogger(exp_dir, name='lightning_logs')




    # Callbacks
    checkpoint_callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            mode='min'
        ),
        ModelCheckpoint( # this is the vc_init cfd 
            monitor='train_vc_cfd',
            mode='min'
        ),
        ModelCheckpoint(
            monitor='val_vc_cfd',
            mode='min'
        ),
        ModelCheckpoint( # this is the vp_cfd
            monitor='train_vp_cfd',
            mode='min'
        ),
        ModelCheckpoint( # this is the vp_cfd
            monitor='val_vp_cfd',
            mode='min'
        )
    ]
    trainer = pl.Trainer(
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        accelerator=cfg.SPEEDUP.ACCELERATOR,
        num_nodes=1,
        devices="auto", 
        strategy="auto",
        callbacks=checkpoint_callbacks,
        logger=tensorboard_logger,
    )

    trainer.test(model, datamodule, ckpt_path=resume_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_dir', 
        '-E', 
        type=str,
        help='Path to directory where logs and checkpoints are saved.'
    )
    parser.add_argument(
        '--resume_training_states', 
        '-R', 
        type=str, 
        default=None,
        help='Load training state. For resuming.'
    )
    parser.add_argument(
        '--load_from_ckpt', 
        '-L', 
        type=str, 
        default=None,
        help='Path to checkpoint. Load for finetuning'
    )
    parser.add_argument(
        "--gpus", 
        type=str, 
        default=None, 
        help="Comma-separated list of GPU indices to use. E.g., '0,1,2'"
    )    
    parser.add_argument(
        "--dev", 
        action="store_true"
    )  
    parser.add_argument(
        "--plot", 
        action="store_true"
    ) 
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Device: {device}')


    assert ((args.resume_training_states is not None) * (args.load_from_ckpt is not None) == 0), 'Specify either resume_training_states or load_from_ckpt, not both'

    run_test(
        exp_dir=args.experiment_dir,
        dev=args.dev,
        resume_path=args.resume_training_states,
        load_path=args.load_from_ckpt,
        plot=args.plot
    )
