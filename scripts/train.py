import os
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
from loguru import logger

import sys
sys.path.append('.')

# import warnings 
# warnings.filterwarnings('ignore', message=r'`torch.cuda.amp.autocast\(args...\)`', category=FutureWarning)

from core.configs.cch_cfg import get_cch_cfg_defaults

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from core.models.trainer import CCHTrainer
from core.data.cch_datamodule import CCHDataModule


def run_train(exp_dir, cfg_opts=None, resume_from_epoch=None, dev=False, device_ids=None):
    # Get config
    cfg = get_cch_cfg_defaults()
    if cfg_opts is not None:
        cfg.merge_from_list(cfg_opts)

    if dev:
        cfg.TRAIN.BATCH_SIZE = 2

    # Create directories
    model_save_dir = os.path.join(exp_dir, 'saved_models')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    vis_save_dir = os.path.join(exp_dir, 'vis')
    if not os.path.exists(vis_save_dir):
        os.makedirs(vis_save_dir)

    # Initialize Lightning modules
    model = CCHTrainer(
        cfg=cfg,
        dev=dev,
        vis_save_dir=vis_save_dir
    )

    datamodule = CCHDataModule(cfg)

    # Callbacks
    checkpoint_callbacks = [
        ModelCheckpoint(
            dirpath=model_save_dir,
            filename='loss_{epoch:03d}',
            save_top_k=1,
            save_last=False,
            verbose=True,
            monitor='loss',
            mode='min'
        ),
    ]

    # Logger
    logger = TensorBoardLogger(exp_dir, name='lightning_logs')

    # Load checkpoint if resuming
    if resume_from_epoch is not None:
        checkpoint_path = os.path.join(model_save_dir, f'epoch_{resume_from_epoch:03d}.ckpt')
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found at {checkpoint_path}")
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = None

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        accelerator='gpu',
        devices= [np.array(device_ids).max().item()] if dev else device_ids, 
        strategy=DDPStrategy(find_unused_parameters=True) if not dev else 'auto',
        callbacks=checkpoint_callbacks,
        logger=logger,
        log_every_n_steps=100,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_dir', 
        '-E', 
        type=str,
        help='Path to directory where logs and checkpoints are saved.'
    )
    parser.add_argument(
        '--cfg_opts', 
        '-O', 
        nargs='*', 
        default=None,
        help='Command line options to modify experiment config e.g. ''-O TRAIN.NUM_EPOCHS 120'' '
                'will change number of training epochs to 120 in the config.'
    )
    parser.add_argument(
        '--resume_from_epoch', 
        '-R', 
        type=int, 
        default=None,
        help='Epoch to resume experiment from. If resuming, experiment_dir must already exist, '
                'with saved model checkpoints and config yaml file.'
    )
    parser.add_argument(
        "--gpus", 
        type=str, 
        default='0,1', 
        help="Comma-separated list of GPU indices to use. E.g., '0,1,2'"
    )    
    parser.add_argument(
        "--dev", 
        action="store_true"
    )  
    args = parser.parse_args()

    # If in dev mode, override GPU settings to use single GPU
    if args.dev:
        args.gpus = '0'
        logger.info('Dev mode: Using single GPU')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Device: {device}')

    device_ids = list(map(int, args.gpus.split(",")))
    logger.info(f"Using GPUs: {args.gpus} (Device IDs: {device_ids})")

    run_train(
        exp_dir=args.experiment_dir,
        cfg_opts=args.cfg_opts,
        resume_from_epoch=args.resume_from_epoch,
        dev=args.dev,
        device_ids=device_ids
    )
