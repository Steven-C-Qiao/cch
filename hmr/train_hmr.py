import os
import torch
import argparse

from loguru import logger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import seed_everything

import sys
sys.path.append('.')

from hmr.hmr_cfg import get_hmr_cfg_defaults
from hmr.hmr_trainer import HMRTrainer
from hmr.smpl_datamodule import SmplDataModule
from core.data.cch_datamodule import CCHDataModule


def run_train(exp_dir, cfg_opts=None, dev=False, device_ids=None, resume_path=None, load_path=None):
    seed_everything(42)
    
    # Get config
    cfg = get_hmr_cfg_defaults()
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

    model = HMRTrainer(
        cfg=cfg,
        dev=dev,
        vis_save_dir=vis_save_dir
    )

    if cfg.DATA.TYPE == 'cape':
        datamodule = CCHDataModule(cfg)
    else:
        datamodule = SmplDataModule(cfg)

    # Callbacks
    checkpoint_callbacks = [
        ModelCheckpoint(
            dirpath=model_save_dir,
            filename='val_loss_{epoch:03d}',
            save_top_k=1,
            save_last=True,
            verbose=True,
            monitor='val_loss',
            mode='min'
        ),
    ]

    tensorboard_logger = TensorBoardLogger(exp_dir, name='lightning_logs')

    trainer = pl.Trainer(
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        accelerator='gpu',
        devices=device_ids, 
        strategy='auto',
        callbacks=checkpoint_callbacks,
        logger=tensorboard_logger,
        log_every_n_steps=100,
        gradient_clip_val=1.0,
    )


    if load_path is not None:
        logger.info(f"Loading checkpoint: {load_path}")
        ckpt = torch.load(load_path, weights_only=False, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)

    trainer.fit(model, datamodule, ckpt_path=resume_path)


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
        default='0,1', 
        help="Comma-separated list of GPU indices to use. E.g., '0,1,2'"
    )    
    parser.add_argument(
        "--dev", 
        action="store_true"
    )  
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Device: {device}')

    device_ids = list(map(int, args.gpus.split(",")))
    logger.info(f"Using GPUs: {args.gpus} (Device IDs: {device_ids})")

    assert ((args.resume_training_states is not None) * (args.load_from_ckpt is not None) == 0), 'Specify either resume_training_states or load_from_ckpt, not both'

    run_train(
        exp_dir=args.experiment_dir,
        cfg_opts=args.cfg_opts,
        dev=args.dev,
        device_ids=device_ids,
        resume_path=args.resume_training_states,
        load_path=args.load_from_ckpt
    )
