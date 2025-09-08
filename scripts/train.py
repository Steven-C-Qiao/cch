import os
import torch
import argparse
import glob
from pathlib import Path

from loguru import logger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

import sys
sys.path.append('.')

from core.configs.cch_cfg import get_cch_cfg_defaults
# from core.models.trainer import CCHTrainer
# from core.data.cch_datamodule import CCHDataModule
from core.models.trainer_4ddress import CCHTrainer
from core.data.d4dress_datamodule import CCHDataModule

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_train(exp_dir, cfg_opts=None, dev=False, device_ids=None, resume_path=None, load_path=None, plot=False):
    set_seed(42)
    
    # Get config
    cfg = get_cch_cfg_defaults()
    if cfg_opts is not None:
        cfg.merge_from_list(cfg_opts)

    # path = resume_path if resume_path is not None else load_path
    # if path is not None:
    #     path = str(Path(path).parent.parent / 'lightning_logs')
    #     if os.path.exists(path):
    #         version_dirs = sorted(glob.glob(os.path.join(path, 'version_*')))
    #         if version_dirs:
    #             latest_version = version_dirs[-1]
    #             hparams_file = os.path.join(latest_version, 'hparams.yaml')
    #             if os.path.exists(hparams_file):
    #                 cfg.merge_from_file(hparams_file)
    #                 logger.info(f"Loaded hyperparameters from: {hparams_file}")

    if dev:
        cfg.TRAIN.BATCH_SIZE = 1

    # Create directories
    model_save_dir = os.path.join(exp_dir, 'saved_models')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    vis_save_dir = os.path.join(exp_dir, 'vis')
    if not os.path.exists(vis_save_dir):
        os.makedirs(vis_save_dir)



    model = CCHTrainer(
        cfg=cfg,
        dev=dev,
        vis_save_dir=vis_save_dir,
        plot=plot
    )

    datamodule = CCHDataModule(cfg)



    # Callbacks
    checkpoint_callbacks = [
        # ModelCheckpoint(
        #     dirpath=model_save_dir,
        #     filename='val_vp_cfd_{epoch:03d}',
        #     save_top_k=1,
        #     save_last=False,
        #     verbose=True,
        #     monitor='val_vp_cfd',
        #     mode='min'
        # ),
        ModelCheckpoint(
            dirpath=model_save_dir,
            filename='val_loss_{epoch:03d}',#'val_vpp2vp_cfd_{epoch:03d}',
            save_top_k=1,
            save_last=True,
            verbose=True,
            monitor='val_loss',#'val_vpp2vp_cfd',
            mode='min'
        ),
    ]

    tensorboard_logger = TensorBoardLogger(exp_dir, name='lightning_logs')

    trainer = pl.Trainer(
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        accelerator='gpu',
        devices=device_ids, 
        # strategy=DDPStrategy(find_unused_parameters=True) if not dev else 'auto',
        strategy=DDPStrategy() if not dev else 'auto',
        callbacks=checkpoint_callbacks,
        logger=tensorboard_logger,
        log_every_n_steps=100,
        gradient_clip_val=1.0,
    )


    if load_path is not None:
        logger.info(f"Loading checkpoint: {load_path}")
        ckpt = torch.load(load_path, weights_only=False, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        # logger.log_hyperparams(ckpt['hyper_parameters'])

    trainer.fit(model, datamodule, ckpt_path=resume_path)
    # trainer.test(model, datamodule)


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
    parser.add_argument(
        "--plot", 
        action="store_true"
    )  
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Device: {device}')

    device_ids = list(map(int, args.gpus.split(",")))
    logger.info(f"Using GPUs: {args.gpus} (Device IDs: {device_ids})")

    assert ((args.resume_training_states is not None) * (args.load_from_ckpt is not None) == 0), 'Specify either resume_training_states or load_from_ckpt, not both'

    torch.set_float32_matmul_precision('high')
    run_train(
        exp_dir=args.experiment_dir,
        cfg_opts=args.cfg_opts,
        dev=args.dev,
        device_ids=device_ids,
        resume_path=args.resume_training_states,
        load_path=args.load_from_ckpt,
        plot=args.plot

    )
