#!/usr/bin/env python3

import os
import torch
import argparse
import glob
from tqdm import tqdm
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

def run_train(exp_dir, cfg_opts=None, dev=False, resume_path=None, load_path=None, plot=False):
    set_seed(42)
    
    # Get config
    cfg = get_cch_cfg_defaults()
    if cfg_opts is not None:
        cfg.merge_from_list(cfg_opts)


    cfg.TRAIN.BATCH_SIZE = 1
    cfg.TRAIN.LR = 0.

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
        plot=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if load_path is not None:
        logger.info(f"Loading checkpoint: {load_path}")
        ckpt = torch.load(load_path, weights_only=False, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=True)
        # logger.log_hyperparams(ckpt['hyper_parameters'])

    datamodule = CCHDataModule(cfg)
    datamodule.setup('val')
    train_dataloader = datamodule.train_dataloader()

    def _move_to_device(data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        if isinstance(data, dict):
            return {k: _move_to_device(v, device) for k, v in data.items()}
        if isinstance(data, list):
            return [_move_to_device(v, device) for v in data]
        if isinstance(data, tuple):
            return tuple(_move_to_device(v, device) for v in data)
        return data

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(train_dataloader)):
            batch = _move_to_device(batch, device)
            batch_proc = model.process_4ddress(batch, batch_idx, normalise=model.normalise)
            preds = model(batch_proc)
            model.visualiser.visualise(preds, batch_proc, batch_idx=batch_idx, split='plot')




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
        action='append',
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

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # device_ids = list(map(int, args.gpus.split(",")))
    # logger.info(f"Using GPUs: {args.gpus} (Device IDs: {device_ids})")

    assert ((args.resume_training_states is not None) * (args.load_from_ckpt is not None) == 0), 'Specify either resume_training_states or load_from_ckpt, not both'

    # Flatten cfg_opts if it's a list of lists (from multiple -O flags)
    if args.cfg_opts is not None:
        flattened_cfg_opts = []
        for opt_list in args.cfg_opts:
            flattened_cfg_opts.extend(opt_list)
        args.cfg_opts = flattened_cfg_opts

    torch.set_float32_matmul_precision('high')
    run_train(
        exp_dir=args.experiment_dir,
        cfg_opts=args.cfg_opts,
        dev=args.dev,
        # device_ids=device_ids,
        resume_path=args.resume_training_states,
        load_path=args.load_from_ckpt,
        plot=args.plot
    )
