#!/usr/bin/env python3

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
from core.utils.general import load_pickle

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

if __name__ == "__main__":
    set_seed(42)
    
    # Get config
    cfg = get_cch_cfg_defaults()
    cfg.TRAIN.BATCH_SIZE = 1

    model = CCHTrainer(
        cfg=cfg,
        dev=dev,
        vis_save_dir=vis_save_dir,
        plot=plot
    )
    load_path = "/scratch/u5au/chexuan.u5au/cch/exp/exp_011_s3/saved_models/last.ckpt"
    if load_path is not None:
        logger.info(f"Loading checkpoint: {load_path}")
        ckpt = torch.load(load_path, weights_only=False, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=True)

    datamodule = CCHDataModule(cfg)

    batch = datamodule.train_dataloader().dataset[0]

    preds = model(batch)

    vc_init = preds['vc_init']
    print(vc_init.shape)




    take_dir = batch['take_dir']
    print(take_dir)

    pose_seq_dir = os.path.join(take_dir, 'SMPL')
    def get_pose_seq(pose_seq_dir):
        pose_seq = []
        for file in os.listdir(pose_seq_dir):
            if file.endswith('.pkl'):
                pose_seq.append(load_pickle(os.path.join(pose_seq_dir, file))['pose'])
        return pose_seq

    pose_seq = get_pose_seq(pose_seq_dir)
    print(pose_seq.shape)
    

    import ipdb; ipdb.set_trace()