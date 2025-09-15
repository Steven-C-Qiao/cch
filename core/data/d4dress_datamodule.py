import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from core.data.d4dress_dataset import D4DressDataset, d4dress_collate_fn

class CCHDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fraction = 1.0

    def setup(self, stage=None):
        dataset = D4DressDataset(cfg=self.cfg)
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=self.cfg.TRAIN.PIN_MEMORY,
            collate_fn=d4dress_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            drop_last=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=self.cfg.TRAIN.PIN_MEMORY,
            collate_fn=d4dress_collate_fn
        ) 
    
    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            drop_last=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=self.cfg.TRAIN.PIN_MEMORY,
            collate_fn=d4dress_collate_fn
        )