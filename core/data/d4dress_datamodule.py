import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from core.data.d4dress_dataset import D4DressDataset, d4dress_collate_fn

class CCHDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fraction = 1.0

        self.train_ids = [
            '00122', '00123', '00127', '00129', '00134', '00135', '00136', '00137', 
            '00140', '00147', '00148', '00149', '00151', '00152', '00154', '00156', 
            '00160', '00163', '00167', '00168', '00169', '00170', '00174', '00175', 
            '00176', '00179', '00180', '00185', '00187', '00190'
        ]  
        self.val_ids = ['00188', '00191']

    def setup(self, stage):

        self.train_dataset = D4DressDataset(cfg=self.cfg, ids=self.train_ids)
        self.val_dataset = D4DressDataset(cfg=self.cfg, ids=self.val_ids)

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