import torch
import pytorch_lightning as pl
from collections import defaultdict
from torch.utils.data import DataLoader, DistributedSampler
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from core.data.thuman_dataset import THumanDataset
from core.data.d4dress_dataset import D4DressDataset
from core.data.thuman_metadata import THuman_metadata   


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized mesh data.
    Returns lists for mesh data that can't be stacked, and tensors for data that can.
    """
    collated = defaultdict(list)
    
    for sample in batch:
        for key, value in sample.items():
            collated[key].append(value)
    
    nonstackable_keys = [
        #THuman
        'scan_verts', 'scan_faces', 'smplx_param', 'gender', 'scan_ids', 'camera_ids', 'dataset',
        #4DDress
        'scan_mesh', 'scan_mesh_verts', 'scan_mesh_faces', 'scan_mesh_verts_centered', 'scan_mesh_colors',
        'template_mesh', 'template_mesh_verts', 'template_mesh_faces', 'template_full_mesh', 
        'template_full_lbs_weights', 'gender', 'take_dir', 'dataset'
    ]


    for key in collated.keys():
        if collated[key] and (key not in nonstackable_keys):
            try:
                collated[key] = torch.stack(collated[key])
            except RuntimeError as e:
                print(f"Warning: Could not stack {key}, keeping as list. Error: {e}")
                # Keep as list if stacking fails
    
    # Keep mesh data as lists since they have different vertex counts
    for key in nonstackable_keys:
        if key in collated:
            # Keep as list - don't try to stack
            pass
    
    return dict(collated)



class FullDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 4DDress splits (reuse those used in the standalone datamodule)
        self.d4dress_train_ids = [
            '00122', '00123', '00127', '00129', '00135', '00136', '00137',
            '00140', '00147', '00149', '00151', '00152', '00154', '00156',
            '00160', '00163', '00167', '00168', '00169', '00170', '00174', '00175',
            '00176', '00179', '00180', '00185', '00187', '00188', '00190', '00191'
        ]
        # self.d4dress_val_ids = ['00188', '00191']
        self.d4dress_val_ids = ['00134', '00148']

        self.thuman_train_ids = sorted(THuman_metadata.keys())[:-50]
        self.thuman_val_ids = sorted(THuman_metadata.keys())[-50:]

        self.train_thuman = None
        self.train_d4dress = None
        self.val_d4dress = None

    def setup(self, stage=None):
        # THuman used for training
        self.train_thuman = THumanDataset(self.cfg, ids=self.thuman_train_ids)
        self.val_thuman = THumanDataset(self.cfg, ids=self.thuman_val_ids)
        # 4DDress train/val
        self.train_d4dress = D4DressDataset(cfg=self.cfg, ids=self.d4dress_train_ids)
        self.val_d4dress = D4DressDataset(cfg=self.cfg, ids=self.d4dress_val_ids)

        print(f"THuman train samples: {len(self.train_thuman)}")
        print(f"4DDress train samples: {len(self.train_d4dress)}")
        print(f"4DDress val samples: {len(self.val_d4dress)}")


    # def train_dataloader(self):
    #     thuman_loader = DataLoader(
    #         self.val_d4dress,
    #         batch_size=self.cfg.TRAIN.BATCH_SIZE,
    #         shuffle=True,
    #         drop_last=True,
    #         num_workers=self.cfg.TRAIN.NUM_WORKERS,
    #         pin_memory=self.cfg.TRAIN.PIN_MEMORY,
    #         collate_fn=custom_collate_fn,
    #     )

    #     d4dress_loader = DataLoader(
    #         self.val_d4dress,
    #         batch_size=self.cfg.TRAIN.BATCH_SIZE,
    #         shuffle=True,
    #         drop_last=True,
    #         num_workers=self.cfg.TRAIN.NUM_WORKERS,
    #         pin_memory=self.cfg.TRAIN.PIN_MEMORY,
    #         collate_fn=custom_collate_fn,
    #     )

    #     return [thuman_loader, d4dress_loader]

        


    def train_dataloader(self):
        thuman_loader = DataLoader(
            self.train_thuman,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=self.cfg.TRAIN.PIN_MEMORY,
            collate_fn=custom_collate_fn,
        )

        d4dress_loader = DataLoader(
            self.train_d4dress,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=self.cfg.TRAIN.PIN_MEMORY,
            collate_fn=custom_collate_fn,
        )

        return [thuman_loader, d4dress_loader]



    def val_dataloader(self):
        # thuman_val_loader = DataLoader(
        #     self.val_thuman,
        #     batch_size=self.cfg.TRAIN.BATCH_SIZE,
        #     shuffle=False,
        #     drop_last=True,
        #     num_workers=self.cfg.TRAIN.NUM_WORKERS,
        #     pin_memory=self.cfg.TRAIN.PIN_MEMORY,
        #     collate_fn=custom_collate_fn,
        # )
        d4dress_val_loader = DataLoader(
            self.val_d4dress,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            drop_last=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=self.cfg.TRAIN.PIN_MEMORY,
            collate_fn=custom_collate_fn,
        )

        return d4dress_val_loader

    def test_dataloader(self):
        return self.val_dataloader()


