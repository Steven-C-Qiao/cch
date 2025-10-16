from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl
from typing import Any, Dict, List, Optional

import torch

from core.data.thuman_dataset import THumanDataset
from core.data.d4dress_dataset import D4DressDataset


def full_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function that works for both D4Dress and THuman samples.
    It attempts to stack tensors when possible and leaves inherently
    variable-sized structures (e.g., meshes) as lists.
    """
    from collections import defaultdict

    collated = defaultdict(list)

    for sample in batch:
        for key, value in sample.items():
            collated[key].append(value)

    # Union of non-stackable keys across datasets
    nonstackable_keys = set([
        # From D4Dress
        'scan_mesh', 'scan_mesh_verts', 'scan_mesh_faces', 'scan_mesh_verts_centered', 'scan_mesh_colors',
        'template_mesh', 'template_mesh_verts', 'template_mesh_faces', 'template_full_mesh', 'template_full_lbs_weights',
        'gender', 'take_dir',
        # From THuman
        'smplx_param',
    ])

    # Try stacking for everything else
    for key in list(collated.keys()):
        if key in nonstackable_keys:
            continue
        values = collated[key]
        if len(values) == 0:
            continue
        try:
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                # Best-effort tensor conversion if all elements are stackable
                collated[key] = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in values])
        except Exception:
            # Leave as list if stacking fails
            pass

    return dict(collated)


class FullDataset(ConcatDataset):
    """
    Concatenation of D4Dress and THuman datasets for unified training.
    """

    def __init__(self, cfg, d4dress_ids: List[str] = None):
        d4dress_ids = d4dress_ids or [
            '00122', '00123', '00127', '00129', '00134', '00135', '00136', '00137',
            '00140', '00147', '00148', '00149', '00151', '00152', '00154', '00156',
            '00160', '00163', '00167', '00168', '00169', '00170', '00174', '00175',
            '00176', '00179', '00180', '00185', '00187', '00190'
        ]

        d4dress = D4DressDataset(cfg=cfg, ids=d4dress_ids)
        thuman = THumanDataset(cfg=cfg)
        super().__init__([d4dress, thuman])


class FullDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Same ids as the D4Dress-only datamodule for consistency
        self.train_ids = [
            '00122', '00123', '00127', '00129', '00134', '00135', '00136', '00137',
            '00140', '00147', '00148', '00149', '00151', '00152', '00154', '00156',
            '00160', '00163', '00167', '00168', '00169', '00170', '00174', '00175',
            '00176', '00179', '00180', '00185', '00187', '00190'
        ]
        self.val_ids = ['00188', '00191']

        self._train_dataset = None
        self._val_dataset = None

    def setup(self, stage: Optional[str] = None):
        d4dress_train = D4DressDataset(cfg=self.cfg, ids=self.train_ids)
        d4dress_val = D4DressDataset(cfg=self.cfg, ids=self.val_ids)
        thuman_all = THumanDataset(cfg=self.cfg)

        # Use THuman for both train and val to provide pose/shape diversity
        # self._train_dataset = ConcatDataset([thuman_all,d4dress_train])
        self._train_dataset = ConcatDataset([thuman_all])
        self._val_dataset = ConcatDataset([d4dress_val])

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=self.cfg.TRAIN.PIN_MEMORY,
            collate_fn=full_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            drop_last=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=self.cfg.TRAIN.PIN_MEMORY,
            collate_fn=full_collate_fn,
        )

    def test_dataloader(self):
        return self.val_dataloader()

