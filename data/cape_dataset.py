from torch.utils.data import Dataset
import numpy as np
import os
import random
import pickle
import torch
import trimesh
import logging
from tqdm import tqdm

from loguru import logger

PATH_TO_DATA = "/scratches/kyuban/cq244/datasets/CAPE"

ids = [ '00032', '00096', '00122', '00127', '00134', 
        '00145', '00159', '00215', '02474', '03223', 
        '03284', '03331', '03375', '03383', '03394']

class CapeDataset(Dataset):
    """
    Each subject has a bunch of sequencies
    """

    def __init__(self):
        self.ids = ids
        self.num_frames = 4
        # get sequence ids from seq_list_0xxxx.txt
        self.all_seqs = []
        # self.seq_ids = {id: [] for id in self.ids}
        for id in self.ids:
            with open(os.path.join(PATH_TO_DATA, f'seq_lists/seq_list_{id}.txt'), 'r') as f:
                # Skip first 3 lines 
                for _ in range(3):
                    f.readline()
                
                # Process each sequence line
                for line in f:
                    # reached end of list
                    if not line.strip():
                        break 

                    splitted_line = line.split()
                    if len(splitted_line) == 3:
                        sequence_name, valid_frames, removed_frames = splitted_line
                    elif len(splitted_line) == 2:
                        sequence_name, valid_frames = splitted_line
                        removed_frames = None
                    else:
                        raise ValueError(f"Invalid line: {line}")

                    # self.seq_ids[id].append((sequence_name, valid_frames, removed_frames))
                    self.all_seqs.append((id, sequence_name, valid_frames, removed_frames))
        self.total_num_sequences = len(self.all_seqs)
        self.total_num_frames = sum(int(valid_frames) for _, _, valid_frames, _ in self.all_seqs)

        assert self.total_num_sequences == 609
        assert self.total_num_frames == 148411
        logger.info(f"Total number of sequences: {self.total_num_sequences}")
        logger.info(f"Total number of frames: {self.total_num_frames}")

    def __len__(self):
        return self.total_num_sequences

    def __getitem__(self, index):
        # get sequence id
        id, sequence_name, valid_frames, removed_frames = self.all_seqs[index]
        valid_frames_indices = np.arange(int(valid_frames))
        
        if removed_frames is not None:
            all_removed = removed_frames.split(',')
            print(all_removed)
            for i in range(len(all_removed)):
                # Parse removed frames range if present
                removed_start, removed_end = map(int, all_removed[i].split('-'))
                
                valid_frames_indices[removed_start:] += removed_end - removed_start + 1

        # randomly sample frames
        sampled_frames_indices = np.random.choice(valid_frames_indices, self.num_frames, replace=False)

        ret = {}
        # Load npz data
        fpath = os.path.join(PATH_TO_DATA, f'sequences/{id}/{sequence_name}/{sequence_name}.{sampled_frames_indices[0]:06d}.npz')
        data = np.load(fpath)

        ret['transl'] = torch.from_numpy(data['transl']).float()
        ret['v_cano'] = torch.from_numpy(data['v_cano']).float()
        ret['pose'] = torch.from_numpy(data['pose']).float()
        ret['v_posed'] = torch.from_numpy(data['v_posed']).float()

        # TODO sample cameras and render images


        return ret


if __name__ == "__main__":
    dataset = CapeDataset()

    # build test dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    for batch in dataloader:
        for k, v in batch.items():
            print(k, v.shape)
        import ipdb; ipdb.set_trace()
