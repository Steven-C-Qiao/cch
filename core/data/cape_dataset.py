from torch.utils.data import Dataset
import numpy as np
import os
import random
import pickle
import torch
import trimesh
import logging
from tqdm import tqdm
from collections import defaultdict
from trimesh.sample import sample_surface_even

from loguru import logger

PATH_TO_DATA = "/scratches/kyuban/cq244/datasets/CAPE"

ids = [ '00032', '00096', '00122', '00127', '00134', 
        '00145', '00159', '00215', '02474', '03223', 
        '03284', '03331', '03375', '03383', '03394']

corrupted_frames = [
    'sequences/00032/shortlong_pose_model/shortlong_pose_model.000047.npz',
    'sequences/03394/longlong_ROM_lower/longlong_ROM_lower.000100.npz',
    'sequences/00215/poloshort_basketball/poloshort_basketball.000061.npz',
    'sequences/00096/shirtshort_chicken_wings/shirtshort_chicken_wings.000108.npz',
    'sequences/00122/shortshort_punching/shortshort_punching.000147.npz',
    'sequences/02474/longlong_ROM_lower/longlong_ROM_lower.000354.npz',
    'sequences/02474/longlong_rotate_hips/longlong_rotate_hips.000245.npz',
]


import smplx 

smpl_model = smplx.create(
    'model_files',
    gender='neutral',
    num_betas=10
)
smpl_faces = smpl_model.faces

class CapeDataset(Dataset):
    """
    Each subject has a bunch of sequencies
    """

    def __init__(self):
        self.ids = ids
        self.smpl_faces = smpl_faces
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

                    self.all_seqs.append((id, sequence_name, valid_frames, removed_frames))
        self.total_num_sequences = len(self.all_seqs)
        self.total_num_frames = sum(int(valid_frames) for _, _, valid_frames, _ in self.all_seqs)

        assert self.total_num_sequences == 609
        # assert self.total_num_frames == 148411
        logger.info(f"Total number of sequences: {self.total_num_sequences}")
        logger.info(f"Total number of frames: {self.total_num_frames}")

    def __len__(self):
        # Pseudo lengthen CAPE since num_sequences is small, but frames in each sequence is large
        return self.total_num_sequences * 100

    def __getitem__(self, index):
        # get sequence id
        id, sequence_name, valid_frames, removed_frames = self.all_seqs[index % self.total_num_sequences]
        valid_frames_indices = np.arange(int(valid_frames)) + 1
        
        if removed_frames is not None:
            all_removed = removed_frames.split(',')
            for i in range(len(all_removed)):
                # Parse removed frames range if present
                removed_start, removed_end = map(int, all_removed[i].split('-'))

        
                try:
                    idx = np.where(valid_frames_indices==removed_start)[0].item()
                except:
                    continue # this happens when all the rest of the frames are removed
                valid_frames_indices[idx:] += removed_end - removed_start + 1

        # randomly sample frames
        sampled_frames_indices = np.random.choice(valid_frames_indices, self.num_frames, replace=False)

        ret = defaultdict(list)
        for i in sampled_frames_indices:
            data = None
            while data is None:
                try:
                    fpath = os.path.join(PATH_TO_DATA, f'sequences/{id}/{sequence_name}/{sequence_name}.{i:06d}.npz')
                    data = np.load(fpath)

                    transl = torch.from_numpy(data['transl']).float()
                    v_cano = torch.from_numpy(data['v_cano']).float()
                    pose = torch.from_numpy(data['pose']).float() 
                    v_posed = torch.from_numpy(data['v_posed']).float()
   
                except:
                    # Sample a new random frame if loading fails
                    i = np.random.choice(valid_frames_indices, 1, replace=False)[0]
                    continue

            # print(transl.shape, v_cano.shape, pose.shape, v_posed.shape)
            assert transl.shape == (3,), f"Expected transl shape (3,), got {transl.shape}"
            assert v_cano.shape == (6890, 3), f"Expected v_cano shape (6890, 3), got {v_cano.shape}"
            assert pose.shape == (72,), f"Expected pose shape (72,), got {pose.shape}"
            assert v_posed.shape == (6890, 3), f"Expected v_posed shape (6890, 3), got {v_posed.shape}"
            ret['transl'].append(transl)
            ret['v_cano'].append(v_cano)
            ret['pose'].append(pose) 
            ret['v_posed'].append(v_posed)

                
            # Sample from vp for chamfer
            posed_mesh = trimesh.Trimesh(
                vertices=data['v_posed'],
                faces=self.smpl_faces
            )
            sampled_points, _ = sample_surface_even(posed_mesh, 6890)  # Sample 2048 points
            ret['v_posed_samples'].append(torch.from_numpy(sampled_points).float())

        # Convert lists to tensors
        ret = {k: torch.stack(v) for k, v in ret.items()}

        # Load per person attributes
        fpath = os.path.join(PATH_TO_DATA, f'minimal_body_shape/{id}/{id}_param.pkl')
        with open(fpath, 'rb') as f:
            data = pickle.load(f)

        ret['betas'] = torch.from_numpy(data['betas']).float()



        # Load the first frame of the sequence, and use its v_cano as the canonical mesh
        i=1
        frame = f'{i:06d}'
        fpath = os.path.join(PATH_TO_DATA, f'sequences/{id}/{sequence_name}/{sequence_name}.{frame}.npz')
        while not os.path.exists(fpath):
            i+=1
            frame = f'{i:06d}'
            fpath = os.path.join(PATH_TO_DATA, f'sequences/{id}/{sequence_name}/{sequence_name}.{frame}.npz')
        data = np.load(fpath)
        ret['first_frame_v_cano'] = torch.from_numpy(data['v_cano']).float()


        



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
