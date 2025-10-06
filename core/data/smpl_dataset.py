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



import smplx 

smpl_model = smplx.create(
    'model_files',
    gender='neutral',
    num_betas=10
)
smpl_faces = smpl_model.faces

class SmplDataset(Dataset):
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

                    # self.seq_ids[id].append((sequence_name, valid_frames, removed_frames))
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
            try:
                fpath = os.path.join(PATH_TO_DATA, f'sequences/{id}/{sequence_name}/{sequence_name}.{i:06d}.npz')
                data = np.load(fpath)
            except:
                data = None
                while data is None:
                    new_idx = np.random.choice(valid_frames_indices, 1, replace=False) # sample a new random frame
                    fpath = os.path.join(PATH_TO_DATA, f'sequences/{id}/{sequence_name}/{sequence_name}.{new_idx[0]:06d}.npz')
                    try: # some frames are corrupted, so loading here still might fail
                        data = np.load(fpath)
                    except: # iterate until a valid frame is loaded
                        continue

            ret['pose'].append(torch.from_numpy(data['pose']).float()[3:])

        # Convert lists to tensors
        ret = {k: torch.stack(v) for k, v in ret.items()}

        # betas = torch.randn(10)
        # with torch.no_grad():
        #     smpl_output = smpl_model(betas=betas, 
        #                                 body_pose=ret['pose'], 
        #                                 global_orient=torch.zeros(self.num_frames, 3))
        #     ret['v_posed'] = smpl_output.vertices
        #     ret['joints'] = smpl_output.joints[:, :24]

        #     t_smpl_output = smpl_model(betas=betas, 
        #                                 body_pose=torch.zeros_like(ret['pose']), 
        #                                 global_orient=torch.zeros(self.num_frames, 3))
        #     ret['v_cano'] = t_smpl_output.vertices
        #     ret['joints_cano'] = t_smpl_output.joints[:, :24]

        # ret['betas'] = betas

        # posed_samples = []
        # for i in range(self.num_frames):
        #     # Sample from vp for chamfer
        #     posed_mesh = trimesh.Trimesh(
        #         vertices=ret['v_posed'][i].cpu().detach().numpy(),
        #         faces=self.smpl_faces
        #     )
        #     sampled_points, _ = sample_surface_even(posed_mesh, 6890)  # Sample 2048 points
        #     posed_samples.append(torch.from_numpy(sampled_points).float())

        # ret['v_posed_sample'] = torch.stack(posed_samples)


        return ret


if __name__ == "__main__":
    dataset = SmplDataset()

    # build test dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=4)
    for batch in dataloader:
        for k, v in batch.items():
            print(k, v.shape)
        import ipdb; ipdb.set_trace()
