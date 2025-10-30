import torch 

from core.configs.paths import THUMAN_PATH
from core.data.thuman_metadata import THuman_metadata
from core.data.d4dress_utils import load_pickle
import os 

import numpy as np

from tqdm import tqdm


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ids = os.listdir(os.path.join(THUMAN_PATH, 'cleaned'))
    ids = list(set([id.split('.')[0] for id in ids]))


    max_diff = 0
    
    for id in (sorted(ids)):
        camera_ids = [
            '000', '010', '020', '030', '040', '050', '060', '070', '080', 
            '090', '100', '110', '120', '130', '140', '150', '160', '170', 
            '180', '190', '200', '210', '220', '230', '240', '250', '260', 
            '270', '280', '290', '300', '310', '320', '330', '340', '350', 
        ]

        path = os.path.join(THUMAN_PATH, 'cleaned', f'{id}.pkl')
        scan = load_pickle(path)
        verts = np.array(scan['scan_verts'])
        faces = scan['scan_faces']
        diff = verts.max() - verts.min()
        if diff > max_diff:
            max_diff = diff
            max_id = id
        print(id, verts.shape, diff)


        # for camera in camera_ids:
        #         fname = os.path.join(THUMAN_PATH, 'render_persp/thuman2_36views', id, f'{id}_w_maps_sparse', f'{camera}.pt')#.npz')
        #         f = torch.load(fname)
        #         import ipdb; ipdb.set_trace()
        #         f = f.to_dense() 

