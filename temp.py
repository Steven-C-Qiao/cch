import torch 

from core.configs.paths import THUMAN_PATH
from core.data.thuman_metadata import THuman_metadata
from core.data.d4dress_utils import load_pickle
import os 

from tqdm import tqdm


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ids = os.listdir(os.path.join(THUMAN_PATH, 'decimated'))
    ids = list(set([id.split('.')[0] for id in ids]))
    
    for id in tqdm(ids):
        camera_ids = [
            '000', '010', '020', '030', '040', '050', '060', '070', '080', 
            '090', '100', '110', '120', '130', '140', '150', '160', '170', 
            '180', '190', '200', '210', '220', '230', '240', '250', '260', 
            '270', '280', '290', '300', '310', '320', '330', '340', '350', 
        ]


        for camera in camera_ids:
                fname = os.path.join(THUMAN_PATH, 'render_persp/thuman2_36views', id, f'{id}_w_maps_sparse', f'{camera}.pt')#.npz')
                f = torch.load(fname)
                import ipdb; ipdb.set_trace()
                f = f.to_dense() 

