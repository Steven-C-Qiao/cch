import os
import torch 

from scipy.sparse import load_npz

PATH = '/scratch/u5aa/chexuan.u5aa/datasets/THuman/render_persp/thuman2_36views'

def load_w_maps_sparse(w_maps_fname):
    """
    Load sparse w_maps from .npz file and convert back to dense format.
    
    Args:
        w_maps_fname: Path to the .npz file containing sparse matrix
        
    Returns:
        numpy array of shape (512, 512, 55) - dense w_map for single camera
    """
    # Load sparse matrix from .npz file
    sparse_matrix = load_npz(w_maps_fname)
    
    # Convert sparse matrix back to dense format
    dense_data = sparse_matrix.toarray()  # Shape: (512*512, 55)
    
    # Reshape back to original format: (512, 512, 55)
    w_map = dense_data.reshape(512, 512, 55)
    
    return w_map


camera_ids = [
    '000', '010', '020', '030', '040', '050', '060', '070', '080', '090', '100', '110', 
    '120', '130', '140', '150', '160', '170', '180', '190', '200', '210', '220', '230', 
    '240', '250', '260', '270', '280', '290', '300', '310', '320', '330', '340', '350'
]

if __name__=="__main__":
    subject_ids = os.listdir(PATH)
    first_half = sorted(subject_ids)[:len(subject_ids)//2]
    second_half = sorted(subject_ids)[len(subject_ids)//2:]
    second_half = [x for x in second_half if x != '1957']
    for subject_id in second_half:
        print(f"Processing {subject_id}")
        w_maps_dir_fname = os.path.join(PATH, subject_id, f'{subject_id}_w_maps_sparse')
        if not os.path.exists(w_maps_dir_fname):
            continue
            
        # Check if 350.pt exists
        if os.path.exists(os.path.join(w_maps_dir_fname, '350.pt')):
            continue
            
        for camera_id in camera_ids:
            w_map_fname = os.path.join(w_maps_dir_fname, f'{camera_id}.npz')
            w_maps = load_w_maps_sparse(w_map_fname)
            w_maps = torch.tensor(w_maps, dtype=torch.float16)
            # Get indices and values for sparse tensor creation
            indices = w_maps.nonzero().T  # Transpose to get shape (3, nnz)
            values = w_maps[w_maps != 0]  # Extract non-zero values
            # print(indices.shape, values.shape)
            torch_sparse_w = torch.sparse_coo_tensor(indices, values, w_maps.shape, dtype=torch.float16)
            torch.save(torch_sparse_w, os.path.join(w_maps_dir_fname, f'{camera_id}.pt'))

    # # Test loading the saved sparse tensor and converting to dense
    # loaded_sparse = torch.load(os.path.join(w_maps_dir_fname, f'{camera_id}.pt'))
    # loaded_dense = loaded_sparse.to_dense()
    # print(f"Loaded sparse tensor shape: {loaded_sparse.shape}")
    # print(f"Number of non-zero elements: {loaded_sparse._nnz()}")
    # print(f"Dense tensor shape: {loaded_dense.shape}")

    # import ipdb; ipdb.set_trace()
    # print(torch.norm(loaded_dense - w_maps).mean())