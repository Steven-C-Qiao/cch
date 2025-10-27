
from scipy.sparse import load_npz

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


if __name__=="__main__":
    path = '/scratch/u5aa/chexuan.u5aa/datasets/THuman/render_persp/thuman2_36views'
    w_maps_dir_fname = os.path.join(path, scan_id, f'{scan_id}_w_maps_sparse', f'{sampled_cameras[i]}.npz')