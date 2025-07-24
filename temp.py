import pickle 
import numpy as np

def load_pickle(pkl_dir):
    """Load pickle file with proper encoding handling."""
    try:
        # Try with default encoding first
        with open(pkl_dir, "rb") as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        # If that fails, try with latin-1 encoding
        with open(pkl_dir, "rb") as f:
            return pickle.load(f, encoding='latin-1')
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

# Load the SMPL data
data = load_pickle('/scratches/kyuban/cq244/CCH/cch/model_files/smpl/SMPL_NEUTRAL.pkl')

if data is not None:
    print("Successfully loaded SMPL data!")
    print(f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
    
    # Print some basic info about the data
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: numpy array with shape {value.shape}")
            elif isinstance(value, (list, tuple)):
                print(f"{key}: {type(value)} with length {len(value)}")
            else:
                print(f"{key}: {type(value)}")
else:
    print("Failed to load SMPL data")

print(data['J_regressor'].shape)