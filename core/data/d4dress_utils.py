import pickle
import numpy as np
from PIL import Image
from typing import Sequence
from torchvision import transforms

# load data from pkl_dir
def load_pickle(pkl_dir):
    with open(pkl_dir, "rb") as f:
        try:
            return pickle.load(f)
        except LookupError as e:
            # Try with encoding for python2 pickles loaded in python3
            f.seek(0)
            return pickle.load(f, encoding="latin1")

# save data to pkl_dir
def save_pickle(pkl_dir, data):
    pickle.dump(data, open(pkl_dir, "wb"))

# load image as numpy array
def load_image(img_dir):
    return np.array(Image.open(img_dir))

# save numpy array image
def save_image(img_dir, img):
    Image.fromarray(img).save(img_dir)

# get xyz rotation matrix
def rotation_matrix(angle, axis='x'):
    # get cos and sin from angle
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    # get totation matrix
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == 'z':
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R


def convert_intrinsics_to_pytorch3d_convention(intrinsics):
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    px = intrinsics[0, 2]
    py = intrinsics[1, 2]
    K = np.array([
        [-fx, 0, px, 0],
        [0, -fy, py, 0],
        [0, 0,  0,  1],
        [0, 0,  1,  0]
    ])
    return K

def get_R_T_from_extrinsics(extrinsics):
    R = extrinsics[:3, :3]
    T = extrinsics[:3, 3]
    return R, T

def d4dress_cameras_to_pytorch3d_cameras(cameras):
    Rs, Ts, Ks = [], [], []
    for camera_id, camera in cameras.items():
        R, T = get_R_T_from_extrinsics(camera['extrinsics'])
        K = convert_intrinsics_to_pytorch3d_convention(camera['intrinsics'])
        Rs.append(R.T)
        Ts.append(T)
        Ks.append(K)

    Rs = np.stack(Rs)
    Ts = np.stack(Ts)
    Ks = np.stack(Ks)

    return Rs, Ts, Ks



# sequence list for single-view human&cloth reconstruction and image-based human parsing
SUBJ_OUTFIT_SEQ_HUMANCLOTHRECON_HUMANPARSING = {

    '00122': {
        'Gender': 'male',
        'Inner': ['Take5', 'Take8'],
        'Outer': ['Take11', 'Take16'],
    },

    '00123': {
        'Gender': 'female',
        'Inner': ['Take3', 'Take5'],
        'Outer': ['Take10', 'Take11'],
    },

    '00127': {
        'Gender': 'male',
        'Inner': ['Take8', 'Take9'],
        'Outer': ['Take16', 'Take18'],
    },

    '00129': {
        'Gender': 'female',
        'Inner': ['Take3', 'Take5'],
        'Outer': ['Take11', 'Take13'],
    },

    '00134': {
        'Gender': 'male',
        'Inner': ['Take5', 'Take6'],
        'Outer': ['Take12', 'Take19'],
    },

    '00135': {
        'Gender': 'male',
        'Inner': ['Take7', 'Take10'],
        'Outer': ['Take21', 'Take24'],
    },

    '00136': {
        'Gender': 'female',
        'Inner': ['Take8', 'Take12'],
        'Outer': ['Take19', 'Take28'],
    },

    '00137': {
        'Gender': 'female',
        'Inner': ['Take5', 'Take7'],
        'Outer': ['Take16', 'Take19'],
    },

    '00140': {
        'Gender': 'female',
        'Inner': ['Take6', 'Take8'],
        'Outer': ['Take19', 'Take21'],
    },

    '00147': {
        'Gender': 'female',
        'Inner': ['Take11', 'Take12'],
        'Outer': ['Take16', 'Take19'],
    },

    '00148': {
        'Gender': 'female',
        'Inner': ['Take6', 'Take7'],
        'Outer': ['Take16', 'Take19'],
    },

    '00149': {
        'Gender': 'male',
        'Inner': ['Take4', 'Take12'],
        'Outer': ['Take14', 'Take24'],
    },

    '00151': {
        'Gender': 'female',
        'Inner': ['Take4', 'Take9'],
        'Outer': ['Take15', 'Take20'],
    },

    '00152': {
        'Gender': 'female',
        'Inner': ['Take4', 'Take8'],
        'Outer': ['Take17', 'Take18'],
    },

    '00154': {
        'Gender': 'male',
        'Inner': ['Take5', 'Take9'],
        'Outer': ['Take20', 'Take21'],
    },

    '00156': {
        'Gender': 'female',
        'Inner': ['Take4', 'Take8'],
        'Outer': ['Take14', 'Take19'],
    },

    '00160': {
        'Gender': 'male',
        'Inner': ['Take6', 'Take7'],
        'Outer': ['Take17', 'Take18'],
    },

    '00163': {
        'Gender': 'female',
        'Inner': ['Take7', 'Take10'],
        'Outer': ['Take13', 'Take15'],
    },

    '00167': {
        'Gender': 'female',
        'Inner': ['Take7', 'Take9'],
        'Outer': ['Take12', 'Take14'],
    },

    '00168': {
        'Gender': 'male',
        'Inner': ['Take3', 'Take7'],
        'Outer': ['Take11', 'Take16'],
    },

    '00169': {
        'Gender': 'male',
        'Inner': ['Take3', 'Take10'],
        'Outer': ['Take17', 'Take19'],
    },

    '00170': {
        'Gender': 'female',
        'Inner': ['Take9', 'Take11'],
        'Outer': ['Take15', 'Take24'],
    },

    '00174': {
        'Gender': 'male',
        'Inner': ['Take6', 'Take9'],
        'Outer': ['Take13', 'Take15'],
    },

    '00175': {
        'Gender': 'male',
        'Inner': ['Take4', 'Take9'],
        'Outer': ['Take13', 'Take20'],
    },

    '00176': {
        'Gender': 'female',
        'Inner': ['Take3', 'Take6'],
        'Outer': ['Take11', 'Take14'],
    },

    '00179': {
        'Gender': 'male',
        'Inner': ['Take4', 'Take8'],
        'Outer': ['Take13', 'Take15'],
    },

    '00180': {
        'Gender': 'male',
        'Inner': ['Take3', 'Take7'],
        'Outer': ['Take14', 'Take17'],
    },

    '00185': {
        'Gender': 'female',
        'Inner': ['Take7', 'Take8'],
        'Outer': ['Take17', 'Take18'],
    },

    '00187': {
        'Gender': 'female',
        'Inner': ['Take4', 'Take6'],
        'Outer': ['Take10', 'Take15'],
    },

    '00188': {
        'Gender': 'male',
        'Inner': ['Take7', 'Take8'],
        'Outer': ['Take12', 'Take18'],
    },

    '00190': {
        'Gender': 'female',
        'Inner': ['Take2', 'Take7'],
        'Outer': ['Take14', 'Take17'],
    },

    '00191': {
        'Gender': 'female',
        'Inner': ['Take3', 'Take6'],
        'Outer': ['Take13', 'Take19'],
    },

}

# sequence list for video-based human reconstruction and human representation learning
SUBJ_OUTFIT_SEQ_HUMANRECON_HUMANAVATAR = {
    'Inner': {
        '00148': {'Train': ['Take1', 'Take2', 'Take4', 'Take5', 'Take6', 'Take8', 'Take9', 'Take10'], 'Test': ['Take7']},
        '00152': {'Train': ['Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take7', 'Take9'], 'Test': ['Take8']},
        '00154': {'Train': ['Take1', 'Take3', 'Take4', 'Take5', 'Take6', 'Take7', 'Take8', 'Take11'], 'Test': ['Take9']},
        '00185': {'Train': ['Take1', 'Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take8', 'Take9'], 'Test': ['Take7']},
    },
    'Outer': {
        '00127': {'Train': ['Take11', 'Take13', 'Take14', 'Take15', 'Take16', 'Take17', 'Take19'], 'Test': ['Take18']},
        '00137': {'Train': ['Take12', 'Take13', 'Take14', 'Take15', 'Take17', 'Take18', 'Take19', 'Take20', 'Take21'], 'Test': ['Take16']},
        '00149': {'Train': ['Take14', 'Take15', 'Take16', 'Take17', 'Take20', 'Take22', 'Take24', 'Take25'], 'Test': ['Take21']},
        '00188': {'Train': ['Take10', 'Take11', 'Take12', 'Take15', 'Take16', 'Take17', 'Take18'], 'Test': ['Take14']},
    }
}

# sequence list for clothing simulation
SUBJ_OUTFIT_SEQ_CLOTHSIMULATION = {
    'lower': {
        '00129': ['Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take22'],
        '00156': ['Take2', 'Take3', 'Take4', 'Take7', 'Take8', 'Take9'],
        '00152': ['Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take7'],
        '00174': ['Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take7'],
    },
    'upper': {
        '00127': ['Take5', 'Take6', 'Take7', 'Take8', 'Take9', 'Take10'],
        '00140': ['Take1', 'Take3', 'Take4', 'Take6', 'Take7', 'Take8'],
        '00147': ['Take1', 'Take2', 'Take3', 'Take4', 'Take6', 'Take9'],
        '00180': ['Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take7'],
    },
    'dress': {
        '00185': ['Take1', 'Take2', 'Take3', 'Take4', 'Take7', 'Take8'],
        '00148': ['Take4', 'Take5', 'Take6', 'Take7', 'Take8', 'Take9'],
        '00170': ['Take1', 'Take3', 'Take5', 'Take7', 'Take8', 'Take9'],
        '00187': ['Take1', 'Take2', 'Take3', 'Take4', 'Take5', 'Take6'],
    },
    'outer': {
        '00123': ['Take8', 'Take9', 'Take10', 'Take11', 'Take12', 'Take13'],
        '00152': ['Take10', 'Take12', 'Take15', 'Take17', 'Take18', 'Take19'],
        '00176': ['Take9', 'Take10', 'Take11', 'Take12', 'Take13', 'Take14'],
        '00190': ['Take10', 'Take11', 'Take13', 'Take14', 'Take15', 'Take16'],
    },   
}