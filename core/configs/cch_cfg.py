from yacs.config import CfgNode


_C = CfgNode()

_C.VISUALISE_FREQUENCY = 1000

# Train
_C.TRAIN = CfgNode()
_C.TRAIN.NUM_EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.LR = 0.0002
_C.TRAIN.EPOCHS_PER_SAVE = 3
_C.TRAIN.PIN_MEMORY = True
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.LR_SCHEDULER = 'cosine'

# Model
_C.MODEL = CfgNode()
_C.MODEL.GENDER = 'neutral'
_C.MODEL.SMPL_PATH = 'model_files/smpl'
_C.MODEL.SKINNING_WEIGHTS = True

# Loss
_C.LOSS = CfgNode()
_C.LOSS.VC_LOSS_WEIGHT = 1.0
_C.LOSS.VP_LOSS_WEIGHT = 10.0
_C.LOSS.W_REGULARISER_WEIGHT = 100.0
_C.LOSS.CHAMFER_SINGLE_DIRECTIONAL = False

# Input Data
_C.DATA = CfgNode()
_C.DATA.IMG_SIZE = 256
_C.DATA.NORMALISE = True

def get_cch_cfg_defaults():
    return _C.clone()