from yacs.config import CfgNode


_C = CfgNode()

_C.VISUALISE_FREQUENCY = 1000

# Train
_C.TRAIN = CfgNode()
_C.TRAIN.NUM_EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.LR = 0.0001
_C.TRAIN.EPOCHS_PER_SAVE = 3
_C.TRAIN.PIN_MEMORY = True
_C.TRAIN.NUM_WORKERS = 4


# Model
_C.MODEL = CfgNode()
_C.MODEL.GENDER = 'neutral'
_C.MODEL.NUM_SMPL_BETAS = 10
_C.MODEL.MEAN_CAM_T = [0.0, 0.25, 2.5]
_C.MODEL.SMPL_PATH = 'model_files/smpl'

# Input Data
_C.DATA = CfgNode()
_C.DATA.IMG_SIZE = 256


# Loss
_C.LOSS = CfgNode()
_C.LOSS.REDUCTION = 'mean'


def get_cch_cfg_defaults():
    return _C.clone()