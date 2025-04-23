import numpy as np

PATH = '/scratches/kyuban/cq244/datasets/CAPE/sequences/00032/longshort_ATUsquat/longshort_ATUsquat.000001.npz'

def inspect_single_frame():

    data = np.load(PATH)

    for key, value in data.items():
        print(key, value.shape)

    import ipdb 
    ipdb.set_trace()

    return None



if __name__ == "__main__":
    inspect_single_frame()
