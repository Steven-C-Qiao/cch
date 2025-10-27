import os 
import numpy as np 
import trimesh

PATH_TO_DATASET = '/scratch/u5aa/chexuan.u5aa/4DDress'


ids = [
    '00122', '00123', '00127', '00129', '00135', '00136', '00137',
    '00140', '00147', '00149', '00151', '00152', '00154', '00156',
    '00160', '00163', '00167', '00168', '00169', '00170', '00174', '00175',
    '00176', '00179', '00180', '00185', '00187', '00188', '00190', '00191'
]

for id in ids:
    template_dir = os.path.join(PATH_TO_DATASET, '_4D-DRESS_Template', id)
    template_mesh = trimesh.load(os.path.join(template_dir, 'body.ply'))

    height = template_mesh.vertices[:, 1].max() - template_mesh.vertices[:, 1].min()
    print(f'{id}: {height}')