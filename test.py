import trimesh 


path = 'Figures/vis/00134_take1_exp_090/pred_vp_025.ply'

mesh = trimesh.load(path)

print(mesh.vertices.shape)
print(mesh.faces.shape)
print(mesh.vertices)
print(mesh.faces)

import ipdb; ipdb.set_trace()