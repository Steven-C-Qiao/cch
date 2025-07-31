import torch 

images = torch.randn(2, 4, 224, 224, 3)

conf = torch.randn(2, 4, 224, 224)
conf = conf > 0

filtered_points = images[conf]

print(filtered_points.shape)


import ipdb; ipdb.set_trace()