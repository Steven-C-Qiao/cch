# Canonical Human Representation

The core of the CCH network is a warping from **posed partial surface normal maps** to **partial canonical human**, i.e. each input image completes part of the canonical human.

Representation of the canonical human is thus crucial. Below are potential choices, with them comes with different warper strategies.

### Mesh with SMPL topology
Treating the canonical human as a mesh with SMPL topology. Warper transforms surface normal maps to updates to the canonical mesh. The difficulty here is that observations are partial, and output mesh should also be partial, which is difficult for the network to learn since partial mesh dimensions vary. 

However, we can output updates to the full canonical mesh, and alongside a per-vertex confidence.

### SDF
Treating the canonical human as a signed distance field. Warper transforms surface normal maps to updates to the canonical SDF. 

## Weak supervision
**Ground truth canonical human does not exist**. To supervise warper, leverage a cyclic consistency loss.