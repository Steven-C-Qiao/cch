from core.data.thuman_dataset import THumanDataset, thuman_collate_fn
import torch

if __name__ == "__main__":

    from core.configs.cch_cfg import get_cch_cfg_defaults
    cfg = get_cch_cfg_defaults()

    dataset = THumanDataset(cfg)

    # Example of how to use the dataset with custom collate function
    # This prevents PyTorch from trying to stack meshes with different vertex counts
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=thuman_collate_fn)
    
    print("Testing dataset with custom collate function...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: tensor shape {v.shape}")
            else:
                print(f"  {k}: list of {len(v)} items")
                if k in ['scan_mesh', 'smpl_data'] and v:
                    # Show some info about the first mesh
                    first_mesh = v[0][0] if isinstance(v[0], list) else v[0]
                    if isinstance(first_mesh, dict) and 'vertices' in first_mesh:
                        print(f"    First mesh has {len(first_mesh['vertices'])} vertices")
        
        if batch_idx >= 1:  # Only test first few batches
            break


    print(f"batch keys: {batch.keys()}")
    print(f"batch['imgs'].shape: {batch['imgs'].shape}")
    print(f"batch['masks'].shape: {batch['masks'].shape}")
    print(type(batch['scan_mesh'][0]))
    
    import ipdb; ipdb.set_trace()
    print('')