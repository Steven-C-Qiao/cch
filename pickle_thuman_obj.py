#!/usr/bin/env python3
"""
Script to pickle THuman OBJ files for faster loading during training.
This preprocesses all OBJ files and saves vertices, faces, and metadata as pickle files.
"""

import os
import pickle
import trimesh
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

def pickle_thuman_objs(input_path, output_path, force_reprocess=False):
    """
    Preprocess THuman OBJ files and save as pickle files for faster loading.
    
    Args:
        input_path: Path to THuman model directory
        output_path: Path to save pickled files
        force_reprocess: If True, reprocess even if pickle files exist
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all scan directories
    scan_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(scan_dirs)} scan directories")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for scan_dir in tqdm(scan_dirs, desc="Processing scans"):
        scan_id = scan_dir.name
        obj_file = scan_dir / f"{scan_id}.obj"
        pickle_file = output_path / f"{scan_id}.pkl"
        
        # Skip if pickle file exists and not forcing reprocess
        if pickle_file.exists() and not force_reprocess:
            skipped_count += 1
            continue
            
        if not obj_file.exists():
            print(f"Warning: OBJ file not found: {obj_file}")
            error_count += 1
            continue
            
        try:
            # Load mesh with optimized settings for speed
            mesh = trimesh.load(
                str(obj_file), 
                process=False,  # Skip processing for speed
                maintain_order=True,  # Keep vertex order
                skip_materials=True  # Skip material loading
            )
            
            # Extract essential data
            vertices = mesh.vertices.astype(np.float32)  # Convert to float32 for memory efficiency
            faces = mesh.faces.astype(np.uint32)  # Convert to uint32 for memory efficiency
            
            # Create metadata
            metadata = {
                'scan_id': scan_id,
                'num_vertices': len(vertices),
                'num_faces': len(faces),
                'bounds': mesh.bounds,
                'centroid': mesh.centroid,
                'scale': mesh.scale,
                'file_size_mb': obj_file.stat().st_size / (1024 * 1024)
            }
            
            # Save as pickle
            pickle_data = {
                'vertices': vertices,
                'faces': faces,
                'metadata': metadata
            }
            
            with open(pickle_file, 'wb') as f:
                pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {scan_id}: {e}")
            error_count += 1
            continue
    
    print(f"\nProcessing complete!")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Total: {len(scan_dirs)}")

def test_pickle_loading(pickle_path, num_samples=5):
    """
    Test loading speed of pickled files vs original OBJ files.
    """
    import time
    
    pickle_path = Path(pickle_path)
    pickle_files = list(pickle_path.glob("*.pkl"))
    
    if len(pickle_files) == 0:
        print("No pickle files found!")
        return
    
    # Test pickle loading
    print(f"Testing pickle loading with {min(num_samples, len(pickle_files))} files...")
    start_time = time.time()
    
    for i, pickle_file in enumerate(pickle_files[:num_samples]):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {pickle_file.name}: {data['metadata']['num_vertices']} vertices, {data['metadata']['num_faces']} faces")
    
    pickle_time = time.time() - start_time
    print(f"Pickle loading time: {pickle_time:.3f}s")
    
    # Test original OBJ loading for comparison
    print(f"\nTesting original OBJ loading...")
    start_time = time.time()
    
    for i, pickle_file in enumerate(pickle_files[:num_samples]):
        scan_id = pickle_file.stem
        # Assuming original OBJ files are in the same structure
        obj_path = pickle_file.parent.parent / "model" / scan_id / f"{scan_id}.obj"
        if obj_path.exists():
            mesh = trimesh.load(str(obj_path), process=False, maintain_order=True, skip_materials=True)
            print(f"Loaded {obj_path.name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    obj_time = time.time() - start_time
    print(f"OBJ loading time: {obj_time:.3f}s")
    print(f"Speedup: {obj_time/pickle_time:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pickle THuman OBJ files for faster loading")
    parser.add_argument("--input", default="/scratches/kyuban/cq244/datasets/THuman/model", 
                       help="Path to THuman model directory")
    parser.add_argument("--output", default="/scratches/kyuban/cq244/datasets/THuman/pickled", 
                       help="Path to save pickled files")
    parser.add_argument("--force", action="store_true", 
                       help="Force reprocessing even if pickle files exist")
    parser.add_argument("--test", action="store_true", 
                       help="Test loading speed after processing")
    parser.add_argument("--test-samples", type=int, default=5, 
                       help="Number of samples to test loading speed")
    
    args = parser.parse_args()
    
    # Process files
    pickle_thuman_objs(args.input, args.output, args.force)
    
    # Test loading if requested
    if args.test:
        test_pickle_loading(args.output, args.test_samples)