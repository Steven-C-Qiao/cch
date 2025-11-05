

import os
from core.data.thuman_metadata import THuman_metadata

def main():
    # Get all scan IDs from metadata
    metadata_scans = set()
    for subject_id, subject_data in THuman_metadata.items():
        for scan_id in subject_data['scans']:
            metadata_scans.add(scan_id)
    
    print(f"Total scans in metadata: {len(metadata_scans)}")
    
    # Get all scan IDs from decimated directory
    decimated_dir = '/scratch/u5aa/chexuan.u5aa/datasets/THuman/decimated/'
    decimated_scans = set()
    
    if os.path.exists(decimated_dir):
        for filename in os.listdir(decimated_dir):
            if filename.endswith('.obj'):
                scan_id = filename.replace('.obj', '')
                decimated_scans.add(scan_id)
    
    print(f"Total scans in decimated directory: {len(decimated_scans)}")
    
    # Find scans in metadata but not in decimated directory
    missing_in_decimated = metadata_scans - decimated_scans
    missing_in_metadata = decimated_scans - metadata_scans
    
    print(f"\nScans in metadata but NOT in decimated directory ({len(missing_in_decimated)}):")
    for scan_id in sorted(missing_in_decimated):
        print(f"  {scan_id}")
    
    print(f"\nScans in decimated directory but NOT in metadata ({len(missing_in_metadata)}):")
    for scan_id in sorted(missing_in_metadata):
        print(f"  {scan_id}")
    
    # Show some statistics
    print(f"\nStatistics:")
    print(f"  Metadata scans: {len(metadata_scans)}")
    print(f"  Decimated scans: {len(decimated_scans)}")
    print(f"  Missing in decimated: {len(missing_in_decimated)}")
    print(f"  Missing in metadata: {len(missing_in_metadata)}")
    print(f"  Common scans: {len(metadata_scans & decimated_scans)}")

if __name__ == "__main__":
    main()
