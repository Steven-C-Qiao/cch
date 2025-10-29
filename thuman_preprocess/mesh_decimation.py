import os 
import pickle 
import open3d as o3d
import numpy as np
import subprocess
import sys

# Check Open3D version
print(f"Open3D version: {o3d.__version__}")

# import smplx 
# smpl_model = smplx.create(
#     model_type='smplx',
#     model_path="model_files/",
#     num_betas=10,
#     gender='male',
#     num_pca_comps=12,
# )

def process_single_subject(subject_id, input_path, output_path):
    """Process a single subject in a separate process to handle segfaults"""
    try:
        # Check if decimated files already exist
        base_name = subject_id
        pkl_filename = f"{base_name}.pkl"
        obj_filename = f"{base_name}.obj"
        pkl_path = os.path.join(output_path, pkl_filename)
        obj_path = os.path.join(output_path, obj_filename)
        
        # if os.path.exists(pkl_path) and os.path.exists(obj_path):
        #     print(f"Skipping {subject_id} - decimated files already exist")
        #     return True

        # fpath = os.path.join(input_path, subject_id, f'{subject_id}.obj')
        fpath = os.path.join(input_path, f'{subject_id}.obj')
        print(f"Processing {fpath}...")
        
        # Load the mesh using Open3D's tensor API (PR 5199 improvements)
        o3d_obj = o3d.io.read_triangle_mesh(fpath)
        
        # Convert to tensor mesh for improved decimation (PR 5199)
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d_obj)
        
        # Apply improved quadric decimation with tensor API
        # target_reduction=0.75 means reduce by 75%, keeping 25% of original faces
        decimated_t_mesh = t_mesh.simplify_quadric_decimation(target_reduction=0.75)
        
        # Convert back to legacy format for compatibility
        decimated_obj = decimated_t_mesh.to_legacy()
        
        # Extract vertices and faces
        scan_verts = np.asarray(decimated_obj.vertices)
        scan_faces = np.asarray(decimated_obj.triangles)
        
        # Save to pickle
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'scan_verts': scan_verts,
                'scan_faces': scan_faces
            }, f)
        
        # Save decimated mesh as OBJ
        o3d.io.write_triangle_mesh(obj_path, decimated_obj)

        
        # print(f"Saved decimated mesh to {pkl_filename} and {obj_filename}")
        print(f"Original faces: {len(o3d_obj.triangles)}, Decimated faces: {len(scan_faces)}")
        print(f"Original vertices: {len(o3d_obj.vertices)}, Decimated vertices: {len(scan_verts)}")
        # print(f"Reduction: {((len(o3d_obj.triangles) - len(scan_faces)) / len(o3d_obj.triangles) * 100):.1f}%")
        # print("-" * 50)
        return True

    except Exception as e:
        print(f"Error processing {subject_id}: {e}")
        return False

import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=int, default=0)
    parser.add_argument('--num_jobs', type=int, default=8)
    args = parser.parse_args()
    job = args.job
    input_path = '/scratch/u5aa/chexuan.u5aa/datasets/THuman/cleaned'
    output_path = '/scratch/u5aa/chexuan.u5aa/datasets/THuman/decimated'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # subject_ids = os.listdir(input_path)
    subject_ids = [x.split('.')[0] for x in os.listdir(input_path)]

    ids_to_exclude = ['0390', '0400', '0490', '0497', '1221', '1222', '1223', '1224', '1726', '2370']


    subject_ids = [x for x in subject_ids if x not in ids_to_exclude]
    subject_ids = sorted(list(set([x for x in subject_ids if x <= '0526'])))

    # Calculate the portion of subject_ids to process for this job
    total_subjects = len(subject_ids)
    subjects_per_job = total_subjects // args.num_jobs
    remainder = total_subjects % args.num_jobs
    
    # Calculate start and end indices for this job
    start_idx = args.job * subjects_per_job
    if args.job < remainder:
        start_idx += args.job
        end_idx = start_idx + subjects_per_job + 1
    else:
        start_idx += remainder
        end_idx = start_idx + subjects_per_job
    
    # Get the subset of subject_ids for this job
    job_subject_ids = subject_ids[start_idx:end_idx]

    # subject_ids = ['1075', '2281', '0399', '1074', '2285', '2283', 
    #                '2286', '1651', '1077', '2287', '1072', '1647', '1645', '1076', '0397', '1647', '0395', '0398', '1078']

    # subject_ids = ['1649', '2284']
    # subject_ids = ['0400', '0490', '0497', '1726', '2370']
    
    print(f"Job {args.job}/{args.num_jobs}: Processing subjects {start_idx}-{end_idx-1} out of {total_subjects} total subjects")
    print(f"Subjects to process: {len(job_subject_ids)}")

    # Process each subject in a separate subprocess to handle segfaults
    successful_count = 0
    failed_count = 0
    
    for subject_id in job_subject_ids:
        print(f"\n--- Processing subject {subject_id} ---")
        
        # Create a subprocess to handle potential segfaults
        try:
            # Run the processing in a subprocess
            result = subprocess.run([
                sys.executable, '-c', 
                f'''
import sys
sys.path.append("{os.getcwd()}")
from thuman_preprocess.mesh_decimation import process_single_subject
success = process_single_subject("{subject_id}", "{input_path}", "{output_path}")
sys.exit(0 if success else 1)
'''
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                print(f"✓ Successfully processed {subject_id}")
                successful_count += 1
            else:
                print(f"✗ Failed to process {subject_id}")
                print(f"Error output: {result.stderr}")
                failed_count += 1
                
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout processing {subject_id} (exceeded 5 minutes)")
            failed_count += 1
        except Exception as e:
            print(f"✗ Unexpected error processing {subject_id}: {e}")
            failed_count += 1
    
    print(f"\n=== Job {args.job} Summary ===")
    print(f"Successfully processed: {successful_count}")
    print(f"Failed: {failed_count}")
    print(f"Total: {len(job_subject_ids)}")



# from core.configs.paths import THUMAN_PATH
# from core.data.thuman_metadata import THuman_metadata
# from core.data.d4dress_utils import load_pickle
# import os 

# from tqdm import tqdm


# if __name__ == "__main__":
#     ids = os.listdir(os.path.join(THUMAN_PATH, 'decimated'))
#     ids = [id.split('.')[0] for id in ids]
#     for id in tqdm(ids):

#         scan_fname = os.path.join(THUMAN_PATH, 'decimated', f'{id}.pkl')

#         try:
#             scan_data = load_pickle(scan_fname)
#             # print(f"Loaded scan data for ID {id}, number of vertices: {scan_data['vertices'].shape[0]}")
#         except Exception as e:
#             print(f"Failed to load scan data for ID {id}: {e}")