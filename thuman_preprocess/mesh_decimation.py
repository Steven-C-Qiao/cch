import os 
import pickle 
import open3d as o3d
import numpy as np

# Check Open3D version
print(f"Open3D version: {o3d.__version__}")



if __name__=="__main__":
    input_path = '/scratch/u5aa/chexuan.u5aa/datasets/THuman/model'
    output_path = '/scratch/u5aa/chexuan.u5aa/datasets/THuman/decimated'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    subject_ids = os.listdir(input_path)

    ids_to_exclude = ['0390', '0400', '0490', '0497', '1221', '1222', '1223', '1224', '1726', '2370']

    subject_ids = [x for x in subject_ids if x not in ids_to_exclude]

    list_0 = sorted(subject_ids)[:len(subject_ids)//4]
    list_1 = sorted(subject_ids)[len(subject_ids)//4:len(subject_ids)//2]
    list_2 = sorted(subject_ids)[len(subject_ids)//2:len(subject_ids)//4*3]
    list_3 = sorted(subject_ids)[len(subject_ids)//4*3:]

    for subject_id in list_1:
        try:
            # Check if decimated files already exist
            base_name = subject_id
            pkl_filename = f"{base_name}.pkl"
            obj_filename = f"{base_name}.obj"
            pkl_path = os.path.join(output_path, pkl_filename)
            obj_path = os.path.join(output_path, obj_filename)
            
            if os.path.exists(pkl_path) and os.path.exists(obj_path):
                print(f"Skipping {subject_id} - decimated files already exist")
                continue

            fpath = os.path.join(input_path, subject_id, f'{subject_id}.obj')
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

        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
            continue