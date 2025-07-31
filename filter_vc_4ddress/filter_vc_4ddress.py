import numpy as np
import trimesh
from scipy.spatial import cKDTree
import pyrender
import cv2
import os

def render_depth_map(mesh, camera_pos, resolution=(1024, 1024), fov=60):
    """
    Render a depth map of a mesh from a given camera position.
    
    Args:
        mesh: trimesh.Trimesh object
        camera_pos: Camera position [x, y, z]
        resolution: Render resolution (width, height)
        fov: Field of view in degrees
    
    Returns:
        depth_map: 2D numpy array with depth values
    """
    # Create pyrender scene
    scene = pyrender.Scene()
    
    # Convert trimesh to pyrender mesh
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_pyrender)
    
    # Calculate camera parameters
    mesh_center = mesh.centroid
    camera_distance = np.linalg.norm(camera_pos - mesh_center)
    
    # Create camera looking at mesh center
    camera = pyrender.PerspectiveCamera(
        yfov=np.radians(fov),
        aspectRatio=resolution[0] / resolution[1]
    )
    
    # Position camera
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_pos
    
    # Make camera look at mesh center
    look_at = mesh_center - camera_pos
    look_at = look_at / np.linalg.norm(look_at)
    
    # Calculate up vector (assuming Y is up)
    up = np.array([0, 1, 0])
    right = np.cross(look_at, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, look_at)
    
    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = up
    camera_pose[:3, 2] = -look_at
    
    scene.add(camera, pose=camera_pose)
    
    # Render depth map
    renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])
    # depth_map, _ = renderer.render(scene)
    _, depth_map = renderer.render(scene)
    renderer.delete()
    
    return depth_map

def project_vertices(vertices, camera_pos, resolution=(1024, 1024), fov=60):
    """
    Project 3D vertices to 2D screen coordinates.
    
    Args:
        vertices: Nx3 array of 3D vertices
        camera_pos: Camera position [x, y, z]
        resolution: Screen resolution (width, height)
        fov: Field of view in degrees
    
    Returns:
        screen_coords: Nx2 array of screen coordinates
    """
    # Calculate mesh center for camera orientation
    mesh_center = np.mean(vertices, axis=0)
    
    # Camera parameters
    camera_distance = np.linalg.norm(camera_pos - mesh_center)
    aspect_ratio = resolution[0] / resolution[1]
    fov_rad = np.radians(fov)
    
    # Calculate camera basis vectors
    look_at = mesh_center - camera_pos
    look_at = look_at / np.linalg.norm(look_at)
    
    up = np.array([0, 1, 0])
    right = np.cross(look_at, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, look_at)
    
    # Transform vertices to camera space
    camera_to_world = np.eye(4)
    camera_to_world[:3, 0] = right
    camera_to_world[:3, 1] = up
    camera_to_world[:3, 2] = -look_at
    camera_to_world[:3, 3] = camera_pos
    
    world_to_camera = np.linalg.inv(camera_to_world)
    
    # Transform vertices
    vertices_homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
    vertices_camera = (world_to_camera @ vertices_homogeneous.T).T
    
    # Perspective projection
    screen_coords = np.zeros((len(vertices), 2))
    
    for i, vertex in enumerate(vertices_camera):
        if vertex[2] > 0:  # Only project vertices in front of camera
            # Perspective projection
            x = vertex[0] / vertex[2] * np.tan(fov_rad / 2) * aspect_ratio
            y = vertex[1] / vertex[2] * np.tan(fov_rad / 2)
            
            # Convert to screen coordinates
            screen_x = (x + 1) * resolution[0] / 2
            screen_y = (1 - y) * resolution[1] / 2
            
            screen_coords[i] = [screen_x, screen_y]
        else:
            screen_coords[i] = [-1, -1]  # Invalid coordinates
    
    return screen_coords

def filter_occluded_vertices(body_mesh, clothing_mesh, camera_positions, resolution=(1024, 1024)):
    """
    Filter out body vertices that are occluded by clothing from multiple camera angles.
    
    Args:
        body_mesh: trimesh.Trimesh of the naked body
        clothing_mesh: trimesh.Trimesh of the clothing
        camera_positions: List of camera positions to test from
        resolution: Render resolution for depth testing
    
    Returns:
        visible_vertex_indices: Indices of body vertices that are visible
    """
    # Start with all vertices as invisible (False)
    visible_vertices = set()  # Empty set means all vertices start as invisible
    
    print(f"Testing visibility from {len(camera_positions)} camera positions...")
    
    for i, camera_pos in enumerate(camera_positions):
        print(f"Processing camera {i+1}/{len(camera_positions)} at position {camera_pos}")
        
        try:
            # Render depth maps for both meshes
            body_depth = render_depth_map(body_mesh, camera_pos, resolution)
            clothing_depth = render_depth_map(clothing_mesh, camera_pos, resolution)
            clothing_depth[clothing_depth==0] = 100.
            
            # import matplotlib.pyplot as plt
            # # Visualize depth maps
            # plt.figure(figsize=(12, 6))
            
            # plt.subplot(121)
            # plt.imshow(body_depth)
            # plt.colorbar(label='Depth')
            # plt.title('Body Depth Map')
            
            # plt.subplot(122)
            # plt.imshow(clothing_depth)
            # plt.colorbar(label='Depth') 
            # plt.title('Clothing Depth Map')
            
            # plt.tight_layout()
            # plt.savefig(f'depth_maps_camera_{i+1}.png')
            # plt.close()

            # Project body vertices to screen space
            body_screen_coords = project_vertices(body_mesh.vertices, camera_pos, resolution)
            
            # Check each body vertex
            for j, (vertex, screen_coord) in enumerate(zip(body_mesh.vertices, body_screen_coords)):
                x, y = int(screen_coord[0]), int(screen_coord[1])
                if 0 <= x < resolution[0] and 0 <= y < resolution[1]:
                    # If body is visible (not occluded by clothing) at this pixel
                    if body_depth[y, x] < clothing_depth[y, x] - 0.001:  # Small tolerance
                        visible_vertices.add(j)  # Set vertex as visible
        
        except Exception as e:
            print(f"Warning: Failed to process camera {i+1}: {e}")
            continue
    
    print(f"Found {len(visible_vertices)} visible vertices out of {len(body_mesh.vertices)} total")
    return list(visible_vertices)

def generate_camera_positions(mesh_center, radius, num_cameras=8, height_factor=0.5):
    """
    Generate camera positions around the mesh for comprehensive coverage.
    
    Args:
        mesh_center: Center point of the mesh
        radius: Distance from center to place cameras
        num_cameras: Number of cameras to generate
        height_factor: Height variation factor
    
    Returns:
        camera_positions: List of camera positions
    """
    camera_positions = []
    
    # Generate cameras at different heights
    heights = [0, height_factor * radius, -height_factor * radius]
    
    for height in heights:
        for i in range(num_cameras):
            angle = 2 * np.pi * i / num_cameras
            x = mesh_center[0] + radius * np.cos(angle)
            y = mesh_center[1] + radius * np.sin(angle)
            z = mesh_center[2] + height
            camera_positions.append([x, y, z])
    
    return camera_positions

def filter_visible_vertices_raycasting(body_mesh, clothing_mesh, resolution=(1024, 1024), num_cameras=8):
    """
    Complete ray casting implementation for filtering visible vertices.
    
    Args:
        body_mesh: trimesh.Trimesh of the naked body
        clothing_mesh: trimesh.Trimesh of the clothing
        resolution: Render resolution for depth testing
        num_cameras: Number of camera positions to test from
    
    Returns:
        visible_vertex_indices: Indices of body vertices that are visible
    """
    # Calculate mesh bounds and center
    mesh_center = body_mesh.centroid
    mesh_bounds = body_mesh.bounds
    mesh_size = np.max(mesh_bounds[1] - mesh_bounds[0])
    radius = mesh_size * 1.5  # Distance from center
    
    # Generate camera positions
    camera_positions = generate_camera_positions(mesh_center, radius, num_cameras)
    
    # Filter occluded vertices
    visible_indices = filter_occluded_vertices(body_mesh, clothing_mesh, camera_positions, resolution)
    
    return visible_indices

# Example usage function
def example_usage(body_mesh, clothing_mesh):
    """
    Example of how to use the ray casting vertex filtering.
    """
    print(f"Body mesh: {len(body_mesh.vertices)} vertices")
    print(f"Clothing mesh: {len(clothing_mesh.vertices)} vertices")
    
    # Filter visible vertices
    visible_indices = filter_visible_vertices_raycasting(body_mesh, clothing_mesh)
    
    # Create new mesh with only visible vertices
    visible_body_mesh = body_mesh.copy()
    visible_body_mesh.update_vertices(visible_indices)
    visible_body_mesh.remove_unreferenced_vertices()
    
    print(f"Visible body mesh: {len(visible_body_mesh.vertices)} vertices")
    
    # Save result
    visible_body_mesh.export('visible_body.obj')
    print("Saved visible body mesh to 'visible_body.obj'")



if __name__ == "__main__":
    PATH_TO_DATASET = "/scratches/kyuban/cq244/datasets/4DDress"

    ids = [ '00122', '00123', '00127', '00129', '00134', '00135', '00136', '00137', 
            '00140', '00147', '00148', '00149', '00151', '00152', '00154', '00156', 
            '00160', '00163', '00167', '00168', '00169', '00170', '00174', '00175', 
            '00176', '00179', '00180', '00185', '00187', '00188', '00190', '00191']
    
    id = '00122'
    

    # ---- template mesh ----
    template_dir = os.path.join(PATH_TO_DATASET, '_4D-DRESS_Template', id)
        
    upper_mesh = trimesh.load(os.path.join(template_dir, 'upper.ply'))
    body_mesh = trimesh.load(os.path.join(template_dir, 'body.ply'))
    if os.path.exists(os.path.join(template_dir, 'lower.ply')):
        lower_mesh = trimesh.load(os.path.join(template_dir, 'lower.ply'))
        full_mesh = trimesh.util.concatenate([lower_mesh, body_mesh, upper_mesh])
        clothing_mesh = trimesh.util.concatenate([lower_mesh, upper_mesh])

    else:
        full_mesh = trimesh.util.concatenate([body_mesh, upper_mesh])
        clothing_mesh = upper_mesh

    example_usage(body_mesh, clothing_mesh)
