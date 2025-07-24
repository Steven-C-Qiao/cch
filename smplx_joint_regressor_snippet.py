import torch
import numpy as np
from smplx.lbs import vertices2joints

def find_joint_locations_from_vertices(vertices, J_regressor):
    """
    Find joint locations from vertices using a joint regressor.
    
    Args:
        vertices: torch.Tensor of shape (batch_size, 10495, 3) - SMPL-X mesh vertices
        J_regressor: torch.Tensor of shape (num_joints, 10495) - Joint regressor matrix
    
    Returns:
        joints: torch.Tensor of shape (batch_size, num_joints, 3) - Joint locations
    """
    # Use SMPL-X's vertices2joints function to compute joint locations
    joints = vertices2joints(J_regressor, vertices)
    return joints

# Example usage:
if __name__ == "__main__":
    # Example 1: Using SMPL-X model's built-in joint regressor
    try:
        import smplx
        
        # Initialize SMPL-X model (you need SMPL-X model files)
        smplx_model = smplx.create(
            model_path='path/to/smplx/models',  # Update this path
            model_type='smplx',
            gender='neutral',
            batch_size=1
        )
        
        # Get the joint regressor from SMPL-X model
        J_regressor = smplx_model.J_regressor  # Shape: (55, 10495)
        
        print(f"SMPL-X Joint Regressor shape: {J_regressor.shape}")
        
    except Exception as e:
        print(f"Could not load SMPL-X model: {e}")
        print("Creating a dummy joint regressor for demonstration...")
        
        # Create a dummy joint regressor for demonstration
        num_joints = 55  # SMPL-X has 55 joints
        num_vertices = 10495  # SMPL-X has 10495 vertices
        
        # Create a simple joint regressor (this is just for demonstration)
        J_regressor = torch.zeros(num_joints, num_vertices)
        
        # Assign equal weights to vertices for each joint (simplified)
        vertices_per_joint = num_vertices // num_joints
        for i in range(num_joints):
            start_idx = i * vertices_per_joint
            end_idx = min((i + 1) * vertices_per_joint, num_vertices)
            J_regressor[i, start_idx:end_idx] = 1.0 / (end_idx - start_idx)
    
    # Create example vertices (batch_size=2, 10495 vertices, 3 coordinates each)
    batch_size = 2
    vertices = torch.randn(batch_size, 10495, 3)
    
    print(f"Input vertices shape: {vertices.shape}")
    
    # Find joint locations
    joints = find_joint_locations_from_vertices(vertices, J_regressor)
    
    print(f"Output joints shape: {joints.shape}")
    print(f"Number of joints: {joints.shape[1]}")
    print(f"First joint location: {joints[0, 0].tolist()}")
    
    # Example 2: Manual computation (alternative method)
    def manual_joint_computation(vertices, J_regressor):
        """Manual computation of joint locations for understanding."""
        batch_size = vertices.shape[0]
        num_joints = J_regressor.shape[0]
        
        # Reshape vertices to (batch_size, num_vertices * 3)
        vertices_flat = vertices.view(batch_size, -1)
        
        # Expand J_regressor to handle 3D coordinates
        J_expanded = torch.zeros(num_joints, 10495 * 3)
        for i in range(num_joints):
            for j in range(10495):
                weight = J_regressor[i, j]
                J_expanded[i, j*3] = weight     # x
                J_expanded[i, j*3+1] = weight   # y
                J_expanded[i, j*3+2] = weight   # z
        
        # Compute joints: (batch_size, num_vertices*3) @ (num_vertices*3, num_joints)
        joints_manual = torch.matmul(vertices_flat, J_expanded.T)
        joints_manual = joints_manual.view(batch_size, num_joints, 3)
        
        return joints_manual
    
    # Verify both methods give the same result
    joints_manual = manual_joint_computation(vertices, J_regressor)
    print(f"Methods match: {torch.allclose(joints, joints_manual, atol=1e-6)}") 