import torch
import numpy as np
import smplx
from smplx.lbs import vertices2joints
from typing import Optional, Tuple

def create_smplx_joint_regressor_example():
    """
    Example code to use joint regressor to find joint locations from vertices
    for an SMPL-X mesh with 10495 vertices.
    """
    
    # Initialize SMPL-X model
    # Note: You need to have SMPL-X model files in the specified path
    smplx_model = smplx.create(
        model_path='path/to/smplx/models',  # Update this path to your SMPL-X model files
        model_type='smplx',
        gender='neutral',
        num_betas=10,
        num_expression_coeffs=10,
        use_face_contour=True,
        batch_size=1
    )
    
    # Get the joint regressor from the SMPL-X model
    # SMPL-X has 55 joints (including face, hands, and feet joints)
    J_regressor = smplx_model.J_regressor  # Shape: (55, 10495)
    
    print(f"Joint regressor shape: {J_regressor.shape}")
    print(f"Number of joints: {J_regressor.shape[0]}")
    print(f"Number of vertices: {J_regressor.shape[1]}")
    
    # Create example vertices (batch_size=2, num_vertices=10495, 3 coordinates)
    batch_size = 2
    num_vertices = 10495
    vertices = torch.randn(batch_size, num_vertices, 3)
    
    print(f"Input vertices shape: {vertices.shape}")
    
    # Method 1: Using SMPL-X's built-in vertices2joints function
    joints_from_vertices = vertices2joints(J_regressor, vertices)
    print(f"Joints from vertices shape: {joints_from_vertices.shape}")
    
    # Method 2: Manual computation using matrix multiplication
    # Reshape vertices to (batch_size, num_vertices * 3) for matrix multiplication
    vertices_flat = vertices.view(batch_size, -1)  # Shape: (batch_size, 10495 * 3)
    
    # Expand J_regressor to handle 3D coordinates
    # J_regressor: (55, 10495) -> (55, 10495 * 3)
    J_regressor_expanded = torch.zeros(J_regressor.shape[0], J_regressor.shape[1] * 3)
    
    for i in range(J_regressor.shape[0]):  # For each joint
        for j in range(J_regressor.shape[1]):  # For each vertex
            weight = J_regressor[i, j]
            J_regressor_expanded[i, j*3] = weight     # x-coordinate
            J_regressor_expanded[i, j*3+1] = weight   # y-coordinate  
            J_regressor_expanded[i, j*3+2] = weight   # z-coordinate
    
    # Compute joints manually
    joints_manual = torch.matmul(J_regressor_expanded, vertices_flat.T).T
    joints_manual = joints_manual.view(batch_size, -1, 3)
    
    print(f"Manual joints shape: {joints_manual.shape}")
    
    # Verify both methods give the same result
    print(f"Methods match: {torch.allclose(joints_from_vertices, joints_manual, atol=1e-6)}")
    
    return joints_from_vertices, J_regressor

def create_custom_joint_regressor(num_joints: int = 55, num_vertices: int = 10495):
    """
    Create a custom joint regressor matrix for SMPL-X.
    
    Args:
        num_joints: Number of joints (default 55 for SMPL-X)
        num_vertices: Number of vertices (default 10495 for SMPL-X)
    
    Returns:
        J_regressor: Joint regressor matrix of shape (num_joints, num_vertices)
    """
    
    # Create a sparse joint regressor matrix
    # Each joint is computed as a weighted average of nearby vertices
    J_regressor = torch.zeros(num_joints, num_vertices)
    
    # Example: Create a simple joint regressor where each joint is influenced by
    # vertices in a specific region. This is a simplified example.
    # In practice, you would use the actual SMPL-X joint regressor.
    
    # For demonstration, let's create a simple regressor where:
    # - Joint 0 (root) is influenced by vertices in the pelvis region
    # - Joint 1 (left hip) is influenced by vertices in the left hip region
    # - etc.
    
    # This is just an example - you would need the actual vertex-to-joint mapping
    vertices_per_joint = num_vertices // num_joints
    
    for joint_idx in range(num_joints):
        start_vertex = joint_idx * vertices_per_joint
        end_vertex = min((joint_idx + 1) * vertices_per_joint, num_vertices)
        
        # Assign equal weights to vertices in this joint's region
        num_vertices_for_joint = end_vertex - start_vertex
        if num_vertices_for_joint > 0:
            weight = 1.0 / num_vertices_for_joint
            J_regressor[joint_idx, start_vertex:end_vertex] = weight
    
    # Normalize each row to sum to 1
    row_sums = J_regressor.sum(dim=1, keepdim=True)
    J_regressor = J_regressor / (row_sums + 1e-8)
    
    return J_regressor

def apply_joint_regressor_to_vertices(vertices: torch.Tensor, 
                                    J_regressor: torch.Tensor) -> torch.Tensor:
    """
    Apply joint regressor to vertices to get joint locations.
    
    Args:
        vertices: Tensor of shape (batch_size, num_vertices, 3)
        J_regressor: Joint regressor matrix of shape (num_joints, num_vertices)
    
    Returns:
        joints: Tensor of shape (batch_size, num_joints, 3)
    """
    
    # Use SMPL-X's vertices2joints function
    joints = vertices2joints(J_regressor, vertices)
    
    return joints

def visualize_joint_regressor_weights(J_regressor: torch.Tensor, 
                                    joint_idx: int = 0,
                                    save_path: Optional[str] = None):
    """
    Visualize the weights of a joint regressor for a specific joint.
    
    Args:
        J_regressor: Joint regressor matrix
        joint_idx: Index of joint to visualize
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    
    weights = J_regressor[joint_idx].detach().cpu().numpy()
    
    plt.figure(figsize=(12, 4))
    plt.plot(weights)
    plt.title(f'Joint Regressor Weights for Joint {joint_idx}')
    plt.xlabel('Vertex Index')
    plt.ylabel('Weight')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

def main():
    """
    Main function demonstrating SMPL-X joint regressor usage.
    """
    
    print("=== SMPL-X Joint Regressor Example ===")
    print("This example shows how to use a joint regressor to find joint locations")
    print("from vertices for an SMPL-X mesh with 10495 vertices.\n")
    
    try:
        # Try to use actual SMPL-X model
        joints, J_regressor = create_smplx_joint_regressor_example()
        print("✓ Successfully used SMPL-X model")
        
    except Exception as e:
        print(f"⚠ Could not load SMPL-X model: {e}")
        print("Creating custom joint regressor for demonstration...\n")
        
        # Create custom joint regressor for demonstration
        num_joints = 55  # SMPL-X has 55 joints
        num_vertices = 10495  # SMPL-X has 10495 vertices
        
        J_regressor = create_custom_joint_regressor(num_joints, num_vertices)
        
        # Create example vertices
        batch_size = 2
        vertices = torch.randn(batch_size, num_vertices, 3)
        
        # Apply joint regressor
        joints = apply_joint_regressor_to_vertices(vertices, J_regressor)
        
        print(f"✓ Created custom joint regressor")
        print(f"  - Joint regressor shape: {J_regressor.shape}")
        print(f"  - Input vertices shape: {vertices.shape}")
        print(f"  - Output joints shape: {joints.shape}")
    
    # Demonstrate the joint regressor usage
    print("\n=== Joint Regressor Usage ===")
    
    # Example: Get joint locations for a single mesh
    single_vertices = torch.randn(1, 10495, 3)  # Single mesh with 10495 vertices
    single_joints = apply_joint_regressor_to_vertices(single_vertices, J_regressor)
    
    print(f"Single mesh vertices: {single_vertices.shape}")
    print(f"Single mesh joints: {single_joints.shape}")
    print(f"First joint location: {single_joints[0, 0].tolist()}")
    
    # Example: Get joint locations for multiple meshes
    multiple_vertices = torch.randn(5, 10495, 3)  # 5 meshes with 10495 vertices each
    multiple_joints = apply_joint_regressor_to_vertices(multiple_vertices, J_regressor)
    
    print(f"\nMultiple meshes vertices: {multiple_vertices.shape}")
    print(f"Multiple meshes joints: {multiple_joints.shape}")
    
    # Show joint statistics
    print(f"\n=== Joint Statistics ===")
    print(f"Mean joint positions: {multiple_joints.mean(dim=0).shape}")
    print(f"Joint position std: {multiple_joints.std(dim=0).shape}")
    
    # Visualize joint regressor weights for the first joint
    print(f"\n=== Visualizing Joint Regressor Weights ===")
    try:
        visualize_joint_regressor_weights(J_regressor, joint_idx=0)
        print("✓ Created visualization of joint regressor weights")
    except ImportError:
        print("⚠ matplotlib not available, skipping visualization")
    
    print("\n=== Summary ===")
    print("The joint regressor is a matrix that maps vertex positions to joint positions.")
    print("For SMPL-X:")
    print(f"  - Input: {num_vertices} vertices (3D coordinates each)")
    print(f"  - Output: {J_regressor.shape[0]} joints (3D coordinates each)")
    print("  - The regressor matrix has shape: (num_joints, num_vertices)")
    print("  - Each joint is computed as a weighted average of vertices")
    
    return joints, J_regressor

if __name__ == "__main__":
    main() 