#!/usr/bin/env python3

import pyvista as pv
import sys
import os
import numpy as np

def visualize_ply(ply_path):
    """
    Display a PLY mesh in an interactive viewer using PyVista.
    
    Args:
        ply_path (str): Path to the PLY file
    """
    # Check if file exists
    if not os.path.exists(ply_path):
        print(f"Error: File {ply_path} does not exist.")
        sys.exit(1)
        
    # Check if file is a PLY file
    if not ply_path.lower().endswith('.ply'):
        print("Error: Input file must be a PLY file.")
        sys.exit(1)
    
    try:
        # Create interactive plotter
        plotter = pv.Plotter()
        
        # Load the PLY file
        print(f"Loading PLY file: {ply_path}")
        mesh = pv.read(ply_path)
        
        # Print some vertex positions
        print("\nSample vertex positions:")
        vertices = mesh.points
        for i in range(min(5, len(vertices))):
            print(f"Vertex {i}: {vertices[i]}")
            
        # Print vertex colors if they exist
        if 'RGB' in mesh.array_names:
            print("\nSample vertex colors:")
            colors = mesh.get_array('RGB')
            for i in range(min(5, len(colors))):
                print(f"Vertex {i} color: {colors[i]}")

        import ipdb; ipdb.set_trace()

        
        
        # Check if mesh is empty
        if mesh.n_points == 0:
            print("Error: The mesh contains no vertices.")
            sys.exit(1)
            
        # Print basic mesh information
        print(f"Mesh contains {mesh.n_points} vertices and {mesh.n_cells} faces.")
        
        # Set up the visualization
        plotter.add_mesh(
            mesh,
            show_edges=False,
            smooth_shading=True,
            specular=0.5,
            specular_power=15,
        )
        
        # Set background color (dark gray)
        plotter.background_color = '#1a1a1a'
        
        # Set camera position for initial view
        plotter.camera_position = 'iso'  # isometric view
        plotter.camera.zoom(1.5)  # adjust zoom level
        
        # Enable shadows for better depth perception
        plotter.enable_shadows()
        
        # Add text with controls information
        control_text = """
        Controls:
        Left Mouse: Rotate
        Middle Mouse: Pan
        Right Mouse: Zoom
        R: Reset Camera
        Q: Quit
        """
        plotter.add_text(control_text, position='upper_left', font_size=12)
        
        # Show the interactive viewer
        plotter.show()
        
    except Exception as e:
        print(f"Error occurred while processing the PLY file: {str(e)}")
        sys.exit(1)

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python vis_ply.py <path_to_ply_file>")
        sys.exit(1)
        
    ply_path = sys.argv[1]
    visualize_ply(ply_path)

if __name__ == "__main__":
    main()