# o3d-utility
This module provides a set of utilities for working with point clouds and 3D meshes using Open3D.  
It also includes tools for transformations, coloring, bounding volumes, and geodesic distance computation on meshes.

## 🔄 Transformations
 * Predefined rotation matrices (90° and inverse).
 * Functions for random scaling, shifting, and rotation of point clouds/meshes.

## 🎨 Colors
 * Predefined normalized RGB colors such as RED, GREEN, BLUE, etc.

## ☁️ Point Clouds
 * `create_point_cloud_from_vertices`: create a point cloud from vertices/normals/colors.
 * `load_o3d_point_cloud`: load a point cloud from file and remove duplicates.
 * `random_scale_pcd`, `random_shift_pcd`, `random_rotation_pcd`, `random_trasformation`: random transformations.

## 🔺 Meshes
 * `create_mesh_from_vertices_faces`: create a triangle mesh from vertices and faces.
 * `load_o3d_mesh`: load a mesh from file and remove duplicates.
 * `create_lines`: generate a LineSet to visualize mesh edges.
 * `adjust_mesh_density`: subdivide a mesh until it reaches a target vertex density.
 * `highlight_points_on_mesh`: color specific vertices of a mesh for visualization.

## 📏 Distances & Graph Utilities
 * `euclidian_distances`: compute an approximate path between two vertices using Euclidean distances.
 * `a_star_geodesic`: compute geodesic distances on a mesh using the A* algorithm.
 * `compute_vertex_adj`: build vertex adjacency lists (supports multiple levels).

## 🔲 Bounding Volumes
 * `create_aabb`: compute the Axis-Aligned Bounding Box (AABB).
 * `create_obb`: compute the Oriented Bounding Box (OBB).
