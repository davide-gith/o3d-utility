import math
import heapq
import numpy as np
import open3d as o3d
from tqdm import tqdm
from typing import Union


# --------------------------------------- TRANSFORMATIONS ---------------------------------------
# Rotation angle90
angle90 = np.radians(90)

# Rotation matrices
rot_matrix = np.array([[1, 0, 0, 0],
                       [0, np.cos(angle90), -np.sin(angle90), 0],
                       [0, np.sin(angle90), np.cos(angle90), 0],
                       [0, 0, 0, 1]])

rot_matrix_inverse = np.array([[1, 0, 0, 0],
                               [0, np.cos(-angle90), -np.sin(-angle90), 0],
                               [0, np.sin(-angle90), np.cos(-angle90), 0],
                               [0, 0, 0, 1]])

# --------------------------------------- COLORS ---------------------------------------
YELLOW = [1.0, 1.0, 0.0]
ORANGE = [1.0, 0.5, 0.0]
PURPLE = [0.5, 0, 1]
RED = [1.0, 0.0, 0.0]
GREEN = [0.0, 1.0, 0.0]
BLUE = [0.0, 0.0, 1.0]
WHITE = [1.0, 1.0, 1.0]
BLACK = [0.0, 0.0, 0.0]
GRAY = [0.5, 0.5, 0.5]


# --------------------------------------- POINT CLOUD & MESH FUNCTIONS ---------------------------------------
def create_point_cloud_from_vertices(pc_vertices: np.ndarray, pc_normals: np.ndarray = None, pc_colors: list = None) -> o3d.geometry.PointCloud:
    """ Given vertices and faces, this function creates a triangular mesh.
    :param pc_vertices: vertices of the point cloud
    :param pc_normals: normals of the point cloud (optional)
    :param pc_colors: colors of the point cloud (optional)
    :return: point cloud """

    # Create new point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc_vertices)

    # Apply normals if exists
    if pc_normals is not None:
        point_cloud.normals = o3d.utility.Vector3dVector(pc_normals)

    # Apply colors if exists, otherwise apply GRAY
    if pc_colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(pc_colors)
    else:
        colors = [GRAY] * len(point_cloud.points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


def load_o3d_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """ Import a point cloud with open3d and remove duplicate points
    :param path: path to the point cloud
    :return: point cloud """

    point_cloud = o3d.io.read_point_cloud(path)
    point_cloud.remove_duplicated_points()

    return point_cloud


def create_mesh_from_vertices_faces(m_vertices: np.ndarray, m_faces: np.ndarray, m_normals: np.ndarray = None, m_colors: np.ndarray = None) -> o3d.geometry.TriangleMesh:
    """ Given vertices and faces, this function creates a triangular mesh.
     :param m_vertices: vertices of the mesh
     :param m_faces: faces of the mesh
     :param m_normals: normals of the mesh (optional)
     :param m_colors: colors of the mesh (optional)
     :return: triangular mesh """

    # Create new mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(m_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(m_faces)

    # Apply normals if exists
    if m_normals is not None:
        mesh.vertex_normals = o3d.utility.Vector3dVector(m_normals)

    # Apply colors if exists, otherwise apply GRAY
    if m_colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(m_colors)
    else:
        colors = [GRAY] * len(mesh.vertices)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh


def load_o3d_mesh(path: str) -> o3d.geometry.TriangleMesh:
    """ Import a mesh with open3d and remove duplicate vertices and faces
    :param path: path to the mesh
    :return: mesh """

    mesh = o3d.io.read_triangle_mesh(path)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()

    return mesh


def create_lines(mesh_vertices: np.ndarray, edges: o3d.utility.Vector2iVector, color: list = None) -> o3d.geometry.LineSet:
    """ Create a LineSet from vertices and edges of a mesh.
    :param mesh_vertices: vertices of the mesh
    :param edges: edges of the mesh, saved in a numpy array of shape (n, 2). Ex: [[0,5], [5,8], [8, 10]]
    :param color: color to apply to the edges, ex: GREEN, RED, ... (optional)
    :return: Lineset """

    # LineSet creation
    lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(mesh_vertices),
        lines=o3d.utility.Vector2iVector(np.array(edges)),
    )

    # Apply colors if exists, otherwise apply BLACK
    if color is not None:
        lines_color = [color] * len(edges)
    else:
        lines_color = [BLACK] * len(edges)
    lines.colors = o3d.utility.Vector3dVector(lines_color)

    return lines


def adjust_mesh_density(mesh: o3d.geometry.TriangleMesh, target_density: int) -> o3d.geometry.TriangleMesh:
    """ Given a triangular mesh and a target number of points, the function densifies the initial mesh until the number of points exceeds the target
    :param mesh: the initial mesh to densifies
    :param target_density: number of points to overcome
    :return: the densified mesh """

    count = 1
    while True:
        denser_mesh = mesh.subdivide_midpoint(number_of_iterations=count)
        if len(denser_mesh.vertices) > target_density:
            return denser_mesh
        count += 1


def highlight_points_on_mesh(mesh: o3d.geometry.TriangleMesh, points_to_highlight: list, dir: str, color: list = None) -> None:
    """ Given a mesh and some points to highlight, this function shows that colored mesh.
    :param mesh: starting mesh
    :param points_to_highlight: list of indexes on the mesh to highlight
    :param dir: current directory
    :param color: color to apply to the edges, ex: GREEN, RED, ... (optional) """

    # Apply WHITE if color is None
    if color is None: color = WHITE

    mesh_colors = [GRAY] * len(mesh.vertices)
    for z in range(len(points_to_highlight)):
        mesh_colors[points_to_highlight[z]] = color

    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    o3d.visualization.draw_geometries([mesh], window_name=dir, width=800, height=800)


def random_scale_pcd(pcd, scale_low: float = 0.8, scale_high: float = 1.25) -> o3d.geometry.PointCloud:
    """ This function randomly scale the input point cloud.
    :param pcd: the initial point cloud
    :param scale_low: low bound of scaling
    :param scale_high: high bound of scaling
    :return: the scaled point cloud """

    scale = np.random.uniform(scale_low, scale_high)
    pcd.scale(scale, center=pcd.get_center())

    return pcd


def random_shift_pcd(pcd: o3d.geometry.PointCloud, shift_range: float = 0.1) -> o3d.geometry.PointCloud:
    """ This function randomly shift the input point cloud.
    :param pcd: the initial point cloud
    :param shift_range: range of shift value
    :return pcd_mesh: the shifted point cloud """

    shift = np.random.uniform(-shift_range, shift_range, 3)
    pcd.translate(shift)

    return pcd


def random_rotation_pcd(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """ This function randomly rotate the point cloud.
    :param pcd: the initial point cloud
    :return: the rotated point cloud """

    rotation_angle = np.random.uniform() * 2 * np.pi
    rotation_matrix = pcd.get_rotation_matrix_from_axis_angle([rotation_angle, rotation_angle, rotation_angle])
    pcd.rotate(rotation_matrix, center=pcd.get_center())

    return pcd


def random_trasformation(pcd: o3d.geometry.TriangleMesh) -> o3d.geometry.PointCloud:
    """ This function randomly translate, rotate and shift a point cloud
    :param pcd: the initial point cloud
    :return: the transformed point cloud """

    pcd = random_scale_pcd(pcd, scale_low=0.2, scale_high=10)
    pcd = random_shift_pcd(pcd, shift_range=10)
    pcd = random_rotation_pcd(pcd)

    return pcd


def euclidian_distances(mesh: o3d.geometry.TriangleMesh, mesh_adj: list, source_point_idx: int, target_point_idx: int) -> list:
    """ This function compute the distances between two points computing euclidian distances between points
    :param vertices: vertices of the mesh
    :param mesh_adj: adjacency matrix of the mesh
    :param source_adj: adjacency matrix of the source
    :param target_point_idx: index of the point in the mesh
    :return: list of indexes that form the path between two points with euclidian distances """

    vertices = np.array(mesh.vertices)
    source_adj = list(mesh_adj[source_point_idx])

    path = []

    while True:
        distance = math.inf
        new_point_idx = -1000
        num_visited_points = 0

        # Compute closest point
        for elem in source_adj:
            new_distance = np.linalg.norm(vertices[elem] - vertices[target_point_idx])
            if new_distance < distance and len(list(mesh_adj[elem])) > 0:
                # If new_distance is usable, but elem is already in path, updated num_visited_points, otherwise save new_point_idx and distance
                if elem in path:
                    num_visited_points += 1
                else:
                    distance = new_distance
                    new_point_idx = elem

        # If new_point_idx hasn't been changed, something has gone wrong
        if new_point_idx == -1000:
            # We can tolerate that all adjeacent points have already been visited
            if num_visited_points == len(source_adj):
                break
            else:
                return []

        # Add to the path
        path.append(new_point_idx)

        # Update source_adj
        source_adj = list(mesh_adj[new_point_idx])

        # Check if the path is complete (case 2)
        if target_point_idx in source_adj:
            break

    return path


def heuristic(current: int, goal: int, vertices: np.ndarray):
    """ The euristic of A* for geodesic distances is the distance between two points
    :param current: current point
    :param goal: goal point
    :param vertices: vertices of the mesh """
    return np.linalg.norm(vertices[current] - vertices[goal])


def a_star_geodesic(mesh: o3d.geometry.TriangleMesh, start_idx: int, terget_idx: int) -> list:
    """ Given a mesh, a start point and a target point, this function computes the geodesic distances through A*
    :param mesh: 3d mesh
    :param start_idx: index of the start point
    :param terget_idx: index of the target point
    :return: list of indexes that form the path between two points with geodesic distances """

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Costruisce il grafo dai triangoli
    graph = {}
    for triangle in triangles:
        for i in range(3):
            if triangle[i] not in graph:
                graph[triangle[i]] = []
            for j in range(3):
                if i != j:
                    graph[triangle[i]].append(triangle[j])

    open_set = []
    heapq.heappush(open_set, (0, start_idx))

    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start_idx] = 0

    f_score = {node: float('inf') for node in graph}
    f_score[start_idx] = heuristic(start_idx, terget_idx, vertices)

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == terget_idx:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_idx)
            path.reverse()
            return path

        for neighbor in graph[current]:
            tentative_g_score = g_score[current] + np.linalg.norm(vertices[current] - vertices[neighbor])

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, terget_idx, vertices)

                if not any(neighbor == item[1] for item in open_set):
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []


# --------------------------------------- BOUNDING VOLUMS ---------------------------------------
def create_aabb(object_3d: Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]) -> o3d.geometry.LineSet:
    """ Given a point cloud or a mesh this function computes the aabb of the 3d object
    :param object_3d: point cloud or mesh
    :return: aabb of the point cloud """

    if isinstance(object_3d, o3d.geometry.PointCloud):
        vertices = np.asarray(object_3d.points)
    elif isinstance(object_3d, o3d.geometry.TriangleMesh):
        vertices = np.asarray(object_3d.vertices)
    else:
        raise TypeError("Input must be either a PointCloud or TriangleMesh.")
      
    max_x = np.max(vertices[:, 0])  # max on x-axis
    max_y = np.max(vertices[:, 1])  # max on y-axis
    max_z = np.max(vertices[:, 2])  # max on z-axis

    min_x = np.min(vertices[:, 0])  # min on x-axis
    min_y = np.min(vertices[:, 1])  # min on y-axis
    min_z = np.min(vertices[:, 2])  # min on z-axis

    box_vertices = np.array([[max_x, max_y, min_z],
                             [max_x, max_y, max_z],
                             [min_x, max_y, max_z],
                             [min_x, max_y, min_z],
                             [max_x, min_y, min_z],
                             [max_x, min_y, max_z],
                             [min_x, min_y, max_z],
                             [min_x, min_y, min_z]])

    box_edges = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0],
                                            [4, 5], [5, 6], [6, 7], [7, 4],
                                            [0, 4], [1, 5], [2, 6], [3, 7]])

    aabb = create_lines(box_vertices, box_edges, RED)

    return aabb


def create_obb(object_3d: Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]) -> o3d.geometry.OrientedBoundingBox:
    """ Given a point cloud or a mesh this function computes the obb of the object
    :param object_3d: point cloud or mesh
    :return: obb of the object """

    obb = object_3d.get_oriented_bounding_box()
    obb.color = BLUE

    return obb


def compute_vertex_adj(mesh: o3d.geometry.TriangleMesh, level: int=1) -> list[list[int]]:
    """ Given a 3d mesh this function computes the vertex adjacency list by levels.
    If the level is 1 you will get the neighbors of each point, otherwise for levels greater than 1 you will have
    neighbors excluding points from previous levels.
    :param mesh: 3d mesh
    :param level: level to compute adjacency list
    :return: vertex adjacency list """

    if level < 1:
        raise ValueError("Level musts be greater or equals than 0.")

    mesh.compute_adjacency_list()
    adj_mesh = list(mesh.adjacency_list)

    # Compute adjacency for level = 1
    if level == 1:
        return [list(adj_mesh[idx]) for idx in range(len(adj_mesh))]


    # Compute adjacency for levels > 1
    result_adj = []
    for idx in range(len(adj_mesh)):
        prev_set = {idx}  # Vertices from previous levels
        current_set = set(adj_mesh[idx])  # Immediate neighbors

        # Iterate until the desired level
        for _ in range(1, level):
            next_set = set(
                neighbor
                for vertex in current_set
                for neighbor in adj_mesh[vertex]
                if neighbor not in prev_set and neighbor not in current_set
            )
            # Update
            prev_set.update(current_set)
            current_set = next_set

        # Collect neighbors at the specified level
        result_adj.append(list(current_set))

    return result_adj