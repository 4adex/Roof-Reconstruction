# Description: This file contains the handcrafted solution for the task of wireframe reconstruction 

import io
from collections import defaultdict
from typing import Tuple, List

import cv2
import hoho
import numpy as np
import scipy.interpolate as si
from PIL import Image as PImage
from hoho.color_mappings import gestalt_color_mapping
from hoho.read_write_colmap import read_cameras_binary, read_images_binary, read_points3D_binary
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

from enum import Enum


apex_color = gestalt_color_mapping["apex"]
eave_end_point = gestalt_color_mapping["eave_end_point"]
flashing_end_point = gestalt_color_mapping["flashing_end_point"]


apex_color, eave_end_point, flashing_end_point = [np.array(i) for i in [apex_color, eave_end_point, flashing_end_point]]
unclassified = np.array([(215, 62, 138)])
line_classes = ['eave', 'ridge', 'rake', 'valley']


class VertexType(Enum):
    APEX = 0
    EAVE_END_POINT = 1


class NearestNDInterpolatorWithThreshold(si.NearestNDInterpolator):
    def __init__(self, points, values, max_distance):
        super().__init__(points, values)
        self.max_distance = max_distance
        self.tree = cKDTree(points)

    def __call__(self, *args):
        # Convert the input to a 2D array of query points
        query_points = np.array(args).T
        distances, indices = self.tree.query(query_points, k=5, distance_upper_bound=self.max_distance)

        found_mask = indices != len(self.values)
        temp_values = np.concatenate([self.values, [0]])
        values = temp_values[indices]

        values = np.sum(values, axis=1)
        found_mask_sum = np.sum(found_mask, axis=1)
        found_mask = found_mask_sum != 0
        values[found_mask] /= found_mask_sum[found_mask]

        values[~found_mask] = np.nan

        return values.T
    

def empty_solution():
    '''Return a minimal valid solution, i.e. 2 vertices and 1 edge.'''
    return np.zeros((2, 3)), [(0, 1)]


def convert_entry_to_human_readable(entry):
    out = {}
    already_good = {'__key__', 'wf_vertices', 'wf_edges', 'edge_semantics', 'mesh_vertices', 'mesh_faces',
                    'face_semantics', 'K', 'R', 't'}
    for k, v in entry.items():
        if k in already_good:
            out[k] = v
            continue
        match k:
            case 'points3d':
                out[k] = read_points3D_binary(fid=io.BytesIO(v))
            case 'cameras':
                out[k] = read_cameras_binary(fid=io.BytesIO(v))
            case 'images':
                out[k] = read_images_binary(fid=io.BytesIO(v))
            case 'ade20k' | 'gestalt':
                out[k] = [PImage.open(io.BytesIO(x)).convert('RGB') for x in v]
            case 'depthcm':
                out[k] = [PImage.open(io.BytesIO(x)) for x in entry['depthcm']]
    return out



def get_vertices(image_gestalt, *, color_range=3.5, dialations=2, erosions=1, kernel_size=11):
    ### detects the apex and eave end and flashing end points
    apex_mask = cv2.inRange(image_gestalt, apex_color - color_range, apex_color + color_range)
    eave_end_point_mask = cv2.inRange(image_gestalt, eave_end_point - color_range, eave_end_point + color_range)
    flashing_end_point_mask = cv2.inRange(image_gestalt, flashing_end_point - color_range,
                                          flashing_end_point + color_range)
    eave_end_point_mask = cv2.bitwise_or(eave_end_point_mask, flashing_end_point_mask)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    apex_mask = cv2.morphologyEx(apex_mask, cv2.MORPH_DILATE, kernel, iterations=dialations)
    apex_mask = cv2.morphologyEx(apex_mask, cv2.MORPH_ERODE, kernel, iterations=erosions)

    eave_end_point_mask = cv2.morphologyEx(eave_end_point_mask, cv2.MORPH_DILATE, kernel, iterations=dialations)
    eave_end_point_mask = cv2.morphologyEx(eave_end_point_mask, cv2.MORPH_ERODE, kernel, iterations=erosions)

    *_, apex_stats, apex_centroids = cv2.connectedComponentsWithStats(apex_mask, connectivity=4, stats=cv2.CV_32S)
    *_, other_stats, other_centroids = cv2.connectedComponentsWithStats(eave_end_point_mask, connectivity=4, stats=cv2.CV_32S)

    return (apex_centroids[1:],
            other_centroids[1:],
            apex_mask,
            eave_end_point_mask,
            np.maximum(apex_stats[1:, cv2.CC_STAT_WIDTH], apex_stats[1:, cv2.CC_STAT_HEIGHT])/2,
            np.maximum(other_stats[1:, cv2.CC_STAT_WIDTH], other_stats[1:, cv2.CC_STAT_HEIGHT])/2)


def get_missed_vertices(vertices, inferred_centroids, *, min_missing_distance=200.0, **kwargs):
    vertices = KDTree(vertices)
    closest = vertices.query(inferred_centroids, k=1, distance_upper_bound=min_missing_distance)
    missed_points = inferred_centroids[closest[1] == len(vertices.data)]

    return missed_points


def get_lines_and_directions(gest_seg_np, edge_class, *, color_range=4., rho, theta, threshold, min_line_length,
                             max_line_gap, extend=30, kernel_size=3, dilation_iterations=1, **kwargs):
    edge_color = np.array(gestalt_color_mapping[edge_class])

    mask = cv2.inRange(gest_seg_np,
                       edge_color - color_range,
                       edge_color + color_range)
    mask = cv2.morphologyEx(mask,
                            cv2.MORPH_DILATE, np.ones((kernel_size, kernel_size)), iterations=dilation_iterations)

    if not np.any(mask):
        return [], []

    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(mask, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    if lines is None:
        return [], []

    line_directions = []
    edges = []

    for line_idx, line in enumerate(lines):
        for x1, y1, x2, y2 in line:
            if x1 < x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            direction = (np.array([x2 - x1, y2 - y1]))
            direction = direction / np.linalg.norm(direction)

            for extend_value in range(0, int(extend), 5):
                new_direction = extend_value * direction

                x1, y1 = -new_direction + (x1, y1)
                x2, y2 = + new_direction + (x2, y2)

                line_directions.append(direction)
                edges.append((x1, y1, x2, y2))
    return edges, line_directions


def infer_missing_vertices(ridge_edges, rake_edges):
    ridge_edges = np.array(ridge_edges)
    rake_edges = np.array(rake_edges)
    ridge_ends = np.concatenate([ridge_edges[:, 2:], ridge_edges[:, :2]])
    rake_ends = np.concatenate([rake_edges[:, 2:], rake_edges[:, :2]])
    ridge_ends = KDTree(ridge_ends)
    rake_ends = KDTree(rake_ends)
    missing_candidates = rake_ends.query_ball_tree(ridge_ends, 10)
    missing_candidates = np.concatenate([*missing_candidates])
    missing_candidates = np.unique(missing_candidates).astype(np.int32)

    return ridge_ends.data[missing_candidates]


def get_vertices_and_edges_from_segmentation(gest_seg_np, *,
                                             point_radius=30,
                                             max_angle=5.,
                                             point_radius_scale=1,
                                             **kwargs):
    '''Get the vertices and edges from the gestalt segmentation mask of the house'''
    # Apex
    connections = []
    deviation_threshold = np.cos(np.deg2rad(max_angle))

    (apex_centroids, eave_end_point_centroids,
     apex_mask, eave_end_point_mask,
     apex_radii, eave_radii) = get_vertices(gest_seg_np)

    vertices = np.concatenate([apex_centroids, eave_end_point_centroids])


    edges = []
    line_directions = []

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 40  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 60  # minimum number of pixels making up a line
    max_line_gap = 40  # maximum gap in pixels between connectable line segments
    ridge_edges, ridge_directions = get_lines_and_directions(gest_seg_np, "ridge",
                                                             rho=rho,
                                                             theta=theta,
                                                             threshold=threshold,
                                                             min_line_length=min_line_length,
                                                             max_line_gap=max_line_gap,
                                                             kernel_size=3,
                                                             dilation_iterations=3,
                                                             **kwargs)

    rake_edges, rake_directions = get_lines_and_directions(gest_seg_np, "rake",
                                                           rho=rho,
                                                           theta=theta,
                                                           threshold=threshold,
                                                           min_line_length=min_line_length,
                                                           max_line_gap=max_line_gap,
                                                           **kwargs)

    if len(ridge_edges) > 0:
        edges.append(ridge_edges)
        line_directions.append(ridge_directions)

    if len(rake_edges) > 0:
        edges.append(rake_edges)
        line_directions.append(rake_directions)

    missed_vertices = []
    if len(ridge_edges) > 0 and len(rake_edges) > 0:
        inferred_vertices = infer_missing_vertices(ridge_edges, rake_edges)
        missed_vertices = get_missed_vertices(vertices, inferred_vertices, **kwargs)
        vertices = np.concatenate([vertices, missed_vertices])
    if len(vertices) < 2:
        return [], []

    vertex_size = np.full(len(vertices), point_radius/2)
    if len(apex_radii) > 0 and len(eave_radii) > 0:
        apex_radii *= point_radius_scale
        eave_radii *= point_radius_scale
        apex_radii = np.maximum(apex_radii, 10)
        eave_radii = np.maximum(eave_radii, 10)
        point_radius = np.max([np.max(apex_radii), np.max(eave_radii)])
        vertex_size[:len(apex_radii)] = apex_radii
        vertex_size[len(apex_radii):len(apex_radii) + len(eave_radii)] = eave_radii


    vertices = KDTree(vertices)

    for edge_class in ['eave',
                       'step_flashing',
                       'flashing',
                       # 'post',
                       'valley',
                       'hip',
                       'transition_line',
                       'fascia',
                       'soffit',]:
        class_edges, class_directions = get_lines_and_directions(gest_seg_np, edge_class,
                                                                 rho=rho,
                                                                 theta=theta,
                                                                 threshold=threshold,
                                                                 min_line_length=min_line_length,
                                                                 max_line_gap=max_line_gap,
                                                                 **kwargs)

        if len(class_edges) > 0:
            edges.append(class_edges)
            line_directions.append(class_directions)

    edges = np.concatenate(edges).astype(np.float64)
    if len(edges) < 1:
        return [], []
    line_directions = np.concatenate(line_directions).astype(np.float64)

    # calculate the distances between the vertices and the edge ends

    begin_edges = KDTree(edges[:, :2])
    end_edges = KDTree(edges[:, 2:])

    begin_indices = begin_edges.query_ball_tree(vertices, point_radius)
    end_indices = end_edges.query_ball_tree(vertices, point_radius)

    line_indices = np.where(np.array([len(i) and len(j) for i, j in zip(begin_indices, end_indices)]))[0]

    # create all possible connections between begin and end candidates that correspond to a line
    begin_vertex_list = []
    end_vertex_list = []
    line_idx_list = []
    for line_idx in line_indices:
        begin_vertices, end_vertices = begin_indices[line_idx], end_indices[line_idx]
        begin_vertices, end_vertices = np.array(begin_vertices), np.array(end_vertices)
        begin_value = begin_edges.data[line_idx]
        end_value = end_edges.data[line_idx]
        begin_in_range_indices = np.where(
            np.linalg.norm(vertices.data[begin_vertices] - begin_value, axis=1)
            <
            vertex_size[begin_vertices])[0]
        end_in_range_indices = np.where(
            np.linalg.norm(vertices.data[end_vertices] - end_value, axis=1)
            <
            vertex_size[end_vertices])[0]
        begin_vertices = begin_vertices[begin_in_range_indices]
        end_vertices = end_vertices[end_in_range_indices]
        if len(begin_vertices) < 1 or len(end_vertices) < 1:
            continue


        begin_vertices, end_vertices = np.meshgrid(begin_vertices, end_vertices)
        begin_vertex_list.extend(begin_vertices.flatten())
        end_vertex_list.extend(end_vertices.flatten())

        line_idx_list.extend([line_idx] * len(begin_vertices.flatten()))

    line_idx_list = np.array(line_idx_list)
    all_connections = np.array([begin_vertex_list, end_vertex_list])

    # decrease the number of possible connections to reduce number of calculations
    possible_connections = np.unique(all_connections, axis=1)
    possible_connections = np.sort(possible_connections, axis=0)
    possible_connections = np.unique(possible_connections, axis=1)
    possible_connections = possible_connections[:, possible_connections[0, :] != possible_connections[1, :]]

    if possible_connections.shape[1] < 1:
        return [], []

    # precalculate the possible direction vectors
    possible_direction_vectors = vertices.data[possible_connections[0]] - vertices.data[possible_connections[1]]
    possible_direction_vectors = possible_direction_vectors / np.linalg.norm(possible_direction_vectors, axis=1)[:,
                                                              np.newaxis]

    owned_lines_per_possible_connections = [list() for i in range(possible_connections.shape[1])]

    # assign lines to possible connections
    for line_idx, i, j in zip(line_idx_list, begin_vertex_list, end_vertex_list):
        if i == j:
            continue
        i, j = min(i, j), max(i, j)
        for connection_idx, connection in enumerate(possible_connections.T):
            if np.all((i, j) == connection):
                owned_lines_per_possible_connections[connection_idx].append(line_idx)
                break

    # check if the lines are in the same direction as the possible connection
    for fitted_line_idx, owned_lines_per_possible_connection in enumerate(owned_lines_per_possible_connections):
        line_deviations = np.abs(
            np.dot(line_directions[owned_lines_per_possible_connection], possible_direction_vectors[fitted_line_idx]))
        if np.any(line_deviations > deviation_threshold):
            connections.append(possible_connections[:, fitted_line_idx])

    vertices = [{"xy": v, "type": VertexType.APEX} for v in apex_centroids]
    vertices += [{"xy": v, "type": VertexType.APEX} for v in missed_vertices]
    vertices += [{"xy": v, "type": VertexType.EAVE_END_POINT} for v in eave_end_point_centroids]
    return vertices, connections


def get_uv_depth(vertices, depth):
    '''Get the depth of the vertices from the depth image'''

    depth[depth > 3000] = np.nan
    uv = np.array([v['xy'] for v in vertices])
    uv_int = uv.astype(np.int32)
    H, W = depth.shape[:2]
    uv_int[:, 0] = np.clip(uv_int[:, 0], 0, W - 1)
    uv_int[:, 1] = np.clip(uv_int[:, 1], 0, H - 1)
    vertex_depth = depth[(uv_int[:, 1], uv_int[:, 0])]
    return uv, vertex_depth


def merge_vertices_3d(vert_edge_per_image, merge_th=0.1, **kwargs):
    '''Merge vertices that are close to each other in 3D space and are of same types'''
    all_3d_vertices = []
    connections_3d = []
    cur_start = 0
    types = []

    for cimg_idx, (vertices, connections, vertices_3d) in vert_edge_per_image.items():
        # remove nan values and remap the connections
        connections = [[a, b]
                       for (a, b) in connections
                       if
                       not np.any(np.isnan(vertices_3d[a]))
                       and
                       not np.any(np.isnan(vertices_3d[b]))
                       ]
        left_vertex_indices = np.where(np.all(~np.isnan(vertices_3d), axis=1))[0]

        new_indices = np.arange(len(left_vertex_indices))

        new_vertex_mapping = dict(zip(left_vertex_indices, new_indices))

        vertices = [v for i, v in enumerate(vertices) if i in new_vertex_mapping]
        types += [int(v['type'] == VertexType.APEX) for v in vertices]
        vertices_3d = vertices_3d[left_vertex_indices]
        connections = [[new_vertex_mapping[a] + cur_start, new_vertex_mapping[b] + cur_start] for a, b in connections]



        all_3d_vertices.append(vertices_3d)
        connections_3d += connections
        cur_start += len(vertices_3d)

    all_3d_vertices = np.concatenate(all_3d_vertices, axis=0)

    distmat = cdist(all_3d_vertices, all_3d_vertices)
    types = np.array(types).reshape(-1, 1)
    same_types = cdist(types, types)
    mask_to_merge = (distmat <= merge_th) & (same_types == 0)
    new_vertices = []
    new_connections = []
    to_merge = sorted(list(set([tuple(a.nonzero()[0].tolist()) for a in mask_to_merge])))
    to_merge_final = defaultdict(list)
    for i in range(len(all_3d_vertices)):
        for j in to_merge:
            if i in j:
                to_merge_final[i] += j
    for k, v in to_merge_final.items():
        to_merge_final[k] = list(set(v))
    already_there = set()
    merged = []
    for k, v in to_merge_final.items():
        if k in already_there:
            continue
        merged.append(v)
        for vv in v:
            already_there.add(vv)
    old_idx_to_new = {}
    for count, idxs in enumerate(merged):
        new_vertices.append(all_3d_vertices[idxs].mean(axis=0))
        for idx in idxs:
            old_idx_to_new[idx] = count
    new_vertices = np.array(new_vertices)
    for conn in connections_3d:
        new_con = sorted((old_idx_to_new[conn[0]], old_idx_to_new[conn[1]]))
        if new_con[0] == new_con[1]:
            continue
        if new_con not in new_connections:
            new_connections.append(new_con)
    return new_vertices, new_connections


def clean_points3d(entry, clustering_eps):
    image_dict = {}
    for k, v in entry["images"].items():
        image_dict[v.name] = v
    points = [v.xyz for k, v in entry["points3d"].items()]
    
    points = np.array(points)
    point_keys = [k for k, v in entry["points3d"].items()]
    point_keys = np.array(point_keys)
    
    clustered = DBSCAN(eps=clustering_eps, min_samples=5).fit(points).labels_
    clustered_indices = np.argsort(clustered)
    
    points = points[clustered_indices]
    point_keys = point_keys[clustered_indices]
    clustered = clustered[clustered_indices]
    
    _, cluster_indices = np.unique(clustered, return_index=True)
    
    clustered_points = np.split(points, cluster_indices[1:])
    clustered_keys = np.split(point_keys, cluster_indices[1:])
    
    biggest_cluster_index = np.argmax([len(i) for i in clustered_points])
    biggest_cluster = clustered_points[biggest_cluster_index]
    biggest_cluster_keys = clustered_keys[biggest_cluster_index]
    biggest_cluster_keys = set(biggest_cluster_keys)
    
    points3d_kdtree = KDTree(biggest_cluster)
    
    return points3d_kdtree, biggest_cluster_keys, image_dict


    
def get_depth_from_pointcloud(image, pointcloud, biggest_cluster_keys, R, t):
    belonging_points3d = []
    belonging_points2d = []
    point_indices = np.where(image.point3D_ids != -1)[0]
    for idx, point_id in zip(point_indices, image.point3D_ids[point_indices]):
        if point_id in biggest_cluster_keys:
            belonging_points3d.append(pointcloud[point_id].xyz)
            belonging_points2d.append(image.xys[idx])
    
    if len(belonging_points3d) < 1:
        print(f'No 3D points in image {image.name}')
        raise KeyError
    belonging_points3d = np.array(belonging_points3d)
    belonging_points2d = np.array(belonging_points2d)
    # projected2d, _ = cv2.projectPoints(belonging_points3d, R, t, K, dist_coeff)
    important = np.where(np.all(belonging_points2d >= 0, axis=1))
    # Normalize the uv to the camera intrinsics
    world_to_cam = np.eye(4)
    world_to_cam[:3, :3] = R
    world_to_cam[:3, 3] = t
    
    homo_belonging_points = cv2.convertPointsToHomogeneous(belonging_points3d)
    depth = cv2.convertPointsFromHomogeneous(cv2.transform(homo_belonging_points, world_to_cam))
    depth = depth[:, 0, 2]
    # projected2d = projected2d[:, 0, :]
    depth = depth[important[0]]
    # projected2d = projected2d[important[0]]
    projected2d = belonging_points2d[important[0]]
    return projected2d, depth

def predict(entry, visualize=False,
            scale_estimation_coefficient=2.5,
            clustering_eps=100,
            dist_coeff=0,
            pointcloud_depth_coeff = 1,
            interpolation_radius=200,
            **kwargs) -> Tuple[np.ndarray, List[int]]:
    if 'gestalt' not in entry or 'depthcm' not in entry or 'K' not in entry or 'R' not in entry or 't' not in entry:
        print('Missing required fields in the entry')
        return (entry['__key__'], *empty_solution())
    entry = hoho.decode(entry)

    vert_edge_per_image = {}

    points3d_kdtree, biggest_cluster_keys, image_dict = clean_points3d(entry, clustering_eps)



    for i, (gest, depthcm, K, R, t, imagekey) in enumerate(zip(entry['gestalt'],
                                                               entry['depthcm'],
                                                               entry['K'],
                                                               entry['R'],
                                                               entry['t'],
                                                               entry['__imagekey__']
                                                               )):

        gest_seg = gest.resize(depthcm.size)
        gest_seg_np = np.array(gest_seg).astype(np.uint8)
        vertices, connections = get_vertices_and_edges_from_segmentation(gest_seg_np, **kwargs)

        if (len(vertices) < 2) or (len(connections) < 1):
            print(f'Not enough vertices or connections in image {i}')
            vert_edge_per_image[i] = np.empty((0, 2)), [], np.empty((0, 3))
            continue

        depth_np = np.array(depthcm) / scale_estimation_coefficient
        uv, depth_vert_from_depth_map = get_uv_depth(vertices, depth_np)

        try:
            image = image_dict[imagekey]

            projected2d, depth = get_depth_from_pointcloud(image, entry["points3d"], biggest_cluster_keys, R, t)
            if len(depth) < 1:
                print(f'No 3D points in image {i}')
                raise KeyError
            depth *= pointcloud_depth_coeff

            interpolator = NearestNDInterpolatorWithThreshold(projected2d, depth, interpolation_radius)

            uv = np.array([v['xy'] for v in vertices])
            xi, yi = uv[:, 0], uv[:, 1]
            depth_vert_from_pointcloud = interpolator(xi, yi)
            depthmap_used = False

        except KeyError:
            depthmap_used = True

        # Normalize the uv to the camera intrinsics

        xy_local = np.ones((len(uv), 3))
        xy_local[:, 0] = (uv[:, 0] - K[0, 2]) / K[0, 0]
        xy_local[:, 1] = (uv[:, 1] - K[1, 2]) / K[1, 1]
        # Get the 3D vertices

        depth_vert_nan_idxs = None
        if depthmap_used:
            depth_vert = depth_vert_from_depth_map
        else:
            depth_vert_nan_idxs = np.where(np.isnan(depth_vert_from_pointcloud))[0]
            depth_vert_from_pointcloud[depth_vert_nan_idxs] = depth_vert_from_depth_map[depth_vert_nan_idxs]
            depth_vert = depth_vert_from_pointcloud

        norm_factor = np.linalg.norm(xy_local, axis=1)[..., None]
        if depth_vert_nan_idxs is not None and len(depth_vert_nan_idxs) > 0:
            norm_factor_min = np.min(norm_factor[depth_vert_nan_idxs])
            if len(depth_vert_nan_idxs) != len(norm_factor):
                norm_factor_max = np.max(norm_factor[~np.isin(np.arange(len(norm_factor)), depth_vert_nan_idxs)])
            else:
                norm_factor_max = np.max(norm_factor)
        else:
            norm_factor_min = np.min(norm_factor)
            norm_factor_max = np.max(norm_factor)

        vertices_3d_local = depth_vert[..., None] * xy_local
        if depthmap_used:
            vertices_3d_local /= norm_factor_max
        else:
            vertices_3d_local[depth_vert_nan_idxs] /= norm_factor_max
            vertices_3d_local[~np.isin(np.arange(len(vertices_3d_local)), depth_vert_nan_idxs)] /= norm_factor_max

        world_to_cam = np.eye(4)
        world_to_cam[:3, :3] = R
        world_to_cam[:3, 3] = t

        cam_to_world = np.linalg.inv(world_to_cam)
        vertices_3d = cv2.transform(cv2.convertPointsToHomogeneous(vertices_3d_local), cam_to_world)
        vertices_3d = cv2.convertPointsFromHomogeneous(vertices_3d).reshape(-1, 3)


        vert_edge_per_image[i] = vertices, connections, vertices_3d
    all_3d_vertices, connections_3d = merge_vertices_3d(vert_edge_per_image, **kwargs)
    all_3d_vertices_clean, connections_3d_clean = all_3d_vertices, connections_3d
    


    if (len(all_3d_vertices_clean) < 2) or len(connections_3d_clean) < 1:
        print(f'Not enough vertices or connections in the 3D vertices')
        return (entry['__key__'], *empty_solution())
    if visualize:
        from hoho.viz3d import plot_estimate_and_gt
        plot_estimate_and_gt(all_3d_vertices_clean,
                             connections_3d_clean,
                             entry['wf_vertices'],
                             entry['wf_edges'])
    return entry['__key__'], all_3d_vertices_clean, connections_3d_clean