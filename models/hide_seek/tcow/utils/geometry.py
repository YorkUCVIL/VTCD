'''
Tools / utils / helper methods pertaining to camera projections and other 3D stuff.
Created by Basile Van Hoorick, Jul 2022.
'''

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'utils/'))

from __init__ import *

# Library imports.
import numpy as np


def calculate_3d_point_trajectory_kubric(data_stack, metadata, query_time, query_uv):
    '''
    Given freshly generated Kubric data, calculate the 3D trajectories of a set of query points over
        time. The query points are specified in normalized perspective (image space) coordinates.
    :param data_stack (dict).
    :param metadata (dict).
    :param query_time (Q) array: Frame indices (int) of where to apply the query points.
    :param query_uv (Q, 2) array: Batch of normalized image coordinates (u, v) in pixel space.
    :return See calculate_3d_point_trajectory().
    '''
    depth = data_stack['depth']  # (T, H, W, 1).
    segmentation = data_stack['segmentation']  # (T, H, W, 1).
    object_coordinates = data_stack['object_coordinates'] / 65535.0  # (T, H, W, 3) in [0, 1].
    frame_inds = np.arange(depth.shape[0])

    return calculate_3d_point_trajectory_frames(
        depth, segmentation, object_coordinates, metadata, query_time, query_uv, frame_inds)


def calculate_3d_point_trajectory_frames(
        depth, segmentation, object_coordinates, metadata, query_time, query_uv, frame_inds):
    '''
    Given a video annotated with instance segmentation and depth, calculate the 3D trajectories of a
        set of query points over time. The query points are specified in normalized perspective
        (image space) coordinates.
    :param depth (T, H, W, 1) array: Distance from camera in meters per pixel.
    :param segmentation (T, H, W, 1) array: Integer object instance indices per pixel.
    :param object_coordinates (T, H, W, 3) array: (x, y, z) in [0, 1] with offsets relative to
        object local coordinate systems per pixel.
    :param metadata (dict).
    :param query_time (Q) array: Frame indices (int) of where to apply the query points.
    :param query_uv (Q, 2) array: Batch of normalized image coordinates (u, v) in pixel space.
    :param frame_inds (list): Subset of used frame indices.
    :return traject_retval (dict)
        query_time (Q) array: Copy of input.
        query_uv (Q, 2) array: Copy of input.
        query_depth (Q) array.
        query_xyz (Q, 3) array.
        uvdxyz (Q, T, 6) array: (u, v, d, x, y, z) that describes both perspective (UVD) and
            canonical (XYZ) coordinates.
        uvdxyz_static (Q, T, 6) array: Assumes the point to track never moves in 3D world space.
        inst_ids (Q): Selected instance indices (int); equals segmentation map values minus 1.
        inst_coords (Q, 3) array: (x, y, z) in [0, 1] relative to instance local coordinate system.
        inst_bboxes (Q, T, 8, 3) array: (x, y, z) for each 3D cube encompassing instances over time.
    '''
    Q = query_uv.shape[0]
    T = len(frame_inds)
    (W, H) = metadata['scene']['resolution']
    query_u = query_uv[..., 0]  # (Q).
    query_v = query_uv[..., 1]  # (Q).
    qu_idx = np.clip(np.floor(query_u * W).astype(np.int32), 0, W - 1)  # (Q).
    qv_idx = np.clip(np.floor(query_v * H).astype(np.int32), 0, H - 1)  # (Q).

    assert np.all(query_time == query_time[0]), 'Currently all times must be the same.'

    # NOTE / WARNING: In Kubric, depth is distance from the camera itself, not the offset from the
    # camera plane!
    # Sanity check while debugging: metadata['camera']['K'] should match camera_K[-1];
    # same for camera_R (but only approximately because the former is one frame after the end).
    (camera_K, camera_R) = get_camera_matrices_numpy(metadata)  # (T, 3, 3), (T, 4, 4).
    camera_K = camera_K[frame_inds]  # (T, 3, 3).
    camera_R = camera_R[frame_inds]  # (T, 3, 3).

    query_thw_inds = np.stack([query_time, qv_idx, qu_idx], axis=0)  # (3, Q).
    query_thw_inds_flat = np.ravel_multi_index(query_thw_inds, (T, H, W))  # (Q).
    query_depth = depth.ravel()[query_thw_inds_flat]  # (Q).
    query_uvd = np.stack([query_u, query_v, query_depth], axis=-1)  # (Q, 3).
    query_xyz = unproject_points_2d_to_3d(
        query_uvd, camera_K[query_time[0]], camera_R[query_time[0]])  # (Q, 3).

    # inst_id = segmentation[query_time, qv_idx, qu_idx, 0] - 1  # (Q).
    # inst_coords = object_coordinates[query_time, qv_idx, qu_idx]  # (Q, 3) with (x, y, z) in [0, 1].

    inst_ids = segmentation.astype(np.int32).ravel()[query_thw_inds_flat] - 1  # (Q).
    inst_coords = object_coordinates.ravel().reshape(-1, 3)[query_thw_inds_flat]
    # (Q, 3) with (x, y, z) in [0, 1].
    instances = []
    inst_bboxes_3d = []
    for inst_id in inst_ids:

        if inst_id == -1:
            if 'dome' in metadata:
                instances.append(metadata['dome'])
                bbox_3d = metadata['dome']['bboxes_3d']
                bbox_3d = np.array(bbox_3d)[frame_inds]
                inst_bboxes_3d.append(bbox_3d)
            else:
                instances.append(None)
                inst_bboxes_3d.append(np.zeros((T, 8, 3)))

        else:
            # bboxes_3d denotes the 8 corners in 3D XYZ world space over time.
            instances.append(metadata['instances'][inst_id])
            bbox_3d = metadata['instances'][inst_id]['bboxes_3d']
            bbox_3d = np.array(bbox_3d)[frame_inds]
            inst_bboxes_3d.append(bbox_3d)

    inst_bboxes_3d = np.stack(inst_bboxes_3d)  # (Q, T, 8, 3).

    traject_retval = calculate_3d_point_trajectory(
        query_time, query_uv, query_depth, query_xyz, inst_ids, inst_coords, inst_bboxes_3d,
        camera_K, camera_R)

    return traject_retval


def calculate_3d_point_trajectory(
        query_time, query_uv, query_depth, query_xyz, inst_ids, inst_coords, inst_bboxes_3d,
        camera_K, camera_R):
    (Q, T) = inst_bboxes_3d.shape[:2]

    uvdxyz = np.zeros((Q, T, 6), dtype=np.float32)
    uvdxyz_static = np.zeros((Q, T, 6), dtype=np.float32)

    # NOTE: If inst_id == -1, this is a background (dome) pixel, which does not have an associated
    # foreground instance. In this case, we assume the point has static world coordinates since the
    # dome never moves. We always calculate static tracks for baseline purposes anyway.
    for t in range(T):
        tracked_xyz = query_xyz  # (Q, 3).
        tracked_uvd = project_points_3d_to_2d(tracked_xyz, camera_K[t], camera_R[t])  # (Q, 3).
        uvdxyz_static[..., t, 0:3] = tracked_uvd
        uvdxyz_static[..., t, 3:6] = tracked_xyz

    for t in range(T):
        # One way to obtain the local to world transform of the tracked object is to calculate the
        # rotation matrix from the quaternion, then apply it to the object coordinates (after
        # manually scaling them correctly in each dimension), then apply the object position.
        # But we can also estimate an end-to-end transform via least squares instead, which combines
        # these steps automatically. To do this, we have to define a source coordinate box that
        # matches the range of object_coordinates (and thus inst_coords) calculated before.
        coord_box = np.array(list(itertools.product([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])))  # (8, 3).
        coord_box_homo = np.concatenate(
            [coord_box, np.ones_like(coord_box[..., 0:1])], axis=-1)  # (8, 4).
        coord_box_homo = repeat(coord_box_homo, 'C X -> Q C X', Q=Q)  # (Q, 8, 4).
        inst_bbox_3d = inst_bboxes_3d[..., t, :, :]  # (Q, 8, 3).
        inst_bbox_3d_homo = np.concatenate(
            [inst_bbox_3d, np.ones_like(inst_bbox_3d[..., 0:1])], axis=-1)  # (Q, 8, 4).
        inst_coords_homo = np.concatenate(
            [inst_coords, np.ones_like(inst_coords[..., 0:1])], axis=-1)  # (Q, 4).

        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
        # Doesn't seem to support solving multiple systems independently at once.
        # TODX: Optimize by calculating only once per instance (many points fall on the same bbox).
        world_to_locals = []
        for n in range(Q):
            world_to_local = np.linalg.lstsq(coord_box_homo[n], inst_bbox_3d_homo[n], rcond=None)
            world_to_locals.append(world_to_local[0])  # (4, 4).
        world_to_local = np.stack(world_to_locals, axis=0)  # (Q, 4, 4).

        # We right-multiply object coordinates with: transpose of local to world = world to local.
        # Apply batch equivalent of: matmul(inst_coords_homo, world_to_local).
        # tracked_point_homo = np.matmul(inst_coords_homo[:, None, :], world_to_local).squeeze(-2)  # (Q, 4).
        tracked_point_homo = np.einsum('ni,nij->nj', inst_coords_homo, world_to_local)  # (Q, 4).
        tracked_xyz = tracked_point_homo[..., 0:3] / tracked_point_homo[..., 3:4]  # (Q, 3).
        tracked_uvd = project_points_3d_to_2d(tracked_xyz, camera_K[t], camera_R[t])  # (Q, 3).
        uvdxyz[..., t, 0:3] = tracked_uvd
        uvdxyz[..., t, 3:6] = tracked_xyz

    # Dome is not supported in older exports, so replace invalid rows.
    uvdxyz[inst_ids == -1] = uvdxyz_static[inst_ids == -1]

    # NOTE: Sanity checks (comparing query info with uvdxyz) are performed in pipeline.py.
    # NOTE: The biggest error in this process is due to object_coordinates, which are essentially
    # just 16-bit values. This means that occlusion estimation will not be extremely precise. In
    # general, we should avoid trusting the projection of XYZ point trajectories back to UV image
    # space, because even though those UV values will be close, it could occasionally correspond to
    # an entirely different object and thus completely wrong depth / XYZ coordinates.

    # Organize & return results.
    traject_retval = dict()
    traject_retval['query_time'] = query_time  # (Q).
    traject_retval['query_uv'] = query_uv  # (Q, 2).
    traject_retval['query_depth'] = query_depth  # (Q).
    traject_retval['query_xyz'] = query_xyz  # (Q, 3).
    traject_retval['uvdxyz'] = uvdxyz  # (Q, T, 6).
    traject_retval['uvdxyz_static'] = uvdxyz_static  # (Q, T, 6).
    traject_retval['inst_ids'] = inst_ids  # (Q).
    traject_retval['inst_coords'] = inst_coords  # (Q, 3).
    traject_retval['inst_bboxes_3d'] = inst_bboxes_3d  # (Q, T, 8, 3).

    return traject_retval


def unproject_points_2d_to_3d(uvd, camera_K, camera_R):
    '''
    Given a set of 2D points in image space along with depth and camera parameters, calculate the
        corresponding 3D canonical coordinates.
    :param uvd (*, 3) array: Image coordinates (u, v, d) in pixel space (values between 0 and 1)
        and depth in meters. Any leading dimensions are supported.
    :param camera_K (3, 3) array: Camera intrinsics matrix.
    :param camera_R (4, 4) array: Camera extrinstics matrix (i.e. pose = rotation + translation).
    :return xyz (*, 3) array: World coordinates (x, y, z).
    '''

    # Given uvd = 2D coordinates + depth in image space.
    uvd = uvd.astype(np.float64)
    camera_K = camera_K.astype(np.float64)
    camera_R = camera_R.astype(np.float64)
    depth = uvd[..., 2:3]

    # Get intermediate coordinates in camera space using inverse intrinsics.
    uvd_cam_plane = uvd.copy()
    uvd_cam_plane[..., 2] = 1.0
    uvd_cam_plane = np.matmul(uvd_cam_plane, np.linalg.inv(camera_K.T))

    # Because depth assumes distance from a camera point instead of camera plane, most depth values
    # are "too high". Before applying depth, correct for this by turning the coordinates from
    # residing on a Z=1 plane into a ball-ish shape by dividing them by this mismatch factor.
    uvd_cam_ball = uvd_cam_plane / np.sqrt(np.sum(np.square(uvd_cam_plane), axis=-1, keepdims=True))

    # Multiply by depth values to obtain real 3D coordinates in camera space.
    xyz_cam = uvd_cam_ball * depth

    # Get 3D coordinates in world space using extrinsics.
    xyz_cam = np.concatenate([xyz_cam, np.ones_like(xyz_cam[..., 0:1])], axis=-1)
    xyz_world = np.matmul(xyz_cam, camera_R.T)
    xyz = xyz_world[..., 0:3] / xyz_world[..., 3:4]

    xyz = xyz.astype(np.float32)

    return xyz


def project_points_3d_to_2d(xyz, camera_K, camera_R):
    '''
    Given a set of 3D points in world space along with camera parameters, calculate the
        corresponding 2D perspective coordinates.
    :param xyz (*, 3) array: World coordinates (x, y, z). Any leading dimensions are supported.
    :param camera_K (3, 3) array: Camera intrinsics matrix.
    :param camera_R (4, 4) array: Camera extrinstics matrix (i.e. pose = rotation + translation).
    :return uvd (*, 3) array: Image coordinates (u, v, d) in pixel space (values between 0 and 1)
        and depth in meters.
    '''

    # Given xyz = 3D coordinates in world space.
    xyz = xyz.astype(np.float64)
    camera_K = camera_K.astype(np.float64)
    camera_R = camera_R.astype(np.float64)

    # Get 3D coordinates in camera space using inverse extrinsics.
    xyz_world = np.concatenate([xyz, np.ones_like(xyz[..., 0:1])], axis=-1)
    xyz_cam = np.matmul(xyz_world, np.linalg.inv(camera_R.T))
    xyz_cam = xyz_cam[..., 0:3]

    # Divide by Z values to obtain intermediate coordinates in camera space.
    uvd_cam_plane = xyz_cam / xyz_cam[..., 2:3]

    # Get 2D coordinates in image space using intrinsics.
    uvd_plane = np.matmul(uvd_cam_plane, camera_K.T)

    # Calculate & append depth values as distance from the camera.
    depth = xyz_cam[..., 2:3] * np.sqrt(np.sum(np.square(uvd_cam_plane), axis=-1, keepdims=True))
    uvd = np.concatenate([uvd_plane[..., 0:2], depth], axis=-1)

    uvd = uvd.astype(np.float32)

    # TODX why is this
    uvd = -uvd

    return uvd


def uvd_from_depth(depth_map, wrong_uvd=False):
    '''
    Convert a depth map to a UVD array by simply prepending image space coordinates.
    :param depth_map (H, W, 1?) array.
    :return uvd (H, W, 3) array.
    '''
    while len(depth_map.shape) >= 3:
        assert depth_map.shape[-1] == 1
        depth_map = depth_map[..., 0]
    (H, W) = depth_map.shape[:2]

    uvd = np.zeros((H, W, 3), dtype=depth_map.dtype)
    uvd[..., 2] = depth_map

    if wrong_uvd:
        # For legacy purposes, i.e. debugging previously trained models.
        uvd[..., 0] = np.arange(H, dtype=depth_map.dtype)[:, None] / H
        uvd[..., 1] = np.arange(W, dtype=depth_map.dtype)[None, :] / W

    else:
        # NOTE: Getting the order right here is VERY important, otherwise you will be "transposing" the
        # image space when points get mapped to 3D, and then the XYZ space will be all weird and wrong.
        # This has caused me a lot of headaches. Even though the shape (H, W) is (vertical, horizontal),
        # the actual meaning of (u, v) is (horizontal, vertical) respectively! So we have to carefully
        # assign coordinates in this order, relying on numpy to broadcast along the other dimension.
        uvd[..., 0] = np.arange(W, dtype=depth_map.dtype)[None, :] / W
        uvd[..., 1] = np.arange(H, dtype=depth_map.dtype)[:, None] / H

    return uvd


def get_camera_matrices_numpy(metadata):
    '''
    :param metadata (dict) with keys camera, scene.
    :return (camera_K, camera_R).
        camera_K (T, 3, 3) array: Camera intrinsics matrix over time.
        camera_R (T, 4, 4) array: Camera extrinstics matrix over time.
    '''
    # TODX: Look at other Kubric methods: cameras intrinsics and objects matrix_world.
    focal_length = metadata['camera']['focal_length']
    positions = metadata['camera']['positions']
    quaternions = metadata['camera']['quaternions']  # (w, x, y, z) in Kubric / pyquat.
    sensor_width = metadata['camera']['sensor_width']
    (width, height) = metadata['scene']['resolution']
    sensor_height = sensor_width * height / width
    positions = np.array(positions)
    quaternions = np.array(quaternions)
    (camera_K, camera_R) = get_camera_matrices_from_params(
        focal_length,
        positions,
        quaternions,
        sensor_width,
        sensor_height,
    )
    return (camera_K, camera_R)


def get_camera_matrices_from_params(
        focal_length,
        positions,
        quaternions,
        sensor_width,
        sensor_height):
    '''
    :param focal_length (float).
    :param positions (T, 3) array.
    :param quaternions (T, 4) array.
    :param sensor_width, sensor_height (float).
    :return (camera_K, camera_R).
        camera_K (T, 3, 3) array: Camera intrinsics matrix over time.
        camera_R (T, 4, 4) array: Camera extrinstics matrix over time.
    '''
    intrinsics = []
    matrix_world = []
    for frame_idx in range(quaternions.shape[0]):
        f_x = focal_length / sensor_width
        f_y = focal_length / sensor_height
        p_x = 0.5
        p_y = 0.5
        intrinsics.append(np.array([[f_x, 0.0, -p_x],
                                    [0.0, -f_y, -p_y],
                                    [0.0, 0.0, -1.0]]))

        position = positions[frame_idx]
        quat = quaternions[frame_idx]
        # NOTE: The callee assumes x, y, z, w, but Kubric is w, x, y, z.
        rotation_matrix = rotation_matrix_from_quaternion_numpy(
            np.concatenate([quat[1:], quat[0:1]], axis=0))
        transformation = np.concatenate(
            [rotation_matrix, position[:, None]],
            axis=1,
        )
        transformation = np.concatenate(
            [transformation,
             np.array([0.0, 0.0, 0.0, 1.0])[None, :]],
            axis=0,
        )
        matrix_world.append(transformation)

    return (np.stack(intrinsics, axis=0).astype(np.float64),
            np.stack(matrix_world, axis=0).astype(np.float64))


def rotation_matrix_from_quaternion_numpy(quaternion):
    (x, y, z, w) = (quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3])

    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    matrix = np.stack([1.0 - (tyy + tzz), txy - twz, txz + twy,
                       txy + twz, 1.0 - (txx + tzz), tyz - twx,
                       txz - twy, tyz + twx, 1.0 - (txx + tyy)],
                      axis=-1)  # pyformat: disable
    matrix = matrix.reshape(*matrix.shape[:-1], 3, 3)

    return matrix


def calculate_uv_xyz_sensitivity(xyz, camera_K, camera_R):
    '''
    Calculates the approximate rate of change in 2D image space coordinates as a function of unit
        displacements along individual axes in 3D world space.
    :param xyz (Q, 3) array.
    :param camera_K (3, 3) array.
    :param camera_R (4, 4) array.
    :return uv_sensitivity (Q, 3, 2) array: For each given point, partial derivatives of (u, v) with
        respect to (x, y, z).
    '''
    uv_x1 = project_points_3d_to_2d(xyz + [-0.5, 0.0, 0.0], camera_K, camera_R)[..., 0:2]
    uv_x2 = project_points_3d_to_2d(xyz + [0.5, 0.0, 0.0], camera_K, camera_R)[..., 0:2]
    uv_dx = uv_x2 - uv_x1  # (Q, 2).

    uv_y1 = project_points_3d_to_2d(xyz + [0.0, -0.5, 0.0], camera_K, camera_R)[..., 0:2]
    uv_y2 = project_points_3d_to_2d(xyz + [0.0, 0.5, 0.0], camera_K, camera_R)[..., 0:2]
    uv_dy = uv_y2 - uv_y1  # (Q, 2).

    uv_z1 = project_points_3d_to_2d(xyz + [0.0, 0.0, -0.5], camera_K, camera_R)[..., 0:2]
    uv_z2 = project_points_3d_to_2d(xyz + [0.0, 0.0, 0.5], camera_K, camera_R)[..., 0:2]
    uv_dz = uv_z2 - uv_z1  # (Q, 2).

    uv_sensitivity = np.stack([uv_dx, uv_dy, uv_dz], axis=1)  # (Q, 3, 2).

    return uv_sensitivity


def project_points_3d_to_2d_multi(xyz, camera_K, camera_R, has_batch=True, has_time=True):
    '''
    Given a set of 3D points in world space along with camera parameters, calculate the
        corresponding 2D perspective coordinates.
    :param xyz (B?, *, T?, 3) array: World coordinates (x, y, z) over time. The batch dimension must
        come first, and the frame dimension must come second last.
    :param camera_K (B?, T?, 3, 3) array: Camera intrinsics matrix over time.
    :param camera_R (B?, T?, 4, 4) array: Camera extrinstics matrix (i.e. pose = rotation + translation)
        over time.
    :return uvd (B?, *, T?, 3) array: Image coordinates (u, v, d) in pixel space (values between 0 and 1)
        and depth in meters over time.
    '''
    is_tensor = torch.is_tensor(xyz)
    if is_tensor:
        xyz_numpy = xyz.detach().cpu().numpy()
        camera_K_numpy = camera_K.detach().cpu().numpy()
        camera_R_numpy = camera_R.detach().cpu().numpy()
    else:
        xyz_numpy = xyz
        camera_K_numpy = camera_K
        camera_R_numpy = camera_R

    if has_batch:
        B = xyz.shape[0]
        uvd_numpy = []
        for b in range(B):
            uvd_numpy.append(project_points_3d_to_2d_multi(
                xyz_numpy[b], camera_K_numpy[b], camera_R_numpy[b],
                has_batch=False, has_time=has_time))
        uvd_numpy = np.stack(uvd_numpy, axis=0)

    elif has_time:
        T = xyz.shape[-2]
        uvd_numpy = []
        for t in range(T):
            uvd_numpy.append(project_points_3d_to_2d_multi(
                xyz_numpy[..., t, :], camera_K_numpy[t], camera_R_numpy[t],
                has_batch=False, has_time=False))
        uvd_numpy = np.stack(uvd_numpy, axis=-2)

    else:
        uvd_numpy = project_points_3d_to_2d(xyz_numpy, camera_K_numpy, camera_R_numpy)

    if is_tensor:
        uvd = torch.tensor(uvd_numpy, device=xyz.device, dtype=xyz.dtype)
    else:
        uvd = uvd_numpy
    return uvd


def unproject_points_2d_to_3d_multi(uvd, camera_K, camera_R, has_batch=True, has_time=True):
    '''
    Given a set of 2D points in image space along with depth and camera parameters, calculate the
        corresponding 3D canonical coordinates.
    :param uvd (B?, *, T?, 3) array: Image coordinates (u, v, d) in pixel space (values between 0
        and 1) and depth in meters over time. The batch dimension must come first, and the frame
        dimension must come second last.
    :param camera_K (B?, T?, 3, 3) array: Camera intrinsics matrix over time.
    :param camera_R (B?, T?, 4, 4) array: Camera extrinstics matrix (i.e. pose = rotation + translation)
        over time.
    :return xyz (B?, *, T? 3) array: World coordinates (x, y, z) over time.
    '''
    is_tensor = torch.is_tensor(uvd)
    if is_tensor:
        uvd_numpy = uvd.detach().cpu().numpy()
        camera_K_numpy = camera_K.detach().cpu().numpy()
        camera_R_numpy = camera_R.detach().cpu().numpy()
    else:
        uvd_numpy = uvd
        camera_K_numpy = camera_K
        camera_R_numpy = camera_R

    if has_batch:
        B = uvd.shape[0]
        xyz_numpy = []
        for b in range(B):
            xyz_numpy.append(unproject_points_2d_to_3d_multi(
                uvd_numpy[b], camera_K_numpy[b], camera_R_numpy[b],
                has_batch=False, has_time=has_time))
        xyz_numpy = np.stack(xyz_numpy, axis=0)

    elif has_time:
        T = uvd.shape[-2]
        xyz_numpy = []
        for t in range(T):
            xyz_numpy.append(unproject_points_2d_to_3d_multi(
                uvd_numpy[..., t, :], camera_K_numpy[t], camera_R_numpy[t],
                has_batch=False, has_time=False))
        xyz_numpy = np.stack(xyz_numpy, axis=-2)

    else:
        xyz_numpy = unproject_points_2d_to_3d(uvd_numpy, camera_K_numpy, camera_R_numpy)

    if is_tensor:
        xyz = torch.tensor(xyz_numpy, device=uvd.device, dtype=uvd.dtype)
    else:
        xyz = xyz_numpy
    return xyz


# def create_occlusion_dag(div_segm, div_depth):
#     '''
#     :param div_segm (H, W, 1, K) array of uint8: Xray instance segmentation for a single frame.
#     :param div_depth (H, W, 1, K) array of float32: Xray depth map for a single frame.
#     '''
#     # TODX
#     pass


def box_to_tf_matrix(box, rows):
    '''
    :param box (8, 3) array: All corners in XYZ space of 3D cube surrounding object.
    :param (tf_matrix, rows).
        tf_matrix (4, 4) array: Coordinate transformation matrix from object space to world space.
        rows (3) array: Indices of rows in box that form an edge with the first row (= origin).
    '''
    # We make minimal assumptions about the ordering of the 3D points, except that the first two
    # rows must make up an edge of the box. Then, we look for the next two orthogonal vectors.
    origin = box[0]

    if rows is None:
        axis1 = box[1] - origin
        axis2 = None
        axis3 = None
        row1 = 1
        row2 = None
        row3 = None

        for i in range(2, 8):
            cand_axis = box[i] - origin
            if axis2 is None:
                if np.abs(np.dot(axis1, cand_axis)) < 1e-7:
                    axis2 = cand_axis
                    row2 = i
            elif axis3 is None:
                if np.abs(np.dot(axis1, cand_axis)) < 1e-7 and np.abs(np.dot(axis2, cand_axis)) < 1e-7:
                    axis3 = cand_axis
                    row3 = i

        assert axis2 is not None and axis3 is not None, \
            'Could not find orthogonal axes for object_box'
        rows = np.array([row1, row2, row3])

    else:
        axis1 = box[rows[0]] - origin
        axis2 = box[rows[1]] - origin
        axis3 = box[rows[2]] - origin

    object_to_world = np.stack([axis1, axis2, axis3, origin], axis=1)
    object_to_world = np.concatenate([object_to_world, [[0.0, 0.0, 0.0, 1.0]]], axis=0)
    # Sanity check while debugging: origin + axis1 must be close to object_to_world @ [1, 0, 0, 1].
    # NOTE: object_to_world is generally not orthonormal, because the axis lengths follow the size
    # of the container box, not unit vectors.

    return (object_to_world, rows)


# def get_containment_fraction_lower_bound(inside_box, outside_box):
def get_containment_fraction_approx(inside_box, outside_box):
    '''
    Calculates a sampled approximation of how much volume of a non-aligned 3D bounding box of a
        candidate object intersects (i.e. is inside of) that of a reference object.
    :param inside_box (8, 3) array of float: All corners in XYZ space of candidate containee cube.
    :param outside_box (8, 3) array of float: All corners in XYZ space of reference container cube.
    :return cflb (float).
    '''
    # NEW: Work with sampling. This is kind of brute-force, but at least it is simple and correct.
    # https://stackoverflow.com/questions/1827489/numpy-meshgrid-in-3d
    (x, y, z) = np.meshgrid(np.linspace(0, 1, 6), np.linspace(0, 1, 6), np.linspace(0, 1, 6),
                            indexing='ij')
    xyz = np.stack([x, y, z], axis=-1).reshape((-1, 3))  # (216, 3).
    xyz_homo = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)  # (216, 4).

    # Study the inside box in the coordinate system of the outside box.
    (outside_to_world, rows) = box_to_tf_matrix(outside_box, None)
    (inside_to_world, rows) = box_to_tf_matrix(inside_box, None)
    world_to_outside = np.linalg.inv(outside_to_world)
    inside_to_outside = np.matmul(world_to_outside, inside_to_world)
    # # NOTE: Unlike outside_to_world, world_to_outside (and inside_to_outside) are generally not
    # even orthogonal, because outside_to_world is not orthonormal!

    xyz_warped = np.matmul(inside_to_outside, xyz_homo.T).T
    assert np.all(np.abs(xyz_warped[..., -1] - 1.0) < 1e-5), \
        'Homogeneous coordinate is not 1'
    xyz_warped = xyz_warped[..., 0:3]
    points_contained = np.logical_and(np.all(xyz_warped >= 0.0, axis=1),
                                      np.all(xyz_warped <= 1.0, axis=1))
    cf_approx = np.mean(points_contained.astype(np.float32))

    return cf_approx

    # Sanity check:
    # NOTE: I verified that warp_matrix and inside_to_outside are exactly the same.
    # inside_box_homo = np.concatenate([inside_box, np.ones((8, 1))], axis=1)
    # warped_inside_box = np.matmul(world_to_outside, inside_box_homo.T).T
    # assert np.all(np.abs(warped_inside_box[..., -1] - 1.0) < 1e-5), \
    #     'Homogeneous coordinate is not 1'
    # warped_inside_box = warped_inside_box[..., 0:3]
    # We can now use warped_inside_box as if the outside box is [0, 1]^3.

    # (warp_matrix, _) = box_to_tf_matrix(warped_inside_box, rows)
    # # NOTE: Unlike outside_to_world, warp_matrix is generally not even orthogonal!

    # xyz_warped = np.matmul(warp_matrix, xyz_homo.T).T
    # assert np.all(np.abs(xyz_warped[..., -1] - 1.0) < 1e-5), \
    #     'Homogeneous coordinate is not 1'
    # xyz_warped = xyz_warped[..., 0:3]


    # OLD: This is too approximate and too strict because it treats extremal coordinates in each
    # dimension separately.
    # cflb = 1.0
    # warped_xyz_bounds = np.stack([warped_inside_box.min(
    #     axis=0), warped_inside_box.max(axis=0)], axis=1)
    # # (3, 2) with (min_x, max_x), (min_y, max_y), (min_z, max_z) in each row.
    # for i in range(3):
    #     total_range = warped_xyz_bounds[i, 1] - warped_xyz_bounds[i, 0]
    #     inside_range = max(min(warped_xyz_bounds[i, 1], 1.0) -
    #                        max(warped_xyz_bounds[i, 0], 0.0), 0.0)
    #     cur_factor = inside_range / max(total_range, 1e-7)
    #     assert cur_factor >= 0.0 and cur_factor <= 1.0, f'Invalid cur_factor: {cur_factor}'
    #     # print(total_range, inside_range, cur_factor)  # DEBUG
    #     cflb *= cur_factor

    # return cflb


def create_boxes_3d_video(rgb_shape, frame_inds, metadata, camera_K, camera_R):
    '''
    :param metadata (dict).
    :param camera_K (T, 3, 3) array: Camera intrinsics matrix.
    :param camera_R (T, 4, 4) array: Camera extrinstics matrix (i.e. pose = rotation + translation).
    '''
    (T, H, W, _) = rgb_shape
    boxes_3d_vis = np.zeros(rgb_shape, dtype=np.uint8)
    K = metadata['scene']['num_instances']
    cube_edges = [(0, 1), (0, 2), (1, 3), (2, 3),
                  (0, 4), (1, 5), (2, 6), (3, 7),
                  (4, 5), (4, 6), (5, 7), (6, 7)]

    for f, t in enumerate(frame_inds):
        # NOTE: This part is copied from get_thing_occl_cont_dag:
        cam_3d_pos = np.array(metadata['camera']['positions'][t])[None, :]
        # (1, 3).
        obj_3d_pos = np.array([metadata['instances'][k]['positions'][t] for k in range(K)])
        # (K, 3).
        distances = np.linalg.norm(cam_3d_pos - obj_3d_pos, ord=2, axis=-1)
        # (K).
        obj_order = np.argsort(distances)[::-1]  # Large to small = far to close = back to front.
        # (K).

        for inst_id in obj_order:
            cur_rgb = matplotlib.colors.hsv_to_rgb([inst_id / K, 1.0, 1.0])
            cur_rgb = np.array(cur_rgb * 255, dtype=np.uint8)
            cur_rgb = tuple(map(int, cur_rgb))

            box_xyz = np.array(metadata['instances'][inst_id]['bboxes_3d'][t])  # (8, 3).
            box_uvd = project_points_3d_to_2d(box_xyz, camera_K[t], camera_R[t])

            for row1, row2 in cube_edges:
                if box_uvd[row1, 2] > 0.0 and box_uvd[row2, 2] > 0.0:
                    u1 = int(round(box_uvd[row1, 0] * W))
                    v1 = int(round(box_uvd[row1, 1] * H))
                    u2 = int(round(box_uvd[row2, 0] * W))
                    v2 = int(round(box_uvd[row2, 1] * H))
                    # if (u1 >= 0 and v1 >= 0 and u1 < W and v1 < H) or \
                    #         (u2 >= 0 and v2 >= 0 and u2 < W and v2 < H):
                    boxes_3d_vis[f] = cv2.line(boxes_3d_vis[f], (u1, v1), (u2, v2), cur_rgb, 1)

    boxes_3d_vis = (boxes_3d_vis / 255.0).astype(np.float32)
    return boxes_3d_vis


if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    (H, W) = (120, 160)
    depth_map = np.random.uniform(2.0, 10.0, (H, W)).astype(np.float64)
    # camera_K = np.array([[W / 2, 0, -W / 2], [0, H / 2, -H / 2], [0, 0, -1]])
    camera_K = np.array([[40.0 / 32.0, 0.0, -0.5],
                         [0.0, -40.0 / 24.0, -0.5],
                         [0.0, 0.0, -1.0]])
    camera_R = np.array([[np.sqrt(0.5), -np.sqrt(0.5), 0.0, 1.0],
                         [np.sqrt(0.5), np.sqrt(0.5), 0.0, 2.0],
                         [0.0, 0.0, 1.0, 3.0],
                         [0.0, 0.0, 0.0, 1.0]])

    print('depth_map:', stmmm(depth_map))
    print('camera_K:', camera_K)
    print('camera_R:', camera_R)

    uvd = uvd_from_depth(depth_map)
    xyz = unproject_points_2d_to_3d(uvd, camera_K, camera_R)
    uvd_reconstructed = project_points_3d_to_2d(xyz, camera_K, camera_R)

    print('uvd:', stmmm(uvd))
    print('xyz:', stmmm(xyz))
    print('uvd_reconstructed:', stmmm(uvd_reconstructed))
    print('error:', stmmm(uvd - uvd_reconstructed))

    print('x')

    pass
