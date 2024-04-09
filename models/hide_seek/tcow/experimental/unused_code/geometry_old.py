
# ====================================================================================
# TEMPORARY:
# Adapted from kubric point tracking pr_dataset.py and tensorflow_graphics rotation_matrix_3d:

import tensorflow as tf


def get_camera_matrices_tf(
        cam_focal_length,
        cam_positions,
        cam_quaternions,
        cam_sensor_width,
        cam_sensor_height):
    """Tf function that converts camera positions into projection matrices."""
    intrinsics = []
    matrix_world = []
    for frame_idx in range(cam_quaternions.shape[0]):
        focal_length = tf.cast(cam_focal_length, tf.float32)
        sensor_width = tf.cast(cam_sensor_width, tf.float32)
        sensor_height = tf.cast(cam_sensor_height, tf.float32)
        f_x = focal_length / sensor_width
        f_y = focal_length / sensor_height
        p_x = 0.5
        p_y = 0.5
        intrinsics.append(
            tf.stack([
                tf.stack([f_x, 0., -p_x]),
                tf.stack([0., -f_y, -p_y]),
                tf.stack([0., 0., -1.]),
            ]))

        position = cam_positions[frame_idx]
        quat = cam_quaternions[frame_idx]
        # NOTE: The callee assumes x, y, z, w.
        rotation_matrix = rotation_matrix_from_quaternion_tf(
            tf.concat([quat[1:], quat[0:1]], axis=0))
        transformation = tf.concat(
            [rotation_matrix, position[:, tf.newaxis]],
            axis=1,
        )
        transformation = tf.concat(
            [transformation,
             tf.constant([0.0, 0.0, 0.0, 1.0])[tf.newaxis, :]],
            axis=0,
        )
        matrix_world.append(transformation)

    return tf.cast(tf.stack(intrinsics),
                   tf.float32), tf.cast(tf.stack(matrix_world), tf.float32)


def rotation_matrix_from_quaternion_tf(quaternion):
    """Convert a quaternion to a rotation matrix."""
    quaternion = tf.convert_to_tensor(value=quaternion)

    x, y, z, w = tf.unstack(quaternion, axis=-1)
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
    matrix = tf.stack((1.0 - (tyy + tzz), txy - twz, txz + twy,
                       txy + twz, 1.0 - (txx + tzz), tyz - twx,
                       txz - twy, tyz + twx, 1.0 - (txx + tyy)),
                      axis=-1)  # pyformat: disable
    output_shape = tf.concat((tf.shape(input=quaternion)[:-1], (3, 3)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)

====================================================================================
