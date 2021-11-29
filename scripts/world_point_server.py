#!/usr/bin/env python3
from os import path

import json
import numpy as np
import re
import rospy
from geometry_msgs.msg import Point

from ark_ros_cv_task.srv import WorldPoint, WorldPointResponse


def read_file(file_id):
    """
    Read required Pose data file and return a dictionary
    """
    # Get current directory path and relative path to img file
    current_path = path.split(path.abspath(__file__))[0]
    relative_path = f'../data/Pose/{file_id}_2.txt'

    # Get absolute file path, read in img, return
    absolute_path = path.join(current_path, relative_path)
    with open(absolute_path) as f:
        data = f.read()

    # Remove unnecessary chars (all chars in < >) and make a valid JSON string
    stripped = re.sub("[<].*?[>]", "", data).replace('\'', '\"')

    # Deserialize new string and return
    return json.loads(stripped)


def get_rotation_matrix(orientation_dict):
    """
    Convert from ROS orientation quaternion to rotation matrix
    """
    # Extract the values from the quaternion dict
    w, x, y, z = (orientation_dict[key + '_val'] for key in 'wxyz')

    # First row
    r00 = 2 * (w * w + x * x) - 1
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)

    # Second row
    r10 = 2 * (x * y + w * z)
    r11 = 2 * (w * w + y * y) - 1
    r12 = 2 * (y * z - w * x)

    # Third row
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 2 * (w * w + z * z) - 1

    # Return 3x3 matrix
    return np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ])


def get_translation_vector(position_dict, r_matrix):
    """
    Convert from camera position (in world coordinate system) to position of
    world origin in camera coordinate system.
    """
    # Get position of camera in world coordinate system
    x, y, z = (position_dict[key + '_val'] for key in 'xyz')
    camera_position = np.array([[x, y, z]]).T

    # Reverse position vector and rotate it to get position of world origin
    # in camera coordinate system
    return r_matrix @ -camera_position


def get_extrinsics_matrix(file_id):
    data_dict = read_file(file_id)

    # Rotation matrix from world coordinate to camera coordinate system
    r_matrix = get_rotation_matrix(data_dict['orientation'])

    # Position of world origin in camera coordinate system
    t_vector_c = get_translation_vector(data_dict['position'], r_matrix)

    # Create the transformation matrix
    transformation = np.eye(4)
    transformation[:3, -1:] = t_vector_c  # Translation vector in last column
    transformation[:3, :3] = r_matrix  # Rotation matrix in top-left 3x3

    return transformation


def get_intrinsics_matrix():
    f = 128
    return np.array([
        [f, 0, 0],
        [0, f, 0],
        [0, 0, 1],
    ])


def world_point_handler(req):
    ext = get_extrinsics_matrix(req.file_id)
    cam = get_intrinsics_matrix()
    cx, cy = 256 / 2, 144 / 2

    ext_inv = np.linalg.inv(ext)
    cam_inv = np.linalg.inv(cam)

    # cent = np.array([[0, 0, 1]]).T
    cent = np.array([[cx, cy, 0]]).T
    c = cam @ cent
    pc = cam_inv @ c
    pc_h = np.vstack((pc, np.ones((1, 1))))
    cam = ext_inv @ pc_h  # World position of origin

    response = WorldPointResponse()
    response.cam = Point(*cam[:3])
    response.points = []

    # cam_inv multiplication, vstack to get homogenous, and ext_inv
    for corner_object in req.corners:
        # Convert corner pixel coordinate to a homogenous vector
        corner = np.ones((3, 1))
        corner[:2, 0] = corner_object.coords
        corner[0, 0] = corner[0, 0] - cx
        corner[1, 0] = cy - corner[1, 0]

        # Get position of corner pixel in camera frame
        corner_cam = cam_inv @ corner
        print("Corner cam:", corner_cam)

        # Homogenize and convert to world frame
        corner_cam = np.vstack((corner_cam, np.ones((1, 1))))
        corner_world = ext_inv @ corner_cam
        print(corner_world)
        response.points.append(Point(*corner_world[:3]))
        print(corner_object.key)

    print(req)
    print(response)
    print(cam_inv)
    print(ext_inv)
    return response


def world_point_server():
    rospy.init_node('world_point_server')
    s = rospy.Service('world_point', WorldPoint, world_point_handler)
    print("World Point Server serving.")
    rospy.spin()


if __name__ == "__main__":
    world_point_server()
