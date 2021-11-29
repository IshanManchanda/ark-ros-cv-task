#!/usr/bin/env python3
import json
import re
from os import path

import numpy as np
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

    # REVIEW: Read in as an ROS Pose object instead of parsing with JSON?
    # Remove unnecessary chars (all chars in < >) and make a valid JSON string
    stripped = re.sub("[<].*?[>]", "", data).replace('\'', '\"')

    # Deserialize new string and return
    return json.loads(stripped)


def get_rotation_matrix(orientation_dict):
    """
    Convert from ROS orientation quaternion to rotation matrix
    """
    # REVIEW: Use ROS Transform lib to convert instead?
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
    """
    Get extrinsics/transformation matrix for a particular camera
    """
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
    """
    Get intrinsics matrix for perfect pinhole camera with 90deg horizontal FoV
    that takes an image 256 pixels wide.
    """
    f = 128
    return np.array([
        [f, 0, 0],
        [0, f, 0],
        [0, 0, 1],
    ])


def world_point_handler(req):
    """
    Request handler for the world point service
    """
    print("--- Received Request ---")
    print(req)
    print("--- End Request ---")
    # Get camera matrices and compute their inverses
    ext = get_extrinsics_matrix(req.file_id)
    cam = get_intrinsics_matrix()
    ext_inv = np.linalg.inv(ext)
    cam_inv = np.linalg.inv(cam)

    # Reverse project the optical center to get camera position in world frame
    # This value is also directly read-in from the Pose file,
    # this procedure is only used as a sanity check
    cx, cy = 256 / 2, 144 / 2
    center_cam = np.array([[cx, cy, 0, 1]]).T
    cam = ext_inv @ center_cam

    # Construct the response object and save the world position of the camera
    response = WorldPointResponse()
    response.cam = Point(*cam[:3])
    response.points = []

    # Iterate over all corners to get their projection points
    for corner_object in req.corners:
        # Convert corner pixel coordinate to a homogenous vector
        # Also deal with cx, cy offsets as well as y axis inversion
        corner = np.ones((3, 1))
        corner[0, 0] = corner_object.coords[0] - cx
        corner[1, 0] = cy - corner_object.coords[1]

        # Get position of corner pixel in camera frame
        corner_cam = cam_inv @ corner

        # Homogenize and convert to world frame
        corner_cam = np.vstack((corner_cam, np.ones((1, 1))))
        corner_world = ext_inv @ corner_cam

        # Add point to response object
        response.points.append(Point(*corner_world[:3]))

    print("--- Sending Response ---")
    print(response)
    print("--- End Response ---\n")
    return response


def world_point_server():
    """
    Start the corner info ROS Service
    """
    rospy.init_node('world_point_server')
    s = rospy.Service('world_point', WorldPoint, world_point_handler)
    print("World Point Server serving.")
    rospy.spin()


if __name__ == "__main__":
    world_point_server()
