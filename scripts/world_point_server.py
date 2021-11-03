#!/usr/bin/env python3

import json
import numpy as np
import re
import rospy
from os import path
from ark_ros_cv_task.srv import WorldPoint, WorldPointResponse


def read_file(file_id):
    """
    Read required Pose data file and return a dictionary
    """
    # Get current directory path and relative path to img file
    current_path = path.split(path.abspath(__file__))[0]
    relative_path = f'../data/Pose/{file_id}_2.png'

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
    cx, cy = 256 / 2, 144 / 2
    return np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1],
    ])


def main():
    for i in range(10):
        print(f"\n--- File {i} ---")
        ext = get_extrinsics_matrix(i)
        cam = get_intrinsics_matrix()

        ext_inv = np.linalg.inv(ext)
        cam_inv = np.linalg.inv(cam)

        # cent = np.array([[0, 0, 1]]).T
        cent = np.array([[0, 0, 0]]).T
        c = cam @ cent
        pc = cam_inv @ c
        pc_h = np.vstack((pc, np.ones((1, 1))))
        print(ext_inv @ pc_h)

        # TODO: Need to do stuff from above 3 lines
        # cam_inv multiplication, vstack to get homogenous, and ext_inv
        pass


if __name__ == '__main__':
    main()


def handle_add_two_ints(req):
    print("Returning [%s + %s = %s]" % (req.a, req.b, (req.a + req.b)))
    return WorldPointResponse(req.a + req.b)


def add_two_ints_server():
    rospy.init_node('add_two_ints_server')
    s = rospy.Service('add_two_ints', AddTwoInts, handle_add_two_ints)
    print("Ready to add two ints.")
    rospy.spin()


if __name__ == "__main__":
    add_two_ints_server()
