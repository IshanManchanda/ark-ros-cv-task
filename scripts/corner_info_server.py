#!/usr/bin/env python3
from os import path

import cv2
import numpy as np
import rospy

from ark_ros_cv_task.msg import Corner
from ark_ros_cv_task.srv import CornerInfo, CornerInfoResponse


def read_image(file_id):
    """
    Read RGB image corresponding to file_id
    """
    # Get current directory path and relative path to img file
    current_path = path.split(path.abspath(__file__))[0]
    relative_path = f'../data/RGB/{file_id}_0.png'

    # Get absolute file path, read in img, return
    absolute_path = path.join(current_path, relative_path)
    img = cv2.imread(absolute_path)
    return img


def get_potential_corners(img):
    """
    Get list of potential corners in image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Params: Max number of detections, Minimum quality (between 0 and 1),
    #         Minimum distance between detections
    corners = cv2.goodFeaturesToTrack(gray, 11, 0.06, 9)

    # Convert to int type, reshape as n x 2, return corner list
    corners = np.int64(corners)
    corners = corners.reshape((-1, 2))
    return corners


def perform_voting(hs_list):
    """
    Perform voting for the colors present around a corner
    """
    # Hue value bounds for 5 face colors
    bounds = [
        [0, 8],  # red
        [12, 24],  # yellow
        [33, 50],  # green
        [119, 126],  # blue
        [128, 145],  # purple
        [170, 180]  # red, we will vote in % 5 to account for wrap-around
    ]
    # Threshold for saturation and minimum number of votes
    sat_thresh = 120

    # Vote counter for 5 colors
    votes = np.zeros(5, dtype=np.int64)

    # Perform voting
    for pixel in hs_list:
        # If pixel saturation is less than threshold then skip it
        if pixel[1] < sat_thresh:
            continue

        # Iterate over the colors and check Hue bounds
        for i, bound in enumerate(bounds):
            # Check if Hue value within bounds of this color
            if bound[0] <= pixel[0] <= bound[1]:
                # Use modulo to provide wrap-around for red
                votes[i % 5] += 1
                break

    return votes


def get_corner_colors(img, corner):
    # def get_corner_colors(img, corner, j):
    """
    Get a set of the colors present near the corner of the image
    """
    # Image size and corner coordinates
    ix, iy, _ = img.shape
    cy, cx = corner

    # Extract square window of size (2 * step + 1)
    step = 4
    x1, y1 = max(0, cx - step), max(0, cy - step)
    x2, y2 = 1 + min(ix, cx + step), 1 + min(iy, cy + step)
    roi = img[x1:x2, y1:y2, :]

    # Convert to HSV then extract H-S pairs and flatten
    hs_list = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, :2].reshape((-1, 2))

    # Minimum vote threshold
    vote_thresh = 5
    votes = perform_voting(hs_list)

    # Get color indices which have more votes than threshold
    colors = np.argwhere(votes >= vote_thresh).ravel()

    # A corner can at-most have 3 colors, loop until true
    while colors.size > 3:
        # Remove the least-voted color
        colors = np.delete(colors, np.argmin(votes[colors]))

    # Return as a set
    return set(colors)


def get_corner_info(file_id):
    # Opposite face colors
    # We use these to get the third face color for corners with only
    # two visible faces in a particular image
    opposites = {0: 6, 1: 3, 2: 4, 3: 1, 4: 2, 6: 0}

    # Read in image
    img = read_image(file_id)
    corner_coords = get_potential_corners(img)

    # TODO: Refactor and optimize this function
    # Make a set of votes colors
    all_colors = set()
    colors_list = []
    ids = []
    p_candidates = {}

    for k, corner in enumerate(corner_coords):
        colors = get_corner_colors(img, corner)

        # Add color to the common set for this image
        all_colors |= colors
        colors_list.append(colors)

        # Not a corner if detected colors 0 or 1, continue
        if len(colors) < 2:
            continue

        ids.append(k)
        key = ''.join(str(x) for x in sorted(colors))
        # p_candidates[key] = p_candidates.get(key, 0) + 1
        p_candidates[key] = k

    print(all_colors)
    candidates = {}
    for key, val in p_candidates.items():
        missing = all_colors - colors_list[val]
        if not missing:
            candidates[key] = corner_coords[val]
            continue

        missing = opposites[missing.pop()]
        colors_list[val].add(missing)
        key = ''.join(str(x) for x in sorted(colors_list[val]))
        # print(missing, key)
        candidates[key] = corner_coords[val]
    print(p_candidates)
    print(candidates)
    return candidates


def handle_corner_info(req):
    corner_info = get_corner_info(req.file_id)
    response = CornerInfoResponse()

    for key, val in corner_info.items():
        print(key, val)
        corner = Corner(key, val)
        response.corners += [corner]

    print(response)
    return response


def corner_info_server():
    rospy.init_node('corner_info_server')
    s = rospy.Service('corner_info', CornerInfo, handle_corner_info)
    print("Ready to add two ints.")
    rospy.spin()


if __name__ == "__main__":
    class Temp:
        pass
    req = Temp()
    req.file_id = 0
    handle_corner_info(req)
    # corner_info_server()
