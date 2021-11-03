#!/usr/bin/env python3

from os import path
import numpy as np
import rospy
# from cv2 import cv2 as cv2
import cv2

from ark_ros_cv_task.srv import CornerInfo, CornerInfoResponse
from ark_ros_cv_task.msg import Corner


def get_potential_corners(file_id):
    # Works, find all 3 relev corners in all 10 images but has extra ones as well
    # file_name = f'data/Segmentation/{idx}_1.png'
    file_name = f'../data/RGB/{file_id}_0.png'
    bin_path = path.split(path.abspath(__file__))[0]
    file_path = path.join(bin_path, file_name)
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Max detections, minimum quality (between 0 and 1), minimum distance
    corners = cv2.goodFeaturesToTrack(gray, 11, 0.06, 9)
    corners = np.int0(corners)
    corners = corners.reshape((corners.shape[0], 2))

    for i in corners:
        # x, y = i.ravel()
        x, y = i
        img[y, x] = (0, 255, 0)
        # cv2.circle(img, (x, y), 3, 255, -1)
    cv2.imwrite(f'out/{file_id}_out.png', img)

    return img, corners


def get_corner_colors(img, corner):
    # def get_corner_colors(img, corner, j):
    """
    Get a list of the colors present near the corner of the image
    """
    # We perform voting using the HSV values of the pixels in a window
    # around the corner. Colors with more votes than a threshold are selected,
    # upto a limit of 3 colors.
    ix, iy, _ = img.shape
    cy, cx = corner

    # Square window of size (2 * step + 1)
    step = 4
    x1, y1 = max(0, cx - step), max(0, cy - step)
    x2, y2 = 1 + min(ix, cx + step), 1 + min(iy, cy + step)
    roi = img[x1:x2, y1:y2, :]

    roi_hs = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, :2].reshape((-1, 2))

    # print(roi.shape)
    # cv2.imshow('temp', roi)
    # img[max(0, cx - 2):1 + min(ix, cx + 2), max(0, cy - 2):1 + min(iy, cy + 2), :] = (0, 255, 0)
    # cv2.imwrite(f'out/out{j}.png', roi)

    votes = np.zeros(5, dtype=np.int64)
    bounds = [
        [0, 8],  # red
        [12, 24],  # yellow
        [33, 50],  # green
        [119, 126],  # blue
        [128, 145],  # purple
        [170, 180]  # red, we will vote in % 5 to account for wrap-around
    ]
    sat_thresh = 120
    vote_thresh = 5

    # Perform voting
    for pixel in roi_hs:
        for i, bound in enumerate(bounds):
            if pixel[1] >= sat_thresh and bound[0] <= pixel[0] <= bound[1]:
                # Use modulo to provide wrap-around for red
                votes[i % 5] += 1

    # Get color indices which have more votes than threshold
    colors = np.argwhere(votes >= vote_thresh).ravel()

    # A corner can at-most have 3 colors
    while colors.size > 3:
        # Remove the least-voted color
        colors = np.delete(colors, np.argmin(votes[colors]))
    # print(colors)
    # print()
    return set(colors)


def get_corner_info(file_id):
    # Opposite face colors
    opposites = {0: 6, 1: 3, 2: 4, 3: 1, 4: 2, 6: 0}
    img, corners = get_potential_corners(file_id)

    # Make a set of votes colors
    all_colors = set()
    colors_list = []
    ids = []
    p_candidates = {}

    for k, corner in enumerate(corners):
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
            candidates[key] = corners[val]
            continue

        missing = opposites[missing.pop()]
        colors_list[val].add(missing)
        key = ''.join(str(x) for x in sorted(colors_list[val]))
        # print(missing, key)
        candidates[key] = corners[val]
    print(p_candidates)
    print(candidates)
    return candidates


def handle_corner_info(req):
    corner_info = get_corner_info(req.file_id)
    response = CornerInfoResponse()

    for key, val in corner_info.items():
        print(key, val)
        # TODO: Create ROS message by importing the msg file and
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
