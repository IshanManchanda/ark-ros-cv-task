#!/usr/bin/env python3

import numpy as np
import rospy
import matplotlib.pyplot as plt

from ark_ros_cv_task.srv import CornerInfo, WorldPoint


def corner_info_client(file_id):
    rospy.wait_for_service('corner_info')
    try:
        corner_info = rospy.ServiceProxy('corner_info', CornerInfo)
        resp = corner_info(file_id)
        return resp
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def world_point_client(file_id, corners):
    rospy.wait_for_service('corner_info')
    try:
        world_point = rospy.ServiceProxy('world_point', WorldPoint)
        resp = world_point(file_id, corners.corners)
        return resp
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def plot3d():
    pass


def main():
    detections = {}
    for i in range(0, 10):
        corners = corner_info_client(i)
        # print(corners)
        # TODO: Request world point server with this info, get world points
        points = world_point_client(i, corners)
        # print(points)
        cam = points.cam
        # print(cam)
        cam = np.array([cam.x, cam.y, cam.z])
        # print(cam)
        for corner, point in zip(corners.corners, points.points):
            pt = np.array([point.x, point.y, point.z])
            try:
                detections[corner.key] += [(cam, pt)]
            except KeyError:
                detections[corner.key] = [(cam, pt)]
            # detections[corner.key] = detections.get(corner.key, []) + [(cam, pt)]
            # print(pt, corner.key)
        # return
        # print(detections)
        # TODO: Use world point + camera pos to parameterize equation of ray
        # TODO: Accumulate all rays corresponding to a single corner
        #       across all images
        # print(info.corners)
        # print(type(info.corners[0].coords))
        # break
    # print(detections)
    return

    # plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    # print(detections)
    for key, lines in detections.items():
        # if len(lines) <= 2: continue
        # if len(lines) <= 2 or key == '034' or key == '012' or key == '014': continue
        # print(key, len(lines))
        # continue
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        for line in lines:
            x, y, z = np.array(line).T
            # print(x, y, z)
            ax.scatter(x, y, z, c='red', s=100)
            ax.plot(x, y, z, color='red')
        # break
    plt.show()


    return

    for key, lines in detections.items():
        print(key)
        # Can't do anything if only one line equation for corner
        if len(lines) < 2:
            continue

        # Iterate over all pairs of lines
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines[i + 1:]):
                # FIXME: Absolutely gon output
                #  Draw these lines in 3D and see what's happening
                # print(line1)
                # print(line2)

                unit_a = line1[0] - line1[1]
                unit_b = line2[0] - line2[1]
                unit_a /= np.linalg.norm(unit_a)
                unit_b /= np.linalg.norm(unit_b)
                unit_c = np.cross(unit_b, unit_a)
                unit_c /= np.linalg.norm(unit_c)

                rhs = line2[1] - line1[1]
                lhs = np.array([unit_a, -unit_b, unit_c]).T
                # print(lhs, rhs)
                print(np.linalg.solve(lhs, rhs))
        # break
    # TODO: For all corners with more than one line, find the point that
    #       minimizes (sum of squared?) distances to all the lines.
    # REVIEW: Is gradient descent viable?
    #         Or is there a convenient closed-form?
    # TODO: Find distances between all pairs of found corners.
    #       Take into account the number of common faces and determine
    #       what factor they are equal to side_length times, sqrt2, sqrt3, etc.
    # TODO: Use information from last step, compute best estimate for s_length
    # TODO: Also get a z-value-of-ground estimate from each viable corner,
    #       subtracting side_length as needed from the top ones.
    # top corners will all have red (1)
    # corners with 2 colors common will always be along an edge,
    # those with one common will be along a face diagonal,
    # and those with none will be along a body diagonal (opposite)


if __name__ == "__main__":
    main()
