#!/usr/bin/env python3

import rospy

from ark_ros_cv_task.srv import CornerInfo


def corner_info_client(file_id):
    rospy.wait_for_service('corner_info')
    try:
        corner_info = rospy.ServiceProxy('corner_info', CornerInfo)
        resp = corner_info(file_id)
        return resp
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def main():
    for i in range(10):
        info = corner_info_client(i)
        # TODO: Request world point server with this info, get world points
        # TODO: Use world point + camera pos to parameterize equation of ray
        # TODO: Accumulate all rays corresponding to a single corner
        #       across all images
        print(info)
        break
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
