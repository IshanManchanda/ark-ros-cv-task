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
        print(info)
        break


if __name__ == "__main__":
    main()
