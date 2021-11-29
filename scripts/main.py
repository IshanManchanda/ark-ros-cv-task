#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import rospy
import tensorflow as tf
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


def get_center_point(lines):
    # List of tuples of points.
    a = np.zeros((3, 3))
    b = np.zeros((3, 1))
    for line in lines:
        ui = line[0] - line[1]
        ui /= np.linalg.norm(ui)
        ui = np.atleast_2d(ui)
        # print(ui, ui.shape)
        proj = np.eye(3) - (ui.T @ ui)
        pi = proj @ np.atleast_2d(line[1]).T

        a += proj
        b += pi
        # print(proj)
        # break
    ans = np.linalg.inv(a) @ b
    return ans


def fit_cube(detections):
    # Initialize tf.Variables
    # Define cost function
    # Evaluate cost function within Gradient Tape
    # Pass gradient to optimizer function
    # Repeat until loss small enough

    # centroid = tf.Variable([0.0, 0.0, 0.0], dtype=tf.float64)
    x = tf.Variable(1.0, dtype=tf.float64, trainable=True)
    y = tf.Variable(1.0, dtype=tf.float64, trainable=True)
    z = tf.Variable(1.0, dtype=tf.float64, trainable=True)
    theta = tf.Variable(1.0, dtype=tf.float64, trainable=True)
    s = tf.Variable(1.0, dtype=tf.float64, trainable=True)

    optimizer = tf.keras.optimizers.Adam()

    steps = 1
    print("\n\n")
    for step in range(steps):
        with tf.GradientTape() as tape:
            x_delta = s * np.sqrt(2) * tf.cos(theta + np.pi / 4)
            y_delta = s * np.sqrt(2) * tf.sin(theta + np.pi / 4)
            '''
            corners = {
                # '126': (x + x_delta, y + y_delta, z - s),
                # '023': (x - x_delta, y + y_delta, z + s),
                '126': (x + x_delta, y + y_delta, z - s),
                '236': (x - x_delta, y + y_delta, z - s),
                '346': (x - x_delta, y - y_delta, z - s),
                '146': (x + x_delta, y - y_delta, z - s),
                '012': (x + x_delta, y + y_delta, z + s),
                '023': (x - x_delta, y + y_delta, z + s),
                '034': (x - x_delta, y - y_delta, z + s),
                '014': (x + x_delta, y - y_delta, z + s),
            }

            loss = 0
            for key, lines in detections.items():
                for line in lines:
                    px, py, pz = corners[key]
                    loss += px * px + py * py + pz * pz
            # loss = x * y
            # x_delta = s * np.sqrt(2) * tf.cos(theta + np.pi / 4)
            # y_delta = s * np.sqrt(2) * tf.sin(theta + np.pi / 4)
            # loss = x_delta * y_delta
            # px = x + x_delta
            # py = y + y_delta
            # pz = z - s

            # loss = px * py * pz
            # loss = 0
            # loss += tf.reduce_sum(np.array([x + x_delta, y + y_delta, z - s]))
            '''
            """
            corners = {
                '126': np.array([x + x_delta, y + y_delta, z - s]),
                '236': np.array([x - x_delta, y + y_delta, z - s]),
                '346': np.array([x - x_delta, y - y_delta, z - s]),
                '146': np.array([x + x_delta, y - y_delta, z - s]),
                '012': np.array([x + x_delta, y + y_delta, z + s]),
                '023': np.array([x - x_delta, y + y_delta, z + s]),
                '034': np.array([x - x_delta, y - y_delta, z + s]),
                '014': np.array([x + x_delta, y - y_delta, z + s]),
            }
            """
            corners = {
                # '126': (x + x_delta, y + y_delta, z - s),
                # '023': (x - x_delta, y + y_delta, z + s),
                '126': (x + x_delta, y + y_delta, z - s),
                '236': (x - x_delta, y + y_delta, z - s),
                '346': (x - x_delta, y - y_delta, z - s),
                '146': (x + x_delta, y - y_delta, z - s),
                '012': (x + x_delta, y + y_delta, z + s),
                '023': (x - x_delta, y + y_delta, z + s),
                '034': (x - x_delta, y - y_delta, z + s),
                '014': (x + x_delta, y - y_delta, z + s),
            }
            loss = 0
            for key, lines in detections.items():
                # Get dist from corners[key] to each line
                for line in lines:
                    loss += corners[key][0] + corners[key][1] + corners[key][2]
                    # loss += tf.reduce_sum(corners[key])
                    # loss += tf.norm(corners[key] - line[0])
                    # d = line[1] - line[0]
                    # d = d / np.linalg.norm(d)
                    # v = corners[key] - line[0]
                    # t = tf.reduce_sum(tf.multiply(v, d))
                    # p = line[0] + t * d
                    # loss += tf.norm(corners[key] - p)
                    # ba = line[0] - corners[key]
                    # bc = line[0] - line[1]
                    # loss += tf.norm(tf.linalg.cross(ba, bc)) / tf.norm(bc)
        # print(loss)
        print(f'Step {step}:', loss)
        print()
        grads = tape.gradient(loss, [x, y, z, theta, s])
        print('Grads:', grads)
        print("\n\n")
        optimizer.apply_gradients(zip(grads, [x, y, z, theta, s]))
    # print(x, y, z, theta, s)
    print(x)
    print(y)
    print(z)
    print(theta)
    print(s)


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
    # return

    # plt.rcParams["figure.autolayout"] = True

    # print(detections)
    # for key, lines in detections.items():
    #     # if len(lines) <= 2: continue
    #     # if len(lines) <= 2 or key == '034' or key == '012' or key == '014': continue
    #     # print(key, len(lines))
    #     # continue
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection="3d")
    #
    #     for line in lines:
    #         x, y, z = np.array(line).T
    #         # print(x, y, z)
    #         ax.scatter(x, y, z, c='red', s=100)
    #         ax.plot(x, y, z, color='red')
    #     # break
    # plt.show()

    fit_cube(detections)
    return
    for key, lines in detections.items():
        print(key)
        print(np.array2string(np.array(lines), separator=','))

    return

    # return
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for key, lines in detections.items():
        print(key)
        # Can't do anything if only one line equation for corner
        if len(lines) < 2:
            continue

        # for line in lines:
        #     l1 = np.array(line)
        #     x, y, z = l1.T
        #     ax.scatter(x, y, z, c='green', s=100)
        #     ax.plot(x, y, z, color='green')
        cp = get_center_point(lines)
        x, y, z = cp
        ax.scatter(x, y, z, c='red', s=100)
        ax.plot(x, y, z, color='red')

        """
        acc = np.zeros(3)
        ctr = 0
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
                ans = np.linalg.solve(lhs, rhs)
                acc += ans
                ctr += 1
        # if x ** 2 + y ** 2 + z ** 2 > 200:
        #     continue
        print(acc/ctr)
        x, y, z = acc / ctr
        ax.scatter(x, y, z, c='red', s=100)
        ax.plot(x, y, z, color='red')
        """

    plt.show()
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
