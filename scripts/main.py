#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import rospy
import tensorflow as tf

from ark_ros_cv_task.srv import CornerInfo, WorldPoint


def corner_info_client(file_id):
    """
    Client for the corner info service
    """
    rospy.wait_for_service('corner_info')
    try:
        corner_info = rospy.ServiceProxy('corner_info', CornerInfo)
        resp = corner_info(file_id)
        return resp
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def world_point_client(file_id, corners):
    """
    Client for the world point
    """
    rospy.wait_for_service('world_point')
    try:
        world_point = rospy.ServiceProxy('world_point', WorldPoint)
        resp = world_point(file_id, corners.corners)
        return resp
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def get_parameterized_corners(x, y, z, theta, s, math_lib=tf):
    """
    Evaluate expressions for the corners of a cube parameterized using
    centroid coordinates, angle of rotation about vertical, semi side length.
    math_lib argument allows using Numpy functions instead of Tensorflow ones
    so that output isn't a tf.Variable.
    """
    # Offsets of corners from centroid in the x and y directions
    x_delta = s * np.sqrt(2) * math_lib.cos(theta + np.pi / 4)
    y_delta = s * np.sqrt(2) * math_lib.sin(theta + np.pi / 4)

    # Parameterized expressions of the corners
    # The keys correspond to the color-based corner descriptors
    corners = {
        # Lower four in counter-clockwise direction starting top-right
        '126': (x + x_delta, y + y_delta, z - s),
        '236': (x - x_delta, y + y_delta, z - s),
        '346': (x - x_delta, y - y_delta, z - s),
        '146': (x + x_delta, y - y_delta, z - s),
        # Upper four in counter-clockwise direction starting top-right
        '012': (x + x_delta, y + y_delta, z + s),
        '023': (x - x_delta, y + y_delta, z + s),
        '034': (x - x_delta, y - y_delta, z + s),
        '014': (x + x_delta, y - y_delta, z + s),
    }
    return corners


def compute_loss(x, y, z, theta, s, detections):
    """
    Compute loss as the sum of squared distances of all corners
    from all their projection lines.
    """
    # Evaluate parametric expressions for all corners
    corners = get_parameterized_corners(x, y, z, theta, s)

    # The loss is the sum of squared distances of each corner
    # from all its projection lines
    loss = 0
    # Iterate over all corners for which we have projection lines
    for key, lines in detections.items():
        # And then over each line
        for line in lines:
            # For some reason Tensorflow's auto differentiation
            # didn't work with numpy vector operations. Thus using scalar
            px, py, pz = corners[key]
            bx, by, bz = line[0]
            dx, dy, dz = line[1]

            # Vector from point on line to corner
            vx = px - bx
            vy = py - by
            vz = pz - bz

            # Compute dot product with unit vector of the line
            t = vx * dx + vy * dy + vz * dz

            # Get point of projection/foot of perpendicular on line
            px1 = bx + dx * t
            py1 = by + dy * t
            pz1 = bz + dz * t

            # Take squared distance
            dx2 = (px - px1) ** 2
            dy2 = (py - py1) ** 2
            dz2 = (pz - pz1) ** 2
            loss += dx2 + dy2 + dz2

    return loss


def fit_cube(detections):
    """
    Fit a parametric cube to the found projection lines
    """
    # Parameters of the cube: The parameterization leverages the information
    # that the cube is lying on a perfectly flat surface.
    # Thus we need only 5 parameters instead of 7.
    # The parameters are: the x, y, z coordinates of the centroid,
    # the angle of rotation about the vertical (z-axis), and the side_length/2.
    # Initial values are eye-balled figures from previous training.
    x = tf.Variable(10.0, dtype=tf.float64, trainable=True)
    y = tf.Variable(3.0, dtype=tf.float64, trainable=True)
    z = tf.Variable(-10.0, dtype=tf.float64, trainable=True)
    theta = tf.Variable(np.pi / 2, dtype=tf.float64, trainable=True)
    s = tf.Variable(5.0, dtype=tf.float64, trainable=True)

    # Convert the 3D lines from 2-point representation to
    # line + unit vector representation.
    processed = {}
    for key, lines in detections.items():
        new_lines = []
        for line in lines:
            # Simply subtract the two points and normalize to get unit vector
            line_unit_vec = line[1] - line[0]
            line_unit_vec /= np.linalg.norm(line_unit_vec)
            new_lines.append((line[0], line_unit_vec))
        processed[key] = new_lines

    # Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

    # Hyperparameters for training
    max_steps = 100
    prev_loss = np.inf
    early_stop_thresh = 0.2

    # Catch KeyboardInterrupt to terminate learning early if needed
    try:
        for step in range(max_steps):
            # Compute loss within a GradientTape
            with tf.GradientTape() as tape:
                loss = compute_loss(x, y, z, theta, s, processed)

            # Print loss and check for early stopping
            print(f'Step {step}:', loss.numpy())
            if prev_loss - loss < early_stop_thresh:
                print('Early stopping.')
                break
            prev_loss = loss

            # Get gradients using automatic differentiation
            # and apply them using the optimizer
            grads = tape.gradient(loss, [x, y, z, theta, s])
            optimizer.apply_gradients(zip(grads, [x, y, z, theta, s]))
    except KeyboardInterrupt:
        pass

    # Return numpy values instead of TF Variables
    return x.numpy(), y.numpy(), z.numpy(), theta.numpy(), s.numpy()


def main():
    detections = {}
    for i in range(0, 10):
        # Get coordinates and descriptions of cube corners in image
        corners = corner_info_client(i)

        # Convert pixel coordinates to world-frame coordinates
        points = world_point_client(i, corners)

        # Generate projection lines by taking camera center and world point
        # Each projection line is identified by the descriptor of the corner
        # it corresponds to
        cam = points.cam
        cam = np.array([cam.x, cam.y, cam.z])
        for corner, point in zip(corners.corners, points.points):
            # Line is a tuple of (camera_position, corner_position)
            # Store it with the appropriate key (descriptor)
            pt = np.array([point.x, point.y, point.z])
            try:
                detections[corner.key] += [(cam, pt)]
            except KeyError:
                detections[corner.key] = [(cam, pt)]

    # Fit a parametric cube to the detected projection lines using TF
    x, y, z, theta, s = fit_cube(detections)
    print('\n--- Final Output ---')
    print(f'Centroid coordinates: ({x}, {y}, {z})')
    print(f'Cube side length: {s * 2}')

    # Pass numpy as math library to prevent getting tf Variables as output
    corner_coords = get_parameterized_corners(x, y, z, theta, s, np)

    # Plot final corner coordinates
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for key, point in corner_coords.items():
        x, y, z = point
        ax.scatter(x, y, z, c='red', s=100)

    # Plot projection lines
    # for key, lines in detections.items():
    #     for line in lines:
    #         x, y, z = np.array(line).T
    #         ax.scatter(x, y, z, c='red', s=100)
    #         ax.plot(x, y, z, color='red')

    plt.show()


if __name__ == "__main__":
    main()
