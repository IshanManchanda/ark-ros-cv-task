# ark-ros-cv-task

ROS (Robot Operating System) + CV (Computer Vision) task for
Aerial Robotics Kharagpur Research Group.

## Project Structure

The project is a ROS package written in Python.
It comprises of 2 ROS services which are invoked by a main script.

The input data as provided is present in the data folder.
The srv folder contains the ROS service descriptions which specify
the request and reponse parameters.
These parameters include the Point msg from geometry_msgs and
a custom Corner msg which is defined in the msg directory.

The scripts folder contains 3 files - the 2 services and the main file.

### Dependencies
- Python ROS packages
- Numpy
- OpenCV-Python
- Tensorflow 2
- Optional: Matplotlib with a GUI backend (Used to render the cube corners 
and detection lines. Code is commented out by default in main.py.)

## Approach

The overarching idea is as given in the Problem Statement hint.

The corner_info service finds the pixel coordinates of the cube corners and
provides a description based on the colors of the faces of the corner.
Then the world_point service uses the provided Pose data to get projection lines
for each corner through the camera center in the world frame of reference.
Finally, a parametric cube is fit to these projection lines using Tensorflow.

### 1. Corner Info

A few different approaches were tried for the corner detection
including the Harris detector. However, these approaches found
several false positives (particularly due to the shadows) 
before all the required corners were found.
The final approach uses the Shi-Tomasi method as implemented in
OpenCV's goodFeaturesToTrack function. In each image, up to 4 corners
(1 with 3 faces visible and 3 with 2 faces) are finally used.
The method is able to detect all required corners with
a reasonable number of false positives.
However, the identified corner pixels are inexact and this stage introduces 
a lot of the noise that the third step tries to deal with.

Visualizations of these detected corners were generated and saved.
These will be uploaded to this repository soon, along with other
exploration/output files and jupyter notebooks.

The corners are then described using the colors of the neighboring faces. This
is done by taking a small window/RoI around each corner and performing voting.
The voting is done by thresholding the Hue and Sat values of pixels in the RoI.
The colors present in the image are also compiled to deal with the occlusion
of one face for 3 out of 4 corners in each image.
This method is able to get a perfect description of all corners of interest.

### 2. World Point

This step involves finding projection lines for each of the detected corners.
The corner coordinates are received along with the request. The corresponding
Pose data is read from the file and the camera matrices are constructed.
The intrinsics matrix can be derived for the perfect pinhole camera with a 
90 deg FoV which takes an image 256 pixels wide.
The focal length is found to be 128 pixels.

The inverses of these matrices are used to find the projection lines.
Each line is represented by 2 points on it - the camera origin and a point
on the projection line at an arbitrary depth (z-coordinate) of 1
from the origin in the camera frame.

### 3. Main (Cube Fitting)

The final step involves fitting the cube to the detected projection lines.
As mentioned above, the corner detection introduces a lot of noise in the lines
and thus the intersection points mostly do not exist.

To overcome this, the first approach tried was to find points
of closest approach between every combination of 2 projection lines that
correspond to the same corner and take the mean.
This method provided extremely poor results as these closest-approach points
lied in a huge range (x-coordinate for a single corner ranged from -200 to 500).
Outlier removal was then tried on these points.
Dropping points outside a certain distance from the origin as well as
dropping points more than 2 standard deviations from the median were tried.
These provided slightly better but still poor results.

Next, instead of considering each line individually, all projection lines
for a single corner were taken together.
The point which minimizes the sum of squared distances from each line was found
for each corner. These points were much better than the previous ones but
still didn't describe a cube well (as per the 3D plot).

The final approach was to create a parametric model for a cube in 3D space
lying on a horizontal surface. Such a cube is sufficiently described by 5
parameters - the x, y, z coordinates of its centroid, the angle of rotation
about the vertical (z-axis), and the side length. The parameterization used
has s as the semi side length for a cleaner formula representation.
The corners of this cube can be found in terms of these parameters and 
keys/color descriptors can be assigned in cyclic order.
These keys were manually assigned, although it is possible to automatically
assign them using the detected color data.
With this description, the distance of each corner can be found to all
the corresponding projection lines and the sum of the squares
of these distances is taken as a loss function.
Tensorflow is then used to automatically differentiate and optimize the values
of these parameters using the Adam optimizer.

## Result
Centroid coordinates: (9.63274322320409, 3.1255089563633005, -9.60482568775914)  
Cube side length: 10.001632076348924
