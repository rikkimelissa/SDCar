This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9). I worked on this project individually.

### Native Installation

I used a native installation with Ubuntu 14.04 and ROS Indigo.

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```

### Code structure

I found the walkthroughs of the nodes by Steven and Eric to be extremely helpful in getting the system up and running. I did not need to make modifications to most of the instruction they provided. The only structural change I made was to turn the tl_classifier class into a separately running ROS node. Calling the image_classifier on every cycle of tl_detection added too much latency to the system and had undesirable effects on the twist_controller. Instead, I run the tl_classifier node at 10hz and publish the latest classification results, which tl_detection subscribes to. This allows the slowness of the classification to remain outside of the control loops.

I've found in testing that this code works most of the time, but if too many processes are running and slowing down the communication between the simulator and ROS, the car reacts pretty poorly to the latency. In general, running tensorflow uses a lot of memory on my computer.

#### tl_detector

This node takes in data from /image_color, /current_pose, and /base_waypoints topics and publishes the locations to stop for red traffic lights to the /traffic_waypoint topic. 

The traffic light classification node uses a frozen neural net model from TensorFlow's Object Detection API, ssdlite_mobilenet_v2_coco_2918_05_09. This neural net is able to classify traffic lights (and many other objects) with a high degree of detection and do so pretty quickly. I tried out a handful of other trained neural nets from the Object Detection API with varying results of successful classification and speed, and picked this one as a good balance of the two. I found Daniel Stang's tutorial on medium to be a really helpful guide on getting started with this API.

I use the trained neural net graph to detect traffic lights in the images by only keeping detected objects that are classified as traffic lights. The neural net also returns a bounding box and a classification score for each detected object. I use the highest scoring detected traffic light and the bounding box area of the image to figure out what color the traffic light is. I transform the image into the HSV spectrum, then look at which third, from top to bottom, of the bounding box has the highest average Value (of HSV). This position determines the state of the traffic light.

#### waypoint_updater

The purpose of this node is to update the target velocity property of each waypoint based on traffic light and obstacle detection data.

#### twist_controller

This node publishes throttle, brake, and steering commands based on the /current_velocity and /twist_cmd topics to receive target linear and angular velocities.



