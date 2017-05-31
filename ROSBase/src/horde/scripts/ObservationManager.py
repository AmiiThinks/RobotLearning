#!/usr/bin/env python

"""
Author: David Quail, May30, 2017.

Description:
The observation manager will
1. listen for various different sensor data coming from various different data sources (servos, cameras, etc).
2. Maintain the data structure of most recent observations across data sources sources.
3. Publish the most recent observations at a predictable time interval.

By massaging the sensory data and publishing it at frequent, predictable time intervals, a learning system can subscribe
to this message. Once subscribed, the learner can perform control and learning at this interval, rather than having
to poll for the information after actions.

TODO:
Publish a message of specific format that encapsulates all the data of interest. Not just a primitive int of angle
"""
from cv_bridge.core import CvBridge
import kobuki_msgs.msg as kob_msg
import math
import msgpack
import numpy as np
from Queue import Queue
import rospy
import signal
import sensor_msgs.msg as sens_msg
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg as std_msg
import sys
import threading
import time

from SensorParser import SensorParser
from Tools import merge_dicts

# for some reason we need this to be able to ctrl-c the program, but
# there is probably a more graceful way
signal.signal(signal.SIGINT, lambda x, y: sys.exit(1))

"""
sets up the subscribers and starts to broadcast the results in a thread every 0.1 seconds
"""

thread = False

class ObservationManager:

    def __init__(self, topics, publishing_frequency):
        # maps each topic to its data format
        topic_format = {
            "/camera/depth/image":sens_msg.Image,
            "/camera/depth/points":sens_msg.PointCloud2,
            "/camera/ir/image":sens_msg.Image,
            "/camera/rgb/image_raw":sens_msg.Image,
            "/camera/rgb/image_rect_color":sens_msg.Image,
            "/mobile_base/sensors/core":kob_msg.SensorState,
            "/mobile_base/sensors/dock_ir":kob_msg.DockInfraRed,
            "/mobile_base/sensors/imu_data":sens_msg.Imu,
            }

        # maps each format to its parser
        format_parser = {
            sens_msg.Image: self.image_parse,
            sens_msg.PointCloud2: self.pc2_parse,
            kob_msg.SensorState: self.sensor_state_parse,
            kob_msg.DockInfraRed: self.dock_ir_parse,
            sens_msg.Imu: self.imu_parse
            }

        self.publishing_frequency = publishing_frequency
        self.topics = topics
        self.formats = [topic_format[topic] for topic in topics]
        self.parsers = [format_parser[fmt] for fmt in self.formats]
        self.most_recent = {topic:dict() for topic in topics}

    def publish_observations(self):

        # Keep track of time for when to avoid sleeping
        self.publishing_frequency -= 0.0001
        tic = time.time()

        while True:
            # To avoid the drift of just calling time.sleep()
            while time.time() < tic:
                time.sleep(0.0001)

            # create the message
            msg = std_msg.Byte()
            msg.data = msgpack.packb(merge_dicts(*self.most_recent.values()))
            
            # publish the message
            self.publisher.publish(msg)

            # reset tic
            tic += self.publishing_frequency

    def start_thread(self, topic):
        sp = SensorParser(topic)
        sp.start(self.most_recent[topic])

    def start(self):
        rospy.init_node('observation_manager', anonymous=True)
        self.publisher = rospy.Publisher('observation_manager/combined', 
                                         std_msg.Byte,
                                         queue_size=1)

        # setup sensor parsers
        for topic in self.topics:
            t = threading.Thread(target=self.start_thread, args=[topic])
            t.setDaemon(True)
            t.start()

        # publish the most recent observation
        self.publish_observations()

        rospy.spin()

    def image_parse(self, img, enc="passthrough"):
        # convert ros image to numpy array
        br = CvBridge()
        image = np.asarray(br.imgmsg_to_cv2(img, desired_encoding=enc))
        return {self.topic: image.tolist()}

    def pc2_parse(self, dat):
        # pc2.read_points returns a generator of (x,y,z) tuples
        gen = pc2.read_points(dat, skip_nans=True, field_names=("x","y","z"))
        return {self.topic: list(gen)}

    def sensor_state_parse(self, data):
        return {
        "bump_right": True if data.bumper % 2 else False,
        "bump_center": True if data.bumper % 4 > 1 else False,
        "bump_left": True if data.bumper >= 4 else False,
        "wheeldrop_right": bool(data.WHEEL_DROP_RIGHT),
        "wheeldrop_left": bool(data.WHEEL_DROP_LEFT),
        "cliff_right": bool(data.CLIFF_RIGHT),
        "cliff_center": bool(data.CLIFF_CENTRE),
        "cliff_left": bool(data.CLIFF_LEFT),
        "ticks_left": data.left_encoder,   # number of wheel ticks since 
        "ticks_right": data.right_encoder, # kobuki turned on; max 65535
        # button number
        "button": int(math.log(data.buttons, 2)) if data.buttons else 0, 
        "current_left_mA": data.current[0] * 10,
        "current_right_mA": data.current[1] * 10,
        "overcurrent_left": True if data.over_current % 2 else False,
        "overcurrent_right": True if data.over_current > 2 else False,
        "battery_voltage": data.battery * 0.1,
        "bottom_dist_left": data.bottom[2],  # cliff PSD sensor (0 - 4095, 
        "bottom_dist_right": data.bottom[0], # distance measure is non-linear)
        "bottom_dist_center": data.bottom[1],
        }

    def dock_ir_parse(self, dock_ir):
        return {
        "ir_near_left": bool(dock_ir.NEAR_LEFT),
        "ir_near_center": bool(dock_ir.NEAR_CENTER),
        "ir_near_right": bool(dock_ir.NEAR_RIGHT),
        "ir_far_left": bool(dock_ir.FAR_LEFT),
        "ir_far_center": bool(dock_ir.FAR_CENTER),
        "ir_far_right": bool(dock_ir.FAR_RIGHT),
        }

    def imu_parse(self, data):
        covar = [data.orientation_covariance,
                 data.angular_velocity_covariance,
                 data.linear_acceleration_covariance]
        covar = [np.asarray(cov).reshape(3,3).tolist() for cov in covar]

        return {
        "orient_x": data.orientation.x,
        "orient_y": data.orientation.y,
        "orient_z": data.orientation.z,
        "orient_w": data.orientation.w,
        "orient_covar": covar[0],
        "ang_vel_x": data.angular_velocity.x,
        "ang_vel_y": data.angular_velocity.y,
        "ang_vel_z": data.angular_velocity.z,
        "ang_vel_covar": covar[1],
        "lin_accel_x": data.linear_acceleration.x,
        "lin_accel_y": data.linear_acceleration.y,
        "lin_accel_z": data.linear_acceleration.z,
        "lin_accel_covar": covar[2],
        }

if __name__ == '__main__':
    try:
        topics = [
            # "/camera/depth/image",
            # "/camera/depth/points",
            # "/camera/ir/image",
            # "/camera/rgb/image_raw",
            # "/camera/rgb/image_rect_color",
            "/mobile_base/sensors/core",
            "/mobile_base/sensors/dock_ir",
            "/mobile_base/sensors/imu_data",
            ]
        manager = ObservationManager(topics)
        manager.start()
    except rospy.ROSInterruptException:
        pass

