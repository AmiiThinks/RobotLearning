import rospy
import yaml
import math
import threading
import time
import message_filters as mf
import std_msgs.msg as std_msg
import sensor_msgs.msg as sens_msg
import kobuki_msgs.msg as kob_msg
import numpy as np
from cv_bridge.core import CvBridge
import sensor_msgs.point_cloud2 as pc2

class SensorParser:

    def __init__(self, sensor_topics=None):
        self.interval = 0.1 # Seconds between updates

        # maps each topic to the format of message in the topic
        topic_format = {
        # "/camera/depth/image":sens_msg.Image,
        # "/camera/depth/points":sens_msg.PointCloud2,
        # "/camera/ir/image":sens_msg.Image,
        # "/camera/rgb/image_raw":sens_msg.Image,
        # "/camera/rgb/image_rect_color":sens_msg.Image,
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
        }

        # use all sensors as default
        sensor_topics = topic_format.keys() if sensor_topics == None else sensor_topics
        
        # build list of sensors 
        self.sensors = [(topic, topic_format[topic]) for topic in sensor_topics]
        self.pubSensor = rospy.Publisher('sensor_parser/state_update', 
                                         # std_msg.Byte,
                                         std_msg.String,
                                         queue_size=1)

        # build callback function
        def publishSensorPacket(*sensors):
            # parsers = [format_parser[sens[1]] for sens in self.sensors]
            # data = {sens:pars(sensors)}

            # msg = std_msg.Byte()
            msg = std_msg.String()
            msg.data = "hello {}".format(time.time())
            
            self.pubSensor.publish(msg)

        # attach callback function to object
        self.publishSensorPacket = publishSensorPacket

    def start(self):
        rospy.init_node('sensor_parser', anonymous=True)

        subs = [mf.Subscriber(topic, frmt) for topic, frmt in self.sensors]

        # Approximate time synchronizer will take care of merging input from topics.
        time_sync = mf.ApproximateTimeSynchronizer(subs, queue_size=1, slop=0.1)
        time_sync.registerCallback(self.publishSensorPacket)

        # keep python awake until the ROS node dies
        rospy.spin()


    def image_parse(self, img, enc="passthrough"):
        # convert ros image to numpy array
        br = CvBridge()
        return np.asarray(br.imgmsg_to_cv2(image_message, desired_encoding=enc)) 

    def pc2_parse(self, cloud):
        # pc2.read_points returns a generator
        return list(pc2.read_points(cloud, 
                                    skip_nans=True, 
                                    field_names=("x", "y", "z")))

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
        "button": int(math.log(data.buttons, 2)), # button number
        "current_left_mA": data.current[0] * 10,
        "current_right_mA": data.current[1] * 10,
        "overcurrent_left": True if data.over_current % 2 else False,
        "overcurrent_right": True if data.over_current > 2 else False,
        "battery_voltage": data.battery * 0.1,
        "dist_left": data.bottom[2],   # cliff PSD sensor (0 - 4095, distance
        "dist_right": data.bottom[0],  # measure is non-linear)
        "dist_center": data.bottom[1],
        }

    def dock_ir_parse(self, dock_ir):
        return {
        "near_left": bool(dock_ir.NEAR_LEFT),
        "near_center": bool(dock_ir.NEAR_CENTER),
        "near_right": bool(dock_ir.NEAR_RIGHT),
        "far_left": bool(dock_ir.FAR_LEFT),
        "far_center": bool(dock_ir.FAR_CENTER),
        "far_right": bool(dock_ir.FAR_RIGHT),
        }

    def imu_parse(self, data):
        covar = [data.orientation_covariance,
                 data.angular_velocity_covariance,
                 data.linear_acceleration_covariance]
        covar = [np.asarray(cov).reshape(3,3) for cov in covar]

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
        manager = SensorParser()
        manager.start()
    except rospy.ROSInterruptException:
        pass



