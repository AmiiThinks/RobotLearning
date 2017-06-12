from functools import wraps
import kobuki_msgs.msg as kob_msg
import rospy
import sensor_msgs.msg as sens_msg
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg as std_msg
from turtlesim.msg import Pose
import numpy as np

from time import time


topic_format = {
    "/camera/depth/image":sens_msg.Image,
    "/camera/depth/points":sens_msg.PointCloud2,
    "/camera/ir/image":sens_msg.Image,
    "/camera/rgb/image_raw":sens_msg.Image,
    "/camera/rgb/image_rect_color":sens_msg.Image,
    "/mobile_base/sensors/core":kob_msg.SensorState,
    "/mobile_base/sensors/dock_ir":kob_msg.DockInfraRed,
    "/mobile_base/sensors/imu_data":sens_msg.Imu,
    "/turtle1/pose":Pose,
    }

def equal_twists(t1, t2):
    return all([np.isclose(t1.linear.x, t2.linear.x),
                np.isclose(t1.linear.y, t2.linear.y),
                np.isclose(t1.linear.z, t2.linear.z),
                np.isclose(t1.angular.x, t2.angular.x),
                np.isclose(t1.angular.y, t2.angular.y),
                np.isclose(t1.angular.z, t2.angular.z)])

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


"""
Decorator that print how long a function takes to execute
"""
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # print 'func:%r args:[%r, %r] took: %2.4f sec' % \
        # (f.__name__, args, kw, te-ts)
        rospy.loginfo('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap


"""
Parsers for easy (but maybe heavy) state creation
"""
def image_parse(img, enc="passthrough"):
    # convert ros image to numpy array
    br = CvBridge()
    image = np.asarray(br.imgmsg_to_cv2(img, desired_encoding=enc)) 
    return {self.topic: image}

def pc2_parse(data):
    # PointCloud2 parser
    # pc2.read_points returns a generator of (x,y,z) tuples
    gen = pc2.read_points(data, skip_nans=True, field_names=("x","y","z"))
    return {self.topic: list(gen)}

def sensor_state_parse(data):
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

def dock_ir_parse(dock_ir):
    return {
    "ir_near_left": bool(dock_ir.NEAR_LEFT),
    "ir_near_center": bool(dock_ir.NEAR_CENTER),
    "ir_near_right": bool(dock_ir.NEAR_RIGHT),
    "ir_far_left": bool(dock_ir.FAR_LEFT),
    "ir_far_center": bool(dock_ir.FAR_CENTER),
    "ir_far_right": bool(dock_ir.FAR_RIGHT),
    }

def imu_parse(data):
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
