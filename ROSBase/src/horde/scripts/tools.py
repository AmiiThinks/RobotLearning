from functools import wraps
import kobuki_msgs.msg as kob_msg
import sensor_msgs.msg as sens_msg
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg as std_msg
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
    }

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
        print 'func:%r took: %2.4f sec' % (f.__name__, te-ts)
        return result
    return wrap