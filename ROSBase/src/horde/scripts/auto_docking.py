import multiprocessing as mp
import numpy as np
import random
import rospy
from geometry_msgs.msg import Twist, Vector3

from action_manager import start_action_manager
from gtd import GTD
from egq import GreedyGQ
from gvf import GVF
from learning_foreground import start_learning_foreground
from auto_docking_policies import eGreedy, Learned_Policy

if __name__ == "__main__":
    try:
        # random.seed(20170612)

        time_scale = 0.5
        forward_speed = 0.2
        turn_speed = 2

        alpha0 = 5
        lambda_ = 0.9
        num_features = 14404
        alpha = (1 - lambda_) * alpha0 / num_features
        parameters = {'alpha': alpha,
                     'beta': 0.01 * alpha,
                     'lambda': lambda_,
                     'alpha0': alpha0}

        one_if_bump = lambda observation: int(any(observation['bump'])) if observation is not None else 0
        one_if_ir = lambda observation: int(any(observation['ir'])) if observation is not None else 0

        theta = np.zeros(num_features*5)
        phi = np.zeros(num_features)
        observation = None
        learningRate = 0.1/(4*900)
        secondaryLearningRate = learningRate/10
        epsilon = 0.1
        lambda_ = lambda observation: 0.95
        learned_policy = Learned_Policy()

        action_space = [None]*5
        num_features = len(action_space)
        learner_parameters = {'theta' : theta,
                        'gamma' : 0.9,
                        '_lambda' : lambda_,
                        'cumulant' : one_if_ir,
                        'alpha' : learningRate,
                        'beta' : secondaryLearningRate,
                        'epsilon' : epsilon,
                        'learned_policy': learned_policy}

        learner = GreedyGQ(**learner_parameters)
        auto_docking = GVF(num_features=14404*num_features,
                        parameters=parameters,
                        gamma= lambda observation: 0.9,
                        cumulant=one_if_ir,
                        learner=learner,
                        target_policy=learner.learned_policy,
                        name='auto_docking',
                        logger=rospy.loginfo)

        behavior_policy = auto_docking.learner.behavior_policy

        topics = [
            # "/camera/depth/image",
            # "/camera/depth/points",
            # "/camera/ir/image",
            # "/camera/rgb/image_raw",
            "/camera/rgb/image_rect_color",
            "/mobile_base/sensors/core",
            "/mobile_base/sensors/dock_ir",
            # "/mobile_base/sensors/imu_data",
            ]

        foreground_process = mp.Process(target=start_learning_foreground,
                                        name="foreground",
                                        args=(time_scale,
                                              [auto_docking],
                                              topics,
                                              behavior_policy,auto_docking))

        action_manager_process = mp.Process(target=start_action_manager,
                                            name="action_manager",
                                            args=())
        foreground_process.start()
        action_manager_process.start()

    except rospy.ROSInterruptException as detail:
        rospy.loginfo("Handling: {}".format(detail))
    finally:
        try:
            foreground_process.join()
            action_manager_process.join()  
        except NameError:
            pass    
