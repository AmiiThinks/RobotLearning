""" Runs a demo that learns how to align the IR emitters of the TurtleBot 
to the IR recievers of the charger.

This module specifies the agent's learning algorithm, parameters, policies, 
features, and actions. The module also interfaces with the 
:doc:`learning_foreground` and the :doc:`action_manager` to run the main 
learning loop and publish actions respectively.

NOTE: When run, this example will learn very slowly as a result of epsilon-
    greedy being ineffective at exploring. However one will see notable 
    progress in 10 minutes.

Authors:
    Shibhansh Dohare, Niko Yasui, Parash Rahman.

"""

import multiprocessing as mp
import numpy as np
import random
import rospy

from action_manager import start_action_manager
from auto_docking_policies import EGreedy
from geometry_msgs.msg import Twist, Vector3
from greedy_gq import GreedyGQ
from gvf import GVF
from learning_foreground import start_learning_foreground
from state_representation import StateConstants

if __name__=='__main__':
    try:
        time_scale = 0.3 # time step length
        turn_speed = 2   # rad/sec turn speed

        features_to_use = ['imu', 'bias', 'ir', 'last_action']
        feature_indices = np.concatenate(
                [StateConstants.indices_in_phi[f] for f in features_to_use])
        num_features = feature_indices.size
        num_active_features = sum(
                StateConstants.num_active_features[f] for f in features_to_use)
        
        lmbda = 0.9
        alpha0 = 0.1
        alpha = alpha0 / num_active_features
        learningRate = alpha
        secondaryLearningRate = learningRate / 10
        epsilon = 0.1
        
        parameters = {'alpha': alpha,
                      'beta': 0.01 * alpha,
                      'lmbda': lmbda,
                      'alpha0': alpha0
                      }

        action_space = np.array([
            Twist(Vector3(0, 0, 0), Vector3(0, 0, 0.3)),  # turn acw/cw
            Twist(Vector3(0, 0, 0), Vector3(0, 0, -0.3))  # turn cw/acw
        ])

        # cumulant is 1 if center is aligned with charging station, 0 otherwise
        def cumulant(observation, phi):
            ir_data_center = observation['ir'][1]
            return 1 if ir_data_center & 2 or ir_data_center & 8 else 0

        # end episode when agent gets 1 as a cumulant (i.e. aligns center with charging station)
        finished_episode = lambda cum: cum == 1

        # when the episode is finished align the robot in a random direction
        def reset_episode():
            random_action = random.choice(action_space)
            return [random_action for i in range(random.randint(1, 80))]


        learner_parameters = {'alpha': learningRate,
                              'beta': secondaryLearningRate,
                              'lmbda': lmbda,
                              'num_features': num_features * action_space.size,
                              'action_space': action_space,
                              'finished_episode': finished_episode
                              }

        learner = GreedyGQ(**learner_parameters)

        target_policy = EGreedy(epsilon=0,
                               feature_indices=feature_indices,
                               action_space=action_space,
                               value_function=learner.predict)

        behavior_GVF = GVF(num_features=num_features*len(action_space),
                            gamma=lambda observation: 0.9,
                            cumulant=cumulant,
                            learner=learner,
                            target_policy=target_policy,
                            name='auto_docking',
                            logger=rospy.loginfo,
                            feature_indices=feature_indices,
                            **parameters)

        behavior_policy = EGreedy(epsilon=epsilon,
                                value_function=behavior_GVF.learner.predict,
                                action_space=action_space,
                                feature_indices=feature_indices)


        foreground_process = mp.Process(target=start_learning_foreground,
                                        name="foreground",
                                        args=(time_scale,
                                              [behavior_GVF],
                                              features_to_use,
                                              behavior_policy,
                                              behavior_GVF,
                                              None,
                                              reset_episode))

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