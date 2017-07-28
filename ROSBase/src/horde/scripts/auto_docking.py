from __future__ import division
import multiprocessing as mp
import numpy as np
import random
import rospy
from geometry_msgs.msg import Twist, Vector3
import tools
import subprocess

from action_manager import start_action_manager
from gtd import GTD
from greedy_GQ import GreedyGQ
from gvf import GVF
from learning_foreground import start_learning_foreground
from auto_docking_policies import *
from state_representation import StateConstants
from std_msgs.msg import Bool

class Switch:
    def __init__(self, explorer, exploiter, num_timesteps_explore):
        self.explorer = explorer
        self.exploiter = exploiter
        self.num_timesteps_explore = num_timesteps_explore
        self.t = 0

    def update(self, *args, **kwargs):
        if self.t > self.num_timesteps_explore:
            self.exploiter.update(*args, **kwargs)
            rospy.loginfo('Greedy policy is the behaviour policy')
        else:
            self.explorer.update(*args, **kwargs)
            rospy.loginfo('Explorer policy is the behaviour policy')
        self.t += 1
        self.t %= 2*self.num_timesteps_explore

    def get_probability(self, *args, **kwargs):
        if self.t > self.num_timesteps_explore:
            prob = self.exploiter.get_probability(*args, **kwargs)
        else:
            prob = self.explorer.get_probability(*args, **kwargs)
        return prob

    def choose_action(self, *args, **kwargs):
        if self.t > self.num_timesteps_explore:
            action = self.exploiter.choose_action(*args, **kwargs)
        else:
            action = self.explorer.choose_action(*args, **kwargs)
        return action

if __name__ == "__main__":
    try:

        time_scale = 0.5
        forward_speed = 0.2
        turn_speed = 2

        alpha0 = 5
        lmbda = 0.9
        features_to_use = ['ir', 'bias']
        feature_indices = np.concatenate([StateConstants.indices_in_phi[f] for f in features_to_use])
        num_features = feature_indices.size
        alpha = alpha0 / num_features
        parameters = {'alpha': alpha,
                     'beta': 0.01 * alpha,
                     'lmbda': lmbda,
                     'alpha0': alpha0}

        task_to_learn = 3
        if task_to_learn == 1: #reach the IR region
            def reward_function(action_space):
                def award(observation):
                    return int(any(observation['ir'])) if observation is not None else 0
                return award

            action_space = [#Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)), #stop
                            Twist(Vector3(0.2, 0, 0), Vector3(0, 0, 0)), # forward
                            Twist(Vector3(-0.2, 0, 0), Vector3(0, 0, 0)), # backward
                            Twist(Vector3(0, 0, 0), Vector3(0, 0, 1.5)), # turn acw/cw
                            Twist(Vector3(0, 0, 0), Vector3(0, 0, -1.5)) # turn cw/acw
                            ]
            # learningRate = 0.1/(4*900)
            learningRate = 0.1/(100)
            secondaryLearningRate = learningRate/10
            epsilon = 0.1
            # lmbda = lambda observation: 0.95
            lmbda = 0.95

        if task_to_learn == 2: #reach the center IR region
            def reward_function(action_space):
                def award(observation):
                    # if not in IR region it will get 0 reward
                    if not any(observation['ir']):
                        return -1
                    return 1000*int(any([(b%16)/8 or (b%4)/2 for b in observation['ir']])) if observation is not None else 0
                return award

            action_space = [#Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)), #stop
                            Twist(Vector3(0.2, 0, 0), Vector3(0, 0, 0)), # forward
                            Twist(Vector3(-0.2, 0, 0), Vector3(0, 0, 0)), # backward
                            Twist(Vector3(0, 0, 0), Vector3(0, 0, 1.0)), # turn acw/cw
                            Twist(Vector3(0, 0, 0), Vector3(0, 0, -1.0)) # turn cw/acw
                            ]

            # learningRate = 0.1/(4*900)
            learningRate = 0.1/(10)
            secondaryLearningRate = learningRate/10
            epsilon = 0.2
            # lmbda = lambda observation: 0.95
            lmbda = 0.4

        if task_to_learn == 3: #align center IR reciever and sender
            def reward_function(action_space):
                def award(observation):
                    ir_data_center = observation['ir'][1]
                    reward = int((ir_data_center%16)/8 or (ir_data_center%4)/2) if observation is not None else 0
                    return reward
                return award

            action_space = [#Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)), #stop
                            # Twist(Vector3(0.05, 0, 0), Vector3(0, 0, 0)), # forward
                            # Twist(Vector3(-0.05, 0, 0), Vector3(0, 0, 0)), # backward
                            Twist(Vector3(0, 0, 0), Vector3(0, 0, 0.3)), # turn acw/cw
                            Twist(Vector3(0, 0, 0), Vector3(0, 0, -0.3)) # turn cw/acw
                            ]

            # learningRate = 0.1/(4*900)
            learningRate = 0.1/(10)
            secondaryLearningRate = learningRate/10
            epsilon = 0.5
            # lmbda = lambda observation: 0.95
            lmbda = 0.9

        if task_to_learn == 4: #align center IR reciever and sender
            global times_field_reward_is_zero
            times_field_reward_is_zero = 0
            def reward_function(action_space):
                def award(observation):
                    global times_field_reward_is_zero

                    aligned_near = (observation['ir'][1]%4)/2
                    aligned_far = (observation['ir'][1]%16)/8
                    aligned = aligned_near or aligned_far
                    print 'ir data: ', observation['ir']
                    field_award = 0
                    action_award = 0
                    success_award = 0
                    if aligned:
                        # pass
                        # field_award = 1
                        if aligned_far:
                            field_award = 1
                        if aligned_near:
                            field_award = 1
                        if tools.equal_twists(action_space[0] ,observation['action']):
                            action_award = 2
                    else:
                        field_award = -1
                    if observation['charging']:
                        print '====================charging===================='
                        success_award = 50
                    if field_award == -1:
                        times_field_reward_is_zero += 1
                        print times_field_reward_is_zero
                    else:
                        times_field_reward_is_zero = 0
                    if times_field_reward_is_zero >= 15:
                        print 'field reward is negative'
                        times_field_reward_is_zero = 0
                        return -10
                    print field_award, action_award
                    return field_award + success_award + action_award
                return award

            action_space = [#Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)), #stop
                            Twist(Vector3(0.08, 0, 0), Vector3(0, 0, 0)), # forward
                            # Twist(Vector3(-0.05, 0, 0), Vector3(0, 0, 0)), # backward
                            Twist(Vector3(0, 0, 0), Vector3(0, 0, 0.3)), # turn acw/cw
                            Twist(Vector3(0, 0, 0), Vector3(0, 0, -0.3)) # turn cw/acw
                            ]

            # learningRate = 0.1/(4*900)
            learningRate = 0.1/(10)
            secondaryLearningRate = learningRate/10
            epsilon = 0.1
            # lmbda = lambda observation: 0.95
            lmbda = 0.95

        action_space = np.array(action_space)

        learner_parameters = {'alpha' : learningRate,
                              'beta' : secondaryLearningRate,
                              'lmbda': lmbda,
                              'num_features_state_action': num_features*action_space.size,
                              'action_space': action_space,
                              'finished_episode': lambda x: x > 0 or x < -100
                             }

        learner = GreedyGQ(**learner_parameters)

        target_policy = Greedy(feature_indices=feature_indices,
                       action_space=action_space,
                       value_function=learner.predict)

        auto_docking = GVF(num_features=num_features*len(action_space),
                        gamma= lambda observation: 0.9,
                        cumulant=reward_function(action_space),
                        learner=learner,
                        target_policy=target_policy,
                        name='auto_docking',
                        logger=rospy.loginfo,
                        feature_indices=feature_indices,
                        **parameters)



        # behavior_policy = eGreedy(epsilon = epsilon,
        #                           value_function=auto_docking.learner.predict,
        #                           action_space=action_space,
        #                           feature_indices=feature_indices)

        exploring_policy = Alternating_Rotation(epsilon = epsilon,
                                  value_function=auto_docking.learner.predict,
                                  action_space=action_space,
                                  feature_indices=feature_indices)


        behavior_policy = Switch(explorer=exploring_policy, exploiter=target_policy, num_timesteps_explore=200)


        foreground_process = mp.Process(target=start_learning_foreground,
                                        name="foreground",
                                        args=(time_scale,
                                              [auto_docking],
                                              features_to_use,
                                              behavior_policy,
                                              auto_docking))

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
