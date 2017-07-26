import multiprocessing as mp
import numpy as np
import random
import rospy
from geometry_msgs.msg import Twist, Vector3
import tools
import subprocess

from action_manager import start_action_manager
from gtd import GTD
from egq import GreedyGQ
from gvf import GVF
from learning_foreground import start_learning_foreground
from auto_docking_policies import eGreedy, Learned_Policy
from state_representation import StateConstants
from std_msgs.msg import Bool

if __name__ == "__main__":
    try:

        time_scale = 0.5
        forward_speed = 0.2
        turn_speed = 2

        alpha0 = 5
        lambda_ = 0.9
        features_to_use = ['imu','ir','bias']
        num_features = np.concatenate([StateConstants.indices_in_phi[f] for f in features_to_use]).size
        alpha = (1 - lambda_) * alpha0 / num_features
        parameters = {'alpha': alpha,
                     'beta': 0.01 * alpha,
                     'lambda': lambda_,
                     'alpha0': alpha0}

        task_to_learn = 4
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
            # theta = np.zeros(num_features*len(action_space))
            theta = np.random.rand(num_features*len(action_space))
            phi = np.zeros(num_features)
            observation = None
            # learningRate = 0.1/(4*900)
            learningRate = 0.1/(100)
            secondaryLearningRate = learningRate/10
            epsilon = 0.1
            # lambda_ = lambda observation: 0.95
            lambda_ = 0.95

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
            # theta = np.zeros(num_features*len(action_space))
            theta = np.random.rand(num_features*len(action_space))

            phi = np.zeros(num_features)
            observation = None
            # learningRate = 0.1/(4*900)
            learningRate = 0.1/(10)
            secondaryLearningRate = learningRate/10
            epsilon = 0.2
            # lambda_ = lambda observation: 0.95
            lambda_ = 0.4

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
            # theta = np.zeros(num_features*len(action_space))
            theta = np.random.rand(num_features*len(action_space))

            phi = np.zeros(num_features)
            observation = None
            # learningRate = 0.1/(4*900)
            learningRate = 0.1/(10)
            secondaryLearningRate = learningRate/10
            epsilon = 0.5
            # lambda_ = lambda observation: 0.95
            lambda_ = 0.4


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
                        pass
                        # field_award = 1
                        # if aligned_far:
                        #     field_award = 1
                        # if aligned_near:
                        #     field_award = 2
                        # if tools.equal_twists(action_space[0] ,observation['action']):
                        #     action_award = 5
                    else:
                        field_award = -1
                    if observation['charging']:
                        print '                     charging===================='
                        success_award = 50
                    if field_award == -1:
                        times_field_reward_is_zero += 1
                        print times_field_reward_is_zero
                    else:
                        times_field_reward_is_zero = 0
                    if times_field_reward_is_zero >= 20:
                        print 'field reward is negative'
                        times_field_reward_is_zero = 0
                        return -1000
                    return field_award + success_award + action_award
                return award

            action_space = [#Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)), #stop
                            Twist(Vector3(0.1, 0, 0), Vector3(0, 0, 0)), # forward
                            # Twist(Vector3(-0.05, 0, 0), Vector3(0, 0, 0)), # backward
                            Twist(Vector3(0, 0, 0), Vector3(0, 0, 0.5)), # turn acw/cw
                            Twist(Vector3(0, 0, 0), Vector3(0, 0, -0.5)) # turn cw/acw
                            ]


            # theta = np.zeros(num_features*len(action_space))
            theta = np.random.rand(num_features*len(action_space))

            phi = np.zeros(num_features)
            observation = None
            # learningRate = 0.1/(4*900)
            learningRate = 0.1/(10)
            secondaryLearningRate = learningRate/10
            epsilon = 0.1
            # lambda_ = lambda observation: 0.95
            lambda_ = 0.95

        learned_policy = Learned_Policy(features_to_use=features_to_use,theta=theta,action_space=action_space)

        learner_parameters = {'theta' : theta,
                        'gamma' : 0.9,
                        '_lambda' : lambda_,
                        'cumulant' : reward_function(action_space),
                        'alpha' : learningRate,
                        'beta' : secondaryLearningRate,
                        'epsilon' : epsilon,
                        'learned_policy': learned_policy,
                        'num_features_state_action': num_features*len(action_space),
                        'features_to_use': features_to_use,
                        'action_space':action_space}

        learner = GreedyGQ(**learner_parameters)
        auto_docking = GVF(num_features=num_features*len(action_space),
                        parameters=parameters,
                        gamma= lambda observation: 0.9,
                        cumulant=reward_function(action_space),
                        learner=learner,
                        target_policy=learner.learned_policy,
                        name='auto_docking',
                        logger=rospy.loginfo,
                        features_to_use=features_to_use)

        behavior_policy = auto_docking.learner.behavior_policy

        foreground_process = mp.Process(target=start_learning_foreground,
                                        name="foreground",
                                        args=(time_scale,
                                              [auto_docking],
                                              features_to_use,
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
