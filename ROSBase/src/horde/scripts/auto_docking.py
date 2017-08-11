"""Describes the auto-docking task.

Contains various parameters for various steps of auto-docking. Very 
similar to 'wall_demo.py'.

Authors: 
    Shibhansh Dohare, Niko Yasui.
"""
from __future__ import division

import multiprocessing as mp

from geometry_msgs.msg import Twist, Vector3

from action_manager import start_action_manager
from auto_docking_policies import *
from greedy_gq import GreedyGQ
from gvf import GVF
from learning_foreground import start_learning_foreground
from state_representation import StateConstants

if __name__ == "__main__":
    """Change the ``task_to_learn`` parameter to learn any sub-task for 
    auto-docking.
    For every sub-task only ``action_space``, ``reward_function``, and
    ``behaviour_policy`` are different.
    """
    try:

        time_scale = 0.3
        forward_speed = 0.2
        turn_speed = 2

        alpha0 = 0.1
        lmbda = 0.9
        features_to_use = ['imu', 'bias', 'ir']
        feature_indices = np.concatenate(
                [StateConstants.indices_in_phi[f] for f in features_to_use])
        num_features = feature_indices.size
        num_active_features = sum(
                StateConstants.num_active_features[f] for f in features_to_use)
        alpha = alpha0 / num_active_features
        parameters = {'alpha': alpha,
                      'beta': 0.01 * alpha,
                      'lmbda': lmbda,
                      'alpha0': alpha0
                      }

        learningRate = alpha
        secondaryLearningRate = learningRate / 10


        def finished_episode(x):
            return x > 0 or x < -100


        epsilon = 0.1

        action_space = []


        def reward_function(x):
            return lambda y: 0


        task_to_learn = 3
        if task_to_learn == 1:  # reach the IR region
            def reward_function(action_space):
                def award(observation):
                    return int(any(observation[
                                       'ir'])) if observation is not None \
                        else 0

                return award


            action_space = [
                # Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)), #stop
                Twist(Vector3(0.2, 0, 0), Vector3(0, 0, 0)),  # forward
                Twist(Vector3(-0.2, 0, 0), Vector3(0, 0, 0)),  # backward
                Twist(Vector3(0, 0, 0), Vector3(0, 0, 1.5)),  # turn acw/cw
                Twist(Vector3(0, 0, 0), Vector3(0, 0, -1.5))  # turn cw/acw
            ]

        if task_to_learn == 2:  # reach the center IR region
            def reward_function(action_space):
                def award(observation):
                    # if not in IR region it will get 0 reward
                    # if not any(observation['ir']):
                    #     return -1
                    return int(any([(b % 16) // 8 or (b % 4) // 2 for b in
                                    observation[
                                        'ir']])) if observation is not None \
                        else 0

                return award


            action_space = [
                # Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)), #stop
                Twist(Vector3(0.1, 0, 0), Vector3(0, 0, 0)),  # forward
                # Twist(Vector3(-0.2, 0, 0), Vector3(0, 0, 0)), # backward
                Twist(Vector3(0, 0, 0), Vector3(0, 0, 1.0)),  # turn acw/cw
                # Twist(Vector3(0, 0, 0), Vector3(0, 0, -1.0)) # turn cw/acw
            ]

        if task_to_learn == 3:  # align center IR reciever and sender
            def reward_function(action_space):
                def award(observation):
                    ir_data_center = observation['ir'][1]
                    reward = int((ir_data_center % 16) / 8 or (
                        ir_data_center % 4) / 2) if observation is not None \
                        else 0
                    return reward

                return award


            action_space = [
                # Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)), #stop
                # Twist(Vector3(0.05, 0, 0), Vector3(0, 0, 0)), # forward
                # Twist(Vector3(-0.05, 0, 0), Vector3(0, 0, 0)), # backward
                Twist(Vector3(0, 0, 0), Vector3(0, 0, 0.3)),  # turn acw/cw
                Twist(Vector3(0, 0, 0), Vector3(0, 0, -0.3))  # turn cw/acw
            ]

        if task_to_learn == 4:  # reach docking station while staying in the
            #  center region
            global times_field_reward_is_zero
            times_field_reward_is_zero = 0


            def reward_function(action_space):
                def award(observation):
                    global times_field_reward_is_zero

                    aligned_near = int((observation['ir'][1] % 4) / 2)
                    aligned_far = int((observation['ir'][1] % 16) / 8)
                    aligned = aligned_near or aligned_far
                    field_award = 0
                    action_award = 0
                    success_award = 0
                    if aligned:
                        print(observation['ir'], aligned_near, aligned_far)
                        # pass
                        # field_award = 1
                        if aligned_far:
                            field_award = 1
                        if aligned_near:
                            field_award = 1
                            # if tools.equal_twists(action_space[0] ,
                            # observation['action']):
                            #     action_award = 2
                    else:
                        field_award = -1
                    if observation['charging']:
                        print(
                            '====================charging===================='
                        )
                        success_award = 50
                    if field_award == -1:
                        times_field_reward_is_zero += 1
                    else:
                        times_field_reward_is_zero = 0
                    print('times_field_reward_is_zero',
                          times_field_reward_is_zero)
                    if times_field_reward_is_zero >= 3:
                        print('field reward is negative')
                        times_field_reward_is_zero = 0
                        return -2
                    # print field_award, action_award
                    return field_award + success_award + action_award

                return award


            finished_episode = lambda x: x > 10 or x < -1

            action_space = [
                # Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)), #stop
                Twist(Vector3(0.08, 0, 0), Vector3(0, 0, 0)),  # forward
                # Twist(Vector3(-0.05, 0, 0), Vector3(0, 0, 0)), # backward
                Twist(Vector3(0, 0, 0), Vector3(0, 0, 0.5)),  # turn acw/cw
                Twist(Vector3(0, 0, 0), Vector3(0, 0, -0.5))  # turn cw/acw
            ]

        action_space = np.array(action_space)

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

        auto_docking = GVF(num_features=num_features * len(action_space),
                           gamma=lambda observation: 0.9,
                           cumulant=reward_function(action_space),
                           learner=learner,
                           target_policy=target_policy,
                           name='auto_docking',
                           logger=rospy.loginfo,
                           feature_indices=feature_indices,
                           **parameters)

        # # for testing any task with e-greedy learning
        # behavior_policy = eGreedy(epsilon = epsilon,
        #
        # value_function=auto_docking.learner.predict,
        #                           action_space=action_space,
        #                           feature_indices=feature_indices)

        # for testing task-3 i.e. aligning robot
        exploring_policy = AlternatingRotation(
                epsilon=epsilon,
                value_function=auto_docking.learner.predict,
                action_space=action_space,
                feature_indices=feature_indices)

        # # for testing task-2 i.e. reaching the center region
        # exploring_policy = ForwardIfClear(action_space=action_space,
        # feature_indices=feature_indices)

        behavior_policy = Switch(explorer=exploring_policy,
                                 exploiter=target_policy,
                                 num_timesteps_explore=1200)

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
