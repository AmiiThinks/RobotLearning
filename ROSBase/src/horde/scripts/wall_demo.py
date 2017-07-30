from __future__ import division
from geometry_msgs.msg import Twist, Vector3
import multiprocessing as mp
import numpy as np
import random
import rospy
from scipy import optimize

from action_manager import start_action_manager
from greedy_gq import GreedyGQ
from gtd import GTD
from gvf import GVF
from learning_foreground import start_learning_foreground
from policy import Policy
from state_representation import StateConstants
import tools

class AvoidWallMEMM(Policy):
    # uses maximum entropy mellowmax
    def __init__(self, omega, *args, **kwargs):
        # where the last action is recorded according
        # to its respective constants
        self.TURN = 2
        self.FORWARD = 1
        self.STOP = 0
        
        Policy.__init__(self, *args, **kwargs)        

        self.mellowmax = lambda x: np.log(np.mean(np.exp(omega * x))) / omega

    def update(self, phi, *args, **kwargs):

        phi = phi[self.feature_indices]

        q_fun = np.vectorize(lambda action: self.value(phi, action))
        q_values = q_fun(self.action_space)
        mm = self.mellowmax(q_values)
        diff = q_values - mm
        
        beta = optimize.brentq(self.find_beta(diff), -10, 10)
        print(beta)
        beta_q = beta * q_values
        exp_q = np.exp(beta_q - np.max(beta_q)) # numerical stability
        self.pi = exp_q / np.sum(exp_q)

        if self.last_index == self.TURN:
            self.pi[self.STOP] += self.pi[self.FORWARD]
            self.pi[self.FORWARD] = 0

        # print(self.pi)

    def find_beta(self, diff):
        def optimize_this(beta, *args):
            return np.sum(np.exp(beta * diff) * diff)
        return optimize_this

    def cumulant(self, observation):
        c = -2
        if observation is not None:
            if int(any(observation['bump'])) and self.last_index != self.STOP:
                c = -1
            elif self.last_index == self.TURN:
                c = 0
            elif self.last_index == self.FORWARD:
                c = 0.5
        return c

class GoForward(Policy):
    def __init__(self, *args, **kwargs):
        kwargs['feature_indices'] = np.array([0])
        Policy.__init__(self, *args, **kwargs)

        self.pi = np.array([0, 1, 0])

    def update(self, phi, observation, *args, **kwargs):
        pass

class ForwardIfClear(Policy):
    def __init__(self, *args, **kwargs):
        # where the last action is recorded according
        # to its respective constants
        self.TURN = 2
        self.FORWARD = 1
        self.STOP = 0

        Policy.__init__(self, *args, **kwargs)

    def update(self, phi, observation, *args, **kwargs):
        phi = phi[self.feature_indices]

        if self.value(phi) > 0.75 or sum(observation['bump']):
            self.last_index = self.TURN
        else:
            if self.last_index == self.TURN:
                self.last_index = self.STOP
            else:
                self.last_index = self.FORWARD

        self.pi = np.zeros(self.action_space.size)
        self.pi[self.last_index] = 1

class Switch:
    def __init__(self, explorer, exploiter, num_timesteps_explore):
        self.explorer = explorer
        self.exploiter = exploiter
        self.num_timesteps_explore = num_timesteps_explore
        self.t = 0

    def update(self, *args, **kwargs):
        if self.t > self.num_timesteps_explore:
            self.exploiter.update(*args, **kwargs)
        else:
            self.explorer.update(*args, **kwargs)
        self.t += 1

    def get_probability(self, *args, **kwargs):
        if self.t > self.num_timesteps_explore:
            prob = self.exploiter.get_probability(*args, **kwargs)
        else:
            prob = self.explorer.get_probability(*args, **kwargs)
        return prob

    def choose_action(self, *args, **kwargs):
        if self.t > self.num_timesteps_explore:
            action = self.exploiter.choose_action(*args, **kwargs)
            msg = "Calling exploitative policy."
        else:
            action = self.explorer.choose_action(*args, **kwargs)
            msg = "Calling exploratory policy."

        rospy.loginfo(msg)
        return action

if __name__ == "__main__":
    try:
        # random.seed(20170612)

        # robotic parameters
        time_scale = 0.3
        forward_speed = 0.2
        turn_speed = 1

        # all available actions
        action_space = np.array([Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)),
                         Twist(Vector3(forward_speed,0,0), Vector3(0,0,0)),
                         Twist(Vector3(0, 0, 0), Vector3(0, 0, turn_speed))])

        # learning parameters
        alpha0 = 0.1
        features_to_use = ['image', 'bias']
        feature_indices = np.concatenate([StateConstants.indices_in_phi[f] for f in features_to_use])
        num_active_features = sum(StateConstants.num_active_features[f] for f in features_to_use)
        num_features = feature_indices.size
        discount_if_bump = lambda obs: 0 if sum(obs["bump"]) else 0.98
        one_if_bump = lambda obs: int(any(obs['bump'])) if obs is not None else 0
        dtb_hp = {'alpha': alpha0 / num_active_features,
                  'beta': 0.005 * alpha0 / num_active_features,
                  'lmbda': 0.9,
                  'alpha0': alpha0,
                 }

        avoid_wall_omega = 10
        alpha0 = 0.01
        avoid_wall_hp = {'alpha': alpha0 / num_active_features,
                         'beta': 0.01 * alpha0 / num_active_features,
                         'lmbda': 0.95,
                         'alpha0': alpha0,
                        }

        # prediction GVF
        dtb_policy = GoForward(action_space=action_space)
        dtb_learner = GTD(num_features, **dtb_hp)
        threshold_policy = ForwardIfClear(action_space=action_space,
                                          feature_indices=feature_indices,
                                          value_function=dtb_learner.predict)
        distance_to_bump = GVF(cumulant = one_if_bump,
                               gamma    = discount_if_bump,
                               target_policy = dtb_policy,
                               num_features = num_features,
                               learner = dtb_learner,
                               name = 'DistanceToBump',
                               logger = rospy.loginfo,
                               feature_indices = feature_indices,
                               **dtb_hp)

        # softmax control GVF
        avoid_wall_learner = GreedyGQ(num_features * action_space.size,
                                      action_space,
                                      finished_episode=lambda x: False,
                                      **avoid_wall_hp)
        avoid_wall_policy = AvoidWallMEMM(omega=avoid_wall_omega,
                                          value_function=avoid_wall_learner.predict,
                                          action_space=action_space,
                                          feature_indices=feature_indices)
        avoid_wall_memm = GVF(cumulant = avoid_wall_policy.cumulant,
                              gamma    = discount_if_bump,
                              target_policy = avoid_wall_policy,
                              num_features = num_features * action_space.size,
                              learner = avoid_wall_learner,
                              name = 'AvoidWall',
                              logger = rospy.loginfo,
                              feature_indices = feature_indices,
                              **avoid_wall_hp)

        # behavior_policy
        behavior_policy = Switch(explorer=threshold_policy,
                                 exploiter=avoid_wall_policy,
                                 num_timesteps_explore=180/time_scale)


        # start processes
        foreground_process = mp.Process(target=start_learning_foreground,
                                        name="foreground",
                                        args=(time_scale,
                                              [distance_to_bump,
                                               avoid_wall_memm],
                                              features_to_use,
                                              behavior_policy))

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