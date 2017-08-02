from __future__ import division
from geometry_msgs.msg import Twist, Vector3
import math
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
from Queue import Queue
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

        self.llast_index = 0
        
        Policy.__init__(self, *args, **kwargs)        

        self.mellowmax = lambda x: np.log(np.mean(np.exp(omega * x))) / omega

    def update(self, phi, *args, **kwargs):

        phi = phi[self.feature_indices]

        q_fun = np.vectorize(lambda action: self.value(phi, action))
        q_values = q_fun(self.action_space)

        mm = self.mellowmax(q_values)
        diff = q_values - mm
        
        beta = optimize.brentq(self.find_beta(diff), -10, 10)
        if beta == 10 or beta == -10:
            rospy.logwarn("Beta = {}".format(beta))
        beta_q = beta * q_values
        self.pi = tools.softmax(beta_q)

        if self.last_index == self.TURN:
            self.pi[self.STOP] += self.pi[self.FORWARD]
            self.pi[self.FORWARD] = 0

    def find_beta(self, diff):
        def optimize_this(beta, *args):
            return np.sum(np.exp(beta * diff) * diff)
        return optimize_this

    def get_probability(self, *args, **kwargs):
        self.llast_index = self.last_index
        return Policy.get_probability(self, *args, **kwargs)

    def cumulant(self, observation):
        c = 0
        if observation is not None:
            if int(any(observation['bump'])) and self.last_index != self.STOP:
                c = -1
            elif self.last_index == self.TURN:
                c = 0
            elif self.last_index == self.FORWARD:
                c = 0.5
        if self.last_index == self.STOP and self.llast_index != self.TURN:
            c = -2
        return c

class AvoidWallSoftmax(Policy):
    # uses maximum entropy mellowmax
    def __init__(self, *args, **kwargs):
        # where the last action is recorded according
        # to its respective constants
        self.TURN = 2
        self.FORWARD = 1
        self.STOP = 0

        self.llast_index = 0

        Policy.__init__(self, *args, **kwargs)

    def update(self, phi, *args, **kwargs):

        phi = phi[self.feature_indices]

        q_fun = np.vectorize(lambda action: self.value(phi, action))
        q_values = q_fun(self.action_space)
        self.pi = tools.softmax(q_values)

        if self.last_index == self.TURN:
            self.pi[self.STOP] += self.pi[self.FORWARD]
            self.pi[self.FORWARD] = 0

    def find_beta(self, diff):
        def optimize_this(beta, *args):
            return np.sum(np.exp(beta * diff) * diff)
        return optimize_this

    def get_probability(self, *args, **kwargs):
        self.llast_index = self.last_index
        return Policy.get_probability(self, *args, **kwargs)

    def cumulant(self, observation):
        c = 0
        if observation is not None:
            if int(any(observation['bump'])) and self.last_index != self.STOP:
                c = -1
            elif self.last_index == self.TURN:
                c = 0
            elif self.last_index == self.FORWARD:
                c = 0.5
        if self.last_index == self.STOP and self.llast_index != self.TURN:
            c = -2
        return c

class GoForward(Policy):
    def __init__(self, *args, **kwargs):
        kwargs['feature_indices'] = np.array([0])
        Policy.__init__(self, *args, **kwargs)

        # self.pi = np.array([0, 1, 0])

        self.pi = np.array([1, 0])

    def update(self, phi, observation, *args, **kwargs):
        pass

class PavlovSoftmax(Policy):
    def __init__(self, time_scale, *args, **kwargs):
        # where the last action is recorded according
        # to its respective constants
        self.TURN = 1
        self.FORWARD = 0

        self.time_scale = time_scale

        Policy.__init__(self, *args, **kwargs)

    def update(self, phi, observation, *args, **kwargs):
        phi = phi[self.feature_indices]
        p = self.value(phi)

        if sum(observation['bump']):
            self.pi = np.zeros(self.action_space.size)
            self.pi[self.TURN] = 1
        else:
            # Joseph Modayil's constants
            T = 1/self.time_scale/5
            k1 = np.log((T-1)*(self.action_space.size-1))
            k2 = k1 * 4

            # make preferences for each action
            prefs = np.zeros(2)
            last = lambda x: self.last_index == x
            prefs[self.TURN] = np.exp(k1 * last(self.TURN))
            prefs[self.FORWARD] = np.exp(k2*(0.5-p) + k1*last(self.FORWARD))

            self.pi = tools.softmax(prefs)


class ForwardIfClear(Policy):
    def __init__(self, *args, **kwargs):
        # where the last action is recorded according
        # to its respective constants
        self.TURN = 1
        self.FORWARD = 0
        # self.STOP = 0

        Policy.__init__(self, *args, **kwargs)

    def update(self, phi, observation, *args, **kwargs):
        phi = phi[self.feature_indices]

        if self.value(phi) > 0.75 or sum(observation['bump']):
            self.last_index = self.TURN
        else:
            # if self.last_index == self.TURN:
            #     self.last_index = self.STOP
            # else:
            self.last_index = self.FORWARD

        self.pi = np.zeros(self.action_space.size)
        self.pi[self.last_index] = 1

class Switch:
    def __init__(self, explorer, exploiter, num_timesteps_explore):
        self.explorer = explorer
        self.exploiter = exploiter
        self.num_timesteps_explore = num_timesteps_explore
        self.action_space = self.explorer.action_space
        self.last_index = 0
        self.t = 0

    def update(self, *args, **kwargs):
        if self.t > self.num_timesteps_explore:
            self.exploiter.update(*args, **kwargs)
        else:
            self.explorer.update(*args, **kwargs)
        self.t += 1

    def get_probability(self, *args, **kwargs):
        if self.t > self.num_timesteps_explore:
            policy = self.exploiter
        else:
            policy = self.explorer

        prob = policy.get_probability(*args, **kwargs)
        self.last_index = policy.last_index

        return prob

    def choose_action(self, *args, **kwargs):
        if self.t > self.num_timesteps_explore:
            policy = self.exploiter
            msg = "Calling exploitative policy."
        else:
            policy = self.explorer
            msg = "Calling exploratory policy."

        action = policy.choose_action(*args, **kwargs)
        self.last_index = policy.last_index

        rospy.loginfo(msg)
        return action

if __name__ == "__main__":
    try:
        # random.seed(20170823)

        # turns on and off the hyperparameter search
        hyperparameter_experiment_mode = True

        action_manager_process = mp.Process(target=start_action_manager,
                                            name="action_manager",
                                            args=())
        action_manager_process.start()
        
        # robotic parameters
        time_scale = 0.05
        forward_speed = 0.2
        turn_speed = 1

        # all available actions
        action_space = np.array([Twist(Vector3(forward_speed, 0, 0),
                                       Vector3(0, 0, 0)),
                                 Twist(Vector3(0, 0, 0),
                                       Vector3(0, 0, turn_speed))])

        # either cycles through hyperparameter possibilities or 
        # runs wall demo once with default hyperparameters
        if (hyperparameter_experiment_mode):
          hyperparameters = [{"alpha0":0.1, "lmbda":0.87}, {"alpha0":0.1, "lmbda":0.9},
                             {"alpha0":0.1, "lmbda":0.93}]
        else:
          hyperparameters = [{}]

        for hps in hyperparameters:
          # learning parameters
          alpha0 = 0.05
          features_to_use = ['image', 'bias']
          feature_indices = np.concatenate([StateConstants.indices_in_phi[f] for f in features_to_use])
          num_active_features = sum(StateConstants.num_active_features[f] for f in features_to_use)
          num_features = feature_indices.size
          turn_sec_to_bump = 2
          # discount = math.pow(0.75, time_scale / turn_sec_to_bump)
          discount = 1 - time_scale
          discount_if_bump = lambda obs: 0 if sum(obs["bump"]) else discount
          one_if_bump = lambda obs: int(any(obs['bump'])) if obs is not None else 0
          dtb_hp = {'alpha': alpha0 / num_active_features,
                    'beta': 0.001 * alpha0 / num_active_features,
                    'lmbda': 0.9,
                    'alpha0': alpha0,
                    'num_features': num_features,
                    'feature_indices': feature_indices,
                   }

          # avoid_wall_omega = 10
          # alpha0 = 0.01
          # avoid_wall_hp = {'alpha': alpha0 / num_active_features,
          #                  'beta': 0.001 * alpha0 / num_active_features,
          #                  'lmbda': 0.1,
          #                  'alpha0': alpha0,
          #                  'num_features': num_features * action_space.size,
          #                  'feature_indices': feature_indices,
          #                 }

          # prediction GVF
          dtb_policy = GoForward(action_space=action_space)
          dtb_learner = GTD(**dtb_hp)

          threshold_policy = PavlovSoftmax(action_space=action_space,
                                      feature_indices=dtb_hp['feature_indices'],
                                      value_function=dtb_learner.predict,
                                      time_scale=time_scale)
          distance_to_bump = GVF(cumulant = one_if_bump,
                                 gamma    = discount_if_bump,
                                 target_policy = dtb_policy,
                                 learner = dtb_learner,
                                 name = 'DistanceToBump',
                                 logger = rospy.loginfo,
                                 **dtb_hp)

          # # softmax control GVF
          # avoid_wall_learner = GreedyGQ(action_space,
          #                               finished_episode=lambda x: False,
          #                               **avoid_wall_hp)
          # avoid_wall_policy = AvoidWallSoftmax(#omega=avoid_wall_omega,
          #                     value_function=avoid_wall_learner.predict,
          #                     action_space=action_space,
          #                     feature_indices=avoid_wall_hp['feature_indices'])
          # avoid_wall_memm = GVF(cumulant = avoid_wall_policy.cumulant,
          #                       gamma    = discount_if_bump,
          #                       target_policy = avoid_wall_policy,
          #                       learner = avoid_wall_learner,
          #                       name = 'AvoidWall',
          #                       logger = rospy.loginfo,
          #                       **avoid_wall_hp)

          # # behavior_policy
          # behavior_policy = Switch(explorer=threshold_policy,
          #                          exploiter=avoid_wall_policy,
          #                          num_timesteps_explore=60/time_scale)

          # start processes
          cumulant_counter = mp.Value('d', 0)
          foreground_process = mp.Process(target=start_learning_foreground,
                                          name="foreground",
                                          args=(time_scale,
                                                [distance_to_bump],
                                                features_to_use,
                                                threshold_policy,
                                                None,
                                                cumulant_counter))

          foreground_process.start()
          if hyperparameter_experiment_mode is False:
            break;
          else:
            # start and stop wall demo if hyper parameter search is on
            rospy.sleep(5)
            foreground_process.terminate()
            print("CUMULANTS: " + str(cumulant_counter.value))

        if (hyperparameter_experiment_mode is True):
          action_manager_process.terminate()
    except rospy.ROSInterruptException as detail:
        rospy.loginfo("Handling: {}".format(detail))
    finally:
        try:
            foreground_process.join()
            action_manager_process.join()  
        except NameError:
            pass    