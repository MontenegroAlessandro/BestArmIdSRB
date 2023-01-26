"""
This script implements the environment in the Gym interface.
"""
# Libraries
import numpy as np
import json


# Implementation of the stochastic rising bandits environment
class SRBEnvironment:
    def __init__(self, horizon, actions, noise):
        """
        This function initialize the variables of the environment
        :param horizon: the horizon of the problem, so the pull budget
        :param actions: this is a list of functions implementing the real underlying processes (rising bandits)
        :param noise: the standard deviation characterizing the stochastic outputs (the same for all the actions)
        """
        assert horizon > 0, "Error in horizon"
        self.horizon = horizon
        self.actions = actions
        self.n = len(self.actions)
        self.state = np.zeros(self.n)
        self.noise = noise
        self.t = 0
        self.noise_matrix = None
        self.reset(0)

    def step(self, action):
        """
        This function implements a step inside the environment. Given an action, it samples from the corresponding
        rising bandit a value, update the state and returns the stochastic sample value
        :param action: this is the id of the rising bandit to sample
        :return: the value sampled
        """
        # check the action id
        assert 0 <= action < self.n, "Error in action id"

        # sample the output from the right bandit
        out = self.actions[action](self.state[action]) + self.noise_matrix[self.t]

        # update the counter for the pulled arm
        self.state[action] += 1

        # increment the time counter
        self.t += 1

        return out

    def reset(self, seed):
        """
        This function resets the environment
        :param seed: the id of the trial that is running
        :return: None
        """
        np.random.seed(seed)
        self.noise_matrix = np.random.normal(0, self.noise, self.horizon)

        # initialize the state vector
        self.state = np.zeros(self.n)

        # initialize the time counter
        self.t = 0

class IMDBEnvironment:
    def __init__(self, horizon):
        self.horizon = horizon
        base_path = '/Users/ale/Desktop/BAISRB/environment/imdb/'
        names = ['NN112', 'NN1', 'NN2', 'NN22', 'NN222', 'OGD', 'LR']
        self.curves = [np.load(base_path + i) for i in names]

        self.n = len(self.curves)
        self.state = np.zeros(self.n)
        self.t = 0
        self.reset(0)

    def step(self, action):
        assert 0 <= action < self.n, "Error in action id"

        # sample the output from the right bandit
        out = self.curves[int(action)][int(self.state[int(action)])]

        # update the counter for the pulled arm
        self.state[action] += 1

        # increment the time counter
        self.t += 1

        return out

    def reset(self, seed):
        # initialize the state vector
        self.state = np.zeros(self.n)

        # initialize the time counter
        self.t = 0
