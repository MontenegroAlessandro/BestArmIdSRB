"""
This module of the Gym interface is acting as a middle point between the agent (who is in charge of running the
proposed algorithm) and the environment. In this way we are trying to decouple the running agent and the environment on
which it is acting on.
"""
# Libraries
import numpy as np
import io
import json
from datetime import datetime
from tqdm.auto import tqdm


# Class runner
class Runner:
    def __init__(self, environment, agent, n_trials, horizon, n_actions, actions=None, log_path=""):
        """
        This is the initializer of the class Runner
        :param environment: the environment on which the actions drawn by the agent are applied
        :param agent: the actor who is selecting the action at each time step
        :param n_trials: this is the number of times we ask to perform the experiment
        :param horizon: the time budget, in terms of pulls, of the experiment
        :param n_actions: the number of actions in the bandit problem
        :param actions: actions to be used
        """
        self.log_path = log_path
        self.environment = environment
        self.agent = agent
        self.n_trials = n_trials
        self.horizon = horizon
        self.n_actions = n_actions
        self.actions = actions

        # data to save the output
        self.results = dict()

    def perform_simulations(self):
        """
        This function is aimed at performing the n_trials experiments of the given agent on the given environment.
        :return: the actions performed by each trial
        """
        # initialize the vector of all the actions, in particular we will have a vector of n_trials element, each of
        # which is the sequence of horizon actions

        scores_all = np.zeros((self.n_trials, self.horizon))
        pulled_arms_all = np.zeros((self.n_trials, self.horizon))
        recommendation_all = np.zeros(self.n_trials)
        estimators_all = np.zeros((self.n_trials, self.n_actions))

        # loop over all the trials
        for sim_i in tqdm(range(self.n_trials)):
            # reset the environment adn the agent
            self.environment.reset(sim_i)
            self.agent.reset()

            # perform a single simulation
            scores, pulled_arms, recommendation, estimators = self._run_simulation()

            scores_all[sim_i, :] = scores
            pulled_arms_all[sim_i, :] = pulled_arms
            recommendation_all[sim_i] = recommendation
            estimators_all[sim_i, :] = estimators

        self.results[self.agent.name] = {
            "scores": scores_all.tolist(),
            "pulled_arms": pulled_arms_all.tolist(),
            "recommendations": recommendation_all.tolist(),
            "estimators": estimators_all.tolist()
        }

    def _run_simulation(self):
        """
        This function has as aim the one of run a single trial of the experiment
        :return: the list of actions played
        """
        scores = np.zeros(self.horizon)
        pulled_arms = np.zeros(self.horizon)

        # loop over the given horizon
        for t in range(self.horizon):
            # pull an action
            action = self.agent.pull_arm()
            pulled_arms[t] = action

            # play the action on the environment and keep the reward
            reward = self.environment.step(action)
            scores[t] = reward

            # register the obtained reward
            self.agent.update(reward)

        # the single simulation has terminated, we need to save all the outputs
        recommendation = self.agent.get_recommendation()
        estimators = np.array(self.agent.get_estimator())

        return scores, pulled_arms, recommendation, estimators

    def save_output(self, config_name):
        # name = self.log_path + "/" + "experiment_" + str(config_name) + "_" + datetime.now().strftime('_%Y%m%d__%H_%M_%S') + ".json"
        name = self.log_path + "/" + config_name + ".json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.results, ensure_ascii=False, indent=4))
