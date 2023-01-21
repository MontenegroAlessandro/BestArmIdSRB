"""
This script implements several agents' classes, each one embedding a strategy for choosing an arm at each step and,
at the end, recommending the best empirical arm
"""

# Libraries
from abc import ABC, abstractmethod
from copy import deepcopy
from random import choice
import numpy as np
import math


# Base class
class Agent(ABC):
    def __init__(self, n_arms, horizon):
        """
        initializer of the base class agent
        :param n_arms: number of arms we explore
        """
        self.n_arms = n_arms
        self.last_pull = None
        self.horizon = horizon
        self.t = 0

    @abstractmethod
    def pull_arm(self):
        """
        This method selects an arm to pull, according to the strategy the agent is implementing
        :return: the arm to pull id
        """
        pass

    @abstractmethod
    def update(self, reward):
        """
        this method is thought to update the various parameters of the agent given the last observed reward
        :param reward: last observed reward from the last pull
        :return: None
        """
        pass

    def reset(self):
        """
        This method just reset the agent for a new experiment
        :return:"""
        self.t = 0
        self.last_pull = None

    @abstractmethod
    def get_estimator(self):
        pass

    @abstractmethod
    def get_recommendation(self):
        pass


# Class implementing the Uniform allocation strategy
class Uniform(Agent):
    def __init__(self, horizon, n_arms):
        super().__init__(n_arms=n_arms, horizon=horizon)
        self.name = "\\uniform"
        self.sum = np.zeros(self.n_arms)
        self.pulls = np.zeros(self.n_arms)

    def pull_arm(self):
        # self.last_pull = self.t % self.n_arms
        self.last_pull = (self.last_pull + 1) % self.n_arms if self.last_pull is not None else 0
        return self.last_pull

    def update(self, reward):
        self.t += 1
        self.pulls[self.last_pull] += 1

        arm = self.last_pull
        self.sum[arm] += reward

    def reset(self):
        super().reset()
        self.pulls = np.zeros(self.n_arms)
        self.sum = np.zeros(self.n_arms)

    def get_estimator(self):
        return self.sum / self.pulls

    def get_recommendation(self):
        return np.argmax(self.sum / self.pulls)


# Class implementing Uniform with teh recommendation applied to an h-wide window
class UniformSmooth(Agent):
    def __init__(self, horizon, n_arms, eps):
        super().__init__(horizon=horizon, n_arms=n_arms)
        self.name = "\\uniformsmooth"
        self.a = np.zeros(self.n_arms)
        self.window = np.zeros(self.n_arms)
        self.pulls = np.zeros(self.n_arms)
        self.scores = np.zeros((self.n_arms, self.horizon))
        self.eps = eps

    def pull_arm(self):
        self.last_pull = (self.last_pull + 1) % self.n_arms if self.last_pull is not None else 0
        return self.last_pull

    def update(self, reward):
        # update the time and the number of pulls of the pulled arm
        self.t += 1
        arm = int(self.last_pull)
        self.pulls[arm] += 1
        n = int(self.pulls[arm])

        # compute the new window
        h = math.floor(self.eps * n)

        # update the sum of last h scores parameter and the global window parameter
        if h == self.window[arm]:
            self.a[arm] += reward - self.scores[arm][n - h - 1]
        else:
            self.a[arm] += reward
        self.window[arm] = h
        self.scores[arm][n - 1] = reward

    def reset(self):
        super().reset()
        self.a = np.zeros(self.n_arms)
        self.window = np.zeros(self.n_arms)
        self.pulls = np.zeros(self.n_arms)
        self.scores = np.zeros((self.n_arms, self.horizon))

    def get_estimator(self):
        return self.a / self.window

    def get_recommendation(self):
        return np.argmax(self.a/self.window)


# Class implementing UCB-E by S. B.
class UcbE(Agent):
    def __init__(self, n_arms, exp_param, horizon):
        """
        Initializer of the class implementing the algorithm UCB-E
        :param n_arms: number of arms
        :param exp_param: exploration parameter
        :param horizon: horizon of the problem
        """
        super().__init__(n_arms=n_arms, horizon=horizon)
        self.name = "\\ucbbubeck"
        self.exp_param = exp_param
        self.B = np.inf * np.ones(self.n_arms)
        self.pulls = np.zeros(self.n_arms)
        self.means = np.zeros(self.n_arms)

    def pull_arm(self):
        # check if the horizon is over
        self.last_pull = np.argmax(self.B)
        return self.last_pull

    def update(self, reward):
        self.t += 1
        self.pulls[self.last_pull] += 1

        # update incrementally the empirical mean
        arm = self.last_pull
        n = self.pulls[arm]
        self.means[arm] = self.means[arm] + (reward - self.means[arm]) / n

        # update the beta parameter just for the arm
        self.B[arm] = self.means[arm] + math.sqrt(self.exp_param / self.pulls[arm])

    def reset(self):
        super().reset()
        self.B = np.inf * np.ones(self.n_arms)
        self.pulls = np.zeros(self.n_arms)
        self.means = np.zeros(self.n_arms)

    def get_recommendation(self):
        return np.argmax(self.means)

    def get_estimator(self):
        return self.means


# Class implementing the Stochastic Rising Bandits version of UCB-E
class UcbSRB(Agent):
    def __init__(self, n_arms, exp_param, horizon, eps, sigma):
        """
        Initializers of the class
        :param n_arms: number of arms to take into account
        :param exp_param: the exploration parameter as described in  UCB-E
        :param horizon: the pull budget
        :param eps: the epsilon parameter needed for the window computation
        """
        # super class initialization
        super().__init__(n_arms=n_arms, horizon=horizon)
        self.name = "\\ucbeshort"

        # mapping arguments
        self.exp_param = exp_param
        self.eps = eps
        self.sigma = sigma

        # useful variables for the execution
        self.warmup = True

        # useful vectors for the execution
        self.pulls = np.zeros(self.n_arms)
        self.window = np.zeros(self.n_arms)
        self.mu_check = np.zeros(self.n_arms)
        self.beta_check = np.zeros(self.n_arms)
        self.upper_bound = np.inf * np.ones(self.n_arms)
        self.a = np.zeros(self.n_arms)
        self.b = np.zeros(self.n_arms)
        self.c = np.zeros(self.n_arms)
        self.d = np.zeros(self.n_arms)
        self.scores = np.zeros((self.n_arms, self.horizon))

    def pull_arm(self):
        """
        Select the arm to pull, here just take the one with the highest upper bound or, if the budget is over, take the
        one marked as best arm (the one that scored the best reward)
        :return: the arm to pull index
        """
        # check if warmup (i.e., no arm with 0 window)
        if self.warmup:
            self.last_pull = self.t % self.n_arms
        # normal pull
        else:
            self.last_pull = np.argmax(self.upper_bound)
        return self.last_pull

    def update(self, reward):
        """
        This function is updating all the parameters of the last pulled agent
        :param reward: the last scored reward
        :return: None
        """
        # SRB-UCB update procedure
        self.t += 1
        arm = int(self.last_pull)
        self.pulls[arm] += 1

        n = int(self.pulls[arm])
        h = math.floor(self.eps * n)

        self.scores[arm][n-1] = reward

        if h == self.window[arm]:
            self.a[arm] += reward - self.scores[arm][n - h - 1]
            self.b[arm] += self.scores[arm][n - h - 1] - self.scores[arm][n - 2 * h - 1]
            self.c[arm] += n * reward - (n - h) * self.scores[arm][n - h - 1]
            self.d[arm] += n * self.scores[arm][n - h - 1] - (n - h) * self.scores[arm][n - 2 * h - 1]
        else:
            self.a[arm] += reward
            self.b[arm] += self.scores[arm][n - 2 * h]
            self.c[arm] += n * reward
            self.d[arm] += (n - h) * self.scores[arm][n - 2 * h] + self.b[arm]

        self.window[arm] = h
        a = self.a[arm]
        b = self.b[arm]
        c = self.c[arm]
        d = self.d[arm]

        self.mu_check[arm] = (1 / h) * (a + (self.horizon * (a - b) / h) - ((c - d) / h)) if h > 0 else 0
        self.beta_check[arm] = self.sigma * (self.horizon - n + h - 1) * math.sqrt(
            (10 * self.exp_param) / (math.pow(h, 3))) if h > 0 else 0
        self.upper_bound[arm] = self.mu_check[arm] + self.beta_check[arm] if h > 0 else 0

        # check if the warmup phase is over
        if 0 not in self.window:
            self.warmup = False

    def reset(self):
        """
        Reset all the parameter of the agent
        :return: None
        """
        # wiping procedure
        super().reset()
        self.pulls = np.zeros(self.n_arms)
        self.window = np.zeros(self.n_arms)
        self.mu_check = np.zeros(self.n_arms)
        self.beta_check = np.zeros(self.n_arms)
        self.upper_bound = [np.inf] * self.n_arms
        self.a = np.zeros(self.n_arms)
        self.b = np.zeros(self.n_arms)
        self.c = np.zeros(self.n_arms)
        self.d = np.zeros(self.n_arms)
        self.warmup = True
        self.scores = np.zeros((self.n_arms, self.horizon))

    def get_estimator(self):
        return self.mu_check

    def get_recommendation(self):
        return np.argmax(self.mu_check)


# Class Implementing Successive Rejects by S. B.
class Sr(Agent):
    def __init__(self, horizon, n_arms):
        # class Agent initialization
        super().__init__(n_arms=n_arms, horizon=horizon)
        self.name = "\\srbubeck"

        # new variables
        self.phases_pulls = self._phases_len_computation()
        self.phase_id = 1

        # vectors
        self.pulls = np.zeros(self.n_arms)
        self.mu_hat = np.zeros(self.n_arms)

        # dictionaries
        self.living_pulls = dict()
        self.living_mu_hat = dict()
        for elem in range(self.n_arms):
            self.living_pulls[elem] = 0
            self.living_mu_hat[elem] = 0

        return

    def pull_arm(self):
        # take a living arm that has not been sufficiently pulled
        lp_copy = deepcopy(self.living_pulls)
        if len(lp_copy) > 1:
            arm_id = None
            while arm_id is None:
                min_pulls = min(lp_copy.values())
                min_ids = [k for k in lp_copy if lp_copy[k] == min_pulls]
                arm_id = choice(min_ids)
                if lp_copy[arm_id] == self.phases_pulls[self.phase_id - 1]:
                    del lp_copy[arm_id]
                    arm_id = None
        else:
            arm_id = min(lp_copy)

        # select the arm
        self.last_pull = arm_id
        return self.last_pull

    def update(self, reward):
        # update the time and the number of pulls of the pulled arm
        self.t += 1
        self.pulls[self.last_pull] += 1
        self.living_pulls[self.last_pull] += 1

        # update the sum of last h scores parameter and the global window parameter
        arm = int(self.last_pull)

        # update the mu_hat for the arm
        self.mu_hat[arm] = self.mu_hat[arm] + (reward - self.mu_hat[arm])/self.pulls[arm]
        self.living_mu_hat[arm] = self.living_mu_hat[arm] + (reward - self.living_mu_hat[arm]) / self.living_pulls[arm]

        # check if the current phase is over
        phase_over = True
        for i in self.living_mu_hat:
            if self.phase_id == self.n_arms:
                phase_over = False
                break
            elif self.pulls[i] < self.phases_pulls[self.phase_id - 1]:
                phase_over = False
                break

        # Elimination procedure
        # if the phase is over delete an arm and update the phase_id
        if phase_over and (self.phase_id <= self.n_arms - 1):
            self.phase_id += 1
            min_value = min(self.living_mu_hat.values())
            min_ids = [k for k in self.living_mu_hat if self.living_mu_hat[k] == min_value]
            to_del = choice(min_ids)

            del self.living_mu_hat[to_del]
            del self.living_pulls[to_del]

    def reset(self):
        super().reset()
        self.phase_id = 1
        self.pulls = np.zeros(self.n_arms)
        self.mu_hat = np.zeros(self.n_arms)
        self.living_pulls = {}
        self.living_mu_hat = {}
        for elem in range(self.n_arms):
            self.living_pulls[elem] = 0
            self.living_mu_hat[elem] = 0

    def get_recommendation(self):
        return list(self.living_pulls.keys())[0]

    def get_estimator(self):
        return self.mu_hat

    def _phases_len_computation(self):
        # compute the log_bar
        log_bar = 0.5
        for i in range(2, self.n_arms + 1):
            log_bar += 1/i

        # build the list with the phases pulls
        phases_pulls = np.zeros(self.n_arms - 1)

        for j in range(1, self.n_arms):
            phases_pulls[j-1] = math.ceil((1/log_bar) * (self.horizon - self.n_arms) / (self.n_arms + 1 - j))

        return phases_pulls


# Class implementing the Stochastic Rising Bandit version of Successive Rejects
class SRSrb(Agent):
    def __init__(self, horizon, n_arms, eps):
        # class Agent initialization
        super().__init__(n_arms=n_arms, horizon=horizon)
        self.name = "\\succrejectshort"

        # new variables
        self.eps = eps
        self.phases_pulls = self._phases_len_computation()
        self.phase_id = 1

        # vectors
        self.pulls = np.zeros(self.n_arms)
        self.window = np.zeros(self.n_arms)
        self.mu_hat = np.zeros(self.n_arms)
        self.a = np.zeros(self.n_arms)
        self.scores = np.zeros((self.n_arms, self.horizon))

        # dictionaries
        self.living_pulls = dict()
        self.living_mu_hat = dict()
        for elem in range(self.n_arms):
            self.living_pulls[elem] = 0
            self.living_mu_hat[elem] = 0

    def pull_arm(self):
        # take a living arm that has not been sufficiently pulled
        lp_copy = deepcopy(self.living_pulls)
        if len(lp_copy) > 1:
            arm_id = None
            while arm_id is None:
                min_pulls = min(lp_copy.values())
                min_ids = [k for k in lp_copy if lp_copy[k] == min_pulls]
                arm_id = choice(min_ids)
                if lp_copy[arm_id] == self.phases_pulls[self.phase_id - 1]:
                    del lp_copy[arm_id]
                    arm_id = None
        else:
            arm_id = min(lp_copy)

        # select the arm
        self.last_pull = arm_id
        return self.last_pull

    def update(self, reward):
        # update the time and the number of pulls of the pulled arm
        self.t += 1
        self.pulls[self.last_pull] += 1
        self.living_pulls[self.last_pull] += 1

        # compute the new window
        h = math.floor(self.eps * self.pulls[self.last_pull])

        # update the sum of last h scores parameter and the global window parameter
        arm = int(self.last_pull)
        n = int(self.pulls[arm])
        if h == self.window[arm]:
            self.a[arm] += reward - self.scores[arm][n - h - 1]
        else:
            self.a[arm] += reward
        self.window[arm] = h
        self.scores[arm][n-1] = reward

        # update the mu_hat for the arm
        self.mu_hat[arm] = self.a[self.last_pull] / h if h > 0 else 0
        self.living_mu_hat[arm] = self.a[self.last_pull] / h if h > 0 else 0

        # check if the current phase is over
        phase_over = True
        for i in self.living_mu_hat:
            if self.phase_id == self.n_arms:
                phase_over = False
                break
            elif self.pulls[i] < self.phases_pulls[self.phase_id - 1]:
                phase_over = False
                break

        # Elimination procedure
        # if the phase is over delete an arm and update the phase_id
        if phase_over and (self.phase_id <= self.n_arms - 1):
            self.phase_id += 1
            min_value = min(self.living_mu_hat.values())
            min_ids = [k for k in self.living_mu_hat if self.living_mu_hat[int(k)] == min_value]
            to_del = choice(min_ids)

            del self.living_mu_hat[to_del]
            del self.living_pulls[to_del]

    def reset(self):
        super().reset()
        self.phase_id = 1

        # vectors
        self.pulls = np.zeros(self.n_arms)
        self.window = np.zeros(self.n_arms)
        self.mu_hat = np.zeros(self.n_arms)
        self.a = np.zeros(self.n_arms)
        self.scores = np.zeros((self.n_arms, self.horizon))

        # dictionaries
        self.living_pulls = dict()
        self.living_mu_hat = dict()
        for elem in range(self.n_arms):
            self.living_pulls[elem] = 0
            self.living_mu_hat[elem] = 0

    def get_estimator(self):
        return self.mu_hat

    def get_recommendation(self):
        return list(self.living_pulls.keys())[0]

    def _phases_len_computation(self):
        # compute the log_bar
        log_bar = 0.5
        for i in range(2, self.n_arms + 1):
            log_bar += 1/i

        # build the list with the phases pulls
        phases_pulls = np.zeros(self.n_arms - 1)

        for j in range(1, self.n_arms):
            phases_pulls[j-1] = math.ceil((1/log_bar) * (self.horizon - self.n_arms) / (self.n_arms + 1 - j))

        return phases_pulls


# Class implementing Prob1 by Yasin
class Prob1(Agent):
    def __init__(self, horizon, n_arms):
        # class Agent initialization
        super().__init__(n_arms=n_arms, horizon=horizon)
        self.name = "\\probone"

        # new variables
        self.log_bar = self._compute_log_bar()

        # vectors
        self.pulls = np.zeros(self.n_arms)
        self.rank = np.ones(self.n_arms)
        self.G = np.zeros(self.n_arms)
        self.probabilities = (1 / self.n_arms) * np.ones(self.n_arms)

    def pull_arm(self):
        self.last_pull = np.random.choice(range(self.n_arms), p=self.probabilities)
        return self.last_pull

    def update(self, reward):
        # update the time
        self.t += 1

        # update the parameters of the pulled agent
        arm = int(self.last_pull)
        self.pulls[arm] += 1
        self.G[arm] += reward / self.probabilities[arm]

        # Update the rank
        order = self.G.argsort()
        for i in range(len(order)):
            self.rank[order[i]] = int(len(order) - i)

        # update the probabilities
        self.probabilities = np.power(self.rank * self.log_bar, -1)

    def reset(self):
        super().reset()
        self.pulls = np.zeros(self.n_arms)
        self.rank = np.ones(self.n_arms)
        self.G = np.zeros(self.n_arms)
        self.probabilities = (1 / self.n_arms) * np.ones(self.n_arms)

    def get_recommendation(self):
        return np.argmax(self.G)

    def get_estimator(self):
        return self.G

    def _compute_log_bar(self):
        res = 0
        for i in range(1, self.n_arms + 1):
            res += 1/i
        return res


# Explore then Commit by Cella
class Etc(Agent):
    def __init__(self, horizon, n_arms, rho, ub_alpha):
        # class Agent initialization
        super().__init__(n_arms=n_arms, horizon=horizon)
        self.name = "\\etccella"
        self.delta = 1/self.horizon
        self.rho = rho
        self.ub_alpha = ub_alpha
        self.phase_id = 1
        self.exploration_phase = True
        self.i_out = None

        # vectors
        self.mu_hat = np.zeros(self.n_arms)
        self.alpha_hat = np.zeros(self.n_arms)
        self.beta_hat = np.zeros(self.n_arms)
        self.pulls = np.zeros(self.n_arms)
        self.x_hat = np.zeros(self.n_arms)
        self.x_tilde = np.zeros(self.n_arms)
        self.a = np.zeros(self.n_arms)
        self.b = np.zeros(self.n_arms)
        self.harmonic_pulls_a = np.zeros(self.n_arms)
        self.harmonic_pulls_b = np.zeros(self.n_arms)
        self.scores = np.zeros((self.n_arms, self.horizon))

    def pull_arm(self):
        if self.exploration_phase:
            self.last_pull = self.t % self.n_arms if self.last_pull is not None else 0
        else:
            self.last_pull = self.i_out
        return self.last_pull

    def update(self, reward):
        # remap the reward in the rested rotting setting
        inv_rew = 1 - reward

        # update the agent parameter
        arm = int(self.last_pull)

        self.t += 1
        self.pulls[arm] += 1

        n = int(self.pulls[arm])
        self.scores[arm][n-1] = inv_rew
        h = math.floor(0.5 * n)

        if n % 2 != 0:
            self.a[arm] += inv_rew - self.scores[arm][n - h - 1]
            # self.b[arm] += self.scores[arm][n - h - 1] - self.scores[arm][n - 2 * h - 1]
            self.b[arm] += self.scores[arm][n - h - 1]
            self.harmonic_pulls_a[arm] += math.pow(n, -self.rho) - math.pow(n - h, -self.rho) if n-h != 0 else math.pow(n, -self.rho)
            # self.harmonic_pulls_b[arm] += math.pow(n - h, -self.rho) - math.pow(n - 2*h, -self.rho) if n-h != 0 else math.pow(n, -self.rho)
            self.harmonic_pulls_b[arm] += math.pow(n - h, -self.rho) if n-h != 0 else 0
        else:
            self.a[arm] += inv_rew
            # self.b[arm] += self.scores[arm][n - 2 * h]
            self.harmonic_pulls_a[arm] += math.pow(n, -self.rho) if n != 0 else 0
            # self.harmonic_pulls_b[arm] += math.pow(n - 2*h, -self.rho) if n - 2*h != 0 else 0

        self.x_hat[arm] = self.b[arm]/h if h != 0 else 0
        self.x_tilde[arm] = self.a[arm] / h if h != 0 else 0
        self.alpha_hat[arm] = (h * (self.x_hat[arm] - self.x_tilde[arm])) / (self.harmonic_pulls_b[arm] - self.harmonic_pulls_a[arm]) if h != 0 else 0
        self.beta_hat[arm] = self.x_hat[arm] - self.harmonic_pulls_b[arm] * (self.alpha_hat[arm] / h) if h != 0 else 0

        # Exploration phase
        if self.exploration_phase:
            # check if we pulled all the arms once in the current phase
            if self.last_pull == self.n_arms - 1:
                tau = self.horizon - self.phase_id*(self.n_arms - 1)

                self.mu_hat = self.beta_hat + (self.alpha_hat / (math.pow(tau, self.rho)))

                for i in range(self.n_arms):
                    mask = np.ones(self.n_arms, dtype=bool)
                    mask[i] = False
                    if self.mu_hat[i] < min(self.mu_hat[mask]) - 2 * self._compute_cb(self.phase_id):
                        self.i_out = i
                        self.exploration_phase = False
                        break

                if self.exploration_phase:
                    self.phase_id += 1
        else:
            tau = self.horizon - self.pulls[arm]*(self.n_arms - 1)
            self.mu_hat[arm] = self.beta_hat[arm] + (self.alpha_hat[arm] / (math.pow(tau, self.rho)))

    def reset(self):
        super().reset()
        self.phase_id = 1
        self.exploration_phase = True
        self.i_out = None

        # vectors
        self.mu_hat = np.zeros(self.n_arms)
        self.alpha_hat = np.zeros(self.n_arms)
        self.beta_hat = np.zeros(self.n_arms)
        self.pulls = np.zeros(self.n_arms)
        self.x_hat = np.zeros(self.n_arms)
        self.x_tilde = np.zeros(self.n_arms)
        self.a = np.zeros(self.n_arms)
        self.b = np.zeros(self.n_arms)
        self.harmonic_pulls_a = np.zeros(self.n_arms)
        self.harmonic_pulls_b = np.zeros(self.n_arms)
        self.scores = np.zeros((self.n_arms, self.horizon))

    def get_estimator(self):
        return self.mu_hat

    def get_recommendation(self):
        return self.i_out

    def _compute_cb(self, tau):
        first_term = (10 * math.pow(math.sqrt(self.ub_alpha) + 1, 2)) / (self.rho * (1 - self.rho))
        second_term = math.log((tau * self.n_arms * self.horizon) / self.delta) / tau
        third_term = math.sqrt(math.pow(tau, -1) * math.log((tau * self.n_arms * self.horizon) / self.delta))
        return first_term * (second_term + third_term)


# Rest-Sure by Cella
class RestSure(Agent):
    def __init__(self, n_arms, horizon, ub_alpha, rho):
        super().__init__(n_arms=n_arms, horizon=horizon)
        self.name = "\\restsurecella"
        self.delta = 1 / self.horizon
        self.ub_alpha = ub_alpha
        self.rho = rho
        self.A = np.arange(self.n_arms)
        self.phase_id = 1
        self.tau_out = self.horizon
        self.exploration_phase = True
        self.i_out = None

        # vectors
        self.mu_hat = np.zeros(self.n_arms)
        self.alpha_hat = np.zeros(self.n_arms)
        self.beta_hat = np.zeros(self.n_arms)
        self.pulls = np.zeros(self.n_arms)
        self.x_hat = np.zeros(self.n_arms)
        self.x_tilde = np.zeros(self.n_arms)
        self.a = np.zeros(self.n_arms)
        self.b = np.zeros(self.n_arms)
        self.harmonic_pulls_a = np.zeros(self.n_arms)
        self.harmonic_pulls_b = np.zeros(self.n_arms)
        self.scores = np.zeros((self.n_arms, self.horizon))

    def pull_arm(self):
        if self.exploration_phase:
            self.last_pull = self.A[self.t % len(self.A)]
        else:
            self.last_pull = self.i_out
        return self.last_pull

    def update(self, reward):
        # remap the reward in the rested rotting setting
        inv_rew = 1 - reward

        # update the agent parameter
        arm = int(self.last_pull)

        self.t += 1
        self.pulls[arm] += 1

        n = int(self.pulls[arm])
        self.scores[arm][n-1] = inv_rew
        h = math.floor(0.5 * n)

        if n % 2 != 0:
            self.a[arm] += inv_rew - self.scores[arm][n - h - 1]
            self.b[arm] += self.scores[arm][n - h - 1]
            self.harmonic_pulls_a[arm] += math.pow(n, -self.rho) - math.pow(n - h, -self.rho) if n - h != 0 else math.pow(n, -self.rho)
            self.harmonic_pulls_b[arm] += math.pow(n - h, -self.rho) if n - h != 0 else 0
        else:
            self.a[arm] += inv_rew
            self.harmonic_pulls_a[arm] += math.pow(n, -self.rho) if n != 0 else 0

        self.x_hat[arm] = self.b[arm] / h if h != 0 else 0
        self.x_tilde[arm] = self.a[arm] / h if h != 0 else 0
        self.alpha_hat[arm] = (h * (self.x_hat[arm] - self.x_tilde[arm])) / (self.harmonic_pulls_b[arm] - self.harmonic_pulls_a[arm]) if h != 0 else 0
        self.beta_hat[arm] = self.x_hat[arm] - self.harmonic_pulls_b[arm] * (self.alpha_hat[arm] / h) if h != 0 else 0

        tau = self.horizon + self.phase_id - self.t
        if self.exploration_phase and self.last_pull == self.A[len(self.A) - 1]:
            # check the part w.h.p.
            self.mu_hat = self.beta_hat + (self.alpha_hat / (math.pow(tau, self.rho)))
            for i in self.A:
                mask = np.ones(self.n_arms, dtype=bool)
                mask[i] = False
                if self.mu_hat[i] < min(self.mu_hat[mask]) - 2 * self._compute_cb(self.phase_id):
                    self.i_out = i
                    self.exploration_phase = False
                    return

            # check the part "no advantage in learning i*"
            tmp = self.beta_hat + (self.alpha_hat / ((tau - len(self.A) + 1)**self.rho))
            if min(tmp) - 2*self._compute_cb(self.phase_id) > min(self.mu_hat):
                self.i_out = choice(self.A)
                self.exploration_phase = False
                return

            # Elimination phase
            if self.t % 30 == 0:
                new_A = self.A
                trash = []
                for i in new_A:
                    count = 0
                    for j in new_A:
                        if i in trash or i >= j or j in trash:
                            continue
                        for m in range(self.phase_id, tau + 1):
                            tmp1 = self.beta_hat[i] + (self.alpha_hat[i] / (math.pow(m, self.rho)))
                            tmp2 = self.beta_hat[j] + (self.alpha_hat[j] / (math.pow(m, self.rho)))
                            if tmp1 - tmp2 > 2*self._compute_cb(self.phase_id):
                                count += 1
                        if count == (tau - self.phase_id):
                            trash.append(i)
                            break
                        else:
                            count = 0
                self.phase_id += 1
                self.A = [arm for arm in new_A if arm not in trash]

            if len(self.A) == 1:
                self.exploration_phase = False
                self.i_out = self.A[0]

        else:
            tau = self.horizon - self.pulls[arm]*(self.n_arms - 1)
            self.mu_hat[arm] = self.beta_hat[arm] + (self.alpha_hat[arm] / (math.pow(tau, self.rho)))

    def reset(self):
        super().reset()
        self.A = np.arange(self.n_arms)
        self.phase_id = 1
        self.tau_out = self.horizon
        self.exploration_phase = True
        self.i_out = None

        # vectors
        self.mu_hat = np.zeros(self.n_arms)
        self.alpha_hat = np.zeros(self.n_arms)
        self.beta_hat = np.zeros(self.n_arms)
        self.pulls = np.zeros(self.n_arms)
        self.x_hat = np.zeros(self.n_arms)
        self.x_tilde = np.zeros(self.n_arms)
        self.a = np.zeros(self.n_arms)
        self.b = np.zeros(self.n_arms)
        self.harmonic_pulls_a = np.zeros(self.n_arms)
        self.harmonic_pulls_b = np.zeros(self.n_arms)
        self.scores = np.zeros((self.n_arms, self.horizon))

    def get_estimator(self):
        return self.mu_hat

    def get_recommendation(self):
        return self.i_out

    def _compute_cb(self, tau):
        first_term = (10 * math.pow(math.sqrt(self.ub_alpha) + 1, 2)) / (self.rho * (1 - self.rho))
        second_term = math.log((tau * self.n_arms * self.horizon) / self.delta) / tau
        third_term = math.sqrt(math.pow(tau, -1) * math.log((tau * self.n_arms * self.horizon) / self.delta))
        return first_term * (second_term + third_term)
