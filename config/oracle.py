import math
import random


class OracleUCBSrb:
    def __init__(self, arms=None, convergence_points=None, eps=-1, horizon=0, sigma=0, beta=0):
        # arguments
        self.arms = arms
        self.convergence_points = convergence_points
        self.eps = eps
        self.horizon = horizon
        self.sigma = sigma
        self.beta = beta

        # new parameters
        self.name = "UCB-SRB"
        self.mus = self.compute_mus()
        self.mus_inf = self.convergence_points
        self.gaps = self.compute_gaps()
        self.gaps_inf = self.compute_gaps_inf()
        self.phi = self.compute_phi()
        self.optimal_a = self.compute_exploration_parameter()
        self.theoretical_horizon = self.compute_theoretical_horizon()
        self.error_ub = self.compute_error_ub()

    def compute_exploration_parameter(self):
        # compute the numerator
        if self.beta >= 1.5:
            num = (pow(self.horizon / self.phi, 0.5) - pow(1 - 2 * self.eps, -self.beta)) ** 2
        elif self.beta > 1:
            num = (pow(self.horizon, self.beta - 1) / pow(self.phi, self.beta) - pow(1 - 2 * self.eps, -self.beta)) ** 2
        else:
            num = 0

        # compute the denominator
        den = 4 * pow(self.sigma, 2) * pow(self.eps, -3)

        # compute a
        a = num / den
        return a

    def compute_mus(self):
        mus = []
        for elem in self.arms:
            mus.append(elem(self.horizon))
        return mus

    def compute_gaps(self):
        # find the mu_star
        mu_star = max(self.mus)

        # compute the optimality gaps
        gaps = []
        for elem in self.mus:
            gaps.append(mu_star - elem)
        return gaps

    def compute_phi(self):
        # compute phi_{2/3}
        phi = 0
        for elem in self.gaps:
            if elem != 0:
                phi += pow(elem, -2 / 3)
        return phi

    def compute_gaps_inf(self):
        gaps_inf = []
        mu_star_inf = max(self.mus_inf)
        for i in range(len(self.mus_inf)):
            gaps_inf.append(mu_star_inf - self.mus_inf[i])
        return gaps_inf

    def compute_theoretical_horizon(self):
        horizon_i = [0] * len(self.arms)
        for i in range(len(self.arms)):
            if self.gaps_inf[i] != 0:
                horizon_i[i] = math.pow(.5 * (self.beta - 1) * self.gaps_inf[i], 1 / (-self.beta + 1))
            else:
                horizon_i[i] = 0
        return math.ceil(max(horizon_i))

    def compute_error_ub(self):
        return 2 * self.horizon * len(self.arms) * math.exp(-0.1*self.optimal_a)

    def represent(self):
        print(self.name)
        print("Gaps: " + str(self.gaps))
        print("Infinity Gaps: " + str(self.gaps_inf))
        print("Phi 2/3: " + str(self.phi))
        print("Optimal a: " + str(self.optimal_a))
        print("Error Upper Bound: " + str(self.error_ub))
        print("Theoretical min horizon: " + str(self.theoretical_horizon))
        print("########################################\n")


class OracleUCBe:
    def __init__(self, arms=None, horizon=None):
        self.name = "UCB-E"
        self.arms = arms
        self.horizon = horizon

        self.mus = self.compute_mus()
        self.gaps = self.compute_gaps()
        self.h_1 = self.compute_h1()
        self.optimal_a = (25*(self.horizon-len(self.arms))) / (36*self.h_1)
        self.error_ub = 2 * len(self.arms) * self.horizon * math.exp(-(2 * self.optimal_a) / 25)

    def compute_mus(self):
        mus = []
        for elem in self.arms:
            mus.append(elem(self.horizon))
        return mus

    def compute_gaps(self):
        # find the mu_star
        mu_star = max(self.mus)

        # compute the optimality gaps
        gaps = []
        for elem in self.mus:
            gaps.append(mu_star - elem)
        return gaps

    def compute_h1(self):
        h_1 = 0
        for elem in self.gaps:
            if elem != 0:
                h_1 += math.pow(elem, -2)
        return h_1

    def represent(self):
        print(self.name)
        print("Gaps: " + str(self.gaps))
        print("H1: " + str(self.h_1))
        print("Optimal a: " + str(self.optimal_a))
        print("Error Upper Bound: " + str(self.error_ub))
        print("########################################\n")


class OracleSR:
    def __init__(self, arms, horizon, convergence_points, eps, beta, sigma):
        self.name = "SR"
        self.arms = arms
        self.horizon = horizon
        self.eps = eps
        self.sigma = sigma
        self.beta = beta
        self.convergence_points = convergence_points

        self.mus = self.compute_mus()
        self.mus_inf = self.convergence_points
        self.gaps = self.compute_gaps()
        self.gaps_inf = self.compute_gaps_inf()
        self.log_bar = self.compute_log_bar()
        self.theoretical_horizon = self.compute_theoretical_horizon()
        self.h_2 = self.compute_h_2()
        self.error_ub = 0.5 * len(self.arms) * (len(self.arms)-1) * math.exp(-self.eps/(8 * math.pow(self.sigma, 2)) * (self.horizon-len(self.arms))/(self.log_bar*self.h_2))
        self.error_bubeck = 0.5 * len(self.arms) * (len(self.arms)-1)*math.exp(-(self.horizon - len(self.arms))/(self.log_bar*self.h_2))

    def compute_mus(self):
        mus = []
        for elem in self.arms:
            mus.append(elem(self.horizon))
        return mus

    def compute_gaps(self):
        # find the mu_star
        mu_star = max(self.mus)

        # compute the optimality gaps
        gaps = []
        for elem in self.mus:
            gaps.append(mu_star - elem)
        return gaps

    def compute_gaps_inf(self):
        gaps_inf = []
        mu_star_inf = max(self.mus_inf)
        for i in range(len(self.mus_inf)):
            gaps_inf.append(mu_star_inf - self.mus_inf[i])
        return gaps_inf

    def compute_theoretical_horizon(self):
        gaps_sr = {}
        for i in range(1, len(self.arms) + 1):
            gaps_sr[i] = self.gaps_inf[i - 1]

        T_isr = [0] * (len(self.arms) - 1)
        for i in range(len(self.arms) - 1):
            j = i + 1
            ph_id = len(self.arms) + 1 - j
            delta = max(gaps_sr.values())

            key = [k for k in gaps_sr if gaps_sr[k] == delta]
            key_max = random.choice(key)
            del gaps_sr[key_max]

            T_isr[i] = math.pow(delta, 1 / (1 - self.beta)) / (
                        math.pow(2, 1 / 1 - self.beta) * math.pow(self.log_bar * ph_id, self.beta / (1 - self.beta)))

        return math.ceil(max(T_isr))

    def compute_h_2(self):
        gaps = self.gaps
        gaps.sort()
        H_list = []
        for i in range(len(self.arms)):
            H_list.append((i + 1) * math.pow(gaps[i], -2)) if gaps[i] != 0 else 0
        return max(H_list)

    def compute_log_bar(self):
        log_bar = 0.5
        for i in range(2, len(self.arms) + 1):
            log_bar += math.pow(i, -1)
        return log_bar

    def represent(self):
        print(self.name)
        print("Gaps: " + str(self.gaps))
        print("Infinity Gaps: " + str(self.gaps_inf))
        print("Phi 2/3: " + str(self.h_2))
        print("Error Upper Bound: " + str(self.error_ub))
        print("Error Upper Bound Bubeck: " + str(self.error_bubeck))
        print("Theoretical min horizon: " + str(self.theoretical_horizon))
        print("########################################\n")
