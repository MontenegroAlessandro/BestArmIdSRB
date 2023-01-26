import io, json, sys
import json
import numpy as np
from oracle import OracleUCBSrb, OracleUCBe

base_path = '/Users/ale/Desktop/BAISRB/environment/imdb/'
names = ['NN112', 'NN1', 'NN2', 'NN22', 'NN222', 'OGD', 'LR']
curves = [np.load(base_path + i) for i in names]

f1 = lambda x: curves[0][int(x)] if x < len(curves[0]) else curves[0][-1]
f2 = lambda x: curves[1][int(x)] if x < len(curves[1]) else curves[1][-1]
f3 = lambda x: curves[2][int(x)] if x < len(curves[2]) else curves[2][-1]
f4 = lambda x: curves[3][int(x)] if x < len(curves[3]) else curves[3][-1]
f5 = lambda x: curves[4][int(x)] if x < len(curves[4]) else curves[4][-1]
f6 = lambda x: curves[5][int(x)] if x < len(curves[5]) else curves[5][-1]
f7 = lambda x: curves[6][int(x)] if x < len(curves[6]) else curves[6][-1]
arms = [f1, f2, f3, f4, f5, f6, f7]

horizon = int(sys.argv[1])
n_trials = 1
# sigma = 0.2
sigma = float(sys.argv[2])
# eps = .4
eps = float(sys.argv[3])
convergence_points = [1, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8]
n_arms = len(arms)

# compose the dictionary for the Agent Uniform
param_agent_unif = dict(
    n_arms=n_arms,
    horizon=horizon
)

# compose the dictionary for the agent Uniform Smooth
param_agent_unif_smooth = dict(
    horizon=horizon,
    n_arms=n_arms,
    eps=eps
)

# compose the dictionary for the Agent UCB
oracle_agent_ucb_e = OracleUCBe(arms=arms, horizon=horizon)
param_agent_ucb = dict(
    n_arms=n_arms,
    horizon=horizon,
    exp_param=oracle_agent_ucb_e.optimal_a
)

# compose the dictionary for the Agent UCB_SRB
oracle_ucb_srb = OracleUCBSrb(arms=arms, convergence_points=convergence_points, eps=eps, horizon=horizon, sigma=sigma,
                              beta=1.5)
param_agent_ucb_srb = dict(
    n_arms=n_arms,
    exp_param=oracle_ucb_srb.optimal_a,
    horizon=horizon,
    eps=eps,
    sigma=sigma
)

# compose the dictionary for the Agent SR
param_agent_sr = dict(
    n_arms=n_arms,
    horizon=horizon
)

# compose the dictionary for the Agent SR_SRB
param_agent_sr_srb = dict(
    n_arms=n_arms,
    horizon=horizon,
    eps=eps,
)

# compose the dictionary for the Agent Prob1
param_agent_prob = dict(
    n_arms=n_arms,
    horizon=horizon
)

# compose the dictionary for the Agent Etc
param_agent_etc = dict(
    n_arms=n_arms,
    horizon=horizon,
    rho=.8,
    ub_alpha=1
)

# compose the dictionary for the Agent RestSure
param_agent_rest_sure = dict(
    n_arms=n_arms,
    horizon=horizon,
    rho=.8,
    ub_alpha=1
)

# Dictionary for the Environment
param_env = dict(
    horizon=horizon
)

# Dictionary final
param = dict(
    horizon=horizon,
    n_trials=n_trials,
    env=param_env,
    agent_unif=param_agent_unif,
    agent_unif_smooth=param_agent_unif_smooth,
    agent_ucb=param_agent_ucb,
    agent_ucb_srb=param_agent_ucb_srb,
    agent_sr=param_agent_sr,
    agent_sr_srb=param_agent_sr_srb,
    agent_prob=param_agent_prob,
    agent_etc=param_agent_etc,
    agent_rest_sure=param_agent_rest_sure
)

# write the JSON file
with io.open(f'imdb.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(param, ensure_ascii=False, indent=4))