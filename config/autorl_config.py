import io, json, sys
import json
import numpy as np
from oracle import OracleUCBSrb, OracleUCBe
import matplotlib.pyplot as plt

arms_dict = dict()
config_name = "../environment/autorl_arms/"
with open(config_name + 'ddpg' + '.json') as json_file:
    arms_dict[0] = json.load(json_file)["ddpg"]

with open(config_name + 'sac' + '.json') as json_file:
    arms_dict[1] = json.load(json_file)["sac"]

#with open(config_name + 'ppo' + '.json') as json_file:
#    arms_dict[0] = json.load(json_file)["ppo"]

# normalize values in 0-1
max_rew = 0
min_rew = -140
arms_dict[0] = [((rew - min_rew) / (max_rew - min_rew)) for rew in arms_dict[0]]
arms_dict[1] = [((rew - min_rew) / (max_rew - min_rew)) for rew in arms_dict[1]]
# arms_dict[2] = [((rew - min_rew) / (max_rew - min_rew)) for rew in arms_dict[2]]

f1 = lambda x: arms_dict[0][int(x)] if x < len(arms_dict[0]) else arms_dict[0][-1]
f2 = lambda x: arms_dict[1][int(x)] if x < len(arms_dict[1]) else arms_dict[1][-1]
#f3 = lambda x: arms_dict[2][int(x)] if x < len(arms_dict[2]) else arms_dict[2][-1]
arms = [f1, f2]

horizon = int(sys.argv[1])
n_trials = 100
# sigma = 0.2
sigma = float(sys.argv[2])
# eps = .4
eps = float(sys.argv[3])
convergence_points = [0.8993156172383346, 0.8935319944176193]
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
    horizon=horizon,
    actions=None
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
with io.open(f'autorl.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(param, ensure_ascii=False, indent=4))
