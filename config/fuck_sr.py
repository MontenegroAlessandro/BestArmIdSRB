import io
import json
from oracle import *

# arms
f1 = lambda x: .0025 * x if .0025 * x < 1 else 1
f2 = lambda x: .47 * (1 - 10 / (10 + x))
arms = [f1, f2]

# common stuff
path = 'experiments'
# from 600 to 800 wins sr-srb
horizon = 3000
n_trials = 100
sigma = 0.01
eps = .25
convergence_points = [1, .47]

# compose the dictionary for the Agent Uniform
param_agent_unif = dict(
    n_arms=len(arms),
    horizon=horizon
)

# compose the dictionary for the agent Uniform Smooth
param_agent_unif_smooth = dict(
    horizon=horizon,
    n_arms=len(arms),
    eps=eps
)

# compose the dictionary for the Agent UCB
oracle_agent_ucb_e = OracleUCBe(arms=arms, horizon=horizon)
param_agent_ucb = dict(
    n_arms=len(arms),
    horizon=horizon,
    exp_param=oracle_agent_ucb_e.optimal_a
)

# compose the dictionary for the Agent UCB_SRB
oracle_ucb_srb = OracleUCBSrb(arms=arms, convergence_points=convergence_points, eps=eps, horizon=horizon, sigma=sigma,
                              beta=2)
param_agent_ucb_srb = dict(
    n_arms=len(arms),
    exp_param=oracle_ucb_srb.optimal_a,
    horizon=horizon,
    eps=eps,
    sigma=sigma
)

# compose the dictionary for the Agent SR
param_agent_sr = dict(
    n_arms=len(arms),
    horizon=horizon
)

# compose the dictionary for the Agent SR_SRB
oracle_sr_srb = OracleSR(arms=arms, convergence_points=convergence_points, eps=eps, horizon=horizon, sigma=sigma,
                         beta=2)
param_agent_sr_srb = dict(
    n_arms=len(arms),
    horizon=horizon,
    eps=eps,
)

# compose the dictionary for the Agent Prob1
param_agent_prob = dict(
    n_arms=len(arms),
    horizon=horizon
)

# compose the dictionary for the Agent Etc
param_agent_etc = dict(
    n_arms=len(arms),
    horizon=horizon,
    rho=.98,
    ub_alpha=1
)

# compose the dictionary for the Agent RestSure
param_agent_rest_sure = dict(
    n_arms=len(arms),
    horizon=horizon,
    rho=.8,
    ub_alpha=1
)

# Dictionary for the Environment
param_env = dict(
    horizon=horizon,
    actions=None,
    noise=sigma
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
with io.open(f'fuck_sr.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(param, ensure_ascii=False, indent=4))

oracle_agent_ucb_e.represent()
oracle_ucb_srb.represent()
oracle_sr_srb.represent()