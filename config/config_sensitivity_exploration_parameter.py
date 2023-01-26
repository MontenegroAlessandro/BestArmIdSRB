# Configuration for UCB-SRB
import io, json, sys
import json
from oracle import OracleUCBSrb

# arms
f1 = lambda x: 1 - 37/(37+x)
f2 = lambda x: .88*(1 - 10/(10+x))
f3 = lambda x: .78*(1 - 1/(1+x))
f4 = lambda x: .7*(1 - 10/(10+x))
f5 = lambda x: .5*(1 - 20/(20+x))
arms = [f1, f2, f3, f4, f5]

# common stuff
horizon = int(sys.argv[1])
n_trials = 1000
sigma = 0.01
eps = .25
convergence_points = [1, .88, .78, .7, .5]

# compose the dictionary for the Agent UCB_SRB
oracle_ucb_srb = OracleUCBSrb(arms=arms, convergence_points=convergence_points, eps=eps, horizon=horizon, sigma=sigma,
                              beta=1.3)
param_agent_ucb_srb_1 = dict(
    n_arms=len(arms),
    exp_param=oracle_ucb_srb.optimal_a*0.02,
    horizon=horizon,
    eps=eps,
    sigma=sigma
)

param_agent_ucb_srb_2 = dict(
    n_arms=len(arms),
    exp_param=oracle_ucb_srb.optimal_a*0.1,
    horizon=horizon,
    eps=eps,
    sigma=sigma
)

param_agent_ucb_srb_3 = dict(
    n_arms=len(arms),
    exp_param=oracle_ucb_srb.optimal_a,
    horizon=horizon,
    eps=eps,
    sigma=sigma
)

param_agent_ucb_srb_4 = dict(
    n_arms=len(arms),
    exp_param=oracle_ucb_srb.optimal_a*10,
    horizon=horizon,
    eps=eps,
    sigma=sigma
)

param_agent_ucb_srb_5 = dict(
    n_arms=len(arms),
    exp_param=oracle_ucb_srb.optimal_a*50,
    horizon=horizon,
    eps=eps,
    sigma=sigma
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
    agent_ucb_srb_1=param_agent_ucb_srb_1,
    agent_ucb_srb_2=param_agent_ucb_srb_2,
    agent_ucb_srb_3=param_agent_ucb_srb_3,
    agent_ucb_srb_4=param_agent_ucb_srb_4,
    agent_ucb_srb_5=param_agent_ucb_srb_5
)

# write the JSON file
with io.open(f'config_sensitivity_exploration_parameter.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(param, ensure_ascii=False, indent=4))
