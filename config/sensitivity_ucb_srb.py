# Configuration for UCB-SRB
import io
import json
from oracle import OracleUCBSrb

# arms
f1 = lambda x: 1 - 40/(40+x)
f2 = lambda x: .88*(1 - 10/(10+x))
f3 = lambda x: .7*(1 - 10/(10+x))
f4 = lambda x: .5*(1 - 20/(20+x))
arms = [f1, f2, f3, f4]

# common stuff
path = 'experiments'
horizon = 3000
n_trials = 100
sigma = 0.01
eps = .25
convergence_points = [1, .88, .7, .5]
shrink = 1/50

# compose the dictionary for the Agent UCB_SRB
oracle_ucb_srb = OracleUCBSrb(arms=arms, convergence_points=convergence_points, eps=eps, horizon=horizon, sigma=sigma,
                              beta=2)
param_agent_ucb_srb = dict(
    n_arms=len(arms),
    exp_param=oracle_ucb_srb.optimal_a*shrink,
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
    shrink=shrink,
    env=param_env,
    agent_ucb_srb=param_agent_ucb_srb,
)

# write the JSON file
with io.open(f'exp_sens.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(param, ensure_ascii=False, indent=4))

oracle_ucb_srb.represent()
