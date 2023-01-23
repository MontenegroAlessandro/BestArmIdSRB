# Libraries
import json
import sys
from environment.environment import SRBEnvironment
from agent.agent import *
from runner.runner import Runner
from config.oracle import OracleUCBSrb

horizon = int(sys.argv[1])

# Arms
f1 = lambda x: 1 - 40 / (40 + x)
f2 = lambda x: .88 * (1 - 10 / (10 + x))
f3 = lambda x: .7 * (1 - 10 / (10 + x))
f4 = lambda x: .5 * (1 - 20 / (20 + x))
arms = [f1, f2, f3, f4]
convergence_points = [1, .88, .7, .5]
eps = .25
sigma = 0.01

oracle_ucb_srb = OracleUCBSrb(arms=arms, convergence_points=convergence_points, eps=eps, horizon=horizon, sigma=sigma,
                              beta=2)

# load parameters
config_name = "exp_sens"
with open('config/' + config_name + '.json') as json_file:
    param = json.load(json_file)
    param["horizon"] = horizon

    shrink = param["shrink"]

    param_agent_ucb_srb = deepcopy(param['agent_ucb_srb'])
    param_agent_ucb_srb["horizon"] = horizon
    param_agent_ucb_srb["exp_param"] = oracle_ucb_srb.optimal_a * shrink

    param_env = deepcopy(param['env'])
    param_env['actions'] = arms
    param_env['horizon'] = horizon


# Build up the blocks
env = SRBEnvironment(**param_env)

agents = [UcbSRB(**param_agent_ucb_srb)]

runner = Runner(
    environment=env,
    agent=None,
    n_trials=param['n_trials'],
    horizon=param['horizon'],
    n_actions=len(arms),
    actions=arms,
    log_path="experiments"
)

print("\n################# T = " + str(horizon) + " #################")
for agent in agents:
    # assign the new agent
    runner.agent = agent

    # perform simulation
    runner.perform_simulations()
    recommendations = runner.results[agent.name]["recommendations"]
    count = recommendations.count(0)
    print("Empirical Error " + str(agent.name) + ": " + str((runner.n_trials - count) / runner.n_trials))

# save the output
name = str(horizon)
runner.save_output(name)
print("##############################################################")