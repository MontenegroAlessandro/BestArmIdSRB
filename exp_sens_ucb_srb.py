# Libraries
import json
from environment.environment import SRBEnvironment
from agent.agent import *
from runner.runner import Runner

f1 = lambda x: 1 - 37 / (37 + x)
f2 = lambda x: .88 * (1 - 10 / (10 + x))
f3 = lambda x: .78 * (1 - 1 / (1 + x))
f4 = lambda x: .7 * (1 - 10 / (10 + x))
f5 = lambda x: .5 * (1 - 20 / (20 + x))
arms = [f1, f2, f3, f4, f5]

# load parameters
config_name = "exp_sens_a"
with open('config/' + config_name + '.json') as json_file:
    param = json.load(json_file)
    param_agent_ucb_srb_1 = deepcopy(param['agent_ucb_srb_1'])
    param_agent_ucb_srb_2 = deepcopy(param['agent_ucb_srb_2'])
    param_agent_ucb_srb_3 = deepcopy(param['agent_ucb_srb_3'])
    param_agent_ucb_srb_4 = deepcopy(param['agent_ucb_srb_4'])
    param_agent_ucb_srb_5 = deepcopy(param['agent_ucb_srb_5'])

    param_env = deepcopy(param['env'])
    param_env['actions'] = arms

# Build up the blocks
env = SRBEnvironment(**param_env)

a1 = UcbSRB(**param_agent_ucb_srb_1)
a1.name += " 0.02"

a2 = UcbSRB(**param_agent_ucb_srb_2)
a2.name += " 0.1"

a3 = UcbSRB(**param_agent_ucb_srb_3)

a4 = UcbSRB(**param_agent_ucb_srb_4)
a4.name += " 10"

a5 = UcbSRB(**param_agent_ucb_srb_5)
a5.name += " 50"

agents = [a1, a2, a3, a4, a5]

runner = Runner(
    environment=env,
    agent=None,
    n_trials=param['n_trials'],
    horizon=param['horizon'],
    n_actions=len(arms),
    actions=arms,
    log_path="experiments/exp_sens_a"
)

print("\n################# T = " + str(param["horizon"]) + " #################")
for agent in agents:
    # assign the new agent
    runner.agent = agent

    # perform simulation
    runner.perform_simulations()
    recommendations = runner.results[agent.name]["recommendations"]
    count = recommendations.count(0) if param["horizon"] >= 200 else recommendations.count(1)
    print("Empirical Error " + str(agent.name) + ": " + str((runner.n_trials - count) / runner.n_trials))

# save the output
name = str(param["horizon"])
runner.save_output(name)
print("##############################################################")
