# Libraries
import json
from environment.environment import SRBEnvironment
from agent.agent import *
from runner.runner import Runner

# arms
f1 = lambda x : .0025*x if .0025*x < 0.99 else 0.99
f2 = lambda x : .47 * (1 - 10/(10+x))
f3 = lambda x : .004*x if .004*x < .7 else .7
arms = [f1, f2, f3]
convergence_points = [1, .47, .7]
eps = .25
sigma = 0.01

# load parameters
config_name = "config_setting_B"
with open('config/' + config_name + '.json') as json_file:
    param = json.load(json_file)

    param_agent_unif = deepcopy(param['agent_unif'])

    param_agent_unif_smooth = deepcopy(param['agent_unif_smooth'])

    param_agent_ucb = deepcopy(param['agent_ucb'])

    param_agent_ucb_srb = deepcopy(param['agent_ucb_srb'])

    param_agent_sr = deepcopy(param['agent_sr'])

    param_agent_sr_srb = deepcopy(param['agent_sr_srb'])

    param_agent_prob = deepcopy(param['agent_prob'])

    param_agent_etc = deepcopy(param['agent_etc'])

    param_agent_rest_sure = deepcopy(param['agent_rest_sure'])

    param_env = deepcopy(param['env'])
    param_env['actions'] = arms


# Build up the blocks
env = SRBEnvironment(**param_env)

agents = [Uniform(**param_agent_unif),
          UniformSmooth(**param_agent_unif_smooth),
          UcbE(**param_agent_ucb),
          UcbSRB(**param_agent_ucb_srb),
          Sr(**param_agent_sr),
          SRSrb(**param_agent_sr_srb),
          Prob1(**param_agent_prob),
          Etc(**param_agent_etc),
          RestSure(**param_agent_rest_sure)]

runner = Runner(
    environment=env,
    agent=None,
    n_trials=param['n_trials'],
    horizon=param['horizon'],
    n_actions=len(arms),
    actions=arms,
    log_path="experiments/exp_setting_B"
)

print("\n################# T = " + str(param["horizon"]) + " #################")
for agent in agents:
    # assign the new agent
    runner.agent = agent

    # perform simulation
    runner.perform_simulations()
    recommendations = runner.results[agent.name]["recommendations"]
    count = recommendations.count(0)
    print("Empirical Error " + str(agent.name) + ": " + str((runner.n_trials - count) / runner.n_trials))

# save the output
name = str(param["horizon"])
# runner.save_output(name)
print("##############################################################")
