import json
from environment.environment import IMDB
from agent.agent import *
from runner.runner import Runner

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

config_name = "imdb"
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

# Build up the blocks
env = IMDB(**param_env)

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
    log_path="experiments/imdb"
)

print("\n################# T = " + str(param["horizon"]) + " #################")
for agent in agents:
    # assign the new agent
    runner.agent = agent

    # perform simulation
    runner.perform_simulations()
    recommendations = runner.results[agent.name]["recommendations"]
    if param["horizon"] >= 2000:
        count = recommendations.count(2)
    else:
        count = recommendations.count(5)
    print("Recommendation: " + str(runner.results[agent.name]["recommendations"]))
    print("Empirical Error " + str(agent.name) + ": " + str((runner.n_trials - count) / runner.n_trials))

# save the output
name = str(param["horizon"])
runner.save_output(name)
print("##############################################################")