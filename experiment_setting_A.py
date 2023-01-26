# Libraries
import json
import sys
from environment.environment import SRBEnvironment
from agent.agent import *
from runner.runner import Runner
from config.oracle import OracleUCBSrb, OracleUCBe, OracleSR

# Arms
f1 = lambda x: 1 - 37/(37+x)
f2 = lambda x: .88*(1 - 10/(10+x))
f3 = lambda x: .78*(1 - 1/(1+x))
f4 = lambda x: .7*(1 - 10/(10+x))
f5 = lambda x: .5*(1 - 20/(20+x))
arms = [f1, f2, f3, f4, f5]
convergence_points = [1, .88, .78, .7, .5]
eps = .25

# load parameters
config_name = "config_setting_A"
with open('config/' + config_name + '.json') as json_file:
    param = json.load(json_file)
    # param["horizon"] = horizon

    param_agent_unif = deepcopy(param['agent_unif'])
    # param_agent_unif["horizon"] = horizon

    param_agent_unif_smooth = deepcopy(param['agent_unif_smooth'])
    # param_agent_unif_smooth["horizon"] = horizon

    param_agent_ucb = deepcopy(param['agent_ucb'])
    # param_agent_ucb["horizon"] = horizon
    # param_agent_ucb["exp_param"] = oracle_agent_ucb_e.optimal_a

    param_agent_ucb_srb = deepcopy(param['agent_ucb_srb'])
    # param_agent_ucb_srb["horizon"] = horizon
    # param_agent_ucb_srb["exp_param"] = oracle_ucb_srb.optimal_a

    param_agent_sr = deepcopy(param['agent_sr'])
    # param_agent_sr["horizon"] = horizon

    param_agent_sr_srb = deepcopy(param['agent_sr_srb'])
    # param_agent_sr_srb["horizon"] = horizon

    param_agent_prob = deepcopy(param['agent_prob'])
    # param_agent_prob["horizon"] = horizon

    param_agent_etc = deepcopy(param['agent_etc'])
    # param_agent_etc["horizon"] = horizon

    param_agent_rest_sure = deepcopy(param['agent_rest_sure'])
    # param_agent_rest_sure["horizon"] = horizon

    param_env = deepcopy(param['env'])
    param_env['actions'] = arms
    # param_env['horizon'] = horizon


# Build up the blocks
env = SRBEnvironment(**param_env)
'''
Uniform(**param_agent_unif)
UniformSmooth(**param_agent_unif_smooth)
UcbE(**param_agent_ucb)
UcbSRB(**param_agent_ucb_srb)
Sr(**param_agent_sr)
SRSrb(**param_agent_sr_srb)
Prob1(**param_agent_prob)
Etc(**param_agent_etc)
RestSure(**param_agent_rest_sure)
'''
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
    log_path="experiments/exp_setting_A"
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
runner.save_output(name)
print("##############################################################")
