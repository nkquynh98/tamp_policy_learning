# Utils
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
import json
import matplotlib.pyplot as plt
import pickle
import time
#Pybewego
from pybewego.numerical_optimization.ipopt_mo import NavigationOptimization
from pybewego.motion_optimization import CostFunctionParameters
from pybewego.workspace_viewer_server import WorkspaceViewerServer, WorkspaceViewerServerPlanar
from pybewego.motion_optimization import MotionOptimization
from pybewego.motion_optimization import CostFunctionParameters

#Pyrieef
from pyrieef.geometry.workspace import *
from pyrieef.motion.trajectory import *
from pyrieef.utils.collision_checking import *
from pyrieef.graph.shortest_path import *
#from trajectory import *

# Working Environment
from toy_gym.envs.toy_tasks.toy_pickplace_tamp import ToyPickPlaceTAMP
from toy_gym.policy.TAMPActionExecutor import TAMPActionExecutor, TAMPActionExecutorFreeFlyer

# Motion planning
from motion_planning.core.workspace import WorkspaceFromEnv
from motion_planning.core.action import *
from motion_planning.core.TAMP_motion_planner import TAMPMotionOptimizer
from motion_planning.core.TAMP_motion_planner_freeflyer import TAMPMotionOptimizerFreeFlyer

# Logic planning
from logic_planning.planner import LogicPlanner
from logic_planning.parser import PDDLParser
from logic_planning.action import DurativeAction
from logic_planning.helpers import frozenset_of_tuples

#Policy
from policy_learning.model.ToyTaskModel import ToyTaskNet
from policy_learning.core.data_process import encode_goal
import torch

#Policy executator 
from policy_learning.demo.learned_policy_excutor import ToyLearnedPolicyExecutor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

domain_file = "/home/nkquynh/gil_ws/tamp_logic_planner/PDDL_scenarios/domain_toy_gym.pddl"
problem_file = "/home/nkquynh/gil_ws/tamp_logic_planner/PDDL_scenarios/problem_toy_gym.pddl"
domain = PDDLParser.parse_domain(domain_file)
problem = PDDLParser.parse_problem(problem_file)

#print("Problem", problem.positive_goals)

pickled_problem = pickle.dumps(domain)
new_problem = pickle.loads(pickled_problem)
print("abcxyz", new_problem)
#problem.positive_goals = [frozenset({('at', 'object_0', 'target_1'), ('agent-free',), ('at', 'object_1', 'target_2'), ('at', 'object_2', 'target_4'), ('at', 'object_4', 'target_0'), ('at', 'object_3', 'target_3')})]
TRAJ_LENGTH = 50
DRAW_MODE = "pyglet2d" 
NB_POINTS = 40
DEBUG = False
NUM_EPS = 10
MAX_STEPS = 20000
VIEWER_ENABLE = False
DATA_FOLDER = os.path.dirname(os.path.realpath(__file__))+"/../data/"
SAVED_MODELS_FOLDER = DATA_FOLDER+"saved_models/"


tasknet = ToyTaskNet()
tasknet.load_state_dict(torch.load(SAVED_MODELS_FOLDER+"tasknet.pth"))

planner = LogicPlanner(domain)
planner.init_planner(problem=problem, ignore_cache=True)

paths, act_seq = planner.plan(alternative=True)
skeleton = act_seq[0]

goal = problem.get_dict()["positive_goal"]
print("PronlemGoal", goal)
# for i in range(5):
#     pick = MoveToPick(parameters=["object_{}".format(i)],duration=10)
#     action_list.append(pick)
#     place = MoveToPlace(parameters=["object_{}".format(i), "target_{}".format(i)], duration=10)
#     action_list.append(place)


#goal = {"object_0": "target_1", "object_1": "target_2", "object_2": "target_0", "object_3": "target_4", "object_4": "target_3"}

# with open("/home/nkquynh/gil_ws/tamp_queue_server/examples/sample_workspace.json") as f:
#     data = json.load(f)
# env = ToyPickPlaceTAMP(render=VIEWER_ENABLE, json_data=data, goal=goal, enable_physic=False)
with open("/home/nkquynh/gil_ws/tamp_queue_server/examples/sample_workspace.json") as f:
    data = json.load(f)
env = ToyPickPlaceTAMP(render=VIEWER_ENABLE, json_data=data, domain = domain, problem = problem, goal=goal, enable_physic=False)
costs = []
dones = []
success_episode = 0

encoded_goal = encode_goal(problem)
executor = ToyLearnedPolicyExecutor(env)

#Error
motion_error = []
task_error = np.zeros((3))
for i in range(NUM_EPS):
    workspace_objects = env.get_workspace_objects()
    workspace = WorkspaceFromEnv(workspace_objects)
    planner = TAMPMotionOptimizerFreeFlyer(workspace, skeleton=skeleton, enable_global_planner=False, enable_viewer=VIEWER_ENABLE, flexible_traj_ratio=6)
    planner.execute_plan()
    planner.visualize_final_trajectory()
    print("Total cost", planner.get_total_cost())
    costs.append(planner.get_total_cost())
    policy=TAMPActionExecutorFreeFlyer(env, threshold=0.02, threshold_angle=0.02)
    for action in skeleton:
        print(action)
        print(action.trajectory)
    policy.set_action_list(skeleton)
    total_step = 0
    for _ in range(MAX_STEPS):
        #action = np.random.rand(4,)
        #print(action)
        observation = env._get_obs()["observation"]
        planning_command = policy.get_action(show_action_name=False)
        predict_command = executor.get_action(observation)
        
        #For logging only
        planning_action =  policy.get_current_action()
        action_decoded, object_decoded, target_decoded = executor.get_task_output(observation)
        #print("[Planning] {}: {}".format(planning_action.name, planning_action.parameters))
        #print("[Predict] {}: {} {}".format(action_decoded,object_decoded,target_decoded))
        #print("[Planning] command: ", planning_command)
        #print("[Predict] command: ", predict_command) 
        #print("Error: ", np.linalg.norm(planning_command-predict_command))
        if planning_action.name != action_decoded:
            task_error[0]+=1.0
            print("predict {}, real {} ".format(planning_action.name, action_decoded))
        else:
            if object_decoded != planning_action.parameters[0]:
                task_error[1]+=1.0
                print("predict {}, real {} ".format(planning_action.parameters[0], object_decoded))
            if len(planning_action.parameters)>1:
                if target_decoded != planning_action.parameters[1]:
                    task_error[2]+=1.0
                    print("predict {}, real {} ".format(planning_action.parameters[1], target_decoded))                    
        error = np.linalg.norm(planning_command-predict_command)
        motion_error.append(error)
        


        obs, reward, done, info = env.step(planning_command)
        input = np.concatenate([encoded_goal, obs['observation']])
        input = torch.from_numpy(input)
        total_step+=1.0
        #print("predict action", tasknet.forward(input.float()))
        if VIEWER_ENABLE:
            time.sleep(0.01)
        #print("logic_state", env.get_logic_state())
        if done:
            dones.append(i)
            success_episode +=1
            break

    
    print("Action selection error: {}:{} = {:.2f}%".format(task_error[0],total_step,task_error[0]/total_step*100))
    print("Object selection error: {}:{} = {:.2f}%".format(task_error[1],total_step,task_error[1]/total_step*100))
    print("Target selection error: {}:{} = {}%".format(task_error[2],total_step,task_error[2]/total_step*100))
    
    plt.plot(motion_error)
    plt.show()
    while(1):
        pass
    env.reset()
    del planner
    del policy


# print("Success rate {}/{}".format(success_episode, NUM_EPS))
# print("Episode dones ", dones)
# print(costs)
# plt.plot(costs)
# plt.show()