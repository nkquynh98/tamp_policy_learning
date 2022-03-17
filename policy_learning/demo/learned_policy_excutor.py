# Utils
from logic_planning.problem import Problem
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
import json
import matplotlib.pyplot as plt
import time
import os

# Working Environment
from toy_gym.envs.toy_tasks.toy_pickplace_tamp import ToyPickPlaceTAMP
from toy_gym.policy.TAMPActionExecutor import TAMPActionExecutor, TAMPActionExecutorFreeFlyer
from logic_planning.parser import PDDLParser
#Policy
from policy_learning.model.ToyTaskModel import ToyTaskNet
from policy_learning.model.ToyMotionNet import Toy_movetopick, Toy_movetoplace
from policy_learning.core.data_process import encode_goal, decode_action

import torch
DATA_FOLDER = os.path.dirname(os.path.realpath(__file__))+"/../../data/"
SAVED_MODELS_FOLDER = DATA_FOLDER+"saved_models/"
CONFIG_FOLDER = DATA_FOLDER+"configuration/"
#Class to execute the learned policy (task policy, motion policy) in a real environment

class ToyLearnedPolicyExecutor:
    def __init__(self, env: ToyPickPlaceTAMP, json_config = CONFIG_FOLDER+"policy_configuration.json", goal_encoder = encode_goal, action_decoder = decode_action):
        self.env = env
        with open(json_config,"r") as f:
            self.json_config = json.load(f)
        print(self.json_config)
        self.model_folder = self.json_config["saved_model_folder"]
        self.encoded_goal = goal_encoder(problem=self.env.problem)
        self.action_decoder = action_decoder
        self.domain = self.env.domain
        #Create the net instance based on a name (string)
        self.tasknet = eval(self.json_config["tasknet"]["type"])()
        #Load network
        self.tasknet.load_state_dict(torch.load(self.model_folder+self.json_config["tasknet"]["file_name"]))
        self.tasknet.eval()
        self.motion_net = {}
        #load the action model
        for action in self.env.domain.get_dict()["actions"]:
            action_name = action["name"]
            assert action_name in self.json_config["motionnet"], "No net for action {}".format(action_name)
            self.motion_net[action_name] = eval(self.json_config["motionnet"][action_name]["type"])(observation_dim=8)
            self.motion_net[action_name].load_state_dict(torch.load(self.model_folder+self.json_config["motionnet"][action_name]["file_name"]))
            self.motion_net[action_name].eval()
    def get_input_for_motionnet(self, action: str, object:str, target:str):
        #hard code the observation for the motion net: 
        #   movetopick: distance between robot and object
        #   movetoplace: distance between object and target

        if action=="movetopick":
            return self.env.gripper_to_object(object)
        elif action=="movetoplace":
            return self.env.object_to_target(object,target)
    def get_action(self, observation):
        input_task = np.concatenate([self.encoded_goal,observation])
        with torch.no_grad():
            action_encoded, object_encoded, target_encoded = self.tasknet(torch.from_numpy(input_task).float())
        
        action_encoded = action_encoded.numpy()
        object_encoded = object_encoded.numpy()
        target_encoded = target_encoded.numpy()

        action_decoded, object_decoded, target_decoded = self.action_decoder(self.domain, action_encoded, object_encoded, target_encoded)
        #print("[Predict] {}: {} {}".format(action_decoded,object_decoded,target_decoded))

        #Get the hardcode observation from the parameters of the action
        hardcoded_parameters = self.get_input_for_motionnet(action_decoded, object_decoded, target_decoded)
        input_motion = np.concatenate([observation[:-20], hardcoded_parameters])
        
        #Get the action command
        with torch.no_grad():
            return_command = self.motion_net[action_decoded].forward(torch.from_numpy(input_motion).float())
        return return_command.numpy()

    def get_task_output(self, observation):
        input_task = np.concatenate([self.encoded_goal,observation])
        with torch.no_grad():
            action_encoded, object_encoded, target_encoded = self.tasknet(torch.from_numpy(input_task).float())
        
        action_encoded = action_encoded.numpy()
        object_encoded = object_encoded.numpy()
        target_encoded = target_encoded.numpy()

        action_decoded, object_decoded, target_decoded = self.action_decoder(self.domain, action_encoded, object_encoded, target_encoded)
        return action_decoded, object_decoded, target_decoded
        
    

if __name__=="__main__":
    domain_file = "/home/nkquynh/gil_ws/tamp_logic_planner/PDDL_scenarios/domain_toy_gym.pddl"
    problem_file = "/home/nkquynh/gil_ws/tamp_logic_planner/PDDL_scenarios/problem_toy_gym.pddl"
    domain = PDDLParser.parse_domain(domain_file)
    problem = PDDLParser.parse_problem(problem_file)

    
    #Create a class instance based on a String
    MAX_STEP = 1000
    with open("/home/nkquynh/gil_ws/tamp_queue_server/examples/sample_workspace.json") as f:
        data = json.load(f)
    env = ToyPickPlaceTAMP(render=True, json_data=data, domain = domain, problem = problem,enable_physic=False)
    
    executor = ToyLearnedPolicyExecutor(env)
    for _ in range(MAX_STEP):
        observation = env._get_obs()["observation"]
        command = executor.get_action(observation)
        action_decoded, object_decoded, target_decoded = executor.get_task_output(observation)
        print("Observation ", observation)
        print("Action {}: {} {}".format(action_decoded,object_decoded,target_decoded))
        print("command " ,command)
        obs, reward, done, info = env.step(command)
        time.sleep(0.01)
        if done:
            break
    pass
