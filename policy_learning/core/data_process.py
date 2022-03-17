from base64 import encode
from ctypes import sizeof
import pickle
import os
import torch
from tkinter import CURRENT
import numpy as np
from datetime import datetime,date
import time
from policy_learning.core.data_object import *
from queue_server.common_object.domain import Domain
from queue_server.common_object.problem import Problem
import h5py

from pyrieef import motion
# Assume 10 last observation is the observation of the task around 10 last obs,
# including displacement from objects to gripper, and from objects to targets
CURRENT_FOLDER = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = CURRENT_FOLDER + "/../../data/"
FILE_NAME = "example_task_data.pickle"

# for object in data.motion_data_list:
#     assert isinstance(object, Expert_motion_data)
    # print(object.action)
    # print(len(object.observation_list))
    # print(object.observation_list[0].shape)
    # print(len(object.command_list))
    # print(object.command_list[0].shape)
def safe_open(path, rw:str):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, rw)

def encode_goal(problem: Problem):
    problem_dict = problem.get_dict()
    goal = problem_dict["positive_goal"][0]
    object_list = problem_dict["objects"]["object"]
    target_list = problem_dict["objects"]["location"]
    goal_encoded = np.zeros((len(object_list)*len(target_list)))
    for sub_goal in goal:
        if sub_goal[0] == "at" :    
            #print(sub_goal)
            # index of the encoded goal: if the goal is placed object 1 at target 3: goalencoded[1*5+3]=1
            index = object_list.index(sub_goal[1])*len(object_list)+target_list.index(sub_goal[2])
            goal_encoded[index] = 1
    return goal_encoded

def encode_action(domain: Domain, action_to_encode: Action):
    domain_dict = domain.get_dict()
    action_list = domain_dict["actions"]
    
    object_list = domain_dict["constants"]["object"]
    target_list = domain_dict["constants"]["location"]

    action_encoded = np.zeros((len(action_list)))
    object_encoded = np.zeros((len(object_list)))
    target_encoded = np.zeros((len(target_list)))

    for i, action in enumerate(action_list):
        if action["name"]==action_to_encode.name:
            action_encoded[i] = 1
    object_encoded[object_list.index(action_to_encode.parameters[0])]=1
    #Assume that action movetopick has only 1 param : object_x, while movetoplace has 2 param: object_x, target_x 
    if len(action_to_encode.parameters)==2:
        target_encoded[object_list.index(action_to_encode.parameters[0])]=1
    return [action_encoded, object_encoded, target_encoded]
def decode_action(domain:Domain, encoded_action, encoded_object, encoded_target):
    domain_dict = domain.get_dict()
    action_list = domain_dict["actions"]
    object_list = domain_dict["constants"]["object"]
    target_list = domain_dict["constants"]["location"]  
    for i, action in enumerate(action_list):
        if i == np.argmax(encoded_action):
            decoded_action = action["name"]

    decoded_object = object_list[np.argmax(encoded_object)]
    decoded_target= target_list[np.argmax(encoded_target)]

    return decoded_action, decoded_object, decoded_target
def get_index_of_displacement(domain: Domain, action: Action):
    domain_dict = domain.get_dict()
    
    object_list = domain_dict["constants"]["object"]
    for i in range(len(object_list)):
        if action.name == "movetopick":
            if action.parameters[0]=="object_{}".format(i):
                return -4*len(object_list)+2*i
        elif action.name == "movetoplace":
            if action.parameters[0]=="object_{}".format(i):
                return -2*len(object_list)+2*i       
    
def process_task_data(data: Expert_task_data):
    env_name = data.env_name
    #file_name = str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))+".pickle"
    file_name = str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))+".h5"
    #processed_task_data = []
    #The prefix number is added to ensure the dict key appears in the desired orders
    task_data = {"0_observation": [], "1_action": [], "2_objects": [], "3_targets": [] }
    motion_data = {}
    #processed_motion_data = {}
    for action in data.domain.get_dict()["actions"]:
        #processed_motion_data[action["name"]]=[]
        motion_data[action["name"]] = {"0_observation": [], "1_command": []} 
    goal_encoded = encode_goal(data.problem)
    for action_data in data.motion_data_list:
        #output_task = encode_action(data.domain, action_data.action)
        action_encoded, object_encoded, target_encoded = encode_action(data.domain, action_data.action)
        displacement_index = get_index_of_displacement(data.domain, action_data.action)
        #print("Displacement_index", displacement_index)
        #print(action_data.action.name)
        #print(len(action_data.observation_list))
        #print(action_data.observation_list[-1])
        for observation, command in zip(action_data.observation_list, action_data.command_list):
            # Concatenate the goal with the observation
            #print("observation shape", observation.shape)
            input_task = np.concatenate([goal_encoded,observation])

            task_data["0_observation"].append(input_task)
            task_data["1_action"].append(action_encoded)
            task_data["2_objects"].append(object_encoded)
            task_data["3_targets"].append(target_encoded)

            #processed_task_data.append((input_task,output_task))
            # Hardcode the displacement of the current position to target based on the action
            # Move to pick: displacement from gripper to object
            # Move to place: displacement from object to target
            if displacement_index+2!=0:
                input_motion = np.concatenate([observation[:-20], observation[displacement_index:displacement_index+2]])
            else:
                input_motion = np.concatenate([observation[:-20], observation[displacement_index:]])

            motion_data[action_data.action.name]["0_observation"].append(input_motion)
            motion_data[action_data.action.name]["1_command"].append(command)


            #processed_motion_data[action_data.action.name].append((input_motion, command))
            
            #process_task_data.append(())
    # save task dataset to hdf5
    save_to_hdf5(DATA_FOLDER+env_name+"/task_data/"+file_name, task_data)

    # save motion dataset
    for action_name, action_value in motion_data.items():
        save_to_hdf5(DATA_FOLDER+env_name+"/motion_data/"+action_name+"/"+file_name, action_value )
    # with safe_open(DATA_FOLDER+env_name+"/task_data/"+file_name,"wb") as f:
    #     #save in the environment
    #     pickle.dump(processed_task_data,f)
    # for key,value in processed_motion_data.items():
    #     with safe_open(DATA_FOLDER+env_name+"/motion_data/"+key+"/"+file_name,"wb") as f:
    #         pickle.dump(value,f)


def save_to_hdf5(file_name:str, data_dict: dict):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    h5_file = h5py.File(file_name, 'a')
    datalength = 0
    for dataset_name, dataset_value in data_dict.items():
        # Save each list of the data as an dataset (multi-dim np array) in the hdf5 file
        dataset_value = np.vstack(dataset_value)
        datalength = dataset_value.shape[0]
        h5_file.create_dataset(dataset_name, data=dataset_value, compression="gzip", chunks=True)
    h5_file.attrs["data_length"]=datalength

if __name__ =="__main__":
    with open(DATA_FOLDER+FILE_NAME, "rb") as f:
        data = pickle.load(f)

    assert isinstance(data, Expert_task_data)
    print(data.env_name)
    print(data.domain)
    print(data.problem)
    print(encode_goal(data.problem))
    print(encode_action(data.domain, data.motion_data_list[1].action))
    process_task_data(data)
