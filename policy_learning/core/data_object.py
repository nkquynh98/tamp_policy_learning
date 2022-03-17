from queue_server.common_object.action import Action, DurativeAction
class Expert_motion_data(object):
    def __init__(self, action: Action):
        self.action = action
        self.observation_list = []
        self.command_list = []
    def add_observation(self, obs):
        self.observation_list.append(obs)
    def add_command(self, command):
        self.command_list.append(command)

    def __str__(self):
        return self.action.__str__() + \
            "\nObservation" + str([str(obs) for obs in self.observation_list]) + \
            "\nCommand" + str([str(cmd) for cmd in self.command_list])
class Expert_task_data(object):
    def __init__(self, env_name:str, domain, problem, motion_data_list = [], is_task_fully_refined=False):
        self.env_name = env_name
        self.domain = domain
        self.problem = problem
        self.is_task_fully_refined = is_task_fully_refined
        self.motion_data_list = motion_data_list
    def add_motion_data(self, motion_data: Expert_motion_data):
        self.motion_data_list.append(motion_data)
