import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyTaskNet(nn.Module):

    def __init__(self, goal_encoded_dim = 25, observation_dim=26, action_dim = 2, object_dim = 5, target_dim=5):
        super(ToyTaskNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(goal_encoded_dim+observation_dim, 96)  # 100 observation
        self.fc2 = nn.Linear(96,96)
        self.fc3_action = nn.Linear(96, action_dim)
        self.fc3_object = nn.Linear(96, object_dim)
        self.fc3_target = nn.Linear(96, target_dim)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        #action = F.softmax(self.fc3_action(x), dim = -1)
        #objects = F.softmax(self.fc3_object(x), dim = -1)
        #targets = F.softmax(self.fc3_target(x), dim = -1)

        action = self.fc3_action(x)
        objects = self.fc3_object(x)
        targets = self.fc3_target(x)
        # Multihead output
        return action, objects, targets

