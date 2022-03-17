from sklearn.exceptions import PositiveSpectrumWarning
import torch 
from policy_learning.model.ToyMotionNet import Toy_movetoplace, Toy_movetopick
from policy_learning.model.ToyTaskModel import ToyTaskNet
from policy_learning.model.utils import initialize_weights
import os
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
tasknet = ToyTaskNet()
tasknet.apply(initialize_weights)

criterion = torch.nn.CrossEntropyLoss()
SAVED_MODELS_FOLDER = os.path.dirname(os.path.realpath(__file__))+"/../data/saved_models/"
print(SAVED_MODELS_FOLDER)
x = torch.rand((1,110))
print(x)
action_true = torch.tensor([[0.0,1.0]])

torch.save(tasknet, SAVED_MODELS_FOLDER+"ToyTaskNet.pth")
print("Model's state_dict:")
for param_tensor in tasknet.state_dict():

    print(param_tensor, "\t", tasknet.state_dict()[param_tensor].size())

action, object, target = tasknet.forward(x)
print(action)
print(action_true)
loss = criterion(action,action_true)
print(loss)
loss.backward()