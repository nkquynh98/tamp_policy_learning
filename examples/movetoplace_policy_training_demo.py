from policy_learning.core.h5df_dataset_loader import HDF5Dataset
from torch.utils import data
from policy_learning.model.ToyMotionNet import Toy_movetoplace
from policy_learning.model.utils import initialize_weights
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

NUM_EPOCHS = 100
BATCH_SIZE = 100
DATA_FOLDER = os.path.dirname(os.path.realpath(__file__))+"/../data/"
SAVED_MODELS_FOLDER = DATA_FOLDER+"saved_models/"
DATASET_FOLDER = DATA_FOLDER+"ToyPickPlaceTAMP_maze_world_toy_gym/motion_data/movetoplace"
TRAIN_TEST_RATE = 0.8
CHECKPOINT_AFTER = 50


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Init the tasknet
motionnet = Toy_movetoplace(observation_dim=8)
motionnet.to(device)
motionnet.apply(initialize_weights)

#Load the Task dataset
loader_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 2}
dataset = HDF5Dataset(DATASET_FOLDER, data_cache_size=1000,recursive=True, load_data=False)

#Split the dataset into two set
trainset_size = int(TRAIN_TEST_RATE*len(dataset))
testset_size = len(dataset)-trainset_size
train_set, test_set = data.random_split(dataset, (trainset_size, testset_size))

data_loader = data.DataLoader(train_set, **loader_params)
test_loader = data.DataLoader(test_set, **loader_params)

print("data size ", len(data_loader))
print("test size", len(test_loader))
#Define the criterion and optimizer
criterion = nn.MSELoss()


#Using SGD optimizer, learning rate 10^-3, and L2 regulization 4*10^-4
optimizer = optim.SGD(motionnet.parameters(), lr=2*10e-4, weight_decay=1e-4)

start = time.time()
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, return_value in enumerate(data_loader):
        observation, command = return_value
        observation = observation.to(device)
        command = command.to(device).float()
        #print(observation.device)
        #Zero gradient
        optimizer.zero_grad()

        #Forward
        command_pred  = motionnet.forward(observation.float())

        #Loss is the summation of the Cross Entropies of Multihead output
        loss = criterion(command_pred, command)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % CHECKPOINT_AFTER == 0:    # print every 5 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / CHECKPOINT_AFTER:.3f}')
            writer.add_scalar("movetoplace/train_lost", running_loss / CHECKPOINT_AFTER, epoch*len(data_loader)+i)
            torch.save(motionnet.state_dict(), SAVED_MODELS_FOLDER+"movetoplace.pth")
            running_loss = 0.0    

    with torch.no_grad():
        for i, return_value in enumerate(test_loader):
            observation, command = return_value
            observation = observation.to(device)
            command = command.to(device).float()
            #print(observation.device)
            #Forward
            command_pred  = motionnet.forward(observation.float())

            #Loss is the summation of the Cross Entropies of Multihead output
            loss = criterion(command_pred, command)

            # print statistics
            running_loss += loss.item()
            if i % CHECKPOINT_AFTER == 0:    # print every 5 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / CHECKPOINT_AFTER:.3f}')
                writer.add_scalar("movetoplace/test_lost", running_loss / CHECKPOINT_AFTER, epoch*len(test_loader)+i)
                running_loss = 0.0         

                
torch.save(motionnet.state_dict(), SAVED_MODELS_FOLDER+"movetoplace.pth")
print('Finished Training')      
total_time = time.time()-start

print("Example", motionnet.forward(dataset.__getitem__(10)[0].float().to(device)))
print("Total time:",total_time)

