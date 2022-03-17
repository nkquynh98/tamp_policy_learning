from policy_learning.core.h5df_dataset_loader import HDF5Dataset
from torch.utils import data
from policy_learning.model.ToyTaskModel import ToyTaskNet
from policy_learning.model.utils import initialize_weights
#For visulizing training process:
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
NUM_EPOCHS = 10
BATCH_SIZE = 100
DATA_FOLDER = os.path.dirname(os.path.realpath(__file__))+"/../data/"
SAVED_MODELS_FOLDER = DATA_FOLDER+"saved_models/"
DATASET_FOLDER = DATA_FOLDER+"ToyPickPlaceTAMP_maze_world_toy_gym/task_data"
CHECKPOINT_AFTER = 50
TRAIN_TEST_RATE = 0.8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Init the tasknet
tasknet = ToyTaskNet()
tasknet.to(device)
tasknet.apply(initialize_weights)

#Load the Task dataset
loader_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 2}
dataset = HDF5Dataset(DATASET_FOLDER, data_cache_size=100000,recursive=True, load_data=False)

#Split the dataset into two set
trainset_size = int(TRAIN_TEST_RATE*len(dataset))
testset_size = len(dataset)-trainset_size
train_set, test_set = data.random_split(dataset, (trainset_size, testset_size))


data_loader = data.DataLoader(train_set, **loader_params)
test_loader = data.DataLoader(test_set, **loader_params)

print("data size ", len(data_loader))
print("test size", len(test_loader))
#Define the criterion and optimizer
criterion = nn.CrossEntropyLoss()


#Using SGD optimizer, learning rate 10^-3, and L2 regulization 4*10^-4
optimizer = optim.SGD(tasknet.parameters(), lr=10e-3, weight_decay=4e-4)

start = time.time()
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, return_value in enumerate(data_loader):
        observation, action, objects, targets = return_value
        observation = observation.to(device)
        action = action.to(device)
        objects = objects.to(device)
        targets= targets.to(device)
        #print(observation.device)
        #Zero gradient
        optimizer.zero_grad()

        #Forward
        action_pred, object_pred, target_pred = tasknet.forward(observation.float())
        #Loss is the summation of the Cross Entropies of Multihead output
        loss = criterion(action_pred, action)+criterion(object_pred,objects)+criterion(target_pred, targets)
        

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % CHECKPOINT_AFTER == 0:    # print every 5 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / CHECKPOINT_AFTER:.3f}')
            writer.add_scalar("Task/train_lost", running_loss / CHECKPOINT_AFTER, epoch*len(data_loader)+i)
            torch.save(tasknet.state_dict(), SAVED_MODELS_FOLDER+"tasknet.pth")
            running_loss = 0.0        


    #Test the current model
    with torch.no_grad():
        for i, return_value in enumerate(test_loader):
            observation, action, objects, targets = return_value
            observation = observation.to(device)
            action = action.to(device)
            objects = objects.to(device)
            targets= targets.to(device)
            #print(observation.device)

            #Forward
            action_pred, object_pred, target_pred = tasknet.forward(observation.float())
            #Loss is the summation of the Cross Entropies of Multihead output
            loss = criterion(action_pred, action)+criterion(object_pred,objects)+criterion(target_pred, targets)

            # print statistics
            running_loss += loss.item()
            if i % CHECKPOINT_AFTER == 0:    # print every 5 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / CHECKPOINT_AFTER:.3f}')
                writer.add_scalar("Task/test_lost", running_loss / CHECKPOINT_AFTER, epoch*len(data_loader)+i)
                running_loss = 0.0        
torch.save(tasknet.state_dict(), SAVED_MODELS_FOLDER+"tasknet.pth")
print('Finished Training')      
total_time = time.time()-start
writer.flush()
print("Example", tasknet.forward(dataset.__getitem__(10)[0].float().to(device)))
print("Total time:",total_time)

