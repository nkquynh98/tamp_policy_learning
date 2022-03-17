from policy_learning.core.h5df_dataset_loader import HDF5Dataset
from torch.utils import data
import time
import os
NUM_EPOCHS = 1

DATA_FOLDER = os.path.dirname(os.path.realpath(__file__))+"/../data/"
SAVED_MODELS_FOLDER = DATA_FOLDER+"saved_models/"
DATASET_FOLDER = DATA_FOLDER+"ToyPickPlaceTAMP_maze_world_toy_gym/motion_data/movetopick"
loader_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 1}
dataset = HDF5Dataset(DATASET_FOLDER, data_cache_size=100,recursive=True, load_data=False)

data_loader = data.DataLoader(dataset, **loader_params)
#print(len(dataset))
#print(dataset.__getitem__(100))
# print(dataset.__getitem__(100))
# print(dataset.__getitem__(500))
# print(dataset.__getitem__(800))
# print(dataset.__getitem__(101))
# print(dataset.__getitem__(501))
start = time.time()
for i in range(NUM_EPOCHS):
   for return_value in data_loader:
       #print(return_value[0].shape)
       print(return_value)
       pass
total_time = time.time()-start
print("Total time:",total_time)

