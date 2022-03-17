from torch.utils import data
from policy_learning.core.dataset_loader import HDF5Dataset

num_epochs = 1
loader_params = {'batch_size': 1, 'shuffle': True, 'num_workers': 1}

dataset = HDF5Dataset('/home/nkquynh/gil_ws/tamp_policy_learning/examples', recursive=True, load_data=False, 
   data_cache_size=4, transform=None)

print("dataset length", dataset.__len__())

data_loader = data.DataLoader(dataset, **loader_params)

for i in range(num_epochs):
   for x,y in data_loader:
      print(x.shape)
      print(y.shape)
      break