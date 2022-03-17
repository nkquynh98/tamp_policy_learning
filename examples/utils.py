
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class NumbersDataset(Dataset):
    def __init__(self, low, high):
        self.samples = list(range(low, high))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        n = self.samples[idx]
        successors = torch.arange(4).float() + n + 1
        noisy = torch.randn(4) + successors
        return str(n), successors, noisy


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = NumbersDataset(100, 120)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    print(next(iter(dataloader)))
    A = np.array([1,2,3])
    B = np.array([3,4,5])
    C = np.array([4,5,6])
    D = [A,B,C]
    E = np.vstack(D)
    print(E)
    task_data = {"observation": [], "action": [], "objects": [], "targets": [] }
    print(task_data)
    #F = np.array([1,2,3,4])
    #print(F[-2:])
    d = {'ac':33, 'gw':20, 'ap':102, 'za':321, 'bs':10}
    print(d.keys())
    pred = torch.tensor([[0.0, 0.5, 0.5]]).float()
    real = torch.tensor([[0.0, 1.0, 0.0]]).float()
    crit = nn.CrossEntropyLoss()
    soft = F.softmax(pred)

    print(soft)
    print(crit(pred,real))

    a = torch.tensor([2,3,3])
    if len(a.size())==1:
        #a = a.view((1, -1))
        a = a.unsqueeze(0)
    print(len(a.size()))
    print(a)