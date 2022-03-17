import numpy as np
import h5py

# d1 = np.random.random(size = (1000,20))
# d2 = np.random.random(size = (1000,200))
# hf = h5py.File('data.h5', 'w')

# hf.create_dataset('dataset_1', data=d1)
# hf.create_dataset('dataset_2', data=d2)

# hf.close()

# hf = h5py.File('data.h5', 'r')
# n1 = hf.get('dataset_1')
# print(n1.__array__())
# print(hf.keys())

# import numpy as np
# import h5py

f = h5py.File('MyDataset.h5', 'a')
    
g1 = f.create_group('group0')
for i in range(100):

  # Data to be appended
  new_data = np.ones(shape=(100, 64)) * i
  new_label = np.ones(shape=(100,1)) * (i+1)
  print(i)
  if i == 0:
    # Create the dataset at first
    g1.create_dataset('data', data=new_data, compression="gzip", chunks=True, maxshape=(None,64))
    g1.create_dataset('label', data=new_label, compression="gzip", chunks=True, maxshape=(None,1)) 
    group1 = f.get("group0")
    print(group1)
  else:
    g = f.create_group('group{}'.format(i))
    g.create_dataset('data', data=new_data, compression="gzip", chunks=True, maxshape=(None,64))
    g.create_dataset('label', data=new_label, compression="gzip", chunks=True, maxshape=(None,1))     
#     group1 = f.get("group1")
#     # Append new data to it
#     f['group1/data'].resize((f['group1/data'].shape[0] + new_data.shape[0]), axis=0)
#     f['group1/data'][-new_data.shape[0]:] = new_data

#     f['group1/label'].resize((f['group1/label'].shape[0] + new_label.shape[0]), axis=0)
#     f['group1/label'][-new_label.shape[0]:] = new_label

#   print("I am on iteration {} and 'data' chunk has shape:{}".format(i,f['group1/data'].shape))

f.close()