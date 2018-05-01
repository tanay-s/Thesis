import numpy as np
import h5py
h5f_train = h5py.File('/data/hula/tanay/CXR8/chest256_test_801010_no_normal.h5', 'r')
y_train = h5f_train['Y_test'][:]
h5f_train.close()

unique = {}
for i in y_train:
    s = [str(int(j)) for j in i]
    key = ''.join(s)
    if key not in unique.keys():
        unique[key] = 1
    else:
        unique[key] += 1

keys_str = unique.keys()
labels = np.zeros((len(keys_str), 14))
for it, key in enumerate(keys_str):
    labels[it] = np.array([int(i) for i in key])

h5f = h5py.File('/data/hula/tanay/CXR8/gan_data_labels.h5', 'w')
h5f.create_dataset('Y_test', data=labels)
h5f.close()