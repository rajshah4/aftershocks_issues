

#import csv
import os
import copy
import gc
import numpy as np
import pdb as check
from collections import defaultdict
import modelfunctions

pathtofiles = '../Data/AllCSV/'

file_names_training = modelfunctions.readHDF('../Data/Training_FileNames.h5', 'file_names_training')
file_names_testing = modelfunctions.readHDF('../Data/Testing_FileNames.h5', 'file_names_testing')

training = defaultdict(list)
testing = defaultdict(list)

field_names = ['stresses_full_xx', 'stresses_full_yy', 'stresses_full_xy', 'stresses_full_xz', 'stresses_full_yz', 'stresses_full_zz']

for i, file_name in enumerate(file_names_testing):
        print(i, file_name)
        print('testing eq')
        data = modelfunctions.read_file_to_dict(pathtofiles + str(file_name))
        grid_aftershock_count = np.double(data['aftershocksyn'])
        if len(np.unique(grid_aftershock_count)) < 2:
            continue
        tmp = grid_aftershock_count.tolist()
        testing['aftershocksyn'].extend(tmp)
        for j in range(0, len(field_names)):            testing[field_names[j]].extend(np.double(data[field_names[j]]))

for i, file_name in enumerate(file_names_training):
        print(i, file_name)
        print('training')
        data = modelfunctions.read_file_to_dict(pathtofiles + str(file_name))
        grid_aftershock_count = np.double(data['aftershocksyn'])
        if len(np.unique(grid_aftershock_count)) < 2:
            continue
        tmp = grid_aftershock_count.tolist()
        training['aftershocksyn'].extend(tmp)
        for j in range(0, len(field_names)):
            training[field_names[j]].extend(np.double(data[field_names[j]]))

field_names.append('aftershocksyn')
modelfunctions.writeHDFfields('Training_Tmp.h5', field_names, training)
modelfunctions.writeHDFfields('Testing_Tmp.h5', field_names, testing)

