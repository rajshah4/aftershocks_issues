

import os
import copy
import gc
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
import pdb as check
from collections import defaultdict
import h5py
import modelfunctions
import dill
import pickle

#load testing data file names
testFileNames = '../Data/Testing_FileNames.h5'
file_names_testing = modelfunctions.readHDF(testFileNames, 'file_names_testing')

#output file name
filenamePickle = './AllROC_OtherQuantities.pkl'

#fields to calculate
field_names_for_indexing = ['stresses_full_cfs_1', 'stresses_full_max_shear', 'von_mises']
field_names = copy.copy(field_names_for_indexing)


fprdict = defaultdict(list)
tprdict = defaultdict(list)
pvalues = defaultdict(list)
aucdict = defaultdict(list)
testingfields = defaultdict(list)

# Loop over each of the csv files in the testing data set
for (ff, file_name) in enumerate(file_names_testing):
      print('Reading in ' + file_name)

      # Read in .csv file to dictionary
      file = '../Data/AllCSV/' + str(file_name)
      data = modelfunctions.read_file_to_dict(file)
      
      # Binarize aftershocks
      grid_aftershock_count = np.double(data['aftershocksyn'])
      aftershock_count = preprocessing.binarize(grid_aftershock_count.reshape(1,-1))[0]
      
      #skip if no aftershocks
      if len(np.unique(aftershock_count)) < 2:
          continue
      
      # find best-performing sign convention for Coulomb failure stress change for this slip distribution
      num = modelfunctions.FindWhichCFS(data, aftershock_count)
      if num == 1: field_names[0] = 'stresses_full_cfs_1'
      if num == 2: field_names[0] = 'stresses_full_cfs_2'
      if num == 3: field_names[0] = 'stresses_full_cfs_3'
      if num == 4: field_names[0] = 'stresses_full_cfs_4'

      # exclude 2010 Darfield slip distribution from Coulomb failure stress change assessment
      if file_name == '2010DARFIE01ATZO_grid.csv':
          testingfields['grid_aftershocks'].extend(aftershock_count.tolist())
      else:
          testingfields['grid_aftershocks'].extend(aftershock_count.tolist())
          testingfields['grid_aftershocks_cfs'].extend(aftershock_count.tolist())

      for i in range(0, len(field_names)):
        # exclude 2010 Darfield from Coulomb failure stress ROC/AUC assessment
        if not (field_names_for_indexing[i] == 'stresses_full_cfs_1' and file_name == '2010DARFIE01ATZO_grid.csv'):
            false_positive_rate, true_positive_rate, _ = metrics.roc_curve(aftershock_count, np.double(data[field_names[i]]))
            auc = metrics.roc_auc_score(aftershock_count, np.double(data[field_names[i]]))
            aucdict[field_names_for_indexing[i]].append(auc)
            tprdict[field_names_for_indexing[i]].append(true_positive_rate)
            fprdict[field_names_for_indexing[i]].append(false_positive_rate)
            testingfields[field_names_for_indexing[i]].extend(np.double(data[field_names[i]]))

print('now merged auc calculations for these three quantities...')
merge_auc = np.zeros(len(field_names))
fpr = []
tpr = []
for i in range(0, len(field_names_for_indexing)):
      if field_names_for_indexing[i] == 'stresses_full_cfs_1':
            grid_aftershocks = np.double(testingfields['grid_aftershocks_cfs'])
      else:
            grid_aftershocks = np.double(testingfields['grid_aftershocks'])
      fieldvals = np.double(testingfields[field_names_for_indexing[i]]);
      false_positive_rate, true_positive_rate, _ = metrics.roc_curve(grid_aftershocks, fieldvals)
      fpr.append(false_positive_rate)
      tpr.append(true_positive_rate)
      merge_auc[i] = metrics.roc_auc_score(grid_aftershocks, fieldvals)
      print(field_names_for_indexing[i], merge_auc[i])

dill.dump_session(filenamePickle)

