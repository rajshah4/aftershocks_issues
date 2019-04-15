

import os
import copy
import gc
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
import pdb as check
from collections import defaultdict
import h5py
import dill
import pickle
import time
import modelfunctions

#load testing data file names
testFileNames = '../Data/Testing_FileNames.h5'
file_names_testing = modelfunctions.readHDF(testFileNames, 'file_names_testing')

#weight file name
filenameWeight = '../Data/TheBestWeights.h5'

#output file name
filenamePickle = './AllROC_NN.pkl'

#load model
model = modelfunctions.create_model()
model.load_weights(filenameWeight)

fprdict = defaultdict(list)
tprdict = defaultdict(list)
pvalues = defaultdict(list)
aucdict = defaultdict(list)
testingfields = defaultdict(list)

#features
field_names_in = ['stresses_full_xx', 'stresses_full_yy', 'stresses_full_xy', 'stresses_full_xz','stresses_full_yz','stresses_full_zz']

#number of realizations for permutation test
NumRandomRealizations = 5000
# Loop over each of the csv files in the testing data set
for (ff, file_name) in enumerate(file_names_testing):
    print('Reading in ' + file_name)
    
    # Read in .csv file to dictionary
    file = '../Data/AllCSV/' + str(file_name)
    data = modelfunctions.read_file_to_dict(file)
    
    # Binarize aftershocks
    grid_aftershock_count = np.double(data['aftershocksyn'])
    aftershock_count = preprocessing.binarize(grid_aftershock_count.reshape(1,-1))[0]

    #copy aftershock information
    aftershock_count_shuffle = copy.copy(aftershock_count)
    np.random.shuffle(aftershock_count_shuffle)
        
    #skip if no aftershocks
    if len(np.unique(aftershock_count)) < 2:
        continue
    
    #load inputs/features for NN forward prediction
    IN = modelfunctions.LoadInputsDict(data, field_names_in)

    #run forward prediction
    fieldvalsEQ = model.predict(IN)

    #assess performance
    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(aftershock_count, fieldvalsEQ)
    auc = metrics.roc_auc_score(aftershock_count, fieldvalsEQ)
    aucs_shuffled = modelfunctions.ShuffledAUCS(NumRandomRealizations, aftershock_count_shuffle, fieldvalsEQ)
    biggies = np.where(aucs_shuffled >= auc)
    pvalues['ANN'].append(np.float(biggies[0].size)/np.float(NumRandomRealizations))
    aucdict['ANN'].append(auc)
    tprdict['ANN'].append(true_positive_rate)
    fprdict['ANN'].append(false_positive_rate)
    testingfields['ANN'].extend(np.double(fieldvalsEQ.transpose()[0]))
    testingfields['grid_aftershocks'].extend(aftershock_count.tolist())

#calculate merged ROC curves and AUC value based on data assembled here
fpr, tpr, _ = metrics.roc_curve(np.double(testingfields['grid_aftershocks']), np.double(testingfields['ANN']))
merge_auc = metrics.roc_auc_score(np.double(testingfields['grid_aftershocks']), np.double(testingfields['ANN']))

print('merged auc value: ' + str(merge_auc))

dill.dump_session(filenamePickle)

