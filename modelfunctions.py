

import numpy as np
import pdb as check
from collections import defaultdict
import h5py
import sklearn as sklearn
from sklearn import metrics
import csv
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras import initializers
from keras import metrics
#from theano import ifelse

#read in csv file to dictionary
def read_file_to_dict(file_name):
    fp = csv.DictReader(open(file_name))
    data = {}
    for row in fp:
        for column, value in row.items():
            data.setdefault(column, []).append(value)
    return data

#write 2 variables to HDF file
def writeHDF(filename, IN, OUT):
    with h5py.File(filename,'w') as hf:
       hf.create_dataset('IN', data=IN)
       hf.create_dataset('OUT', data=OUT)
    return 0

#write dictionary to HDF file
def writeHDFfields(filename, field_names, data):
    with h5py.File(filename,'w') as hf:
        for field_name in field_names:
            hf.create_dataset(field_name, data=data[field_name])
    return 0

#read a field from an HDF file
def readHDF(filename, field):
    with h5py.File(filename,'r') as hf:
        returnfield = np.squeeze(np.array(hf.get(field)))
    return returnfield

#find best performing sign convention for CFS
def FindWhichCFS(data, aftershock_count):
    auc = np.zeros(4)
    auc[0] = sklearn.metrics.roc_auc_score(aftershock_count.transpose(), np.double(data['stresses_full_cfs_1']))
    auc[1] = sklearn.metrics.roc_auc_score(aftershock_count.transpose(), np.double(data['stresses_full_cfs_2']))
    auc[2] = sklearn.metrics.roc_auc_score(aftershock_count.transpose(), np.double(data['stresses_full_cfs_3']))
    auc[3] = sklearn.metrics.roc_auc_score(aftershock_count.transpose(), np.double(data['stresses_full_cfs_4']))
    num = np.argmax(auc)
    num = num+1
    return num

#permutation test shuffle for NN
def ShuffledAUCS(sims, aftershock_count_shuffle, field):
    auc = np.zeros(sims)
    for iii in range(0, sims):
        # Calculate roc curves with reshuffled earthquake locations
        np.random.shuffle(aftershock_count_shuffle)
        auc[iii] = sklearn.metrics.roc_auc_score(aftershock_count_shuffle.transpose(), field)
    return auc

#load features, assemble, and normalize, and load then labels -- from hdf files
def LoadInputs(filename, field_names_in, field_names_out):
    with h5py.File(filename,'r') as hf:
        INtmp = readHDF(filename, field_names_in[0])
        IN = np.zeros([np.size(INtmp),len(field_names_in)*2])
        for i, field_name in enumerate(field_names_in):
            IN[:,i] = np.abs(readHDF(filename, field_name))
        for i, field_name in enumerate(field_names_in):
            IN[:,len(field_names_in)+i] = -1.0*np.abs(readHDF(filename, field_name))
        OUT = readHDF(filename, field_names_out)
        IN = IN/1.0e6
        return IN, OUT

#load features, assemble, and normalize -- from a dictionary
def LoadInputsDict(data, field_names_in):
        IN = np.zeros([np.size(np.double(data[field_names_in[0]])),len(field_names_in)*2])
        for i, field_name in enumerate(field_names_in):
            IN[:,i] = np.abs(np.double(data[field_name]))
        for i, field_name in enumerate(field_names_in):
            IN[:,len(field_names_in)+i] = -1.0*np.abs(np.double(data[field_name]))
        IN = IN/1.0e6
        return IN

#data generator
def generate_data(POSdata, NEGdata, batch_size, posstart, negstart):
    shapepos = np.shape(POSdata)
    shapeneg = np.shape(NEGdata)
    while 1:
        if posstart + round(batch_size/2.) >= shapepos[0]: 
             posstart = 0
             np.random.shuffle(POSdata)
        posend = posstart + int(round(batch_size/2.))
        if negstart + round(batch_size/2) >= shapeneg[0]: 
             negstart = 0
             np.random.shuffle(NEGdata)
        negend = negstart + int(round(batch_size/2.))       
        data = np.row_stack((POSdata[posstart:posend,:], NEGdata[negstart:negend,:]))
        np.random.shuffle(data)
        posstart = posstart + int(round(batch_size/2.))
        negstart = negstart + int(round(batch_size/2.))
        yield (data[:,:12], data[:,12])

#model setup
def create_model():
    model = Sequential()
    model.add(Dense(50, input_dim=12, kernel_initializer='lecun_uniform', activation = 'tanh'))
    model.add(Dropout(0.50))
    model.add(Dense(50, kernel_initializer='lecun_uniform', activation= 'tanh'))
    model.add(Dropout(0.50))
    model.add(Dense(50, kernel_initializer='lecun_uniform', activation= 'tanh'))
    model.add(Dropout(0.50))
    model.add(Dense(50, kernel_initializer='lecun_uniform', activation= 'tanh'))
    model.add(Dropout(0.50))
    model.add(Dense(50, kernel_initializer='lecun_uniform', activation= 'tanh'))
    model.add(Dropout(0.50))
    model.add(Dense(50, kernel_initializer='lecun_uniform', activation= 'tanh'))
    model.add(Dropout(0.50))
    model.add(Dense(1, kernel_initializer='lecun_uniform', activation='sigmoid'))
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=[metrics.binary_accuracy])
    return model


