

import copy
import gc
import numpy as np
import pdb as check
import random
import modelfunctions
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

#set name of weights file and training/testing files
weightFile = '../Data/TheBestWeights.h5'
trainFile = '../Data/Training.h5'
testFile = '../Data/Testing.h5'

#names of features
field_names_in = ['stresses_full_xx', 'stresses_full_yy', 'stresses_full_xy', 'stresses_full_xz','stresses_full_yz','stresses_full_zz']

#name of label
field_names_out = 'aftershocksyn'

#load training data set
IN_Train, OUT_Train = modelfunctions.LoadInputs(trainFile, field_names_in, field_names_out)

#pull out some validation data with positive grid cells upsampled so that validation data has equal numbers of positive and negative samples
posidx = np.where(OUT_Train==1)
numpos = np.size(posidx)
negidx = np.where(OUT_Train==0)
numneg = np.size(negidx)

#divide data into positive and negative samples
POSdata = np.column_stack((IN_Train[posidx,:][0], OUT_Train[posidx].T))
NEGdata = np.column_stack((IN_Train[negidx,:][0], OUT_Train[negidx].T))

np.random.seed(42) #shuffle order of samples (same way every time to ensure validation data does not change)
np.random.shuffle(POSdata)
np.random.shuffle(NEGdata)
np.random.seed() #reseed

cutoff = int(round(numpos/10)) #validation data consists of a random 10% of positive samples, and same number of randomly selected negative samples

#positive validation samples
val_in1 = copy.copy(POSdata[:cutoff,:len(field_names_in)*2])
val_out1 = copy.copy(POSdata[:cutoff,len(field_names_in)*2])

#negative validation samples
val_in2 = copy.copy(NEGdata[:cutoff,:len(field_names_in)*2])
val_out2 = copy.copy(NEGdata[:cutoff,len(field_names_in)*2])

#merge to obtain entire validation data set
val_in = np.row_stack((val_in1, val_in2))
val_out = np.append(val_out1, val_out2)

#the remaining data set is the training data set
POSdataFinal = copy.copy(POSdata[cutoff:,:])
NEGdataFinal = copy.copy(NEGdata[cutoff:,:])

shapepos = np.shape(POSdataFinal)

#set hyperparameters
batch_size = 3500
steps_per_epoch = int(round((shapepos[0])/batch_size)) #total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch, equal to the number of samples in dataset divided by the batch size
epoch_num = 5
posstart = 0
negstart = 0

#train
model = modelfunctions.create_model()
checkpointer = ModelCheckpoint(filepath=weightFile, monitor = 'val_loss', verbose=2, save_best_only = True)
history = model.fit_generator(modelfunctions.generate_data(POSdataFinal, NEGdataFinal, batch_size, posstart, negstart), steps_per_epoch, validation_data = (val_in, val_out),  callbacks = [checkpointer], verbose=2, epochs=epoch_num)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = list(range(1,len(loss)+1))
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.xlabel('Loss')
plt.legend()
plt.show()
