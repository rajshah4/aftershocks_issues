

import csv
import os
import copy
import gc
import numpy as np
import pdb as check
import h5py
import modelfunctions

filenameWeight = '../Data/TheBestWeights.h5'

degvec = np.arange(0, 360, 10)
for deg in degvec:
    IN = modelfunctions.readHDF('./IN_IdealizedOkadaCase_' + str(int(deg)) + '.h5', 'IN')
    shp = np.shape(IN)
    INFinal = np.zeros([shp[0], shp[1]*2])
    INFinal[:,:6] = np.abs(IN)
    INFinal[:,6:] = -1.*np.abs(IN)
    model = modelfunctions.create_model()
    model.load_weights(filenameWeight)
    fieldvalsEQ = model.predict(INFinal)
    modelfunctions.writeHDF('./NN_Outputs_IdealizedOkadaCase_' + str(int(deg)) + '.h5', IN, fieldvalsEQ)
