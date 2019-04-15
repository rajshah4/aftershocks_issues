

import gc
import numpy as np
import sklearn as sklearn
import pdb as check
import modelfunctions

#set name of weights/biases file
weightFile = '../Data/TheBestWeights.h5'
predFile = './Predicted.h5'
testFile = '../Data/Testing.h5'

#names of features
field_names_in = ['stresses_full_xx', 'stresses_full_yy', 'stresses_full_xy', 'stresses_full_xz','stresses_full_yz','stresses_full_zz']

#name of label
field_names_out = 'aftershocksyn'

#load model
model = modelfunctions.create_model()

#assess performance of training network on testing data set
model.load_weights(weightFile)
inTrue, outTrue = modelfunctions.LoadInputs(testFile, field_names_in, field_names_out)
outPred = model.predict(inTrue)
modelfunctions.writeHDF(predFile, inTrue, outPred)
auc = sklearn.metrics.roc_auc_score(outTrue, outPred)
print('merged AUC on testing data set: ' + str(auc))

