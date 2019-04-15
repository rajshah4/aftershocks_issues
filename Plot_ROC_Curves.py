

import os
import gc
import numpy as np
import matplotlib.pyplot as plt

import pdb as check
import dill
import matplotlib.gridspec as gridspec

#picklies to load
filenamePickleOtherQuantities =  'AllROC_OtherQuantities.pkl'
filenamePickleANN = 'AllROC_NN.pkl'

#output figure name
figname = 'ROC_Curves' + '.png'

#define figure parameters
fontsize = 20
figlabels = ['a', 'b', 'c', 'd']
cols = 4
rows = 1
gs = gridspec.GridSpec(rows, cols)
fig = plt.figure(facecolor='w', figsize=(18, 22))
plt.rcParams['font.family'] = 'Arial'
counter = 0

# plot NN ROC curve first
print('NN ROC curves first...')
dill.load_session(filenamePickleANN)
eqcount = len(fprdict['ANN']) # number of slip distributions in the testing set

#use 4th subplot on far right
ax = plt.subplot(gs[0, 3])
for ff in range(eqcount): # plot ROC curve for each slip distribution in testing set
    plt.plot(fprdict['ANN'][ff], tprdict['ANN'][ff], color = [0.5, 0.5, 0.5], linewidth=0.8, alpha = 0.5)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2) # plot 1:1 line
plt.plot(fpr, tpr, 'b-', linewidth=4) # plot merged roc curve for testing set
plt.text(.97, 0.06, 'AUC = '+ str(np.round(merge_auc, decimals=4)), ha='right', va='center', fontsize=fontsize) # display AUC value

# Plot decorations
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.rcParams['xtick.labelsize'] = fontsize
ax.set_xticks([0, 1])
plt.rcParams['xtick.labelsize'] = fontsize
ax.set_yticks([])
plt.tick_params(axis='both', which='major', labelsize=fontsize)
spine_color = [0.5, 0.5, 0.5]
counter = counter + 1
ax.set_aspect('equal', adjustable='box')

# Now on to the rest of the metrics
print('now the other metrics...')
dill.load_session(filenamePickleOtherQuantities)
field_names = ['stresses_full_cfs_1', 'stresses_full_max_shear', 'von_mises']
for i in range(0, 3):
     ax = plt.subplot(gs[0, i])
     eqcount = len(fprdict[field_names[i]]) # number of slip distributions in the testing set
     for ff in range(eqcount):
        plt.plot(fprdict[field_names[i]][ff], tprdict[field_names[i]][ff], color = [0.5, 0.5, 0.5], linewidth=0.8, alpha = 0.5) # plot ROC curve for each slip distribution in testing set
     plt.plot([0, 1], [0, 1], 'k--', linewidth=2) # plot 1:1 line
     plt.plot(fpr[i], tpr[i], 'b-', linewidth=4) # plot merged roc curve for testing set
     aucstr = str(np.round(merge_auc[i], decimals=4))
     plt.text(.97, 0.06, 'AUC = '+ aucstr, ha='right', va='center', fontsize=fontsize) # display AUC values

    # Plot decorations
     plt.xlim(0, 1)
     plt.ylim(0, 1)
     plt.rcParams['xtick.labelsize'] = fontsize
     ax.set_xticks([0, 1])
     if i==0:
        plt.rcParams['ytick.labelsize'] = fontsize
        ax.set_yticks([0, 1])
     else: ax.set_yticks([])
     plt.tick_params(axis='both', which='major', labelsize=fontsize)
     spine_color = [0.5, 0.5, 0.5]
     ax.set_aspect('equal', adjustable='box') # for axes equal

plt.savefig(figname, dpi = 400)
plt.close()
