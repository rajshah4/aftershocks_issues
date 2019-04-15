

import os
import copy
import gc
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn
import pdb as check
from collections import defaultdict
import matplotlib.colors as colors
import dill
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
import modelfunctions

def sigmoid(num, shift, scale):
    sig = 1.0/(1.0+np.exp(-1.0*(num*scale-shift)))
    return sig

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

fprdictCFS = defaultdict(list)
tprdictCFS = defaultdict(list)
fprdictANN = defaultdict(list)
tprdictANN = defaultdict(list)
aucdictCFS = defaultdict(list)
aucdictANN = defaultdict(list)

#define useful variables
filenameWeight = '../Data/TheBestWeights.h5'
pathtofiles = '../Data/AllCSV/'

field_names_in = ['stresses_full_xx', 'stresses_full_yy', 'stresses_full_xy', 'stresses_full_xz','stresses_full_yz','stresses_full_zz']

#define figure parameters
scale = 10.0
shift = 1.0
min_val_big = 0.2
max_val_big = 0.8
fontsize = 22
#slip distributions to plot
files = ['1999CHICHI01MAxx_grid.csv','1995KOBEJA01YOSH_grid.csv','2005KASHMI01SHAO_grid.csv']
labels = ['Chi Chi', 'Kobe', 'Kashmir']
sublabels = [['a. Chi Chi', 'b. Kobe', 'c. Kashmir', 'd. ROC curves'] ,['e. Chi Chi', 'f. Kobe', 'g. Kashmir', 'h. ROC curves']]

fig = plt.figure(facecolor='white', figsize=(30, 15), dpi=100)
depth_vec = [-7500., -7500., -12500.,]
plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
gs = gridspec.GridSpec(4, 8,
                       width_ratios=[1, .2, 1, .2, 1, .2, 1, .001],
                       height_ratios=[30, 1, 30, 1]
                       )
rowscale = 2
cmap = plt.get_cmap('Reds')
new_cmap = truncate_colormap(cmap, 0.0, 0.75)

#loop over slip distributions
for filenum, filename in enumerate(files):
    
    #load fault info
    fn = ['x1Utm', 'y1Utm', 'x2Utm', 'y2Utm', 'x3Utm', 'y3Utm', 'x4Utm', 'y4Utm']
    fault = defaultdict()
    for field in fn:
        fault[field] = modelfunctions.readHDF(filename[:-9] + '.h5', field)
    
    #read in data
    file = str(pathtofiles + str(filename))
    data = modelfunctions.read_file_to_dict(file)
    grid_aftershock_count = np.double(data['aftershocksyn'])

    #load model
    model = modelfunctions.create_model()
    model.load_weights(filenameWeight)
    
    #prepare inputs to NN
    IN = modelfunctions.LoadInputsDict(data, field_names_in)
    
    #run NN prediction for this slip distribution
    data['ANN'] = model.predict(IN)

    # field names to plot
    field_names = ['stresses_full_cfs_1', 'ANN']

    num = modelfunctions.FindWhichCFS(data, grid_aftershock_count)
    if num == 1: field_names[0] = 'stresses_full_cfs_1'
    if num == 2: field_names[0] = 'stresses_full_cfs_2'
    if num == 3: field_names[0] = 'stresses_full_cfs_3'
    if num == 4: field_names[0] = 'stresses_full_cfs_4'

    fprdictCFS[filename], tprdictCFS[filename], _ = sklearn.metrics.roc_curve(grid_aftershock_count, np.double(data[field_names[0]]))
    aucdictCFS[filename] = sklearn.metrics.roc_auc_score(grid_aftershock_count, np.double(data[field_names[0]]))
    fprdictANN[filename], tprdictANN[filename], _ = sklearn.metrics.roc_curve(grid_aftershock_count, np.double(data[field_names[1]]))
    aucdictANN[filename] = sklearn.metrics.roc_auc_score(grid_aftershock_count, np.double(data[field_names[1]]))
    
    idx_temp = list(np.where(np.double(data['z']) == depth_vec[filenum]))[0]
    x_temp = np.double([data['x'][_] for _ in idx_temp])
    y_temp = np.double([data['y'][_] for _ in idx_temp])
    grid_aftershock_count_temp = np.double([data['aftershocksyn'][_] for _ in idx_temp])

    for i in range(0,len(field_names)): # 0 is CFS and 1 is NN
       ax = plt.subplot(gs[i*2, filenum*rowscale])
       contour_levels = np.linspace(min_val_big, max_val_big, 100)
       if i==0: #if CFS
           field_tmp = np.double([data[field_names[i]][_] for _ in idx_temp])/1.0e6
           field_temp = np.array([sigmoid(fff, shift, scale) for fff in field_tmp])
       else: #if NN
           field_temp = np.double([data[field_names[i]][_] for _ in idx_temp])
           field_temp = field_temp[:,0]
               
       field_temp[np.where(field_temp>=max_val_big)] = max_val_big - 0.001
       field_temp[np.where(field_temp<min_val_big)] = min_val_big + 0.001
       cs = plt.tricontourf(x_temp, y_temp, field_temp, contour_levels, cmap=new_cmap, origin='lower', hold='on', vmin=min_val_big, vmax=max_val_big, lw = 0.1)
       cs = plt.tricontourf(x_temp, y_temp, field_temp, contour_levels, cmap=new_cmap, origin='lower', hold='on', vmin=min_val_big, vmax=max_val_big)
       plt.clim(min_val_big, max_val_big)

       #deal with scale bar
       if ((filenum == 0) & (i == 0)): #plot scale bar in first subplot
            range_x = np.max(x_temp)-np.min(x_temp)
            range_y = np.max(y_temp)-np.min(y_temp)
            startx = np.min(x_temp)+ 0.76*range_x
            starty = np.min(y_temp)+ 0.3*range_y
            plt.plot([startx, startx+35000], [starty, starty], 'k', linewidth=3)
            plt.text(startx+17500, starty-12000, '35 km', fontsize=fontsize, ha='center', va='center')

       #deal with colorbars
       if ((filenum == 1) & (i == 0)): # plot first color bar in row 1
            colorax = plt.subplot(gs[1, rowscale])
            colorbar1 = Colorbar(ax = colorax, mappable = cs, orientation = 'horizontal', ticklocation = 'bottom', ticks = [0.2, 0.5, 0.8])
            colorbar1.ax.tick_params(labelsize=fontsize)
            clabel = '$\mathrm{sig}(\mathrm{a}\Delta \mathrm{CFS}-\mathrm{b})$'
            colorbar1.set_label(clabel, size=fontsize)
            pos = colorax.get_position() # get the original position
            colorax.set_position([pos.x0, pos.y0+0.017, pos.width, pos.height])
       if ((filenum == 1) & (i == 1)): # plot first color bar in row 3
            colorax = plt.subplot(gs[i*2+1, filenum*rowscale])
            colorbar2 = Colorbar(ax = colorax, mappable = cs, orientation = 'horizontal', ticklocation = 'bottom', ticks = [0.2, 0.5, 0.8])
            colorbar2.ax.tick_params(labelsize=fontsize)
            colorbar2.set_label('$\mathrm{NN}\,\mathrm{output}$', size=fontsize)
            pos = colorax.get_position() # get the original position
            colorax.set_position([pos.x0, pos.y0, pos.width, pos.height])

       ax = plt.subplot(gs[i*2, filenum*rowscale])

       # plot fault plane
       fault_color = [1,215/255,0]
       fault_color2 = [0.4, 0.4, 0.4]
       for iPatch in range(0, len(fault['x1Utm'])): # Plot the edges of each fault patch fault patches
            plt.plot([fault['x1Utm'][iPatch], fault['x2Utm'][iPatch]], [fault['y1Utm'] [iPatch], fault['y2Utm'][iPatch]], color=fault_color, linewidth=6)
            plt.plot([fault['x2Utm'][iPatch], fault['x4Utm'][iPatch]], [fault['y2Utm'][iPatch], fault['y4Utm'][iPatch]], color=fault_color, linewidth=6)
            plt.plot([fault['x1Utm'][iPatch], fault['x3Utm'][iPatch]], [fault['y1Utm'][iPatch], fault['y3Utm'][iPatch]], color=fault_color, linewidth=6)
            plt.plot([fault['x3Utm'][iPatch], fault['x4Utm'][iPatch]], [fault['y3Utm'][iPatch], fault['y4Utm'][iPatch]], color=fault_color, linewidth=6)
       for iPatch in range(0, len(fault['x1Utm'])): # Plot the edges of each fault patch fault patches
            plt.plot([fault['x1Utm'][iPatch], fault['x2Utm'][iPatch]], [fault['y1Utm'] [iPatch], fault['y2Utm'][iPatch]], color=fault_color2, linewidth=1)
            plt.plot([fault['x2Utm'][iPatch], fault['x4Utm'][iPatch]], [fault['y2Utm'][iPatch], fault['y4Utm'][iPatch]], color=fault_color2, linewidth=1)
            plt.plot([fault['x1Utm'][iPatch], fault['x3Utm'][iPatch]], [fault['y1Utm'][iPatch], fault['y3Utm'][iPatch]], color=fault_color2, linewidth=1)
            plt.plot([fault['x3Utm'][iPatch], fault['x4Utm'][iPatch]], [fault['y3Utm'][iPatch], fault['y4Utm'][iPatch]], color=fault_color2, linewidth=1)

       # count and plot aftershocks at the depth of interest
       n_cells = 0
       for i_isc in range(0, len(x_temp)):
          if grid_aftershock_count_temp[i_isc] > 0:
            plt.plot(x_temp[i_isc], y_temp[i_isc], 's', color = [0.0, 0.0, 0.0], markersize=5)
            n_cells += 1

       # add labels
       xpos = [0.07, 0.3, 0.48, 0.67]
       ypos = [0.87, 0.45]
       spacing = 0.022
       stringlabel = sublabels[i][filenum]
       plt.text(xpos[filenum], ypos[i], stringlabel, fontweight = 'bold', fontsize=fontsize, ha='left', va='center', transform=fig.transFigure)
       if i == 0: plt.text(xpos[filenum], ypos[i]-spacing, '$\Delta \mathrm{CFS}(\mathbf{\sigma}, 0.4)$', fontsize=fontsize, ha='left', va='center', transform=fig.transFigure)
       if i == 1: plt.text(xpos[filenum], ypos[i]-spacing, '$\mathrm{NN}$', fontsize=fontsize, ha='left', va='center',transform=fig.transFigure)
       plt.text(xpos[filenum], ypos[i]-2*spacing, 'd = ' + str(abs(depth_vec[filenum])/1e3) + ' km', fontsize=fontsize, ha='left', va='center', transform=fig.transFigure)
       plt.text(xpos[filenum], ypos[i]-3*spacing, '$\mathrm{n}$ = ' + str(np.int64(n_cells)), fontsize=fontsize, ha='left', va='center', transform=fig.transFigure)
       plt.text(xpos[filenum], ypos[i]-4*spacing, '$\mathrm{n}_{\mathrm{tot}}$ = ' + str(np.int64(np.sum(grid_aftershock_count))), fontsize=fontsize, ha='left', va='center', transform=fig.transFigure)
       pos = ax.get_position() # get the original position
       if i == 1: ax.set_position([pos.x0, pos.y0-0.017, pos.width, pos.height])

       #plot decorations
       plt.axis('equal')
       plt.axis('tight')
       plt.axis('scaled')
       plt.axis('off')
       ax.set_xticks([])
       ax.set_yticks([])

#ROC curves
for j in range(0,len(field_names)):
    colors = ['r', 'b', 'k']
    ax = plt.subplot(gs[j*2,len(files)*rowscale])
    plt.plot([0, 1], [0, 1], linestyle='--', dashes=(7, 7), linewidth=2, color=[0.6, 0.6, 0.6])
    if j == 0: # if CFS
        for i in range(0,3): plt.plot(fprdictANN[files[i]], tprdictANN[files[i]], colors[i], linewidth=3, alpha = 0.2)
        for i in range(0,3): plt.plot(fprdictCFS[files[i]], tprdictCFS[files[i]], colors[i], linewidth=5, label = labels[i] + ' - $\Delta \mathrm{CFS}$')
        for i in range(0,3): 
            strlabel = '$\mathrm{AUC}_{\mathrm{' + labels[i] + '}} = $ ' + '%.3f' % (np.round(aucdictCFS[files[i]], decimals=3))
            plt.text(xpos[3], ypos[j]-(i+2)*spacing, strlabel, fontsize=fontsize, ha='left', va='center', transform=fig.transFigure)
    elif j == 1: # if NN
        for i in range(0,3): plt.plot(fprdictCFS[files[i]], tprdictCFS[files[i]], colors[i], linewidth=3, alpha = 0.2)
        for i in range(0,3): plt.plot(fprdictANN[files[i]], tprdictANN[files[i]], colors[i], linewidth=5, label = labels[i] +  ' - $\mathrm{NN}$')
        for i in range(0,3): 
            strlabel = '$\mathrm{AUC}_{\mathrm{' + labels[i] + '}} = $ ' + '%.3f' % (np.round(aucdictANN[files[i]], decimals=3))
            plt.text(xpos[3], ypos[j]-(i+2)*spacing, strlabel, fontsize=fontsize, ha='left', va='center', transform=fig.transFigure)

    # plot decorations
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_aspect('equal', 'box')
    plt.legend(frameon=False, fontsize = fontsize)
    stringlabel = sublabels[j][len(files)]
    plt.text(xpos[3], ypos[j], stringlabel, fontweight = 'bold', fontsize=fontsize, ha='left', va='center', transform=fig.transFigure)
    plt.text(xpos[3], ypos[j]-spacing, '$\mathrm{d}}$ = 0-50 km', fontsize=fontsize, ha='left', va='center', transform=fig.transFigure)
    ax.set_xlabel('fpr', fontsize = fontsize)
    ax.set_ylabel('tpr', fontsize = fontsize)
    pos = ax.get_position() # get the original position
    if j == 0: ax.set_position([pos.x0+0.03, pos.y0, pos.width, pos.height])
    if j == 1: ax.set_position([pos.x0+0.03, pos.y0-0.017, pos.width, pos.height])

plt.savefig('ThreeTestCases.pdf')
plt.close()
gc.collect()
