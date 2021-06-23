# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:11:34 2021

@author: Ioana
"""

import numpy as np
import tensorflow as tf
from dipy.io.streamline import load_tractogram

path_files= '/home/jbhayet/Desktop/data/001/results_res/151425/'
#gen_path_machine = 'C:/Users/Ioana/Desktop/'
#path_files      = gen_path_machine + 'DSPS/178950/'
#pathToReference = gen_path_machine + 'DSPS/178950/'
pathToReference = './'
show = True

# only for one patient - update to include all

fNameRef  = 't1.nii.gz'

# check/update
classes = ['CST_L_20p', 'FA_L_20p', 'AF_R_20p', 'OR_L_20p', 'FX_L_20p', 'IFOF_L_20p']
classes = ['AC','AF_L','AF_R','CGFP_L','CGFP_R','CGH_L','CGH_R','CG_L','CG_R','CGR_R','FA_L','FA_R','FMA','FX_R','IFOF_L','IFOF_R','ILF_L','MLF_L','OR_L','SLF_L','UF_L','UF_R','VOF_L','VOF_R']

# following Jean-Bernard's LSTM code
all_trajs  = []
all_labels = []
# Reads the .tck files from each specified class
for i,c in enumerate(classes):
    # Load tractogram
    filename   = path_files+c+'_20p.tck'
    print('Reading file:',filename)
    tractogram = load_tractogram(filename, fNameRef, bbox_valid_check=False)
    # Get all the streamlines
    STs = tractogram.streamlines
    all_trajs.extend(STs)
    all_labels.extend(len(STs)*[i])
print('Total number of streamlines:',len(all_trajs))


n = 20 # number of points in streamline
data = all_trajs

# function to convert each streamline into a fibermap

def get_fibermap(data, n):

    fiber_map = np.zeros([2*n, 2*n, 3]) # define empty map for each streamline
    all_fibre_map = np.zeros([len(all_trajs), 2*n, 2*n, 3]) # to store all maps

    for j in range(len(all_trajs)):

        data = all_trajs[j] # choose one streamline

        for i in range(3): # for each dimension in streamline
            stream = data[:,i]
            stream_rev = stream[::-1] # reverse

            block1 = np.concatenate((stream, stream_rev), axis = 0) # build blocks
            block2 = np.concatenate((stream_rev, stream), axis = 0)

            cell = np.vstack((block1, block2)) # stack vertically

            fiber_slice = np.tile(cell, (n,1)) # create fiber map

            fiber_map[:,:,i] = fiber_slice # assign to map for each dimension

        all_fibre_map[j,:,:,:] = fiber_map # save all maps from all streamlines

    return all_fibre_map

map1 = get_fibermap(data, 20)
print(map1[0])
dataset = tf.data.Dataset.from_tensor_slices((map1,all_labels)) # for CNN input
