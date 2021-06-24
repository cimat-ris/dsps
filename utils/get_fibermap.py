# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:11:34 2021

@author: Ioana
"""

import numpy as np
import tensorflow as tf
from dipy.io.streamline import load_tractogram

## Load data and labels as per Jean-Bernard's/Juan Jose's code

# function to convert each streamline into a fibermap

def get_fibermap(all_trajs, n):

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

# run function with streamline data
map1 = get_fibermap(streamlines, 20)

# convert to tensor for CNN with labels
dataset = tf.data.Dataset.from_tensor_slices((map1,all_labels)) # for CNN input
