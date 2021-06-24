#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:31 2021

@author: Armando Cruz (Gory)
"""

import glob
import time
import argparse
import numpy as np
from dipy.io.streamline import load_tractogram
from dipy.viz import window, actor

clusters_labels = {
    'AC': 0, #
    'AF_L': 1, 
    'AF_R': 2, 
    'CC_MID': 3, 
    'CGFP_L': 4, #
    'CGFP_R': 5,#
    'CGH_L': 6,#
    'CGH_R': 7,#
    'CGR_L': 8, #
    'CGR_R': 9, #
    'CG_L': 10, 
    'CG_R': 11, 
    'CST_L': 12, 
    'CST_R': 13, 
    'FA_L': 14, 
    'FA_R': 15, 
    'FMA': 16, 
    'FMI': 17, 
    'FX_L': 18, 
    'FX_R': 19, 
    'IFOF_L': 20, 
    'IFOF_R': 21, 
    'ILF_L': 22, 
    'ILF_R': 23, 
    'MLF_L': 24, 
    'MLF_R': 25, 
    'OR_L': 26, 
    'OR_R': 27, 
    'SLF_L': 28, 
    'SLF_R': 29, 
    'TAPETUM': 30, 
    'UF_L': 31, 
    'UF_R': 32, 
    'VOF_L': 33, 
    'VOF_R': 34
}

raw_data_path = './HCP_tracto/'
training_data_path = './training/'

errors_2 = np.array([ [x1, x2, x3, x4, x5, x6] for x1 in range(2) for x2 in range(2) for x3 in range(2)
                                    for x4 in range(2) for x5 in range(2) for x6 in range(2)], dtype=np.int32)
errors_3 = np.array([ [x1, x2, x3, x4, x5, x6] for x1 in range(-1,2) for x2 in range(-1,2) for x3 in range(-1,2)
                                    for x4 in range(-1,2) for x5 in range(-1,2) for x6 in range(-1,2)], dtype=np.int32)
                            
epsilon = 0.05
streams_in_file = 100000

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str, help='Name of subject folder.')
    parser.add_argument('--ref', '--reference', default='fa.nii.gz', help='Reference file, Nifti or Trk file. (default: fa.nii.gz)')
    parser.add_argument('--npoints',type=int , default='20', help='Number of points in each streamline. (default: 20)')
    parser.add_argument('--save', default='True', help='Save results into memory. (default: True)')
    parser.add_argument('--show_notfound', default='False', help='Show best candidate to notfound streamline. (default: True)')
    args = parser.parse_args()

    TIC = time.time()
    print('-------------------')
    print('Loading full tractogram')
    print('-------------------')
    tic = time.time()
    full_tract = load_tractogram(raw_data_path+args.subject+'_full_20p.tck', raw_data_path+args.subject+'/'+args.ref, bbox_valid_check=False)
    full_streamlines = np.array(full_tract.streamlines)
    full_streamlines_endpts = np.floor(full_streamlines[:,(0,-1)]/epsilon).reshape((full_streamlines.shape[0], 2*3))
    full_streamlines_endpts2 = np.floor(full_streamlines[:,(-1,0)]/epsilon).reshape((full_streamlines.shape[0], 2*3))
    labeled = np.zeros(len(full_streamlines), dtype=bool)
    y = np.ones(len(full_streamlines),dtype=np.int32)*-1
    toc = time.time()
    print('--Loading full tractogram time:', toc-tic)
    print('Full tracto size:', len(full_streamlines))
    
    clusters_paths = glob.glob(raw_data_path+args.subject+'/*.tck')
    clusters_names = [ (path.split('/')[-1])[:-8] for path in clusters_paths ]
    not_found = []
    not_found_endpts = []
    not_found_labels = []

    for c in range(len(clusters_paths)):
        cluster_path = clusters_paths[c]
        cluster_name = clusters_names[c]
        cluster_label = clusters_labels[cluster_name]
        if cluster_label == -1:
            continue

        print('-------------------')
        print('Loading cluster:', cluster_name, c)
        print('-------------------')
        tic = time.time()
        tractogram = load_tractogram(cluster_path, raw_data_path+args.subject+'/'+args.ref, bbox_valid_check=False)
        cluster = np.array(tractogram.streamlines)
        if len(cluster) == 0:
            continue
        cluster_endpts = np.floor(cluster[:,(0,-1)]/epsilon).reshape((cluster.shape[0], 2*3))
        cluster_endpts2 = np.floor(cluster[:,(-1,0)]/epsilon).reshape((cluster.shape[0], 2*3))
        endpts_error = list(map(tuple, cluster_endpts)) + list(map(tuple, cluster_endpts2))
        cluster_set = set( endpts_error )
        toc = time.time()
        print('--Loading cluster time', cluster_name, toc-tic)
        print('Cluster size:', len(cluster))

        print('--------------------')
        print('-First Labeling:', cluster_name, c)
        print('--------------------')
        tic = time.time()
        smalltic = tic
        labeled_set = set()
        for i, sl in enumerate(full_streamlines_endpts):
            sltuple = tuple(sl)
            if sltuple in cluster_set:
                labeled[i] = True
                y[i] = cluster_label
                labeled_set.add(sltuple)
        toc = time.time()
        print('--First labeling time:', toc-tic)
        print('number of first labeled streamlines:', len(labeled_set))
        if len(labeled_set) == len(cluster):
            continue

        print('--------------------')
        print('-Second Labeling:', cluster_name, c)
        print('--------------------')
        tic = time.time()
        # endpts_error = []
        # for endpts in cluster_endpts:
        #     endpts_poserror = list(map(tuple, endpts + errors))
        #     endpts_negerror = list(map(tuple, endpts - errors))
        #     endpts_error += endpts_poserror + endpts_negerror
        endpts_error = []
        for endpts in cluster_endpts:
            endpts_error += list(map(tuple, endpts + errors_3))
        for endpts in cluster_endpts2:
            endpts_error += list(map(tuple, endpts + errors_3))
        cluster_set = set( endpts_error )
        for i, sl in enumerate(full_streamlines_endpts):
            sltuple = tuple(sl)
            if sltuple in cluster_set:
                labeled[i] = True
                y[i] = cluster_label
                labeled_set.add(sltuple)
        toc = time.time()
        print('--Second labeling time:', toc-tic)
        print('number of first and second labeled streamlines:', len(labeled_set))
        if len(labeled_set) == len(cluster):
            continue

        print('--------------------')
        print('-Missing labeled:', len(cluster)-len(labeled_set))
        print('--------------------')
        tic = time.time()
        for i, csl in enumerate(cluster_endpts):
            csltuple = tuple(csl)
            if csltuple not in labeled_set:
                not_found.append(cluster[i])
                not_found_endpts.append(csl)
                not_found_labels.append(cluster_label)
        toc = time.time()
        print('--Unlabeled detection time', toc-tic)

    print('--------------------')
    print('Brute force missing labeled')
    print('--------------------')
    tic = time.time()
    nshow = 2
    notfound_show = []
    best_candidate_show = []
    best_candidate_show2 = []
    for i, csl in enumerate(not_found_endpts[:nshow]):
        norms = np.linalg.norm(full_streamlines_endpts-csl.reshape(1,csl.shape[0]), axis=1)
        best_candidate = np.argmin( norms )
        notfound_show.append(not_found[i])
        best_candidate_show.append(full_streamlines[best_candidate])
        if norms[best_candidate] < epsilon:
            labeled[best_candidate] = True
            print('HEUREKA')
            continue
        norms = np.linalg.norm(full_streamlines_endpts2-csl.reshape(1,csl.shape[0]), axis=1)
        best_candidate2 = np.argmin( norms )
        best_candidate_show2.append(full_streamlines[best_candidate2])
        if norms[best_candidate] < epsilon:
            labeled[best_candidate] = True
            print('HEUREKA')
            continue
    toc = time.time()
    print('--Bruteforce labeling time', toc-tic)

    if args.show_notfound == 'True':
        # Visualization
        color = np.ones((nshow,3)) * np.array([1.0, 0.0, 0.0]).reshape((1,3))
        notfound_actor = actor.line(notfound_show, color)
        color = np.ones((nshow,3)) * np.array([0.0, 1.0, 0.0]).reshape((1,3))
        best_candidate_actor = actor.line(best_candidate_show, color)
        color = np.ones((nshow,3)) * np.array([0.0, 0.0, 1.0]).reshape((1,3))
        best_candidate_actor2 = actor.line(best_candidate_show2, color)
        # Add display objects to canvas
        scene = window.Scene()
        scene.add(notfound_actor)
        scene.add(best_candidate_actor)
        scene.add(best_candidate_actor2)
        window.show(scene)

    if args.save == 'True':
        print('--------------------')
        print('Save results in memory')
        print('--------------------')
        tic = time.time()
        for i in range(0,len(full_streamlines),streams_in_file):
            x_training = full_streamlines[i:i+streams_in_file]
            y_training = y[i:i+streams_in_file]
            with open(training_data_path+str(args.npoints)+'/'+args.subject+'_'+str(i//streams_in_file)+'.npy', 'w+b') as f:
                np.save(f, x_training)
                np.save(f, y_training)
        x_notfoun = np.array(np.array(not_found))
        y_notfoun = np.array(np.array(not_found_labels))
        with open(training_data_path+str(args.npoints)+'/'+args.subject+'_notfound.npy', 'w+b') as f:
            np.save(f, x_notfoun)
            np.save(f, y_notfoun)
        toc = time.time()
        print('-Save results time:', toc-tic)


    print('--------------------')
    print('--------------------')
    print('RESULTS')
    print('--------------------')
    print('--------------------')
    TOC = time.time()
    print('--Data generation time:', TOC-TIC)
    print('Labeled streamlines:', np.sum(labeled))
    print('--Total missing labeled:', len(not_found))