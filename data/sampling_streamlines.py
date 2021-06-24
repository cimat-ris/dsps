#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 4:17 2021

@author: Armando Cruz (Gory)
"""

import sys
import argparse
import datetime
import numpy as np
from dipy.io.streamline import load_tractogram
from dipy.viz import window, actor, colormap as cmap
from dipy.tracking.streamline import set_number_of_points

raw_data_path = './HCP_tracto/'
lenseg_data_path = './lenght_segmentation/'
uniseg_data_path = './uniform_segmentation/'
lenseg_img_path = './img/lenght_segmentation/'
uniseg_img_path = './img/uniform_segmentation/'

def streamline_parametrization(streamline, distances, cum_distances, length):
    def param(index):
        t = (index*length)
        # print('t:', t)
        i = np.searchsorted(cum_distances, t)
        i[-1] = len(streamline)-1
        i[0] = 1
        # print('i:', i)
        delta = (t - cum_distances[i-1])/distances[i-1]
        delta[0] = 0
        delta[-1] = 1.0
        delta = delta.reshape((delta.shape[0],1))
        # print('delta:', delta)
        return streamline[i-1]*(1-delta) + streamline[i]*delta
    return param

def lenght_segmentation(streamline, length):
    distances = np.fromfunction(lambda i: np.linalg.norm(streamline[i] - streamline[i+1], axis=1) , (len(streamline)-1,), dtype=np.int32)
    cum_distances = np.cumsum(np.insert(distances,0,0))
    n = np.int32(np.floor(cum_distances[-1]/length))+1
    # print('Streamline distances:')
    # print(distances)
    # print('Streamline cum_distances:')
    # print(cum_distances)
    sl_param = streamline_parametrization(streamline, distances, cum_distances, length)
    return np.fromfunction(sl_param, (n+1,))

def uniform_segmentation(streamline, n):
    distances = np.fromfunction(lambda i: np.linalg.norm(streamline[i] - streamline[i+1], axis=1) , (len(streamline)-1,), dtype=np.int32)
    cum_distances = np.cumsum(np.insert(distances,0,0))
    length = np.sum(distances)/n
    # print('Streamline distances:')
    # print(distances)
    # print('Streamline cum_distances:')
    # print(cum_distances)
    # print(len(cum_distances), len(streamline))
    sl_param = streamline_parametrization(streamline, distances, cum_distances, length)
    return np.fromfunction(sl_param, (n+1,))

def uniform_segmentation_dipy(streamline, n):
    return set_number_of_points(streamline, nb_points=n+1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', default='s1', help='Name of the subject. (default: s1)')
    parser.add_argument('--tract', '--tractogram', default='FMI.tck', help='Tractogram file, .tck extension. (default: FMI.tck)')
    parser.add_argument('--ref', '--reference', default='t1.nii.gz', help='Reference file, Nifti or Trk file. (default: t1.nii.gz)')
    parser.add_argument('--seg_type', default='uniform', help='Tipe of streamline segmentation: uniform or length. (default: uniform)')
    parser.add_argument('--size', default='4', help='Size of step or number of streamline componnents. (default: 2)')
    parser.add_argument('--n_lines', default='2', help='Number of streamline to segment. (default: 2)')
    parser.add_argument('--save_picture', default='False', help='Indicates if segmentation output is gonna be saved in an image. (default:False')
    args = parser.parse_args()

    now = datetime.datetime.now()
    timestamp = str(now.strftime("%Y%m%d_%H%M%S"))
    image_name = '_'.join([args.subject, args.size, args.n_lines, timestamp]) + '.png'

    tractogram = load_tractogram(raw_data_path+args.subject+'/'+args.tract, raw_data_path+args.subject+'/'+args.ref, bbox_valid_check=False)
    streamlines = tractogram.streamlines[:int(args.n_lines)]
    segmented_streamlines = []
    segmented_streamlines_dipy = []

    if args.seg_type == 'uniform':
        out_img_path = uniseg_img_path
        segmentation = uniform_segmentation
        size = int(args.size)
    elif args.seg_type == 'length':
        out_img_path = lenseg_img_path
        segmentation = lenght_segmentation
        size = float(args.size)

    for streamline in streamlines:
        segmented_streamlines.append(segmentation(streamline, size))
        segmented_streamlines_dipy.append(uniform_segmentation_dipy(streamline, size))

    print('hey')
    print(segmented_streamlines_dipy)

    # Visualization
    color = np.ones((int(args.n_lines),3)) * np.array([1.0, 0.0, 0.0]).reshape((1,3))
    streamlines_actor = actor.line(np.array(streamlines, dtype=object), color)
    color = np.ones((int(args.n_lines),3)) * np.array([0.0, 1.0, 0.0]).reshape((1,3))
    segmented_streamlines_actor = actor.line(segmented_streamlines, color)
    color = np.ones((int(args.n_lines),3)) * np.array([0.0, 0.0, 1.0]).reshape((1,3))
    segmented_streamlines_dipy_actor = actor.line(segmented_streamlines_dipy, color)
    # Add display objects to canvas
    scene = window.Scene()
    scene.add(streamlines_actor)
    scene.add(segmented_streamlines_actor)
    scene.add(segmented_streamlines_dipy_actor)
    if args.save_picture == 'True':
        window.record(scene, out_path=out_img_path+image_name, size=(600, 600))
    window.show(scene)