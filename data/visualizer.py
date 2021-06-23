#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 3:02 2021

@author: Armando Cruz (Gory)
"""

import sys
import argparse
import numpy as np
from dipy.io.streamline import load_tractogram
from dipy.viz import window, actor, colormap as cmap

path_raw_data = './HCP_tracto/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--subject', default='s1', help='Name of the subject. (default: s1)')
    parser.add_argument('--tract', '--tractogram', default='FMI.tck', help='Tractogram file, .tck extension. (default: FMI.tck)')
    parser.add_argument('--ref', '--reference', default='t1.nii.gz', help='Reference file, Nifti or Trk file. (default: t1.nii.gz)')

    args = parser.parse_args()
    
    tractogram = load_tractogram(path_raw_data+args.subject+'/'+args.tract, path_raw_data+args.subject+'/'+args.ref, bbox_valid_check=False)
    streamlines = tractogram.streamlines
    first_streamline = streamlines[0]

    print('Number of streamlines')
    print(len(streamlines))

    print('Number of 3D point in 1st streamline')
    npuntos = len(first_streamline)
    print(npuntos)

    print('Coordinates of the first 2 points from 1st streamline')
    x0, y0, z0 = first_streamline[0]
    print(x0,y0,z0)
    x1, y1, z1 = first_streamline[1]
    print(x1,y1,z1)

    # Visualization
    color = cmap.line_colors(streamlines)
    streamlines_actor = actor.line(np.array(streamlines, dtype=object), color)
    # Add display objects to canvas
    scene = window.Scene()
    scene.add(streamlines_actor)
    window.show(scene)