import argparse
import datetime
import random
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram,Space
from dipy.io.streamline import load_tractogram,save_tractogram

raw_data_path = './data/HCP_tracto/'
result_data_path = './data/results/'

clusters_names = ['AC', 'AF_L', 'AF_R', 'CC_MID', 'CGFP_L', 'CGFP_R', 'CGH_L', 'CGH_R', 'CGR_L', 'CGR_R', 'CG_L', 'CG_R', 'CST_L', 'CST_R', 'FA_L', 'FA_R', 'FMA', 'FMI', 'FX_L', 'FX_R', 'IFOF_L', 'IFOF_R', 'ILF_L', 'ILF_R', 'MLF_L', 'MLF_R', 'OR_L', 'OR_R', 'SLF_L', 'SLF_R', 'TAPETUM', 'UF_L', 'UF_R', 'VOF_L', 'VOF_R', 'GARBAGE']

def npy_2_tck(streamlines, reference_path, output_path):
    stf = StatefulTractogram(streamlines, reference_path, Space.RASMM)
    save_tractogram(stf, output_path) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str, help='Name of subject folder.')
    parser.add_argument('classifier', type=str, help='Classifier to be used.')
    parser.add_argument('--ref', '--reference', default='fa.nii.gz', help='Reference file, Nifti or Trk file. (default: fa.nii.gz)')
    args = parser.parse_args()

    ###
    ### Read Tractogram
    ###
    ref_path = raw_data_path+args.subject+'/'+args.ref
    full_tract = load_tractogram(raw_data_path+args.subject+'_full_20p.tck', ref_path, bbox_valid_check=False)
    streamlines = np.array(full_tract.streamlines)

    ###
    ### Classify data
    ###
    if args.classifier == 'random':
        classifier = lambda x: np.random.randint(low=-1, high=len(clusters_names)-1, size=x.shape[0])
    elif args.classifier == 'CNN':
        classifier = None
    elif args.classifier == 'LSTM':
        classifier = None
    elif args.classifier == 'transformer':
        classifier = None
    elif args.classifier == 'randforest':
        classifier = None
    predictions = classifier(streamlines)

    ###
    ### Cluster streamlines with predictions and save results
    ###
    now = datetime.datetime.now()
    timestamp = str(now.strftime("%Y-%m-%d_%H-%M-%S"))
    for i in range(-1, len(clusters_names)-1):
        cluster = streamlines[predictions==i]
        out_path = result_data_path+args.subject+'/'+args.classifier+'_'+timestamp+'_'+clusters_names[i]+'.tck'
        npy_2_tck(cluster, ref_path, out_path)