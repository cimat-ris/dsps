import argparse
import datetime
import random
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from dipy.io.stateful_tractogram import StatefulTractogram,Space
from dipy.io.streamline import load_tractogram,save_tractogram
from dipy.align.streamlinear import set_number_of_points
from dipy.viz import window, actor, ui, colormap as cmap
from dipy.viz.app import distinguishable_colormap
from models.lstm_classifier import lstmClassifier
from models.tf_classifier import tfClassifier
from models.cnn_classifier import cnnClassifier

raw_data_path = './data/HCP_tracto/'
result_data_path = './data/results/'

clusters_names = ['AC', 'AF_L', 'AF_R', 'CC_MID', 'CGFP_L', 'CGFP_R', 'CGH_L', 'CGH_R', 'CGR_L', 'CGR_R', 'CG_L', 'CG_R', 'CST_L', 'CST_R', 'FA_L', 'FA_R', 'FMA', 'FMI', 'FX_L', 'FX_R', 'IFOF_L', 'IFOF_R', 'ILF_L', 'ILF_R', 'MLF_L', 'MLF_R', 'OR_L', 'OR_R', 'SLF_L', 'SLF_R', 'TAPETUM', 'UF_L', 'UF_R', 'VOF_L', 'VOF_R', 'GARBAGE']
subjects   = ['152831','151425','154936','158843','172029','177645','179245','151728','154229','155938','175237','178142','157942','170631','177241','178950']

def npy_2_tck(streamlines, reference_path, output_path):
    stf = StatefulTractogram(streamlines, reference_path, Space.RASMM)
    save_tractogram(stf, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str, help='Name of subject folder.')
    parser.add_argument('classifier', type=str, help='Classifier to be used.')
    parser.add_argument('--ref', '--reference', default='fa.nii.gz', help='Reference file, Nifti or Trk file. (default: fa.nii.gz)')
    parser.add_argument('--confusion_mtx', action='store_true', help='Display the confusion matrix')
    parser.add_argument('--view_mis', action='store_true', help='Visualize the misclassified')
    parser.add_argument('--train', action='store_true', help='Train the selected network')
    args = parser.parse_args()

    ###
    ### Read Tractogram
    ###
    ref_path = raw_data_path+args.subject+'/'+args.ref
    #full_tract = load_tractogram(raw_data_path+args.subject+'_full_20p.tck', ref_path, bbox_valid_check=False)
    #streamlines = np.array(full_tract.streamlines)

    ###
    ### Reads the .tck files from each specified class
    ##
    gt_streamlines  = []
    gt_labels       = []
    for i,c in enumerate(clusters_names):
        # Load tractogram
        #filename   = path_files+'auto'+c+'.tck'
        filename   = raw_data_path+args.subject+'/'+c+'_20p.tck'
        if not os.path.isfile(filename):
            continue
        print('[INFO] Reading file:',filename)
        tractogram = load_tractogram(filename, './utils/t1.nii.gz', bbox_valid_check=False)
        # Get all the streamlines
        strs  = tractogram.streamlines
        strs  = set_number_of_points(strs,20)
        gt_streamlines.extend(strs)
        gt_labels.extend(len(strs)*[i])
    print('[INFO] Total number of streamlines:',len(gt_streamlines))
    gt_streamlines = np.stack(gt_streamlines,axis=0)
    gt_labels      = np.stack(gt_labels,axis=0)

    ###
    ### Classify data
    ###
    if args.classifier == 'random':
        classifier  = lambda x: np.random.randint(low=-1, high=len(clusters_names)-1, size=x.shape[0])
        predictions = classifier(gt_streamlines)
    elif args.classifier == 'CNN':
        classifier = cnnClassifier(clusters_names)
        classifier.train(subjects,args.subject,raw_data_path,retrain=args.train)
        predictions =  np.argmax(classifier(gt_streamlines), axis=1)
    elif args.classifier == 'LSTM':
        classifier = lstmClassifier(clusters_names)
        classifier.train(subjects,args.subject,raw_data_path,retrain=args.train)
        predictions = np.argmax(classifier(gt_streamlines), axis=1)
    elif args.classifier == 'TF':
        classifier = tfClassifier(clusters_names)
        classifier.train(subjects,args.subject,raw_data_path,retrain=args.train)
        predictions = np.argmax(classifier(gt_streamlines), axis=1)
    elif args.classifier == 'randforest':
        classifier  = None
        predictions = None

    print("[INFO] Accuracy:", len(gt_labels[gt_labels==predictions])/len(gt_streamlines)*100.0)

    ###
    ### Confusion matrix
    ###
    if args.confusion_mtx:
        confusion_mtx = tf.math.confusion_matrix(gt_labels, predictions)
        confusion_mtx = confusion_mtx/(1+tf.math.reduce_sum(confusion_mtx, axis = 1))*10000
        confusion_mtx = tf.cast(confusion_mtx, tf.int32)
        confusion_mtx = confusion_mtx/100

        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx, xticklabels=clusters_names, yticklabels=clusters_names,annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('True label')
        plt.show()

    if args.view_mis:
        # Some visualization of the baddies
        baddies   = (gt_labels!=predictions)
        baddies_x = gt_streamlines[baddies]
        baddies_id= gt_labels[baddies]
        goodies   = (gt_labels==predictions)
        goodies_x = gt_streamlines[baddies]
        goodies_id= gt_labels[baddies]

        # Add display objects to canvas
        scene = window.Scene()
        colors = [c for i, c in zip(range(len(clusters_names)), distinguishable_colormap())]
        for i,c in enumerate(clusters_names):
            if baddies_x[baddies_id==i].shape[0]>0:
                streamlines_actor = actor.line(baddies_x[baddies_id==i],colors[i],linewidth=3, fake_tube=True)
                scene.add(streamlines_actor)
        window.show(scene)

    ###
    ### Cluster streamlines with predictions and save results
    ###
    now = datetime.datetime.now()
    timestamp = str(now.strftime("%Y-%m-%d_%H-%M-%S"))
    for i in range(-1, len(clusters_names)-1):
        cluster = gt_streamlines[predictions==i]
        if cluster.shape[0]==0:
            continue
        out_path = result_data_path+args.subject+'/'+args.classifier+'_'+timestamp+'_'+clusters_names[i]+'.tck'
        npy_2_tck(cluster, ref_path, out_path)
