import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dipy.io.streamline import load_tractogram
from dipy.align.streamlinear import set_number_of_points
from dipy.viz import window, actor, colormap as cmap
import numpy as np
import time
import os
from models.tf_tools.transformer.transformer import Transformer
# HERE ARE ALL THE HYPERPARAMETERS OF THE TRANSFORMER
from models.tf_tools.parameters import *

start_time = time.time()

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

# Our small recurrent model
class tfClassifier(Transformer):
    def __init__(self,classes):
        super(tfClassifier, self).__init__(d_model, num_layers, num_heads, len(classes), dff, rate=0.1)
        self.classes = classes

    def train(self,subjects,test_subject,path_files,retrain=True):
        self.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        # Checkpoints
        checkpoint_dir   = './checkpoints-TF'+'-'+test_subject
        checkpoint_prefix= os.path.join(checkpoint_dir,"ckpt")
        checkpoint       = tf.train.Checkpoint(optimizer=self.optimizer,model=self)

        if retrain==True:
            train_trajs  = []
            train_labels = []
            val_trajs  = []
            val_labels = []
            for k,subject in enumerate(subjects):
                print('[INFO] Reading subject:',subject)                
                # Reads the .tck files from each specified class
                for i,c in enumerate(self.classes):
                    # Load tractogram
                    #filename   = path_files+'auto'+c+'.tck'
                    filename   = path_files+subject+'/'+c+'_20p.tck'
                    if not os.path.isfile(filename):
                        continue
                    print('[INFO] Reading file:',filename)
                    #tractogram = load_tractogram(filename, path_files+fNameRef, bbox_valid_check=False)
                    tractogram = load_tractogram(filename, './utils/t1.nii.gz', bbox_valid_check=False)
                    # Get all the streamlines
                    STs      = tractogram.streamlines
                    scaledSTs= set_number_of_points(STs,20)
                    if subject==test_subject:
                        val_trajs.extend(scaledSTs)
                        val_labels.extend(len(scaledSTs)*[i])
                    else:
                        train_trajs.extend(scaledSTs)
                        train_labels.extend(len(scaledSTs)*[i])

            print('[INFO] Used for testing: ',test_subject)
            print('[INFO] Total number of streamlines for training:',len(train_trajs))
            print('[INFO] Total number of streamlines for validation:',len(val_trajs))
            train_trajs = np.array(train_trajs)
            val_trajs   = np.array(val_trajs)
            train_labels= np.array(train_labels)
            aux = np.zeros([train_labels.shape[0],len(self.classes)])
            for i in range(train_labels.shape[0]): aux[i,train_labels[i]] = 1
            train_labels = aux
            # Training
            history = self.fit(train_trajs, train_labels, batch_size=32, epochs=25, validation_split=0.3)
            checkpoint.save(file_prefix = checkpoint_prefix)
            # Plots
            plt.figure(figsize=(16, 8))
            plt.subplot(1, 2, 1)
            plot_graphs(history, 'accuracy')
            plt.ylim(None, 1)
            plt.subplot(1, 2, 2)
            plot_graphs(history, 'loss')
            plt.ylim(0, None)
            plt.show()
        else:
            # To avoid training, we can just load the parameters we saved in the previous session
            print("[INFO] Restoring last model")
            status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
