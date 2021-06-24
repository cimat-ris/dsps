import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dipy.io.streamline import load_tractogram
from dipy.align.streamlinear import set_number_of_points
from dipy.viz import window, actor, colormap as cmap
import numpy as np
import time
import os
import seaborn as sns

start_time = time.time()

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])


#- function that receives the names of the tck files for each subject and trains and/or
#- outputs the prediction of an especific model based on LOO for a given subject.

def model_raw(model, subjects, classes, test_idx, path_files, fNameRef, checkpoint_prefix, show = True, training = True, epochs = 25):
    print(training)

    train_subjects = [subjects[i] for i in range(len(subjects)) if not i == test_idx]
    n_class = len(classes)

    train_trajs  = []
    train_labels = []
    test_trajs   = []
    test_labels  = []
    # Reads the .tck files from each specified class
    for k,subject in enumerate(train_subjects):
        # Reads the .tck files from each specified class
        for i,c in enumerate(classes):
            if training or k == test_idx:
                # Load tractogram
                #filename   = path_files+'auto'+c+'.tck'
                filename   = path_files+subject+'/'+c+'_20p.tck'
                if not os.path.isfile(filename):
                    continue
                print('[INFO] Reading file:',filename)
                #tractogram = load_tractogram(filename, path_files+fNameRef, bbox_valid_check=False)
                tractogram = load_tractogram(filename, './connectome_test/t1.nii.gz', bbox_valid_check=False)
                # Get all the streamlines
                STs      = tractogram.streamlines
                scaledSTs= set_number_of_points(STs,20)
                if k == test_idx:
                  test_trajs.extend(scaledSTs)
                  test_labels.extend(len(scaledSTs)*[i])
                else:
                  train_trajs.extend(scaledSTs)
                  train_labels.extend(len(scaledSTs)*[i])
    print('[INFO] Used for testing: ',subjects[test_idx])
    print('[INFO] Total number of streamlines:',len(train_trajs))

    train_trajs = np.array(train_trajs)
    test_trajs = np.array(test_trajs)

    train_labels = np.array(train_labels)
    aux = np.zeros([train_labels.shape[0],n_class])
    for i in range(train_labels.shape[0]): aux[i,train_labels[i]] = 1
    train_labels = aux

    if training:  
        # Training
        history = model.fit(train_trajs, train_labels, batch_size=32, epochs, validation_split=0.3)
        checkpoint.save(file_prefix = checkpoint_prefix)
        # Plots
        if show == True:
            plt.figure(figsize=(16, 8))
            plt.subplot(1, 2, 1)
            plot_graphs(history, 'accuracy')
            plt.ylim(None, 1)
            plt.subplot(1, 2, 2)
            plot_graphs(history, 'loss')
            plt.ylim(0, None)
            plt.show()
    else:
        print("[INF] Restoring last model")
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    K = len(test_labels)
    predictions = model.predict(test_trajs)
    preds = []
    for x in predictions: preds.append(np.argmax(x))
    preds = np.array(preds, dtype = int)
    correct = K - np.count_nonzero(preds - test_labels)
    print("accuracy:", correct/K*100)
    confusion_mtx = tf.math.confusion_matrix(test_labels, preds)
    confusion_mtx = confusion_mtx/tf.math.reduce_sum(confusion_mtx, axis = 1)*10000
    confusion_mtx = tf.cast(confusion_mtx, tf.int32)
    confusion_mtx = confusion_mtx/100
    if show == True:
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx, xticklabels=classes, yticklabels=classes,
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('True label')
        plt.show()

    return preds


#------------------------------------------- Main -------------------------------------------

if __name__ == "__main__":

    #------ This section changes what the model will do and the data it will work with ------
    show     = True
    training = True
    epochs = 2
    #index to choose which subject to test on based on LOO
    test_idx = 1

    path_files= './data/'
    fNameRef  = 'autoCC_MID_target_01.nii.gz'

    classes = ['AC','AF_L','AF_R','CGFP_L','CGFP_R','CGH_L','CGH_R','CG_L','CG_R','CGR_R',
              'FA_L','FA_R','FMA','FX_R','FX_L','IFOF_L','IFOF_R','ILF_L','MLF_L','OR_L',
              'SLF_L','UF_L','UF_R','VOF_L','VOF_R','CC_MID','CST_L', 'CST_R','FMI','TAPETUM']

    subjects   = ['152831','151425','154936','158843','172029']
    # subjects   = ['152831','151425','154936','158843','172029','177645','179245','151728',
    #               '154229','155938','175237','178142','157942','170631','177241','178950']


    #---------- This section concerns specifically to the model you are working with --------

    # VERY IMPORTANT: when you change the model, change this variable to change the label that the checkpoints will have
    model_label = 'LSTM'
    #- - - - - - - - - -

    # from tf_tools.transformer.transformer import Transformer

    # # HERE ARE ALL THE HYPERPARAMETERS OF THE TRANSFORMER
    # from tf_tools.parameters import *

    # n_class = len(classes)

    # model = Transformer(d_model, num_layers, num_heads, n_class, dff, rate=0.1)
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    nclasses = len(classes)
    samples = 3
    emb_size= 32
    rnn_size= 32
    int_size= 32
    s_batch = 128
    # Our small recurrent model
    class simpleModel(tf.keras.Model):
        def __init__(self):
            super(simpleModel, self).__init__(name="simpleModel")
            self.embedding = tf.keras.layers.Dense(emb_size)
            self.lstm     = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_size))
            self.dense    = tf.keras.layers.Dense(int_size, activation='relu')
            self.final    = tf.keras.layers.Dense(nclasses, activation='softmax')

        def call(self, inputs):
            x = self.embedding(inputs)
            x = self.lstm(x)
            x = self.dense(x)
            x = self.final(x)
            return x

    model = simpleModel()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    # Checkpoints
    checkpoint_dir    = './checkpoints/'
    checkpoint_prefix = checkpoint_dir + f"ckpt-{model_label}-{subjects[test_idx]}"
    checkpoint        = tf.train.Checkpoint(optimizer=model.optimizer,model=model)

    #------------------------------------ Work! ----------------------------------------------

    model_raw(model, subjects, classes, test_idx, path_files,
            fNameRef, checkpoint_prefix, show, training, epochs)

    