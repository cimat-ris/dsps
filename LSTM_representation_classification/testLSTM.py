#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jbhayet
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dipy.io.streamline import load_tractogram
from dipy.align.streamlinear import set_number_of_points
from dipy.viz import window, actor, ui, colormap as cmap
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

show      = True
training  = False
#path_files= './connectome_test/'
path_files= '/home/jbhayet/opt/repositories/devel/dsps/data/151425/'
path_files= '/home/jbhayet/opt/repositories/devel/dsps/data/155938/'
path_files= '/home/jbhayet/Desktop/data/001/results_res/'
fNameRef  = 't1.nii.gz'
fNameRef  = 'autoFA_L_exclude_interp.nii.gz'
fNameRef  = 'autoCC_MID_target_01.nii.gz'

#classes = ['CG_L','CG_R','CGH_L','CGH_R','CGR_L','CGR_R','CST_L','CST_R','FA_L','FA_R','FMA','FMI']
#classes = ['IFOF_R','ILF_L','ILF_R','MLF_L','MLF_R','OR_R','SLF_L','SLF_R','TAPETUM','UF_L','UF_R','VOF_L']
classes = ['AC','AF_L','AF_R','CGFP_L','CGFP_R','CGH_L','CGH_R','CG_L','CG_R','CGR_R','FA_L','FA_R','FMA','FX_R','FX_L','IFOF_L','IFOF_R','ILF_L','MLF_L','OR_L','SLF_L','UF_L','UF_R','VOF_L','VOF_R','CC_MID','CST','FMI','TAPETUM']
subjects   = ['152831','151425','154936','158843','172029','177645','179245','151728','154229','155938','175237','178142','157942','170631','177241','178950']

nclasses= len(classes)
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
        self.final    = tf.keras.layers.Dense(nclasses)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.dense(x)
        x = self.final(x)
        return x

model = simpleModel()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

# Checkpoints
checkpoint_dir   = './checkpoints'
checkpoint_prefix= os.path.join(checkpoint_dir, "ckpt")
checkpoint       = tf.train.Checkpoint(optimizer=model.optimizer,model=model)

all_trajs  = []
all_labels = []
idx        = 0
for k,subject in enumerate(subjects):
    if training==False:
        if k>0:
            break
    # Reads the .tck files from each specified class
    for i,c in enumerate(classes):
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
        if k==0:
            idx=idx+len(STs)
        scaledSTs= set_number_of_points(STs,20)
        all_trajs.extend(scaledSTs)
        all_labels.extend(len(scaledSTs)*[i])
print('[INFO] Used for testing: ',idx)
print('[INFO] Total number of streamlines:',len(all_trajs))
dataset = tf.data.Dataset.from_tensor_slices((all_trajs,all_labels))
dataset       = dataset.shuffle(600000, reshuffle_each_iteration=False)
dataset       = dataset.batch(s_batch)
# First subject for validation
val_dataset   = dataset.take(idx//s_batch)
# The rest is for training
train_dataset = dataset.skip(idx//s_batch)

if training:
    # Training
    history = model.fit(train_dataset, epochs=30, validation_data=val_dataset)
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
    print("[INF] Restoring last model")
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Evaluate test individual
test_loss, test_acc = model.evaluate(val_dataset)
model.summary()

y_true = np.concatenate([y for x, y in val_dataset], axis=0)
x_true = np.concatenate([x for x, y in val_dataset], axis=0)
o_pred = model.predict(x_true)
y_pred = np.argmax(o_pred, axis=1)

# Confusion matrix
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=classes, yticklabels=classes,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('True label')
plt.show()

# Some visualization of the baddies
baddies   = (y_true!=y_pred)
baddies_x = x_true[baddies]
baddies_id= y_true[baddies]

# Add display objects to canvas
scene = window.Scene()
color = cmap.line_colors(baddies_x)
streamlines_actor = actor.line(baddies_x,cmap.line_colors(baddies_x),linewidth=3, fake_tube=True)
scene.add(streamlines_actor)

#for i,c in enumerate(classes):
#    color = cmap.line_colors(STs)
#    if baddies_x[baddies_id==i].shape[0]>0:
#        streamlines_actor = actor.line(baddies_x[baddies_id==i],(0.5, 0.5, 0),linewidth=3, fake_tube=True)
#        scene.add(streamlines_actor)

window.show(scene)
