import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dipy.io.streamline import load_tractogram
from dipy.align.streamlinear import set_number_of_points
from dipy.viz import window, actor, colormap as cmap
import numpy as np

from tf_tools.transformer.transformer import Transformer
from tf_tools.parameters import *

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

show      = True
path_files= './data/151425/'
path_files= './data/155938/'

fNameRef  = 't1.nii.gz'
fNameRef  = 'autoFA_L_exclude_interp.nii.gz'
fNameRef  = 'autoCC_MID_target_01.nii.gz'

classes = ['IFOF_R','ILF_L','ILF_R','MLF_L','MLF_R','OR_R','SLF_L','SLF_R','TAPETUM','UF_L','UF_R','VOF_L']
n_class = len(classes)

model = Transformer(d_model, num_layers, num_heads, n_class, dff, rate=0.1)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

all_trajs  = []
all_labels = []
# Reads the .tck files from each specified class
for i,c in enumerate(classes):
    # Load tractogram
    filename   = path_files+'auto'+c+'.tck'
    print('Reading file:',filename)
    #tractogram = load_tractogram(filename, path_files+fNameRef, bbox_valid_check=False)
    tractogram = load_tractogram(filename, './connectome_test/t1.nii.gz', bbox_valid_check=False)
    # Get all the streamlines
    STs      = tractogram.streamlines
    scaledSTs= set_number_of_points(STs,20)
    all_trajs.extend(scaledSTs)
    all_labels.extend(len(scaledSTs)*[i])
print('Total number of streamlines:',len(all_trajs))

# Training
all_trajs = np.array(all_trajs)

all_labels = np.array(all_labels)
aux = np.zeros([all_labels.shape[0],n_class])
for i in range(all_labels.shape[0]): aux[i,all_labels[i]] = 1
all_labels = aux

history = model.fit(all_trajs, all_labels, batch_size=32, epochs=50, validation_split=0.1)

# Plots
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)
plt.show()