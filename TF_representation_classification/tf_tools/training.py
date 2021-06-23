import numpy as np
import tensorflow as tf
from transformer.masking import create_look_ahead_mask
from tools.trajectories import convert_to_traj, convert_tensor_to_traj

def loss_function(real,pred):
    # Error for ade/fde
    diff = pred - real
    diff = diff**2
    diff = tf.sqrt(tf.reduce_sum(diff,[1,2]))
    return tf.math.reduce_min(diff)

def ADE_train(real,pred):
    # Error for ade/fde
    diff = pred - real
    res = 0.
    for i in range(real.shape[0]+1):
        for j in range(i):
            aux = tf.reduce_sum(diff[:,:j,:],1)
            aux = aux**2
            aux = tf.sqrt(tf.reduce_sum(aux,1))
    res = aux + res
    return tf.reduce_sum(res)/diff.shape[0]
 

def ADE_FDE(real,pred):
    real_traj = convert_to_traj(tf.constant(np.array([[0,0]],dtype = "float32")), real)
    pred_traj = convert_to_traj(tf.constant(np.array([[0,0]],dtype = "float32")), pred)

    n = real_traj.shape[0]
    diff = pred_traj - real_traj
    diff = diff**2
    FDE = diff[:,-1,:]
    FDE = tf.sqrt(tf.reduce_sum(FDE,axis = 1))
    FDE = tf.math.reduce_min(FDE)

    diff = tf.reduce_sum(diff,[1,2])/n
    ADE = tf.math.reduce_min(diff)

    return ADE.numpy(),FDE.numpy()


def accuracy_function(real,pred):
    # Error for ade/fde
    diff = real - pred
    diff = diff**2
    diff = -tf.sqrt(tf.reduce_sum(diff, axis=1))
    return tf.math.exp(diff)

@tf.function
def train_step(inp, tar, transformer, optimizer, train_loss, train_accuracy):
  tar_train = tar
  tar_train = tar[:-1,:]
  aux = tf.expand_dims(inp[-1,:],0)
  tar_train = tf.concat([aux,tar_train], axis = 0)

  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_train, True)
    # predictions = transformer(inp, inp, True,12)
    # loss = loss_function(tar, predictions)
    loss = ADE_train(tar, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(accuracy_function(tar, predictions))