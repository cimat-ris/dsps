import numpy as np
import tensorflow as tf

def loss_function(real,pred):
    # Error for ade/fde
    diff = pred - real
    diff = diff**2
    diff = tf.sqrt(tf.reduce_sum(diff,[1,2]))
    return tf.math.reduce_min(diff)

# real: n_batch x sequence_length x p
# prediction: n_batch x n_modes x sequence_length x p
def ADE_train(real,prediction, max = False):
    sequence_length = real.shape[1]
    n_batches       = prediction.shape[0]
    n_modes         = prediction.shape[1]
    real_expanded   = tf.expand_dims(real,1)
    # diff: n_batch x n_modes x sequence_length x p
    diff            = prediction - real_expanded
    # Sum over time to get absolute positions and take the squares
    losses = tf.cumsum(diff,2)**2
    losses = tf.sqrt(tf.reduce_sum(losses,3))
    # Average over time
    losses = tf.reduce_sum(losses,2)/sequence_length
    # Over the samples: take the min or the max
    if max == False:
        losses = tf.reduce_min(losses,axis=1)
    else:
        losses = tf.reduce_max(losses,axis=1)
    # Average over batch elements
    return tf.reduce_sum(losses)/n_batches


def min_ADE_FDE(ground_truth,prediction):
    sequence_lenth = ground_truth.shape[1]
    diff = prediction - tf.expand_dims(ground_truth,1)
    diff = diff**2
    # Evaluate FDE
    FDE = diff[:,:,-1,:]
    FDE = tf.sqrt(tf.reduce_sum(FDE,axis = 2))
    FDE = tf.math.reduce_min(FDE,axis=1,keepdims=True)
    diff= tf.reduce_sum(diff,[2,3])/sequence_lenth
    ADE = tf.math.reduce_min(diff,axis=1,keepdims=True)
    return ADE.numpy(),FDE.numpy()


def accuracy_function(real,pred):
    # Error for ade/fde
    diff = real - pred
    diff = diff**2
    diff = -tf.sqrt(tf.reduce_sum(diff, axis=1))
    return tf.math.exp(diff)

@tf.function
def train_step(input, target, transformer, optimizer, train_accuracy, burnout = False):
    # Target
    target_train = target[:,:-1,:]
    # This is to hold one position only
    aux          = tf.expand_dims(input[:,-1,:],1)
    # target_train will hold the last input data + the T_pred-1 first positions of the future
    # size: n_batch x sequence_size x p
    target_train = tf.concat([aux,target_train], axis = 1)
    with tf.GradientTape() as tape:
        # Apply the tranformer network to the input
        predictions, _ = transformer(input, target_train, True)
        loss           = ADE_train(target, predictions, burnout)
    if loss < 1000 or burnout == True:
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    return loss
        # train_loss(loss)
        #train_accuracy(accuracy_function(target, predictions))
