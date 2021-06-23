import tensorflow as tf
from tensorflow.keras.layers import Dense

class class_ffnn(tf.keras.layers.Layer):
  def __init__(self, n_class, dff):
    super(class_ffnn, self).__init__()

    self.relu1 = Dense(dff, activation='relu')
    self.relu2 = Dense(dff, activation='relu')
    self.sfmax = Dense(n_class, activation='softmax')
    self.a = 1

  def call(self, x):

    leng = x.shape[1]*x.shape[2]
    x = tf.reshape(x, [-1,leng])

    x = self.relu1(x)
    x = self.relu2(x)
    x = self.sfmax(x)

    return x