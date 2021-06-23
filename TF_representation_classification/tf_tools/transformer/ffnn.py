import tensorflow as tf

class point_wise_feed_forward_network(tf.keras.layers.Layer):
  def __init__(self, d_model, dff):
    super(point_wise_feed_forward_network, self).__init__()

    self.relu = tf.keras.layers.Dense(dff, activation='relu')
    self.dense = tf.keras.layers.Dense(d_model)
    self.a = 1

  def call(self, x):
    x = self.relu(x)
    x = self.dense(x)

    return x
