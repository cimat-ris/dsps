import tensorflow as tf

import numpy as np

from .encoder import Encoder
from .class_ffnn import class_ffnn

# Our Transformer model
class Transformer(tf.keras.Model):
  def __init__(self, d_model, num_layers, num_heads, n_class, dff, rate=0.1):
    super(Transformer, self).__init__()
    # Encoder
    self.encoder = Encoder(d_model, num_layers, num_heads, dff, 100, rate)
    # classificator
    self.classificator = class_ffnn(n_class,dff)

  # Call to the transformer
  def call(self, input, training):
      # Call the encoder on the inputs
      enc_output = self.encoder(input, training)
      # Calls on the classification method
      classification = self.classificator(enc_output)
      return classification
