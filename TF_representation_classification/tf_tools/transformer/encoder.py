import tensorflow as tf

from .position_enc import positional_encoding
# from .attention import  Multi_headed_attention
from .attention import  Attention
from .ffnn import point_wise_feed_forward_network

# Encoder layer
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()
    # Multi-headed attention layer
    # self.attention = Multi_headed_attention(d_model,num_heads)
    self.attention = Attention(d_model)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training):
    # Attention layer
    attn_output, _ = self.attention(x, x, x, None)
    attn_output    = self.dropout1(attn_output, training=training)
    # Residual connection
    out1           = self.layernorm1(x+attn_output)
    # out2           = out1
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1+ffn_output)

    return out2


class Encoder(tf.keras.layers.Layer):
  def __init__(self, d_model, num_layers, num_heads, dff, maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()
    #Dimensions used for interchanging between attention and the FFNN
    self.d_model    = d_model
    self.dff        = dff
    self.num_layers = num_layers
    # Embedding of the poses
    # TODO: activation function here?
    self.embedding = tf.keras.layers.Dense(d_model)
    # Positional encoding
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    # Encoding layers
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]
    # Dropout layer
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training):
    batch_size    = x.shape[0]
    sequence_size = x.shape[1]
    # Embedding of the displacements
    # Size: n_batch x T_obs-1 x d_model
    x  = self.embedding(x)
    # Scaling
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    # Add the positional encoding
    if sequence_size % 2 == 0:
      pe           = self.pos_encoding[:int(sequence_size/2),:]
      pos_encoding = tf.concat([pe,tf.reverse(pe,[0])],axis = 0)
    else:
      pe           = self.pos_encoding[:int(sequence_size/2),:]
      aux          = self.pos_encoding[:int(sequence_size/2+1),:]
      pos_encoding = tf.concat([aux,tf.reverse(pe,[0])],axis = 0)
    x += pos_encoding
    # Apply dropout to the embedding
    x  = self.dropout(x, training=training)

    # This sends the data through the different encoder Layers
    for i in range(self.num_layers):
        x = self.enc_layers[i](x, training)

    return x
