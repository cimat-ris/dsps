import tensorflow as tf

def scaled_dot_product_attention(q, k, v, mask):
  # Size q,k,v: num_batch x num_heads x sequence_length x depth
  matmul_qk = tf.matmul(q, k, transpose_b=True)
  # Scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
  # Add the mask to the scaled tensor (used in decoder).
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)
  # Attention weights
  # Size attention_weights: num_batch x num_heads x sequence_length x sequence_length
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
  # Output: num_batch x num_heads x sequence_length x depth
  output = tf.matmul(attention_weights, v)
  return output, attention_weights

class Attention(tf.keras.layers.Layer):
  def __init__(self, d_model):
    super(Attention, self).__init__()

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

  def call(self, v, k, q, mask):
    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

    output = scaled_attention

    return output, attention_weights

class Multi_headed_attention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(Multi_headed_attention, self).__init__()
    # Number of heads
    self.num_heads = num_heads
    # Hidden dimension
    self.d_model   = d_model
    assert d_model % self.num_heads == 0
    # Dimension (depth) per head
    self.depth = d_model // self.num_heads

    # Query, Key, Value matrices for the multiple heads
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

  def split_heads(self, x):
    # Reshape the obtained tensor (queries,keys,values)
    # as num_batch x sequence_length x num_heads x depth
    x = tf.reshape(x,[x.shape[0],-1,self.num_heads, self.depth])
    # Reorganize as num_batch x num_heads x sequence_length x depth
    return tf.transpose(x, perm=[0,2,1,3])

  def call(self, v, k, q, mask):
    batch_size    = tf.shape(q)[0]
    sequence_size = tf.shape(q)[1]
    # All inputs size: num_batch x sequence_length x d_model
    # Generate the query, key, value vectors
    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)
    # Split heads before applying dot products
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)
    # Size q,k,v: num_batch x num_heads x sequence_length x depth
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    concat_attention = tf.reshape(scaled_attention,[scaled_attention.shape[0],-1,self.d_model])
    # Size output: num_batch x sequence_length x d_model
    output = concat_attention
    return output, attention_weights
