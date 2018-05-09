import tensorflow as tf

# Create some wrappers for simplicity
def _conv2d(x, shape, strides, scope):
    # Conv2D wrapper
    with tf.variable_scope(scope + "regularize", reuse=False):
        W = tf.Variable(tf.truncated_normal(shape=shape, stddev=5e-2))
    b = tf.Variable(tf.truncated_normal(shape=[shape[3]], stddev=5e-2))
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x


def _bn(x, phase):
    # wrapper for performing batch normalization and elu activation
    x = tf.contrib.layers.batch_norm(x, center=True, scale=True,
                                     variables_collections=["batch_norm_non_trainable_variables_collection"],
                                     updates_collections=None, decay=0.9, is_training=phase,
                                     zero_debias_moving_mean=True, fused=True)
    return tf.nn.elu(x)
def _block(x, n, k, iw, bw, s, dropout, phase, scope):
    # Creates N no. of 3,3 type residual blocks with dropout that consitute the conv2/3/4 blocks
    # with widening factor K and X as input. s is stride and bw is base width (no. of filters before multiplying with k)
    # iw is input width.
    # (see https://arxiv.org/abs/1605.07146 paper for details on the block)
    # In this case, dropout = probability to keep the neuron enabled.
    # phase = true when training, false otherwise.

    o1 = _conv2d(x, [3, 3, iw, bw * k], s, scope)
    o1 = _bn(o1, phase)
    o1 = tf.nn.dropout(o1, dropout)

    o2 = _conv2d(o1, [3, 3, bw * k, bw * k], 1, scope)
    t = _conv2d(x, [1, 1, iw, bw * k], s, scope)  # shortcut connection

    o = tf.add(o2, t)

    # 1st of the N blocks for conv2/3/4 block ends here. The rest of N-1 blocks will be implemented next with a loop.

    for i in range(0, n - 1):
        y = o
        o1 = _bn(y, phase)
        o2 = _conv2d(o1, [3, 3, bw * k, bw * k], 1, scope)
        o2 = _bn(o2, phase)
        o2 = tf.nn.dropout(o2, dropout)
        o2 = _conv2d(o2, [3, 3, bw * k, bw * k], 1, scope)
        o = tf.add(o2, y)

    return _bn(o, phase)