from utils import _conv2d, _bn, _block
import tensorflow as tf

def WideResNet(x, dropout, phase, layers, kval, scope, n_classes = 10):  # Wide residual network

    # 1 conv + 3 convblocks*(3 conv layers *1 group for each block + 2 conv layers*(N-1) groups for each block [total 1+N-1 = N groups]) = layers
    # 3*2*(N-1) = layers - 1 - 3*3
    # N = (layers -10)/6 + 1
    # So N = (layers-4)/6
    N = (layers - 4) / 6
    # o = _conv2d(x, [3, 3, 3, 16], 1, scope)
    # in_shape = x.get_shape()
    print x.get_shape()
    o = _conv2d(x, [3, 3, 1, 16], 1, scope)
    print o.get_shape()
    o = _bn(o, phase)
    print o.get_shape()
    o = _block(o, N, kval, 16, 16, 1, dropout, phase, scope)
    print o.get_shape()
    o = _block(o, N, kval, 16 * kval, 32, 2, dropout, phase, scope)
    print o.get_shape()
    o = _block(o, N, kval, 32 * kval, 64, 2, dropout, phase, scope)
    print o.get_shape()
    pooled = tf.nn.avg_pool(o, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')
    print o.get_shape()
    # Initialize weights and biases for fully connected layers
    with tf.variable_scope(scope + "regularize", reuse=False):
        wd = tf.Variable(tf.truncated_normal([1 * 1 * 64 * kval, 64 * kval], stddev=5e-2))
        wout = tf.Variable(tf.truncated_normal([64 * kval, n_classes]))
    bd1 = tf.Variable(tf.constant(0.1, shape=[64 * kval]))
    bout = tf.Variable(tf.constant(0.1, shape=[n_classes]))

    # Fully connected layer
    # Reshape pooling layer output to fit fully connected layer input
    fc = tf.reshape(pooled, [-1, wd.get_shape().as_list()[0]])
    fc = tf.add(tf.matmul(fc, wd), bd1)
    fc = tf.nn.elu(fc)
    # Output, class prediction
    out = tf.add(tf.matmul(fc, wout), bout)

    return out