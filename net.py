import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op
def build_network(images,
                  num_outputs,
                  alpha,
                  keep_prob=0.5,
                  is_training=True,
                  scope='yolo',
                  reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                activation_fn=leaky_relu(alpha),
                weights_regularizer=slim.l2_regularizer(0.0005),
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
        ):
            net = tf.pad(
                images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
                name='pad_1')
            net = slim.conv2d(
                net, 64, 7, 2, padding='VALID', scope='conv_2')
            net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
            net = slim.conv2d(net, 192, 3, scope='conv_4')
            net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
            net = slim.conv2d(net, 128, 1, scope='conv_6')
            net = slim.conv2d(net, 256, 3, scope='conv_7')
            net = slim.conv2d(net, 256, 1, scope='conv_8')
            net = slim.conv2d(net, 512, 3, scope='conv_9')
            net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
            net = slim.conv2d(net, 256, 1, scope='conv_11')
            net = slim.conv2d(net, 512, 3, scope='conv_12')
            net = slim.conv2d(net, 256, 1, scope='conv_13')
            net = slim.conv2d(net, 512, 3, scope='conv_14')
            net = slim.conv2d(net, 256, 1, scope='conv_15')
            net = slim.conv2d(net, 512, 3, scope='conv_16')
            net = slim.conv2d(net, 256, 1, scope='conv_17')
            net = slim.conv2d(net, 512, 3, scope='conv_18')
            net = slim.conv2d(net, 512, 1, scope='conv_19')
            net = slim.conv2d(net, 1024, 3, scope='conv_20')
            net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
            net = slim.conv2d(net, 512, 1, scope='conv_22')
            net = slim.conv2d(net, 1024, 3, scope='conv_23')
            net = slim.conv2d(net, 512, 1, scope='conv_24')
            net = slim.conv2d(net, 1024, 3, scope='conv_25')
            net = slim.conv2d(net, 1024, 3, scope='conv_26')
            net = tf.pad(
                net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                name='pad_27')
            net = slim.conv2d(
                net, 1024, 3, 2, padding='VALID', scope='conv_28')
            net = slim.conv2d(net, 1024, 3, scope='conv_29')
            net = slim.conv2d(net, 1024, 3, scope='conv_30')
            net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
            net = slim.flatten(net, scope='flat_32')
            net = slim.fully_connected(net, 512, scope='fc_33')
            net = slim.fully_connected(net, 4096, scope='fc_34')
            net = slim.dropout(
                net, keep_prob=keep_prob, is_training=is_training,  # dropout 有个is_training 参数非常合理
                scope='dropout_35')
            # net = slim.conv2d(images, 32, 3, scope='conv_1', padding='SAME')
            # net = slim.conv2d(net, 64, 3, scope='conv_2', padding='SAME')
            # net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_1')
            # # net = slim.conv2d(net, 128, 3, scope='conv_3', padding='SAME')
            # # net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_2')
            # # net = slim.conv2d(net, 256, 3, scope='conv_4', padding='SAME')
            # net = tf.layers.flatten(net)
            # net = slim.fully_connected(net, 512, scope='fc_1')
            net = slim.fully_connected(
                net, num_outputs, activation_fn=None, scope='fc_36')

    return net