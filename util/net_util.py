import tensorflow as tf
from tensorflow.layers import conv2d

def build_fc_net(input_tfs, layers,
                  activation=tf.nn.relu,
                  weight_init=tf.contrib.layers.xavier_initializer(),
                  reuse=False):
    curr_tf = tf.concat(axis=-1, values=input_tfs)       
    for i, size in enumerate(layers):
        with tf.variable_scope(str(i), reuse=reuse):
            curr_tf = tf.layers.dense(inputs=curr_tf,
                                    units=size,
                                    kernel_initializer=weight_init,
                                    activation=activation)
    return curr_tf

def build_conv_net(input_tfs, layers,
                  activation=tf.nn.relu,
                  weight_init=tf.contrib.layers.xavier_initializer(),
                  reuse=False):
    curr_tf = tf.reshape(input_tfs, [-1, 84, 2*84, 3])

    for i in range(3):
        with tf.variable_scope(str(i), reuse=reuse):
            curr_tf = conv2d(inputs=curr_tf,
                                filters=64,
                                kernel_size=3,
                                strides=2,
                                padding='SAME',
                                kernel_initializer=weight_init,
                                activation='linear')
    curr_tf = tf.layers.flatten(inputs=curr_tf)
    for i, size in enumerate(layers):
        with tf.variable_scope(str(3 + i), reuse=reuse):
            curr_tf = tf.layers.dense(inputs=curr_tf,
                                    units=size,
                                    kernel_initializer=weight_init,
                                    activation=activation)
    return curr_tf