import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 1000
batch_size = 64

input_size = 784
image_size = 28
channel_size = 1
class_size = 10

dropout_prob = 0.2

# ----------------------------------------

conv_layer_1_filter_size = 5
conv_layer_1_num_filter = 32

pooling_layer_1_filter_size = 2
pooling_layer_1_stride = 2

conv_layer_2_filter_size = 5
conv_layer_2_num_filter = 64

pooling_layer_2_filter_size = 2
pooling_layer_2_stride = 2

fully_connected_layer_1 = 1024
fully_connected_layer_2 = 256

dropout_prob = tf.placeholder(tf.float32, shape=[], name="DropoutProb")

# ----------------------------------------

with tf.variable_scope("Input"):
    X = tf.placeholder(tf.float32, shape=[None, input_size])
    input_conv = tf.reshape(X, [-1, image_size, image_size, channel_size])

with tf.variable_scope("Conv1"):
    conv1 = tf.layers.conv2d(inputs=input_conv,
                             filters=conv_layer_1_num_filter,
                             kernel_size=[conv_layer_1_filter_size, conv_layer_1_filter_size],
                             padding="same",
                             activation=tf.nn.relu)

with tf.variable_scope("Pooling1"):
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[pooling_layer_1_filter_size, pooling_layer_1_filter_size],
                                    strides=pooling_layer_1_stride)

with tf.variable_scope("Conv2"):
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=conv_layer_2_num_filter,
                             kernel_size=[conv_layer_2_filter_size, conv_layer_2_filter_size],
                             padding="same",
                             activation=tf.nn.relu)

with tf.variable_scope("Pool2"):
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[pooling_layer_2_filter_size, pooling_layer_2_filter_size],
                                    strides=pooling_layer_1_stride)

with tf.variable_scope("Flatten"):
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

with tf.variable_scope("FullyConnected1"):
    dense1 = tf.layers.dense(inputs=pool2_flat,
                             units=fully_connected_layer_1,
                             activation=tf.nn.relu)

    dropout1 = tf.layers.dropout(inputs=dense1, rate=dropout_prob)

with tf.variable_scope("FullyConnected2"):
    dense2 = tf.layers.dense(inputs=dropout1,
                             units=fully_connected_layer_2,
                             activation=tf.nn.relu)

    dropout2 = tf.layers.dropout(inputs=dense2, rate=dropout_prob)

with tf.variable_scope("FullyConnected3"):
    dense3 = tf.layers.dense(inputs=dropout2,
                             units=class_size,
                             activation=tf.nn.relu)

    output = tf.nn.softmax(dense3)






