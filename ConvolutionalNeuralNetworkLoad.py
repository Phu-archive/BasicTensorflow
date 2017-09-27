import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 1000
batch_size = 64

input_size = 784
image_size = 28
channel_size = 1
class_size = 10

dropout_prob_base = 0.2

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
    weight = tf.Variable(tf.truncated_normal([conv_layer_1_filter_size, conv_layer_1_filter_size, channel_size, conv_layer_1_num_filter]
                                             , stddev=0.05))
    bias = tf.Variable(tf.constant(0.05, shape=[conv_layer_1_num_filter]))

    conv1 = tf.nn.conv2d(input=input_conv, filter=weight, strides=[1, 1, 1, 1], padding="SAME")
    out_conv1 = conv1 + bias


with tf.variable_scope("Pooling1"):
    pool1 = tf.nn.max_pool(value=out_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.variable_scope("Layer1"):
    layer1 = tf.nn.relu(pool1)

with tf.variable_scope("Conv2"):
    weight2 = tf.Variable(tf.truncated_normal([conv_layer_2_filter_size, conv_layer_2_filter_size, conv_layer_1_num_filter, conv_layer_2_num_filter],
                                              stddev=0.05))
    bias2 = tf.Variable(tf.constant(0.05, shape=[conv_layer_2_num_filter]))

    conv2 = tf.nn.conv2d(input=layer1, filter=weight2, strides=[1, 1, 1, 1], padding="SAME")
    out_conv2 = conv2 + bias2

with tf.variable_scope("Pool2"):
    pool2 = tf.nn.max_pool(value=out_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

with tf.variable_scope("Layer2"):
    layer2 = tf.nn.relu(pool2)

with tf.variable_scope("Flatten"):
    layer_shape = layer2.get_shape()
    num_features = layer_shape[1:4].num_elements()
    flatten = tf.reshape(layer2, [-1, num_features])

with tf.variable_scope("FullyConnected1"):
    weight3 = tf.Variable(tf.truncated_normal(shape=[num_features, fully_connected_layer_1], stddev=0.05))
    bias3 = tf.Variable(tf.constant(0.05, shape=[fully_connected_layer_1]))

    pre_activation_layer1 = tf.matmul(flatten, weight3) + bias3
    fc_layer1 = tf.nn.relu(pre_activation_layer1)

with tf.variable_scope("FullyConnected2"):
    weight4 = tf.Variable(tf.truncated_normal(shape=[fully_connected_layer_1, fully_connected_layer_2], stddev=0.05))
    bias4 = tf.Variable(tf.constant(0.05, shape=[fully_connected_layer_2]))

    pre_activation_layer2 = tf.matmul(fc_layer1, weight4) + bias4
    fc_layer2 = tf.nn.relu(pre_activation_layer2)

with tf.variable_scope("FullyConnected3"):
    weight5 = tf.Variable(tf.truncated_normal(shape=[fully_connected_layer_2, class_size], stddev=0.05))
    bias5 = tf.Variable(tf.constant(0.05, shape=[class_size]))

    pre_activation_layer3 = tf.matmul(fc_layer2, weight5) + bias5
    output = tf.nn.softmax(pre_activation_layer3)

with tf.variable_scope("Loss"):
    Y = tf.placeholder(tf.float32, shape=[None, class_size])
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(output), reduction_indices=[1]))

with tf.variable_scope("Accuracy"):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    saver.restore(session, "../saveCNN/trained_model900.ckpt")
    final_accuracy = session.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    print("Fianl Accuracy: ", final_accuracy)
