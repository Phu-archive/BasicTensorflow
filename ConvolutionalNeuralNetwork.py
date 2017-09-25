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

with tf.variable_scope("Loss"):
    Y = tf.placeholder(tf.float32, shape=[None, class_size])
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(output), reduction_indices=[1]))

with tf.variable_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.variable_scope("Accuracy"):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.variable_scope("Log"):
    tf.summary.scalar("Current_Cost", cost)
    tf.summary.scalar("Accuracy", accuracy)
    summary = tf.summary.merge_all()

# Training Loop
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    training_writer = tf.summary.FileWriter("LogsCNN/training", sess.graph)
    testing_writer = tf.summary.FileWriter("LogsCNN/testing", sess.graph)
    accuracy_writer = tf.summary.FileWriter("LogsCNN/accuracy", sess.graph)

    for epoch in range(training_epochs):
        train_batch_data, train_batch_labels = mnist.train.next_batch(batch_size)
        test_batch_data, test_batch_labels = mnist.test.next_batch(batch_size)

        sess.run(optimizer, feed_dict={X: train_batch_data,
                                       Y: train_batch_labels,
                                       dropout_prob: dropout_prob_base})

        display_feed_dict = {X: train_batch_data, Y: train_batch_labels, dropout_prob: 0}
        if epoch % 5 == 0:
            training_cost, training_summary = sess.run([cost, summary], feed_dict=display_feed_dict)
            testing_cost, testing_summary = sess.run([cost, summary], feed_dict=display_feed_dict)
            acc, accuracy_summary = sess.run([accuracy, summary], feed_dict={X: mnist.test.images,
                                                                             Y: mnist.test.labels})

            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)
            accuracy_writer.add_summary(accuracy_summary, epoch)

            print("Training Cost: ", training_cost, " Testing Cost: ", testing_cost, " Accuracy: ", acc)

    final_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})

print("Training Complete with: ", final_accuracy, " accuracy.")
