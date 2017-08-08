import tensorflow as tf
import numpy as np
import os
import dataset_format2
from matplotlib import pyplot as plt

x_train, y_train = dataset_format2.get_train_data()
x_test, y_test = dataset_format2.get_test_data()

image_size = 32
num_channels = 3
num_labels = 1

train_n = np.full((73257, image_size, image_size, num_channels), 255.0, dtype=np.float32)
test_n = np.full((26032, image_size, image_size, num_channels), 255.0, dtype=np.float32)
pixel_depth = 255.0

train_dataset = ((x_train - train_n) / 2) / pixel_depth
test_dataset = ((x_test - test_n) / 2) / pixel_depth

# image_ind = 2
# fig, ax = plt.subplots(1)
# ax.imshow(train_dataset[image_ind, :, :, :])
# plt.show()

batch_size = 200
patch_size = 5
depth = 26
num_hidden = 64

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset)

    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1), name='w1')
    layer1_biases = tf.Variable(tf.zeros([depth]), name='b1')
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1), name='w2')
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]), name='b2')
    layer3_weights = tf.Variable(tf.truncated_normal(
        [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1), name='w3')
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name='b3')
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1), name='w4')
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='b4')


    def accuracy(predictions, labels):
        return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases


    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

num_steps = 51

with tf.Session(graph=graph) as sess:
    saver = tf.train.Saver()

    if os.path.exists('modelx.ckpt'):
        saver.restore(sess, 'modelx.ckpt')
        print('Model restored and initialized')
    else:
        tf.initialize_all_variables().run()
        print('Initialized')

    for step in range(num_steps):
        offset = (step * batch_size) % (y_train.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = y_train[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 50 == 0:
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))

    save_path = saver.save(sess, 'modelx.ckpt')
    print('Model saved in file: %s' % save_path)

    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), y_test))
