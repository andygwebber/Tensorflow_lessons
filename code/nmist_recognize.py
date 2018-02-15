# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 at 8:15 pm

@author: Andy Webber but heavy influenced by "Learning Tensorflow" by
Hope, Resheff and Lieder. Found on page 16 of that book
"""

import tensorflow as tf
import input_data
import numpy as np

#Obtain the data
data = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_X, train_y = data.train.next_batch(110000)
test_X, test_y = data.test.next_batch(20000)

NUM_EPOCHS = 2
MINIBATCH_SIZE = 100
images = np.shape(train_X)[0]
batches = images//MINIBATCH_SIZE

x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784,10]))

y_true = tf.placeholder(tf.float32, [None,10])
y_pred = tf.matmul(x,w)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels = y_true, logits = y_pred))

gd_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for _ in range(NUM_EPOCHS):
        for i in range(batches):
            batch_X = train_X[i*MINIBATCH_SIZE:(i+1)*MINIBATCH_SIZE]
            batch_y = train_y[i*MINIBATCH_SIZE:(i+1)*MINIBATCH_SIZE]
            sess.run(gd_step, feed_dict = {y_true: batch_y, x: batch_X})
        
    ans = sess.run(accuracy, feed_dict={x:test_X,
                                        y_true: test_y})
    
print("Accuracy: {:.4}%".format(ans*100))
