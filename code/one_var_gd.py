# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 21:07:57 2018

@author: Andy
"""

import tensorflow as tf
import numpy as np

NUM_EPOCHS = 500

x = tf.Variable(1.0)

loss = tf.exp(x*x)
#loss = x*x
#loss = np.exp(x*x)

gd_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print (sess.run(x))
    for _ in range(NUM_EPOCHS):
        sess.run(gd_step)
        l = sess.run(loss)
#        _, l = sess.run([gd_step, loss])
#        print(l)
    print (sess.run(x))