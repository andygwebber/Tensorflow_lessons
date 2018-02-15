# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 21:07:57 2018

@author: Andy
"""

import tensorflow as tf
import numpy as np

epochs = 50

x = tf.Variable([1.0, -2.0, 3.0], tf.float32, name='x')
y = tf.constant([4.0, 5.0, 6.0], tf.float32)

z = x - y

loss = tf.tensordot(z,z, 1)
dot = tf.tensordot(x,x, 1)
#loss = tf.exp(dot)
#loss = tf.exp(tf.tensordot(x,x, 1))

gd_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print (sess.run(x))
    for _ in range(epochs):
        sess.run(gd_step)
    print (sess.run(x))
    print (sess.run(dot))