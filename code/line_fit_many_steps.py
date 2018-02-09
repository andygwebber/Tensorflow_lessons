import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

npoints = 100

x = np.random.random([npoints])
noise = np.random.normal(0,.1,npoints)
y = x  + 1.0 + noise


num_epochs = 500
g = tf.Graph()

with g.as_default():
    # Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
    # Both have the type float32
    X = tf.placeholder(tf.float32, None, name='X')
    Y = tf.placeholder(tf.float32, None, name='Y')

    # Step 3: create slope and intercept, initialized to 0
    slope = tf.Variable(0.0, dtype = tf.float32, name='slope')
    intercept = tf.Variable(0.0, dtype = tf.float32, name='intercept')
    
    # Create and define intermediary values and update
    Y_predicted = slope*X + intercept
    loss = tf.losses.mean_squared_error(Y, Y_predicted)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.030).minimize(loss)

    
# Phase 2: Train our model
with tf.Session(graph=g) as sess:
     sess.run(tf.global_variables_initializer())
     
     for i in range(num_epochs): # run 100 epochs
         _, l = sess.run([optimizer, loss], feed_dict={X:x, Y:y})
         print(l)

#     print(loss.eval())
#     print(Y_predicted.eval())
     
     slope, intercept = sess.run([slope, intercept])
     print("slope =", slope, "intercept = ", intercept)
     writer = tf.summary.FileWriter('./graphs', sess.graph)
#     print(sess.run(loss,feed_dict={X:x, Y:y}))
	
# plot the results
X, Y = x, y
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * slope + intercept, 'r', label='Predicted data')
plt.legend()
plt.show()