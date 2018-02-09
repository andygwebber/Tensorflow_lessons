import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

npoints = 100

np.random.seed(1)
x = np.random.random([npoints]) * 2.0
noise = np.random.normal(0,.1,npoints)
y = x  + 1.0 + noise

[true_slope, true_intercept] = np.polyfit(x, y, 1)


num_epochs = 100
num_batches = 10
batch_size = npoints//num_batches
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
     
     for iepoch in range(num_epochs): # run 100 epochs
         for ibatch in range(num_batches):
             i_start = ibatch*batch_size
             i_stop = (ibatch+1)*batch_size
             x_batch = x[i_start:i_stop]
             y_batch = y[i_start:i_stop]
             _, l = sess.run([optimizer, loss], feed_dict={X:x_batch, Y:y_batch})
         print(l)

#     print(loss.eval())
#     print(Y_predicted.eval())
     
     slope, intercept = sess.run([slope, intercept])
     print("slope =", slope, "intercept = ", intercept)
     writer = tf.summary.FileWriter('./graphs', sess.graph)
#     print(sess.run(loss,feed_dict={X:x, Y:y}))
	
# plot the results
X, Y = x, y
fig = plt.figure()
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * slope + intercept, 'r', label='Predicted data')
plt.legend()
plt.ylabel('Y', fontsize=18)
plt.xlabel('X', fontsize=18)
plt.title('Data to be fitted',fontsize=18)
plt.show()
#fig.savefig('batch.jpg')

print(slope - true_slope)