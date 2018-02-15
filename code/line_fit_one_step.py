import numpy as np
import tensorflow as tf

x = np.array([0.0, 1.0, 2.0])
y = np.array([1.0, 2.0, 3.0])

NUM_EPOCHS = 1
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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.010).minimize(loss)

    # calculate gradients. Only used to confirm numbers
    grad_loss_ypred = tf.gradients(loss, Y_predicted)
    grad_yp0 = tf.gradients(Y_predicted[0], slope)
    grad_yp1 = tf.gradients(Y_predicted[1], slope)
    grad_yp2 = tf.gradients(Y_predicted[2], slope)
    grad_loss_slope = tf.gradients(loss, slope)
    grad_yp = [0,0,0]
    
with tf.Session(graph=g) as sess:
     sess.run(tf.global_variables_initializer())
     
     for i in range(NUM_EPOCHS): # run 100 epochs
         """print("gradient of loss from y_pred = ",sess.run([grad_loss_ypred], feed_dict={X:x, Y:y}))
         grad_yp[0], grad_yp[1], grad_yp[2] = sess.run([grad_yp0, grad_yp1, grad_yp2], feed_dict={X:x, Y:y})
         print("gradient of y_pred from slope = ", grad_yp)
         print("gradient of loss from y_pred = ",sess.run([grad_loss_slope], feed_dict={X:x, Y:y}))
         print("slope before optimized step = ",sess.run([slope], feed_dict={X:x, Y:y}))"""
         
         _, l = sess.run([optimizer, loss], feed_dict={X:x, Y:y})
         
#         print("slope after optimized step = ",sess.run([slope], feed_dict={X:x, Y:y}))
         print("slope and intercet are ",sess.run([slope, intercept], feed_dict={X:x, Y:y}))