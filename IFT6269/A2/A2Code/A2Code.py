import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#  Question 1 C
#  Implementing MLE for Mixture of Gaussians with shared Covariance to each Dataset.

FILES = ['classificationA.test', 'classificationA.train',
         'classificationB.test', 'classificationB.train',
         'classificationC.test', 'classificationC.test']

test = tf.placeholder(dtype=tf.float32, shape=[None, 3,])
train = tf.placeholder(dtype=tf.float32, shape=[None, 3,])
pi_mle = tf.Variable(initial_value=0, dtype=tf.float32)
mu_0_mle = tf.Variable(initial_value=[0,0], dtype=tf.float32)
mu_1_mle = tf.Variable(initial_value=[0,0], dtype=tf.float32)
sigma_mle = tf.Variable(initial_value=[[0,0], [0,0]], dtype=tf.float32)

init_op = tf.global_variables_initializer()

#  Compute pi MLE
pi_mle_op = pi_mle.assign(tf.reduce_mean(train, axis=0)[2])

#  Compute mu MLE
group_1 = tf.squeeze(tf.gather(train[:,0:2], indices=tf.where(condition=tf.equal(train[:, 2],0))))
mu_0_mle_op = mu_0_mle.assign(tf.reduce_mean(group_1, axis=0))

group_2 = tf.squeeze(tf.gather(train[:,0:2], indices=tf.where(condition=tf.equal(train[:, 2],1))))
mu_1_mle_op = mu_1_mle.assign(tf.reduce_mean(group_2, axis=0))

#  Compute Sigma MLE
size_1 = tf.cast(tf.shape(group_1)[0], tf.float32)
sigma_mle_1 = tf.matmul(tf.transpose(group_1), group_1)/size_1
sigma_mle_1 -= [[mu_0_mle_op[0]**2, mu_0_mle_op[0]*mu_0_mle_op[1]],
                 [mu_0_mle_op[0]*mu_0_mle_op[1], mu_0_mle_op[1]**2]]
size_2 = tf.cast(tf.shape(group_2)[0], tf.float32)
sigma_mle_2 = tf.matmul(tf.transpose(group_2), group_2)/size_2
sigma_mle_2 -= [[mu_1_mle_op[0]**2, mu_1_mle_op[0]*mu_1_mle_op[1]],
                 [mu_1_mle_op[0]*mu_1_mle_op[1], mu_1_mle_op[1]**2]]
sigma_mle_op = sigma_mle.assign((size_1/(size_1+size_2))*sigma_mle_1 + (size_1/(size_1+size_2))*sigma_mle_2)

with tf.Session() as sess:
    train_data = pd.read_table('hwk2data/' + FILES[0], header=None)
    test_data = pd.read_table('hwk2data/' + FILES[1], header=None)
    sess.run(init_op)
    pi_mle, mu_0_mle, mu_1_mle, sigma_mle = sess.run([pi_mle_op, mu_0_mle_op, mu_1_mle_op, sigma_mle_op],
                                                     feed_dict={train: train_data, test: test_data})

#  Compute seperating plane
m_0 = -np.dot(np.transpose(mu_0_mle), np.linalg.inv(sigma_mle))
m_1 = -np.dot(np.transpose(mu_1_mle), np.linalg.inv(sigma_mle))
b_0 = np.dot(m_0, mu_0_mle) + np.log(1-pi_mle)
b_1 = np.dot(m_1, mu_1_mle) + np.log(pi_mle)

def line(x):
    return -(b_0-b_1 + (m_0[0]-m_1[0])*x)/(m_0[1]-m_1[1])

#  Plot data points
plt.figure()
plt.scatter(x=train_data[train_data[2] == 0][0], y=train_data[train_data[2] == 0][1], color='red')
plt.scatter(x=train_data[train_data[2] == 1][0], y=train_data[train_data[2] == 1][1], color='blue')

#  Overlay with contours of Normal Dist.
x = np.linspace(-8, 8)
y = np.linspace(-8, 8)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, sigmax=sigma_mle[0,0], sigmay=sigma_mle[1,1],
                           mux=mu_0_mle[0], muy=mu_0_mle[1], sigmaxy=sigma_mle[1,0])
Z2 = mlab.bivariate_normal(X, Y, sigmax=sigma_mle[0,0], sigmay=sigma_mle[1,1],
                           mux=mu_1_mle[0], muy=mu_1_mle[1], sigmaxy=sigma_mle[1,0])
plt.contour(X, Y, Z1, colors='r')
plt.contour(X, Y, Z2, colors='blue')

x = np.linspace(-5, 5)
plt.plot(x, line(x), 'k-', linewidth=1)
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.title('Separating Plane')
plt.savefig('img')
plt.show()