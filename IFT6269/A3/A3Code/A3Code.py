import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.patches import Ellipse
import os
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from scipy.stats import multivariate_normal


FILES = ['EMGaussian.train', 'EMGaussian.test']

NUM_MEANS = 4
NUM_ITER = 30
NUM_ITER_EM = 100

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

test = tf.placeholder(dtype=tf.float32, shape=[None, 2])
train = tf.placeholder(dtype=tf.float32, shape=[None, 2])

train_data = pd.read_table('hwk3data/' + FILES[0], header=None, sep=' ')
test_data = pd.read_table('hwk3data/' + FILES[1], header=None, sep=' ')
n = train_data.shape[0]

#x = np.random.multivariate_normal(mean=[5,5], cov=[[3,0],[0,1.5]], size=150)
#x = np.concatenate((x, np.random.multivariate_normal(mean=[5,-5], cov=[[2,0],[0,5]], size=200)))
#x = np.concatenate((x, np.random.multivariate_normal(mean=[-5,5], cov=[[4,0],[2,4]], size=75)))
#x = np.concatenate((x, np.random.multivariate_normal(mean=[-5,-5], cov=[[3,1.5],[0,3]], size=75)))
#train_data = pd.DataFrame(x)

################
#### KMeans ####
################

cluster_centers = tf.Variable(initial_value=tf.random_uniform(shape=[1, 2, NUM_MEANS],minval=-4, maxval=4),
                              dtype=tf.float32, validate_shape=False)
hidden = tf.Variable(initial_value=tf.random_uniform(shape=[n], minval=0, maxval=4, dtype=tf.int64))
loss = tf.Variable(0, dtype=tf.float32)

#  Compute distortion measure!
distortion_measure = tf.reduce_sum(tf.squared_difference(tf.expand_dims(train, 2), cluster_centers), axis=1)

#  Assign clusters based on distortion
hidden_op = hidden.assign(tf.argmin(distortion_measure, dimension=1))

#  Compute loss
loss_op = loss.assign(tf.reduce_mean(tf.reduce_min(distortion_measure, axis=1)))

#  Assign cluster
cluster_op = cluster_centers.assign([tf.transpose([tf.reduce_mean(
    tf.squeeze(tf.gather(train, indices=tf.where(condition=tf.equal(hidden, i)))), axis=0) for i in range(NUM_MEANS)])])

init_op = tf.global_variables_initializer()

centers_list = []
with tf.Session() as sess:
    for j in range(1):
        sess.run(init_op)
        for iter in range(NUM_ITER):
            #  E-Step
            hidden, loss = sess.run([hidden_op, loss_op], feed_dict={train: train_data})
            print("current mean square deviation at iter {0}: {1}".format(iter, loss))
            #  M-Stop
            centers = sess.run([cluster_op], feed_dict={train: train_data})
            centers = np.transpose(centers[0][0])  # Makes it look nice
        centers_list.append(centers)
        #print("current clusters: \n {0}".format(centers))

centers = np.mean(centers_list, axis=0)

#  Plot data points and seperating plane
plt.figure()
plt.scatter(x=train_data[hidden==0][0], y=train_data[hidden==0][1], color='pink', alpha=0.3, s=20)
plt.scatter(x=train_data[hidden==1][0], y=train_data[hidden==1][1], color='cyan', alpha=0.3, s=20)
plt.scatter(x=train_data[hidden==2][0], y=train_data[hidden==2][1], color='green', alpha=0.3, s=20)
plt.scatter(x=train_data[hidden==3][0], y=train_data[hidden==3][1], color='yellow', alpha=0.3, s=20)
plt.scatter(x=centers[0][0], y=centers[0][1], color='red', marker=(5, 2), s=40)
plt.scatter(x=centers[1][0], y=centers[1][1], color='blue', marker=(5, 2), s=40)
plt.scatter(x=centers[2][0], y=centers[2][1], color='green', marker=(5, 2), s=40)
plt.scatter(x=centers[3][0], y=centers[3][1], color='orange', marker=(5, 2), s=40)


plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.title('K Means')

plt.savefig('../K_means_img')
plt.show()

#  Sanity test
#kmeans = KMeans(n_clusters=4, n_init=30).fit(train_data)
#print(kmeans.cluster_centers_)

####################
### EM Spherical ###
####################

#  Initialize variables
pi = [hidden.tolist().count(i)/len(hidden) for i in range(NUM_MEANS)]
pi_list = [pi]

mu = centers
mu_list = [mu]

sigmas = [np.sum(((train_data[hidden==i][0]-mu[i][0])**2 + (train_data[hidden==i][1]-mu[i][1])**2))
                  /hidden.tolist().count(i) for i in range(NUM_MEANS)]

covariance = [[[s,0], [0,s]] for s in sigmas]
cov_list = [covariance]

losses = []
for iter_em in range(NUM_ITER_EM):
    pi = pi_list[iter_em]
    mu = mu_list[iter_em]
    covariance = cov_list[iter_em]

    # E Step
    responsibilities = np.transpose([pi[i]*multivariate_normal(mean=mu[i], cov=covariance[i]).pdf(train_data)
                                     for i in range(NUM_MEANS)])

    #  normalize
    responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=1)

    # M Step
    new_pi = responsibilities.mean(axis=0)
    new_mu = np.dot(np.transpose(responsibilities), train_data) / \
             np.transpose([responsibilities.sum(axis=0), responsibilities.sum(axis=0)])

    new_covariance = []
    avg_X2 = np.dot(responsibilities.T, train_data * train_data)/np.sum(responsibilities, axis=0)[:, np.newaxis]
    avg_means2 = mu ** 2
    avg_X_means = mu * np.dot(responsibilities.T, train_data) / np.sum(responsibilities, axis=0)[:, np.newaxis]
    t = (avg_X2 - 2 * avg_X_means + avg_means2).mean(1)

    new_covariance.append([[t[0], 0], [0,t[0]]])
    new_covariance.append([[t[1], 0], [0,t[1]]])
    new_covariance.append([[t[2], 0], [0,t[2]]])
    new_covariance.append([[t[3], 0], [0,t[3]]])

    # Compute loss
    loss = np.sum([pi[i]*multivariate_normal(mean=mu[i], cov=new_covariance[i]).pdf(train_data)
                   for i in range(NUM_MEANS)], axis=0)
    loss = np.mean(np.log(loss))
    losses.append(loss)

    mu_list.append(new_mu)
    cov_list.append(new_covariance)
    pi_list.append(new_pi)

    print('log likelihood for iteration {0}: {1}'.format(iter_em, loss))

# Final Params
mu = mu_list[-1]
covariance = cov_list[-1]
pi = pi_list[-1]
loss = losses[-1]

test_loss = np.sum([pi[i]*multivariate_normal(mean=mu[i], cov=covariance[i]).pdf(test_data)
                   for i in range(NUM_MEANS)], axis=0)
test_loss = np.mean(np.log(test_loss))
print("train loss: {}".format(loss))
print("test loss: {}".format(test_loss))


gmm = GaussianMixture(n_components=4, covariance_type='spherical', verbose=2)
gmm.fit(train_data)

'''
mu=gmm.means_
covariance = [np.diag([gmm.covariances_[0], gmm.covariances_[0]]),
              np.diag([gmm.covariances_[1], gmm.covariances_[1]]),
              np.diag([gmm.covariances_[2], gmm.covariances_[2]]),
              np.diag([gmm.covariances_[3], gmm.covariances_[3]])
              ]
'''
hidden2 = np.argmax(responsibilities, axis=1)
#  Plot data points and seperating plane
plt.figure()
plt.scatter(x=train_data[hidden2==0][0], y=train_data[hidden2==0][1], color='pink', alpha=0.3, s=20)
plt.scatter(x=train_data[hidden2==1][0], y=train_data[hidden2==1][1], color='cyan', alpha=0.3, s=20)
plt.scatter(x=train_data[hidden2==2][0], y=train_data[hidden2==2][1], color='green', alpha=0.3, s=20)
plt.scatter(x=train_data[hidden2==3][0], y=train_data[hidden2==3][1], color='yellow', alpha=0.3, s=20)
plt.scatter(x=mu[0][0], y=mu[0][1], color='red', marker=(5, 2), s=40)
plt.scatter(x=mu[1][0], y=mu[1][1], color='blue', marker=(5, 2), s=40)
plt.scatter(x=mu[2][0], y=mu[2][1], color='green', marker=(5, 2), s=40)
plt.scatter(x=mu[3][0], y=mu[3][1], color='orange', marker=(5, 2), s=40)

#  Overlay with contours of Normal Dist.
x = np.linspace(-10, 10)
y = np.linspace(-10, 10)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, sigmax=covariance[0][0][0], sigmay=covariance[0][1][1],
                           mux=mu[0][0], muy=mu[0][1], sigmaxy=covariance[0][0][1])
Z2 = mlab.bivariate_normal(X, Y, sigmax=covariance[1][0][0], sigmay=covariance[1][1][1],
                           mux=mu[1][0], muy=mu[1][1], sigmaxy=covariance[1][0][1])
Z3 = mlab.bivariate_normal(X, Y, sigmax=covariance[2][0][0], sigmay=covariance[2][1][1],
                           mux=mu[2][0], muy=mu[2][1], sigmaxy=covariance[2][0][1])
Z4 = mlab.bivariate_normal(X, Y, sigmax=covariance[3][0][0], sigmay=covariance[3][1][1],
                           mux=mu[3][0], muy=mu[3][1], sigmaxy=covariance[3][0][1])
plt.contour(X, Y, Z1, colors='red')
plt.contour(X, Y, Z2, colors='blue')
plt.contour(X, Y, Z3, colors='green')
plt.contour(X, Y, Z4, colors='orange')


plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.title('EM With Spherical Covariance')

plt.savefig('../EM_Spherical_img')
plt.show()

#  Initialize variables
pi = [hidden.tolist().count(i)/len(hidden) for i in range(NUM_MEANS)]
pi_list = [pi]

mu = centers
mu_list = [mu]
sigmas = [np.dot(np.transpose(train_data[hidden==i]-mu[0]),train_data[hidden==i]-mu[0])/hidden.tolist().count(i)
          for i in range(NUM_MEANS)]
cov_list = [sigmas]

###################
### EM  General ###
###################

#  Initialize variables using values from before

losses = []
for iter_em in range(NUM_ITER_EM):
    pi = pi_list[iter_em]
    mu = mu_list[iter_em]
    covariance = cov_list[iter_em]

    # E Step
    responsibilities = np.transpose([pi[i]*multivariate_normal(mean=mu[i], cov=covariance[i]).pdf(train_data)
                                     for i in range(NUM_MEANS)])

    #  normalize
    responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=1)

    # M Step
    new_pi = responsibilities.mean(axis=0)
    new_mu = np.dot(np.transpose(responsibilities), train_data) / \
             np.transpose([responsibilities.sum(axis=0), responsibilities.sum(axis=0)])

    new_covariance = np.empty((4, 2, 2))
    for k in range(4):
        diff = train_data - mu[k]
        new_covariance[k] = np.dot(responsibilities[:, k] * diff.T, diff) / np.sum(responsibilities, axis=0)[k]
        new_covariance[k].flat[::2 + 1]


    # Compute loss
    loss = np.sum([pi[i]*multivariate_normal(mean=mu[i], cov=covariance[i]).pdf(train_data)
                   for i in range(NUM_MEANS)], axis=0)
    loss = np.mean(np.log(loss))
    losses.append(loss)

    mu_list.append(new_mu)
    cov_list.append(new_covariance)
    pi_list.append(new_pi)

    print('log likelihood for iteration {0}: {1}'.format(iter_em, loss))

# Final Params
mu = mu_list[-1]
covariance = cov_list[-1]
pi = pi_list[-1]

test_loss = np.sum([pi[i]*multivariate_normal(mean=mu[i], cov=covariance[i]).pdf(test_data)
                   for i in range(NUM_MEANS)], axis=0)
test_loss = np.mean(np.log(test_loss))

print("train loss: {}".format(loss))
print("test loss: {}".format(test_loss))

gmm = GaussianMixture(n_components=4, verbose=2)
gmm.fit(train_data)


#covariance = gmm.covariances_
#mu = gmm.means_

hidden3 = np.argmax(responsibilities, axis=1)
#  Plot data points and seperating plane
plt.figure()
plt.scatter(x=train_data[hidden3==0][0], y=train_data[hidden3==0][1], color='pink', alpha=0.3, s=20)
plt.scatter(x=train_data[hidden3==1][0], y=train_data[hidden3==1][1], color='cyan', alpha=0.3, s=20)
plt.scatter(x=train_data[hidden3==2][0], y=train_data[hidden3==2][1], color='green', alpha=0.3, s=20)
plt.scatter(x=train_data[hidden3==3][0], y=train_data[hidden3==3][1], color='yellow', alpha=0.3, s=20)
plt.scatter(x=mu[0][0], y=mu[0][1], color='red', marker=(5, 2), s=40)
plt.scatter(x=mu[1][0], y=mu[1][1], color='blue', marker=(5, 2), s=40)
plt.scatter(x=mu[2][0], y=mu[2][1], color='green', marker=(5, 2), s=40)
plt.scatter(x=mu[3][0], y=mu[3][1], color='orange', marker=(5, 2), s=40)

#  Overlay with contours of Normal Dist.
x = np.linspace(-10, 10)
y = np.linspace(-10, 10)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, sigmax=covariance[0][0][0], sigmay=covariance[0][1][1],
                           mux=mu[0][0], muy=mu[0][1], sigmaxy=covariance[0][0][1])
Z2 = mlab.bivariate_normal(X, Y, sigmax=covariance[1][0][0], sigmay=covariance[1][1][1],
                           mux=mu[1][0], muy=mu[1][1], sigmaxy=covariance[1][0][1])
Z3 = mlab.bivariate_normal(X, Y, sigmax=covariance[2][0][0], sigmay=covariance[2][1][1],
                           mux=mu[2][0], muy=mu[2][1], sigmaxy=covariance[2][0][1])
Z4 = mlab.bivariate_normal(X, Y, sigmax=covariance[3][0][0], sigmay=covariance[3][1][1],
                           mux=mu[3][0], muy=mu[3][1], sigmaxy=covariance[3][0][1])
plt.contour(X, Y, Z1, colors='red')
plt.contour(X, Y, Z2, colors='blue')
plt.contour(X, Y, Z3, colors='green')
plt.contour(X, Y, Z4, colors='orange')


plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.title('EM With General Covariance')

plt.savefig('../EM_General_img')
plt.show()
