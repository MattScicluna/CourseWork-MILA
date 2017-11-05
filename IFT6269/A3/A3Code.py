import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from scipy.stats import multivariate_normal


#  Parameters
FILES = ['EMGaussian.train', 'EMGaussian.test']
NUM_MEANS = 4
MEANS_COLOR = ['red', 'blue', 'green', 'orange']
CLUSTER_MEANS_COLOR = ['pink', 'cyan', 'green', 'yellow']
NUM_ITER = 100
EPSILON = 1e-10

train_data = np.loadtxt('hwk4data/' + FILES[0])
test_data = np.loadtxt('hwk4data/' + FILES[1])


def compute_loss(data, pi, mu, cov):
    n, _ = data.shape
    resp = np.array([np.log(pi[i]) + np.log(multivariate_normal(mean=mu[i], cov=cov[i]).pdf(data)) for i in range(NUM_MEANS)]).T
    #  normalize
    resp = np.exp(resp)
    resp /= resp.sum(axis=1, keepdims=1)

    loss = np.array([np.log(pi[i]) + np.log(multivariate_normal(mean=mu[i], cov=cov[i]).pdf(data))
                     for i in range(NUM_MEANS)])
    loss = np.sum(loss.T * resp) / n
    return resp, loss


def get_k_means(data):
    n, d = data.shape
    loss_list = list()
    centers_list = [np.random.uniform(low=data.min(), high=data.max(), size=[NUM_MEANS,d])]
    for iter in range(NUM_ITER):
        #  E Step
        resp = np.sum((centers_list[iter][:,np.newaxis]-data[np.newaxis::])**2, axis=d)
        assign = resp.argmin(axis=0)

        #  M Step
        new_centers = np.array([np.mean(data[assign == j], axis=0) for j in range(NUM_MEANS)])
        centers_list.append(new_centers)
        loss = np.sum(np.min(np.sum((centers_list[iter+1][:,np.newaxis]-data[np.newaxis::])**2, axis=d), axis=0))/n
        loss_list.append(loss)
        print("current mean square deviation at iter {0}: {1}".format(iter, loss))
        if iter > 1:
            if loss_list[iter-1] - loss_list[iter] < EPSILON:
                break
    return centers_list, loss_list, assign


def plot_k_means(clusts, assign):
    plt.figure()
    for j in range(NUM_MEANS):
        plt.scatter(x=train_data[assign == j, 0], y=train_data[assign == j, 1], color=MEANS_COLOR[j], alpha=0.3, s=20)
        plt.scatter(x=clusts[j, 0], y=clusts[j, 1], color=MEANS_COLOR[j], marker=(5, 2), s=40)

    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.title('K Means')

    plt.savefig('../K_means_img')
    plt.show()

k_means, losses, assign = get_k_means(train_data)
plot_k_means(clusts=k_means[-1], assign=assign)

#  Sanity test
#kmeans = KMeans(n_clusters=4, n_init=30).fit(train_data)
#print(kmeans.cluster_centers_)

####################
### EM Spherical ###
####################

init_mu = k_means[-1]
init_resp = assign


def spherical_EM(data, init_mu, init_resp):
    #  Initialize variables
    n, d = data.shape
    mu_list = [init_mu]
    pi_list = [np.array([init_resp.tolist().count(i)/n for i in range(NUM_MEANS)])]
    cov_list = [np.array([np.identity(d)*data[init_resp==i].var() for i in range(NUM_MEANS)])]

    losses = []
    for iter_em in range(NUM_ITER):
        pi = pi_list[iter_em]
        mu = mu_list[iter_em]
        cov = cov_list[iter_em]

        # E Step
        resp, loss = compute_loss(data, pi, mu, cov)

        # Compute loss
        losses.append(loss)

        if iter_em > 1:
            if losses[iter_em] - losses[iter_em-1] < EPSILON:
                break

        # M Step
        new_pi = resp.mean(axis=0)
        pi_list.append(new_pi)

        new_mu = np.dot(resp.T, data)/resp.T.sum(axis=1, keepdims=True)
        mu_list.append(new_mu)

        new_cov = (data[:, np.newaxis, :]-new_mu)**2
        new_cov = new_cov * resp[:,:,np.newaxis]
        new_cov /= resp.sum(axis=0)[:,np.newaxis]
        new_cov = new_cov.mean(d).sum(0)  # average over dimensions for spherical
        new_cov = np.array([np.identity(d) * new_cov[i] for i in range(NUM_MEANS)])

        mu_list.append(new_mu)
        cov_list.append(new_cov)
        pi_list.append(new_pi)

        #print(new_pi)
        #print(new_mu)
        #print(new_cov)

        print('log likelihood for iteration {0}: {1}'.format(iter_em, loss))

    return pi_list, mu_list, cov_list, losses, resp

pi_list, mu_list, cov_list, losses, resp = spherical_EM(train_data, init_mu, init_resp)

# Final Params
mu = mu_list[-1]
cov = cov_list[-1]
pi = pi_list[-1]
loss = losses[-1]


def plot_EM_means(mu, cov, resp, data, name, title):
    hidden = np.argmax(resp, axis=1)
    plt.figure()

    x = np.linspace(-10, 10)
    y = np.linspace(-10, 10)
    X, Y = np.meshgrid(x, y)

    for j in range(NUM_MEANS):
        plt.scatter(x=data[hidden == j, 0], y=data[hidden == j, 1], color=MEANS_COLOR[j], alpha=0.3, s=20)
        plt.scatter(x=mu[j, 0], y=mu[j, 1], color=CLUSTER_MEANS_COLOR[j], marker=(5, 2), s=40)

        Z1 = mlab.bivariate_normal(X, Y, sigmax=cov[j][0][0], sigmay=cov[j][1][1],
                               mux=mu[j][0], muy=mu[j][1], sigmaxy=cov[j][0][1])

        #  Overlay with contours of Normal Dist.
        plt.contour(X, Y, Z1, colors=MEANS_COLOR[j])

    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.title('{}'.format(title))

    plt.savefig('../{}'.format(name))
    plt.show()

plot_EM_means(mu, cov, resp, train_data, name='EM_Spherical_img', title='EM With Spherical Covariance')

# Compute loss
_, test_loss = compute_loss(test_data, pi, mu, cov)


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

###################
### EM  General ###
###################


def general_EM(data, init_mu, init_resp):
    #  Initialize variables
    n, d = data.shape
    mu_list = [init_mu]
    pi_list = [np.array([init_resp.tolist().count(i)/n for i in range(NUM_MEANS)])]
    cov_list = [np.array([np.cov(data[init_resp==i].T) for i in range(NUM_MEANS)])]

    losses = []
    for iter_em in range(NUM_ITER):
        pi = pi_list[iter_em]
        mu = mu_list[iter_em]
        cov = cov_list[iter_em]

        # E Step
        resp, loss = compute_loss(data, pi, mu, cov)
        losses.append(loss)

        if iter_em > 1:
            if losses[iter_em] - losses[iter_em-1] < EPSILON:
                break

        # M Step
        new_pi = resp.mean(axis=0)
        pi_list.append(new_pi)

        new_mu = np.dot(resp.T, data)/resp.T.sum(axis=1, keepdims=True)
        mu_list.append(new_mu)

        new_cov = np.empty((NUM_MEANS, d, d))
        for k in range(NUM_MEANS):
            diff = train_data - mu[k]
            new_cov[k] = np.dot(resp[:, k] * diff.T, diff) / np.sum(resp, axis=0)[k]

        mu_list.append(new_mu)
        cov_list.append(new_cov)
        pi_list.append(new_pi)

        #print(new_pi)
        #print(new_mu)
        #print(new_cov)

        print('log likelihood for iteration {0}: {1}'.format(iter_em, loss))

    return pi_list, mu_list, cov_list, losses, resp

#  Initialize variables using values from before
pi_list, mu_list, cov_list, losses, resp = general_EM(train_data, init_mu, init_resp)

# Final Params
mu = mu_list[-1]
cov = cov_list[-1]
pi = pi_list[-1]

# Compute loss
_, test_loss = compute_loss(test_data, pi, mu, cov)


print("train loss: {}".format(loss))
print("test loss: {}".format(test_loss))

gmm = GaussianMixture(n_components=4, verbose=2)
gmm.fit(train_data)

plot_EM_means(mu, cov, resp, train_data, name='EM_General_img', title='EM With General Covariance')

#covariance = gmm.covariances_
#mu = gmm.means_
