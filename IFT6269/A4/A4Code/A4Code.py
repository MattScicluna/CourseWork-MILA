import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import multivariate_normal

#  Parameters
FILES = ['EMGaussian.train', 'EMGaussian.test']
COLOR = ['red', 'blue', 'green', 'orange']
CLUSTER_MEANS_COLOR = ['pink', 'cyan', 'green', 'yellow']
NUM_ITER = 100
EPSILON = 1e-10

train_data = np.loadtxt('hwk4data/' + FILES[0])
test_data = np.loadtxt('hwk4data/' + FILES[1])

pi = np.array([1/4, 1/4, 1/4, 1/4])
A = np.array([[1/2, 1/6, 1/6, 1/6],
              [1/6, 1/2, 1/6, 1/6],
              [1/6, 1/6, 1/2, 1/6],
              [1/6, 1/6, 1/6, 1/2]])
mu = np.array([[-2.0344, 4.1726],
               [ 3.9779, 3.7735],
               [ 3.8007, -3.7972],
               [-3.0620, -3.5345]])
cov = np.array([[[2.9044, 0.2066], [0.2066, 2.7562]],
               [[0.2104, 0.2904], [0.2904, 12.2392]],
               [[0.9213, 0.0574], [0.0574, 1.8660]],
               [[6.2414, 6.0502], [6.0502, 6.1825]]])


def compute_smoothing(data, pi, A, mu, cov):
    n, d = data.shape
    alphas = np.zeros(shape=[n,4])  # alpha pass
    NC = np.zeros(shape=[n])  # normalization constant

    alpha_1 = pi.T*np.array([multivariate_normal(mean=mu[i], cov=cov[i]).pdf(data[0]) for i in range(4)]).T
    nc = np.sum(alpha_1)
    alphas[0, :] = alpha_1/nc
    NC[0] = nc

    # alpha pass
    for k in range(1,n):
        OT = np.array([multivariate_normal(mean=mu[i], cov=cov[i]).pdf(data[k]) for i in range(4)]).T
        alpha_k = OT*np.dot(A,alphas[k-1])
        NC[k] = np.sum(alpha_k)
        alphas[k,:] = alpha_k/NC[k]

    betas = np.zeros(shape=[n,4])
    beta_1 = np.array([1, 1, 1, 1])
    betas[n-1, :] = beta_1

    # beta pass
    for k in range(n-2,-1,-1):
        OT = np.array([multivariate_normal(mean=mu[i], cov=cov[i]).pdf(train_data[k+1]) for i in range(4)]).T
        beta_k = OT*np.dot(A.T,betas[k+1])
        betas[k,:] = beta_k/NC[k+1]

    #  compute smoothing distribution
    smoothing_dist = alphas*betas/np.sum(alphas*betas, axis=1, keepdims=True)

    #  compute pair marginals
    pair_marginals = np.zeros(shape=[n-1,4,4])
    for k in range(n-1):
        OT1 = np.array([multivariate_normal(mean=mu[i], cov=cov[i]).pdf(data[k]) for i in range(4)]).T
        OT2 = np.array([multivariate_normal(mean=mu[i], cov=cov[i]).pdf(data[k+1]) for i in range(4)]).T
        OT = np.outer(OT1, OT2)
        pair_marginals[k] = np.outer(alphas[k], betas[k+1])*OT*A
        pair_marginals[k] /= np.sum(pair_marginals[k])

    return smoothing_dist, pair_marginals

smoothing, pair_marginals = compute_smoothing(test_data, pi, A, mu, cov)


# Estimate params


plt.figure().subplots_adjust(wspace=0, hspace=2)
plt.suptitle('Smoothing Distribution Over Time')
for j in range(4):
    plt.subplot(4, 1, j+1)
    plt.plot(smoothing[:100, j], color=COLOR[j])
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.title(r'$p(z_t={} |x_1, \cdots, x_T )$'.format(j+1))

plt.savefig('../smoothing_dist')
plt.show()
