import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import multivariate_normal

#  Parameters
FILES = ['EMGaussian.train', 'EMGaussian.test']
COLOR = ['red', 'blue', 'green', 'orange']
CLUSTER_MEANS_COLOR = ['pink', 'cyan', 'green', 'yellow']

train_data = np.loadtxt('hwk4data/' + FILES[0])
test_data = np.loadtxt('hwk4data/' + FILES[1])

pi = np.array([1/4, 1/4, 1/4, 1/4])
A = np.array([[1/2, 1/6, 1/6, 1/6],
              [1/6, 1/2, 1/6, 1/6],
              [1/6, 1/6, 1/2, 1/6],
              [1/6, 1/6, 1/6, 1/2]])
mu = np.array([[-2.0344, 4.1726],
               [3.9779, 3.7735],
               [3.8007, -3.7972],
               [-3.0620, -3.5345]])
cov = np.array([[[2.9044, 0.2066], [0.2066, 2.7562]],
               [[0.2104, 0.2904], [0.2904, 12.2392]],
               [[0.9213, 0.0574], [0.0574, 1.8660]],
               [[6.2414, 6.0502], [6.0502, 6.1825]]])


def alpha_pass(in_data, pi, mu, cov, A):
    n, d = in_data.shape
    alphas = np.zeros(shape=[n,4])  # alpha pass
    NC = np.zeros(shape=[n])  # normalization constant

    alpha_1 = pi.T*np.array([multivariate_normal(mean=mu[i], cov=cov[i]).pdf(in_data[0]) for i in range(4)]).T
    nc = np.sum(alpha_1)
    alphas[0, :] = alpha_1/nc
    NC[0] = nc

    # alpha pass
    for k in range(1,n):
        OT = np.array([multivariate_normal(mean=mu[i], cov=cov[i]).pdf(in_data[k]) for i in range(4)]).T
        alpha_k = OT*np.dot(A,alphas[k-1])
        NC[k] = np.sum(alpha_k)
        alphas[k, :] = alpha_k/NC[k]

    return alphas, NC


def beta_pass(in_data, mu, cov, A, NC):
    n, d = in_data.shape
    betas = np.zeros(shape=[n, 4])
    beta_1 = np.array([1, 1, 1, 1])
    betas[n-1, :] = beta_1

    # beta pass
    for k in range(n-2,-1,-1):
        OT = np.array([multivariate_normal(mean=mu[i], cov=cov[i]).pdf(in_data[k+1]) for i in range(4)]).T
        beta_k = np.dot(OT*betas[k+1], A)
        betas[k,:] = beta_k/NC[k]

    return betas


def compute_smoothing(in_data, pi, A, mu, cov):
    n, d = in_data.shape

    alphas, NC = alpha_pass(in_data, pi, mu, cov, A)
    betas = beta_pass(in_data, mu, cov, A, NC)

    #  compute smoothing distribution
    smoothing_dist = alphas*betas/np.sum(alphas*betas, axis=1, keepdims=True)

    #  compute pair marginals
    pair_marginals = np.zeros(shape=[n-1,4,4])
    for k in range(n-1):
        #OT1 = np.array([multivariate_normal(mean=mu[i], cov=cov[i]).pdf(data[k]) for i in range(4)]).T
        #OT2 = np.array([multivariate_normal(mean=mu[i], cov=cov[i]).pdf(data[k+1]) for i in range(4)]).T
        #OT = np.outer(OT1,OT2)
        OT = np.array([multivariate_normal(mean=mu[i], cov=cov[i]).pdf(in_data[k + 1]) for i in range(4)])
        OT = np.outer(OT, np.ones(4)).T
        pair_marginals[k] = np.outer(alphas[k], betas[k+1])*A*OT
        pair_marginals[k] = pair_marginals[k]/np.sum(pair_marginals[k])

    return smoothing_dist, pair_marginals


#  Plot for Q2
def plot_smoothing(smoothing, lim):
    plt.figure().subplots_adjust(wspace=0, hspace=2)
    plt.suptitle('Smoothing Distribution Over Time')
    for j in range(4):
        plt.subplot(4, 1, j+1)
        plt.plot(smoothing[:lim, j], color=COLOR[j])
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.title(r'$p(z_t={} |x_1, \cdots, x_T )$'.format(j+1))

    plt.savefig('../smoothing_dist')
    plt.show()

smoothing, pair_marginals = compute_smoothing(test_data, pi, A, mu, cov)
plot_smoothing(smoothing, 100)

#print(np.sum(pair_marginals[0], axis=1) == smoothing[0])
'''
a = np.where(smoothing[0] == smoothing[0].max())
print("at state {}".format(a[0][0] + 1))
for ind in range(pair_marginals.shape[0]):
    x, y = np.where(pair_marginals[ind] == np.max(pair_marginals[ind]))
    print("went from {0} to {1}".format(y[0]+1, x[0]+1))
    a = np.where(smoothing[ind+1] == smoothing[ind+1].max())
    print("at state {}".format(a[0][0]+1))
'''


def get_point_marker(data_point, ratios):
    ratios = ratios.cumsum()
    ret_val = list()
    x = [0] + np.cos(np.linspace(0, 2 * np.pi * ratios[0], 10)).tolist()
    y = [0] + np.sin(np.linspace(0, 2 * np.pi * ratios[0], 10)).tolist()
    ret_val.append(list(zip(x, y)))
    for k in range(len(ratios)-2):
        x = [0] + np.cos(np.linspace(2 * np.pi * ratios[k], 2 * np.pi * ratios[k+1], 10)).tolist()
        y = [0] + np.sin(np.linspace(2 * np.pi * ratios[k], 2 * np.pi * ratios[k+1], 10)).tolist()
        ret_val.append(list(zip(x, y)))

    x = [0] + np.cos(np.linspace(2 * np.pi * ratios[-2], 2 * np.pi, 10)).tolist()
    y = [0] + np.sin(np.linspace(2 * np.pi * ratios[-2], 2 * np.pi, 10)).tolist()
    ret_val.append(list(zip(x, y)))

    for ind, ret in enumerate(ret_val):
        plt.scatter(data_point[0], data_point[1], marker=(ret, 0), facecolor=COLOR[ind])


# EM implementation for Q4
class HMM_model:
    def __init__(self, pi_init, A_init, mu_init, cov_init):
        self.pi = pi_init
        self.A = A_init
        self.mu = mu_init
        self.cov = cov_init
        self.smoothing = None
        self.pair_marginals = None

    def compute_loss(self, data):
        _, NC = alpha_pass(data, self.pi, self.mu, self.cov, self.A)
        return np.sum(np.log(NC))

        #  compute smoothing distribution
        #smoothing_dist = alphas * betas / np.sum(alphas * betas, axis=1, keepdims=True)

        #  See Eq 2.1 in homework
        #loss = np.sum(self.smoothing[0] * np.log(self.pi))
        #loss += np.sum(self.smoothing * np.array([np.log(multivariate_normal(mean=self.mu[i], cov=self.cov[i]).pdf(data))
        #                                     for i in range(4)]).T)  # observation model
        #loss += np.sum(pair_marginals * np.log(self.A))  # transition model
        #return loss

    def plot_viterbi(self, labels, data):
        plt.figure()

        for j in range(4):
            dat = data[np.where(np.array(labels) == j)]
            plt.scatter(x=dat[:,0], y=dat[:,1], color=COLOR[j], alpha=0.3,
                        s=20)

        x = np.linspace(-10, 10)
        y = np.linspace(-10, 10)
        X, Y = np.meshgrid(x, y)

        for j in range(4):
            plt.scatter(x=self.mu[j, 0], y=self.mu[j, 1], color=CLUSTER_MEANS_COLOR[j], marker=(5, 2), s=40)

            Z1 = mlab.bivariate_normal(X, Y, sigmax=self.cov[j][0][0], sigmay=self.cov[j][1][1],
                                       mux=self.mu[j][0], muy=self.mu[j][1], sigmaxy=self.cov[j][0][1])

            #  Overlay with contours of Normal Dist.
            plt.contour(X, Y, Z1, colors=CLUSTER_MEANS_COLOR[j])

        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.title('{}'.format('Viterbi Decoded Centres'))

        plt.savefig('../Viterbi_Plot'.format())
        plt.show()

    def plot_means(self, data, title):
        plt.figure()

        for ind in range(data.shape[0]):
            get_point_marker(data[ind], self.smoothing[ind])

        x = np.linspace(-10, 10)
        y = np.linspace(-10, 10)
        X, Y = np.meshgrid(x, y)

        for j in range(4):
            plt.scatter(x=self.mu[j, 0], y=self.mu[j, 1], color=CLUSTER_MEANS_COLOR[j], marker=(5, 2), s=40)

            Z1 = mlab.bivariate_normal(X, Y, sigmax=self.cov[j][0][0], sigmay=self.cov[j][1][1],
                                       mux=self.mu[j][0], muy=self.mu[j][1], sigmaxy=self.cov[j][0][1])

            #  Overlay with contours of Normal Dist.
            plt.contour(X, Y, Z1, colors=CLUSTER_MEANS_COLOR[j])

        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.title('{}'.format(title))

        plt.savefig('../EM_Plot'.format())
        plt.show()

    # Estimate params
    def estimate_params(self, data):
        pi_new = self.smoothing[0]
        A_new = self.pair_marginals.sum(axis=0)
        A_new /= A_new.sum(axis=0, keepdims=True)

        mu_new = np.dot(self.smoothing.T, data) / self.smoothing.T.sum(axis=1, keepdims=True)

        cov_new = np.empty((4, 2, 2))

        for k in range(4):
            diff = data - mu_new[k]
            cov_new[k] = np.dot(self.smoothing[:, k] * diff.T, diff) / np.sum(self.smoothing, axis=0)[k]
        return pi_new, A_new, mu_new, cov_new

    def train_model(self, max_iter, epsilon, train_data, test_data):
        train_losses = list()
        test_losses = list()

        for iter in range(max_iter):
            #  E step
            self.smoothing, self.pair_marginals = compute_smoothing(train_data, self.pi, self.A, self.mu, self.cov)

            #  Compute loss
            loss = self.compute_loss(train_data)
            test_loss = self.compute_loss(test_data)

            print("current loss on iter {0}: {1}".format(iter+1, loss))
            train_losses.append(loss)
            test_losses.append(test_loss)
            if iter > 1:
                if train_losses[iter] - train_losses[iter - 1] < epsilon:
                    break

            #  M step
            self.pi, self.A, self.mu, self.cov = self.estimate_params(train_data)

        self.plot_means(train_data, title='EM Algorithm Results')

        return train_losses, test_losses

    #  Implement Vertibi Decoding
    def vertibi_decoding(self, data):
        n, d = data.shape
        path = {}  # final path
        V = np.zeros(shape=[n, 4])  # probabilities

        V[0, :] = np.log(pi.T) + np.log(np.array([multivariate_normal(mean=mu[i], cov=cov[i]).pdf(data[0]) for i in range(4)]).T)
        for y in range(4):
            path[y] = [y]  # initialize with each state

        for k in range(1, n):
            newpath = {}
            OT = np.array([multivariate_normal(mean=mu[i], cov=cov[i]).pdf(data[k]) for i in range(4)]).T

            for next_state in range(4):
                (prob, state) = max((V[k-1, prev_state] +
                                     np.log(A[prev_state, next_state]) +
                                     np.log(OT[next_state]), prev_state) for prev_state in range(4))
                V[k][next_state] = prob
                newpath[next_state] = path[state] + [next_state]

            path = newpath

        prob, path = np.max(V[-1, :]), path[np.argmax(V[-1, :])]

        self.plot_viterbi(path, data)

        return prob, path

hmm = HMM_model(pi_init=pi, A_init=A, mu_init=mu, cov_init=cov)

train_loss, test_loss = hmm.train_model(max_iter=100, epsilon=1e-3, train_data=train_data, test_data=test_data)
prob, path = hmm.vertibi_decoding(train_data)

#  Plot for Q5
plt.figure()
plt.xlabel('Iteration')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood Per Iteration')
plt.plot(train_loss, color='red', label='train ll')
plt.plot(test_loss, color='blue', label='test ll')
plt.legend(loc='bottom right')
plt.savefig('../ll_Plot'.format())
plt.show()