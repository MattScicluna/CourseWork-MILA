import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import multivariate_normal

DATA_COLOR = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
CLUSTER_MEANS_COLOR = ['pink', 'cyan', 'green', 'yellow', 'violet', 'gray']


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
        plt.scatter(data_point[0], data_point[1], marker=(ret, 0), facecolor=DATA_COLOR[ind])

class KMeans:
    def __init__(self, num_means):
        self.num_means = num_means
        self.centers = None
        self.resp = None
        self.assign = None

    def assign_responsibilities(self, data):
        n, d = data.shape
        resp = np.sum((self.centers[:, np.newaxis] - data[np.newaxis::]) ** 2, axis=d)
        assign = resp.argmin(axis=0)
        return resp, assign

    def compute_new_centers(self, data):
        return np.array([np.mean(data[self.assign == j], axis=0) for j in range(self.num_means)])

    def compute_loss(self, data):
        n, d = data.shape
        loss = np.sum(
            np.min(np.sum((self.centers[:, np.newaxis] - data[np.newaxis::]) ** 2, axis=d), axis=0)) / n
        return loss

    def train(self, data, num_iter, epsilon):
        n, d = data.shape
        loss_list = list()
        self.centers = np.random.uniform(low=data.min(), high=data.max(), size=[self.num_means, d])
        for iter in range(num_iter):
            #  E Step
            self.resp, self.assign = self.assign_responsibilities(data)

            #  M Step
            self.centers = self.compute_new_centers(data)

            loss_list.append(self.compute_loss(data))
            print("current mean square deviation at iter {0}: {1}".format(iter, loss_list[iter]))
            if iter > 1:
                if loss_list[iter - 1] - loss_list[iter] < epsilon:
                    break

        return loss_list

    def plot_k_means(self, data):
        _, assign = self.assign_responsibilities(data)
        plt.figure()
        for j in range(self.num_means):
            plt.scatter(x=data[assign == j, 0], y=data[assign == j, 1], color=DATA_COLOR[j], alpha=0.3,
                        s=20)
            plt.scatter(x=self.centers[j, 0], y=self.centers[j, 1], color=CLUSTER_MEANS_COLOR[j], marker=(5, 2), s=40)

        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.title('K Means')

        #plt.savefig('../K_means_img')
        plt.show()


class GMM_model():
    def __init__(self, init_mu, init_pi, init_cov, struct='full'):
        self.pi = init_pi
        self.mu = init_mu
        self.cov = init_cov
        self.k = self.pi.shape[0]
        self.resp = None
        assert struct in ['full', 'sphere']
        self.struct = struct

    def __init__(self, train_data, num_clusters, struct='full'):
        n, d = train_data.shape
        self.k = num_clusters
        assert struct in ['full', 'sphere']
        self.struct = struct
        kmodel = KMeans(self.k)
        kmodel.train(train_data, 100, 1e-10)
        #kmodel.plot_k_means(train_data)
        self.mu = kmodel.centers
        self.resp = kmodel.assign
        self.pi = np.array([self.resp.tolist().count(i) / n for i in range(self.k)])
        if self.struct == 'full':
            self.cov = np.array([np.identity(d) * train_data[self.resp == i].var() for i in range(self.k)])
        if self.struct == 'sphere':
            self.cov = np.array([np.identity(d) * train_data[self.resp == i].var() for i in range(self.k)])

    def get_responsibilities(self, data):
        n, _ = data.shape
        resp = np.array(
            [np.log(self.pi[i]) + np.log(multivariate_normal(mean=self.mu[i], cov=self.cov[i]).pdf(data)) for i in range(self.k)]).T
        #  normalize
        resp = np.exp(resp)
        resp /= resp.sum(axis=1, keepdims=1)
        return resp

    def compute_loss(self, data):
        n, _ = data.shape
        resp = self.get_responsibilities(data)

        loss = np.array([np.log(self.pi[i]) + np.log(multivariate_normal(mean=self.mu[i], cov=self.cov[i]).pdf(data))
                         for i in range(self.k)])
        loss = np.sum(loss.T * resp) / n
        return resp, loss

    def update_params(self, data):
        n, d = data.shape
        new_pi = self.resp.mean(axis=0)

        new_mu = np.dot(self.resp.T, data) / self.resp.T.sum(axis=1, keepdims=True)

        if self.struct == 'full':
            new_cov = np.empty((self.k, d, d))
            for k in range(self.k):
                diff = data - self.mu[k]
                new_cov[k] = np.dot(self.resp[:, k] * diff.T, diff) / np.sum(self.resp, axis=0)[k]
        if self.struct == 'sphere':
            new_cov = (data[:, np.newaxis, :] - new_mu) ** 2
            new_cov = new_cov * self.resp[:, :, np.newaxis]
            new_cov /= self.resp.sum(axis=0)[:, np.newaxis]
            new_cov = new_cov.mean(d).sum(0)  # average over dimensions for spherical
            new_cov = np.array([np.identity(d) * new_cov[i] for i in range(self.k)])

        return new_pi, new_mu, new_cov

    def train(self, data, num_iter, epsilon):
        losses = []
        for iter_em in range(num_iter):

            # E Step
            self.resp, loss = self.compute_loss(data)
            losses.append(loss)

            if iter_em > 1:
                if losses[iter_em] - losses[iter_em-1] < epsilon:
                    break

            # M Step
            self.pi, self.mu, self.cov = self.update_params(data)
            print('log likelihood for iteration {0}: {1}'.format(iter_em, loss))

        return losses

    def plot_means(self, data, name, title):
        n, d = data.shape
        resp = self.get_responsibilities(data)

        plt.figure()

        for ind in range(n):
            get_point_marker(data[ind], resp[ind])

        x = np.linspace(-10, 10)
        y = np.linspace(-10, 10)
        X, Y = np.meshgrid(x, y)

        for j in range(self.k):
            plt.scatter(x=self.mu[j, 0], y=self.mu[j, 1], color=CLUSTER_MEANS_COLOR[j], marker=(5, 2), s=40)

            Z1 = mlab.bivariate_normal(X, Y, sigmax=self.cov[j][0][0], sigmay=self.cov[j][1][1],
                                       mux=self.mu[j][0], muy=self.mu[j][1], sigmaxy=self.cov[j][0][1])

            #  Overlay with contours of Normal Dist.
            plt.contour(X, Y, Z1, colors=CLUSTER_MEANS_COLOR[j])

        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.title('{}'.format(title))

        plt.savefig('../{}'.format(name))
        plt.show()

# EM implementation for Q4
class HMM_model:
    def __init__(self, pi_init, A_init, mu_init, cov_init):
        self.pi = pi_init
        self.A = A_init
        self.mu = mu_init
        self.cov = cov_init
        self.smoothing = None
        self.pair_marginals = None
        self.k, self.d = self.mu.shape
        assert self.d == 2  # ensure 2D data

    def alpha_pass(self, in_data):
        n, _ = in_data.shape
        alphas = np.zeros(shape=[n, self.k])  # alpha pass
        NC = np.zeros(shape=[n])  # normalization constant

        alpha_1 = self.pi.T * np.array([multivariate_normal(mean=self.mu[i], cov=self.cov[i]).pdf(in_data[0]) for i in range(self.k)]).T
        nc = np.sum(alpha_1)
        alphas[0, :] = alpha_1 / nc
        NC[0] = nc

        # alpha pass
        for k in range(1, n):
            OT = np.array([multivariate_normal(mean=self.mu[i], cov=self.cov[i]).pdf(in_data[k]) for i in range(self.k)]).T
            alpha_k = OT * np.dot(self.A, alphas[k - 1])
            NC[k] = np.sum(alpha_k)
            alphas[k, :] = alpha_k / NC[k]

        return alphas, NC

    def beta_pass(self, in_data, NC):
        n, _ = in_data.shape
        betas = np.zeros(shape=[n, self.k])
        beta_1 = np.ones(shape=[self.k])
        betas[n - 1, :] = beta_1

        # beta pass
        for k in range(n - 2, -1, -1):
            OT = np.array([multivariate_normal(mean=self.mu[i], cov=self.cov[i]).pdf(in_data[k + 1])
                           for i in range(self.k)]).T
            beta_k = np.dot(OT * betas[k + 1], self.A)
            betas[k, :] = beta_k / NC[k]

        return betas

    def compute_smoothing(self, in_data):
        n, _ = in_data.shape

        alphas, NC = self.alpha_pass(in_data)
        betas = self.beta_pass(in_data, NC)

        #  compute smoothing distribution
        smoothing_dist = alphas * betas / np.sum(alphas * betas, axis=1, keepdims=True)

        #  compute pair marginals
        pair_marginals = np.zeros(shape=[n - 1, self.k, self.k])
        for k in range(n - 1):
            # OT1 = np.array([multivariate_normal(mean=mu[i], cov=cov[i]).pdf(data[k]) for i in range(4)]).T
            # OT2 = np.array([multivariate_normal(mean=mu[i], cov=cov[i]).pdf(data[k+1]) for i in range(4)]).T
            # OT = np.outer(OT1,OT2)
            OT = np.array([multivariate_normal(mean=self.mu[i], cov=self.cov[i]).pdf(in_data[k + 1]) for i in range(self.k)])
            OT = np.outer(OT, np.ones(self.k)).T
            pair_marginals[k] = np.outer(alphas[k], betas[k + 1]) * self.A * OT
            pair_marginals[k] = pair_marginals[k] / np.sum(pair_marginals[k])

        return smoothing_dist, pair_marginals

    def compute_loss(self, data):
        _, NC = self.alpha_pass(data)
        return np.sum(np.log(NC))

        #  compute smoothing distribution
        #smoothing_dist = alphas * betas / np.sum(alphas * betas, axis=1, keepdims=True)

        #  See Eq 2.1 in homework
        #loss = np.sum(self.smoothing[0] * np.log(self.pi))
        #loss += np.sum(self.smoothing * np.array([np.log(multivariate_normal(mean=self.mu[i], cov=self.cov[i]).pdf(data))
        #                                     for i in range(4)]).T)  # observation model
        #loss += np.sum(pair_marginals * np.log(self.A))  # transition model
        #return loss

    # Estimate params
    def estimate_params(self, data):
        pi_new = self.smoothing[0]
        A_new = self.pair_marginals.sum(axis=0)
        A_new /= A_new.sum(axis=0, keepdims=True)

        mu_new = np.dot(self.smoothing.T, data) / self.smoothing.T.sum(axis=1, keepdims=True)

        cov_new = np.empty((self.k, self.d, self.d))

        for k in range(self.k):
            diff = data - mu_new[k]
            cov_new[k] = np.dot(self.smoothing[:, k] * diff.T, diff) / np.sum(self.smoothing, axis=0)[k]
        return pi_new, A_new, mu_new, cov_new

    def train_model(self, max_iter, epsilon, train_data, test_data):
        train_losses = list()
        test_losses = list()

        for iter in range(max_iter):
            #  E step
            self.smoothing, self.pair_marginals = self.compute_smoothing(train_data)

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

        self.plot_means(data=train_data, smoothing=self.smoothing, title='EM Algorithm Results')

        return train_losses, test_losses

    #  Implement Vertibi Decoding
    def vertibi_decoding(self, data, showplot=True):
        T, _ = data.shape
        V = np.zeros(shape=[T, self.k])  # probabilities
        prev = np.empty(shape=[T, self.k], dtype='int')  # final path

        V[0, :] = np.log(self.pi.T) + np.log(np.array([multivariate_normal(mean=self.mu[j], cov=self.cov[j]).pdf(data[0]) for j in range(self.k)]).T)
        prev[0, :] = -np.ones(shape=[self.k])  # initialize with each state

        for i in range(1, T):
            for j in range(self.k):
                (prob, state) = max((V[i - 1, k] + np.log(self.A[k, j]) +
                                 np.log(multivariate_normal(mean=self.mu[j], cov=self.cov[j]).pdf(data[i])), k)
                                for k in range(self.k))
                V[i][j] = prob
                prev[i][j] = state

        #  Final path
        path = np.empty(shape=[T], dtype='int')
        prob = np.max(V[-1, :])
        path[-1] = np.argmax(V[-1, :])
        #  backtrack
        for i in range(T-1, 0, -1):
            path[i-1] = prev[i, path[i]]

        if showplot:
            self.plot_viterbi(path, data)

        return prob, path

    def plot_viterbi(self, labels, data, plot_covs=False):
        plt.figure()

        for j in range(self.k):
            dat = data[np.where(np.array(labels) == j)]
            plt.scatter(x=dat[:,0], y=dat[:,1], color=DATA_COLOR[j], alpha=0.3,
                        s=20)

        x = np.linspace(-10, 10)
        y = np.linspace(-10, 10)
        X, Y = np.meshgrid(x, y)

        for j in range(self.k):
            plt.scatter(x=self.mu[j, 0], y=self.mu[j, 1], color=CLUSTER_MEANS_COLOR[j], marker=(5, 2), s=40)
            if plot_covs:
                Z1 = mlab.bivariate_normal(X, Y, sigmax=self.cov[j][0][0], sigmay=self.cov[j][1][1],
                                           mux=self.mu[j][0], muy=self.mu[j][1], sigmaxy=self.cov[j][0][1])

                #  Overlay with contours of Normal Dist.
                plt.contour(X, Y, Z1, colors=CLUSTER_MEANS_COLOR[j])

        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.title('Data Clustered using Viterbi Decoding')

        plt.savefig('../Viterbi_Plot'.format())
        plt.show()

    def plot_means(self, data, smoothing, title):
        plt.figure()

        for ind in range(data.shape[0]):
            get_point_marker(data[ind], smoothing[ind])

        x = np.linspace(-10, 10)
        y = np.linspace(-10, 10)
        X, Y = np.meshgrid(x, y)

        for j in range(self.k):
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
