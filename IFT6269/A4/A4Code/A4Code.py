import numpy as np
import matplotlib.pyplot as plt
from models import HMM_model, GMM_model

#  set seed for reproducibility
np.random.seed(2017)

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

hmm = HMM_model(pi_init=pi, A_init=A, mu_init=mu, cov_init=cov)

#  Plot for Q2
def plot_smoothing(smoothing, lim, title):
    plt.figure().subplots_adjust(wspace=0, hspace=2)
    plt.suptitle('Smoothing Distribution Over Time')
    for j in range(4):
        plt.subplot(4, 1, j+1)
        plt.plot(smoothing[:lim, j], color=COLOR[j])
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.title(r'$p(z_t={} |x_1, \cdots, x_T )$'.format(j+1))

    plt.savefig('../{}'.format(title))
    plt.show()

smoothing, pair_marginals = hmm.compute_smoothing(test_data)
plot_smoothing(smoothing, 100, 'smoothing_dist')

'''
print(np.sum(pair_marginals[0], axis=1) == smoothing[0])
a = np.where(smoothing[0] == smoothing[0].max())
print("at state {}".format(a[0][0] + 1))
for ind in range(pair_marginals.shape[0]):
    x, y = np.where(pair_marginals[ind] == np.max(pair_marginals[ind]))
    print("went from {0} to {1}".format(y[0]+1, x[0]+1))
    a = np.where(smoothing[ind+1] == smoothing[ind+1].max())
    print("at state {}".format(a[0][0]+1))
'''

train_loss, test_loss = hmm.train_model(max_iter=100, epsilon=1e-3, train_data=train_data, test_data=test_data)

#  Plot for Q5
plt.figure()
plt.xlabel('Iteration')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood Per Iteration')
plt.plot(train_loss, color='red', label='train ll')
plt.plot(test_loss, color='blue', label='test ll')
plt.legend(loc='lower right')
plt.savefig('../ll_Plot'.format())
plt.show()

#  Q6
GModel = GMM_model(train_data=train_data, num_clusters=4, struct='full')
losses = GModel.train(train_data, 100, 1e-10)
print('training log likelihood for GMM with full cov: {}'.format(losses[-1]*500))
_, test_loss = GModel.compute_loss(test_data)
print('test log likelihood for GMM with full cov: {}'.format(test_loss*500))
#GModel.plot_means(train_data, name='EM_general_img', title='EM With General Covariance')

GModel = GMM_model(train_data=train_data, num_clusters=4, struct='sphere')
losses = GModel.train(train_data, 100, 1e-10)
print('training log likelihood for GMM with spherical cov: {}'.format(losses[-1]*500))
_, test_loss = GModel.compute_loss(test_data)
print('test log likelihood for GMM with spherical cov: {}'.format(test_loss*500))

#  Plot for Q8
prob, _ = hmm.vertibi_decoding(train_data, showplot=True)

#  Plot for Q9
smoothing, _ = hmm.compute_smoothing(test_data)
plot_smoothing(smoothing, 100, 'smoothing_dist_EM')

_, path = hmm.vertibi_decoding(test_data, showplot=False)

#  Plot for Q10 and Q11
plt.figure()
plt.title('Marginally Most Probable Path vs Viterbi Path')
plt.plot((np.argmax(smoothing, axis=1)+1)[:100], color='red', label='Marginal Most Probable')
plt.plot((path+1)[:100], color='blue', label='Viterbi Most Probable')
plt.xlabel('Time')
plt.ylabel('Most Likely State')
plt.legend(loc='upper right')
plt.savefig('../marg_vs_viterbi')
plt.show()

# See where points differed np.where(np.argmax(smoothing, axis=1)!=path)
