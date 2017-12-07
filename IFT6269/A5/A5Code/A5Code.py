import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2017)

i_array = np.array(range(1,50)).reshape([7, 7])
ising_array = np.insert(i_array, 0, i_array[6], axis=0)
ising_array = np.insert(ising_array, 8, i_array[0], axis=0)
ising_array = np.insert(ising_array, 7, ising_array[:, 0], axis=1)
ising_array = np.insert(ising_array, 0, ising_array[:, 6], axis=1)


def find_neighbour(node):
    x, y = np.where(node == ising_array[1:-1, 1:-1])
    x, y = x[0]+1, y[0]+1
    neighbours = [ising_array[x - 1, y], ising_array[x + 1, y],
                  ising_array[x, y - 1], ising_array[x, y + 1]]
    return neighbours


class ising_model():

    def __init__(self, graph, states):
        self.graph = graph
        self.states = states

    def sample_posterior(self, node):
        logit = graph[node]['vertex_weight'] + \
                np.sum([self.states[v]*graph[node]['edges'][v] for v in graph[node]['edges'].keys()])
        prob = 1 / (1 + np.exp(-logit))
        return np.random.binomial(1, prob)

    def gibbs_sampling(self, num_samples, burn_in):
        samples = np.zeros(shape=[num_samples, len(graph)])
        print("burning in ...")
        for t in range(burn_in):
            #  sample each state seperately
            for state in self.states.keys():
                self.states[state] = self.sample_posterior(state)

        print("finished burning in, collecting samples ...")
        for t in range(num_samples):
            #  sample each state seperately
            for state in self.states.keys():
                self.states[state] = self.sample_posterior(state)
            #  Add sample to collection
            for state in self.states.keys():
                samples[t, state-1] = self.states[state]
            if t+1 % 500 == 0:
                print("Finished epoch {}".format(t+1))

        #  compute moments
        return np.mean(samples, axis=0).reshape([7, 7])

    def update_node(self, node):
        logit = self.graph[node]['vertex_weight'] + \
                np.sum([self.states[v]*self.graph[node]['edges'][v] for v in self.graph[node]['edges'].keys()])
        tau = 1 / (1 + np.exp(-logit))
        return tau

    def compute_loss(self):
        loss = 0
        for state in self.states:
            loss += self.states[state]*self.graph[state]['vertex_weight']
            loss += 0.5*np.sum([self.graph[state]['edges'][neighbour]*self.states[state]*self.states[neighbour]
                                for neighbour in self.graph[state]['edges']])
            loss += self.states[state]*np.log(self.states[state]) + (1-self.states[state])*np.log(1-self.states[state])
        return -loss

    def var_mean_field(self, stopping):
        dist = np.inf
        losses = list()
        iter = 1
        #  Stop when distance between updates < stopping critereon
        while dist > stopping:
            dist = 0
            for state in self.states.keys():
                previous_state = self.states[state]
                self.states[state] = self.update_node(state)
                dist += abs(self.states[state]-previous_state)
            losses.append(self.compute_loss())
            print('Loss at iteration {}: {}'.format(iter, losses[-1]))
            iter += 1
        return np.array(list(self.states.values())).reshape([7,7]), losses

#  Q2 - Gibbs Sampling

#  initial states randomly chosen
init_states = dict()
for i in range(1, 50):
    init_states[i] = np.random.binomial(1, 0.5)

# Initialize network as per homework
graph = dict()
for i in range(1, 50):
    node = dict()
    node['vertex_weight'] = (-1) ** i
    node['edges'] = dict()
    neighbours = find_neighbour(i)
    for j in neighbours:
        # if (j, i) not in edges_weight.keys():
        #    edges_weight[(i, j)] = 0.5
        node['edges'][j] = 0.5
    graph[i] = node

#  Collect moments
moments = np.ndarray(shape=[10, 7, 7])
for i in range(10):
    i_mod = ising_model(graph=graph, states=init_states)
    moments[i, :, :] = i_mod.gibbs_sampling(num_samples=5000, burn_in=1000)

np.round(np.sqrt(moments.var(0)), 4)

#  Q2 - Variational Inference

#  initial states randomly chosen
init_states = dict()
for i in range(1, 50):
    init_states[i] = np.random.uniform()

var_ising = ising_model(graph=graph, states=init_states)
var_moments, losses = var_ising.var_mean_field(stopping=0.001)

plt.figure()
plt.title(r'$KL(q||p) - log(Z_p)$ per Epoch')
plt.plot(losses, color='red')
plt.xlabel('Epoch')
plt.ylabel(r'$KL(q||p) - log(Z_p)$')
plt.savefig('../per_epoch')
plt.show()

abs(var_moments-moments.mean(0)).sum()/49
