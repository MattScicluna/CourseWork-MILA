from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import matplotlib.pyplot as plt

from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch


class newsDataset(Dataset):

    def __init__(self, train_data, target_data, transform=None):
        self.train_data = train_data
        self.target_data = target_data
        self.transform = transform
        if transform == 'stdize':
            self.feat_means = np.mean(train_data.toarray(), axis=0)
            self.feat_stds = np.std(train_data.toarray(), axis=0)

    def __len__(self):
        return self.train_data.shape[0]

    def get_data_length(self):
        return self.train_data.shape[1]

    def __getitem__(self, idx):
        if self.transform == 'stdize':
            return (
                np.array((self.train_data[idx]-self.feat_means)/(1e-5+self.feat_stds)).reshape(-1),
                self.target_data[idx])
        else:
            return (self.train_data[idx].toarray(), self.target_data[idx])


def load_data(transformation, batch_size):
    #  Load data
    newsgroups_data = fetch_20newsgroups(data_home='./', subset='all')
    newsgroups_targets = newsgroups_data.target

    count_vect = CountVectorizer(min_df=10)
    newsgroups_input_data = count_vect.fit_transform(newsgroups_data.data)
    del newsgroups_data, count_vect

    if transformation == 'tfidf':
        tfidf_transformer = TfidfTransformer()
        newsgroups_input_data = tfidf_transformer.fit_transform(newsgroups_input_data)
        print('applied tfidf successfully!')
    #if transformation == 'stdize':
    #    feat_means = np.mean(newsgroups_input_data.toarray(), axis=0)
    #    newsgroups_input_data -= feat_means
    #    del feat_means
    #    feat_vars = np.std(newsgroups_input_data.toarray(), axis=0)
    #    newsgroups_input_data /= (feat_vars + 1e-5)
    #    del feat_vars
    #    print('standardized columns successfully!')

    train_ind = int(newsgroups_targets.shape[0]*0.6)
    valid_ind = int(newsgroups_targets.shape[0]*0.8)

    training_data = newsDataset(train_data=newsgroups_input_data[0:train_ind],
                                target_data=newsgroups_targets[0:train_ind],
                                transform=transformation)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=8)

    input_size, output_size = training_data.get_data_length(), 20

    del training_data

    valid_data = newsDataset(train_data=newsgroups_input_data[(train_ind+1):valid_ind],
                             target_data=newsgroups_targets[(train_ind+1):valid_ind],
                             transform=transformation)
    valid_dataloader = DataLoader(valid_data, batch_size=64, shuffle=True, num_workers=8)

    del valid_data, train_ind

    test_data = newsDataset(train_data=newsgroups_input_data[(valid_ind+1):],
                            target_data=newsgroups_targets[(valid_ind+1):],
                            transform=transformation)

    del newsgroups_targets, newsgroups_input_data, valid_ind

    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=8)

    del test_data

    return train_dataloader, valid_dataloader, test_dataloader, input_size, output_size


#  Build the model
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        a = np.sqrt(6 / (100 + input_size))
        torch.nn.init.uniform(self.fc1.weight, -a, a)

        self.fc2 = nn.Linear(100, output_size)
        a = np.sqrt(6 / (100 + output_size))
        torch.nn.init.uniform(self.fc2.weight, -a, a)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


def compute_accuracy(model, dataloader, input_size):
    error = 0
    acc = 0
    total = 0

    for i_batch, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data.float()), Variable(target)
        data = data.view(-1, input_size)

        pred = model(data)
        loss = F.nll_loss(pred, target)
        error += loss.data[0]

        acc += (torch.max(pred, 1)[1].eq(target)).sum().data[0]
        total += len(target)

    return 100 * (acc / total), error / len(dataloader)


def train_model(model, lr, epochs, compute_test_error, train_dataloader,
                valid_dataloader, test_dataloader,
                input_size, transformation, momentum, get_updates=False):

    train_accuracy = []
    valid_accuracy = []
    test_accuracy = []
    if get_updates:
        updates = []

    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):
        train_acc = 0
        total = 0
        model.train()
        end = False

        for i_batch, (data, target) in enumerate(train_dataloader):
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data.float()), Variable(target)
            data = data.view(-1, input_size)
            optimizer.zero_grad()

            pred = model(data)
            loss = F.nll_loss(pred, target)

            if get_updates:
                updates.append(loss.data[0])
                if len(updates) > 5000:
                    return updates

            if loss.data[0] > 10000:
                print('learning is too unstable, terminating ...')
                end = True
                break

            loss.backward()
            optimizer.step()

            if i_batch % 25 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i_batch, len(train_dataloader),
                           100. * i_batch / len(train_dataloader), loss.data[0]))

            train_acc += (torch.max(pred, 1)[1].eq(target)).sum().data[0]
            total += len(target)

        if end:
            train_accuracy.append(0)
            valid_accuracy.append(0)
            test_accuracy.append(0)
            break

        train_accuracy.append(100*(train_acc/total))

        model.eval()
        acc, error = compute_accuracy(model, valid_dataloader, input_size)
        print('Valid Epoch: {} Loss: {:.6f} Accuracy: {:.6f} %'.format(epoch, error, acc))
        valid_accuracy.append(acc)

        if compute_test_error:
            acc, error = compute_accuracy(model, test_dataloader, input_size)
            test_accuracy.append(acc)

    if compute_test_error:
        plt.figure()
        plt.title('Accuracy of Training and Test Sets for Transformation: {}'.format(transformation))
        plt.plot(train_accuracy, color='red', label='Training Set')
        plt.plot(test_accuracy, color='blue', label='Test Set')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper right')
        plt.savefig('Accuracy-{}'.format(transformation))
        plt.axis([0, epochs, 0, 100])
        plt.show()

    return train_accuracy, valid_accuracy, test_accuracy


def find_best_hyperparam(epochs, transformation, batch_size):
    train_dataloader, valid_dataloader, test_dataloader, input_size, output_size = load_data(transformation, batch_size)

    #  Hyperparam Search
    best_hyperparam = 0.00001
    best_acc = 0
    for hyperparam in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        model = Net(input_size=input_size, output_size=output_size)
        model.cuda()
        _, valid_accuracy, _ = \
        train_model(model=model, lr=hyperparam, epochs=epochs, compute_test_error=False,
                    train_dataloader=train_dataloader, valid_dataloader=valid_dataloader,
                    test_dataloader=test_dataloader, input_size=input_size,
                    transformation=transformation, momentum=0.9)
        valid_acc = valid_accuracy[-1]
        print('learning rate {} has accuracy: {:.6f}'.format(hyperparam, valid_acc))
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_hyperparam = hyperparam
            print('New best hyperparam: {:.6f}'.format(best_hyperparam))

    model = Net(input_size=input_size, output_size=output_size)
    model.cuda()
    train_accuracy, valid_accuracy, test_accuracy = \
        train_model(model=model, lr=best_hyperparam, epochs=epochs, compute_test_error=True,
                    train_dataloader=train_dataloader, valid_dataloader=valid_dataloader,
                    test_dataloader=test_dataloader,
                    input_size=input_size, transformation=transformation, momentum=0.9)
    return best_hyperparam, train_accuracy, valid_accuracy, test_accuracy

# Q 2.1
final_outcome = list()
for trans in ['stdize', 'tfidf', 'no transformation']:

    best_hyperparam, train_accuracy, valid_accuracy, test_accuracy = \
        find_best_hyperparam(epochs=20, transformation=trans, batch_size=64)

    final_outcome.append([trans, best_hyperparam, train_accuracy[-1], test_accuracy[-1]])

    print('\n####################################################\n')
    print('Best hyperparameter for transformation {}: {}\n'.format(trans, best_hyperparam))
    print('#####################################################\n')

print(final_outcome)

#Q2.1 Answers

#  Data set           Best Learning Rate  Training Acc      Test Acc
# 'stdize'            0.01                99.90271513221897 86.70734942955691
# 'tfidf'             0.05                99.77005394888123 90.10347572300344
# 'no transformation' 0.005               98.39922172105776 82.40912708941363

#1) Without transformations, and with large learning rates, the model would overfit before the 20th epoch, so learning
# rate had to be small. With standardization, the training was unstable for learning rates > 0.01, and otherwise
# the model performed comparably to the model trained on the original data. With the tfidf, the model performed well
# across learning rates >0.005 (otherwise, the learning would take much longer than 20 epochs).
# Therefore, they all worked well with a learning rate of 0.01, but each performed optimally at different rates.

#2) If \epsilon = 0 for the Standardization preprocessing, we may have had numerical issues coming from very small
# standard deviations. This could occur with very common words or very rare words. One way to deal with the problem is
# to exclude the most common and rare words from the model.

#3) tf-idf has the advantage of adding weight to words which do no appear often and removing weight from common words.
# Words which only appear in a few document often have high predictive power, and words that are very common often
# do not.

#Q2.2 Answers
train_dataloader, valid_dataloader, test_dataloader, input_size, output_size = load_data('tfidf', batch_size=1)
model = Net(input_size=input_size, output_size=output_size)
model.cuda()
updates_1 = train_model(model=model, lr=0.1, epochs=1, compute_test_error=False,
            train_dataloader=train_dataloader, valid_dataloader=valid_dataloader,
            test_dataloader=test_dataloader,
            input_size=input_size, transformation='tfidf', momentum=0, get_updates=True)
del train_dataloader, valid_dataloader, test_dataloader, input_size, output_size

train_dataloader, valid_dataloader, test_dataloader, input_size, output_size = load_data('tfidf', batch_size=100)
updates_2 = train_model(model=model, lr=0.1, epochs=100, compute_test_error=False,
            train_dataloader=train_dataloader, valid_dataloader=valid_dataloader,
            test_dataloader=test_dataloader,
            input_size=input_size, transformation='tfidf', momentum=0, get_updates=True)

plt.figure()
plt.title('Per Update Loss with Different Minibatch Sizes')
plt.plot(updates_1, color='red', label='Minibatch size 1')
plt.plot(updates_2, color='blue', label='Minibatch size 100')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('loss-minibatch')
plt.axis([0, 5000, 0, 5])
plt.show()

#Q2.2 Answers
#1) The variance is higher with minibatches of size 1, since larger minibatches is essentially just averaging over the
# gradients of the minibatch (since variance of average is \frac{1}{n} the variance of the single example by the CLT)
#2) If we had to have a minibatch of size 1, we could decrease the learning rate so that the optimization is less
# sensitive to any perturbations from the higher variance.
