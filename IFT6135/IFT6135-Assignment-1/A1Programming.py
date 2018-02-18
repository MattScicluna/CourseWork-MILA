from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint
from sklearn.feature_extraction.text import TfidfTransformer
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np

#  Helper funs

#  Compute tfidf transformations
#tfidf_transformer = TfidfTransformer()
#train_data_tfidf = tfidf_transformer.fit_transform(train_data)
#train_data_tfidf = torch.FloatTensor(train_data_tfidf.toarray())

class newsDataset(Dataset):

    def __init__(self, used_data, transform=None):
        self.train_data = count_vect.fit_transform(used_data.data)
        self.target_data = used_data.target
        self.transform = transform

    def __len__(self):
        return self.train_data.shape[0]

    def get_data_length(self):
        return self.train_data.shape[1]

    def __getitem__(self, idx):
        return (self.train_data[idx].toarray(), self.target_data[idx])

#  Load data

newsgroups_train = fetch_20newsgroups(data_home='./', subset='train')
pprint(list(newsgroups_train.target_names))
count_vect = CountVectorizer()
training_data = newsDataset(used_data=newsgroups_train, transform=None)

dataloader = DataLoader(training_data, batch_size=64,
                        shuffle=True, num_workers=8)

input_size, output_size = training_data.get_data_length(), 20

#  Build the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
model.cuda()

# create a stochastic gradient descent optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
# create a loss function
criterion = nn.NLLLoss()

for epoch in range(5):
    for i_batch, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data.float()), Variable(target)
        data = data.view(-1, input_size)
        optimizer.zero_grad()

        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        if i_batch % 25 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i_batch, len(dataloader),
                       100. * i_batch / len(dataloader), loss.data[0]))
