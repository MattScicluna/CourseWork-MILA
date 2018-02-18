import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

raw_data = np.loadtxt('20news-bydate/matlab/train.data', dtype=np.int32)

valid_ind = int(np.max(raw_data[:,0])*0.8)

np.where(raw_data[:,0] >= valid_ind)

valid_data = raw_data[np.where(raw_data[:,0] >= valid_ind)]
train_data = raw_data[np.where(raw_data[:,0] < valid_ind)]


def build_dataset(raw_data, name):
    #  load data
    print("loaded data successfully")
    num_articles = np.max(raw_data[:, 0])
    num_words = np.max(raw_data[:, 1])

    def clean_row(k):
        row = raw_data[np.where(raw_data[:, 0] == k)]
        new_row = np.zeros(shape=[num_words], dtype=np.int32)
        try:
            new_row[row[:, 1] - 1] = row[:, 2]
        except:
            print(k)
        return new_row

    #  Build training data
    data = map(clean_row, range(1, num_articles + 1))
    print("processed data successfully")
    np.savetxt(name, np.array(list(data)))


build_dataset(train_data, 'train_data')
