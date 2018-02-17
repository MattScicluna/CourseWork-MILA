from torch.utils.data import Dataset
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

class newsDataset(Dataset):

    def __init__(self, in_data, out_data, transform=None):
        self.in_data = self.build_dataset(np.loadtxt(in_data, dtype=np.int32))
        self.out_data = np.loadtxt(out_data, dtype=np.int32)
        self.transform = transform

        if transform:
            if transform == 'stdize':
                self.data_mean = np.mean(self.in_data, axis=0)
                self.epsilon = 1e-5
                self.data_den = np.std(self.in_data, axis=0, keepdims=True) + self.epsilon
                self.in_data = self.in_data - self.data_mean
                self.in_data /= self.data_den

            if transform == 'tfidf':
                transformer = TfidfTransformer(smooth_idf=False)
                self.in_data = transformer.fit_transform(self.in_data).toarray()

    def build_dataset(self, raw_data):
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
        return np.array(list(data))

    def __len__(self):
        return self.in_data.shape[0]

    def __getitem__(self, idx):
        return {'input': self.in_data[idx], 'output': self.out_data[idx]}

