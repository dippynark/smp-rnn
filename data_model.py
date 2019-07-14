import numpy as np
import os
import pandas as pd
import random
import time

random.seed(time.time())


class StockDataSet(object):
    def __init__(self,
                 stock_sym,
                 input_size=1,
                 num_steps=30,
                 test_ratio=0.1,
                 normalized=True,
                 close_price_only=True):
        self.stock_sym = stock_sym
        self.input_size = input_size
        self.num_steps = num_steps
        self.test_ratio = test_ratio
        self.close_price_only = close_price_only
        self.normalized = normalized

        # Read csv file
        raw_df = pd.read_csv(os.path.join("data", "%s.csv" % stock_sym))

        # Merge into one sequence
        if close_price_only:
            self.raw_seq = raw_df['close'].tolist()
        else:
            self.raw_seq = [price for tup in raw_df[['open', 'close']].values for price in tup]

        self.raw_seq = np.array(self.raw_seq, dtype=np.float32)
        self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_data(self.raw_seq)

    def _prepare_data(self, seq):
        # split into items of input_size
        seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size])
               for i in range(len(seq) // self.input_size)]

        if self.normalized:
            seq = [seq[0] / seq[0][0] - 1.0] + [
                curr / seq[i][-1] - 1.0 for i, curr in enumerate(seq[1:])]

        # split into groups of num_steps
        X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
        y = np.array([seq[i + self.num_steps] for i in range(len(seq) - self.num_steps)])

        train_size = int(len(X) * (1.0 - self.test_ratio))
        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]

        def default_X(data):
            if len(data) == 0:
                return np.empty(shape=(0, self.num_steps, self.input_size))
            return data

        def default_y(data):
            if len(data) == 0:
                return np.empty(shape=(0, self.input_size))
            return data

        return default_X(train_X), default_y(train_y), default_X(test_X), default_y(test_y)

    def train_data(self):
        return self.train_X, self.train_y

    def test_data(self):
        return self.test_X, self.test_y
