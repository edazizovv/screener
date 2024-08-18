import numpy
import pandas
from matplotlib import pyplot

from neura import WrappedNN

import torch
from torch import nn

from futures_runner_overdrive_features import make_close_features


data = pandas.read_csv('./runned_data.csv')

fee_rate = 0.0004

window = 30
future_window = 7
future_q = 0.1
foresearch_limit = 100

target_yield = 0.000404
# target_yield = 0

for j in numpy.arange(start=0, stop=(data.shape[0] - window - foresearch_limit - 1)):

    values = data['Target'].values[j:(j + window)]

    long_pnl = numpy.array([(data.loc[
            data.index[j + window + 1 + i], 'Target'] / data.loc[
            data.index[j + window + 1], 'Target']) * ((1 - fee_rate) / (1 + fee_rate)) - 1 for i in range(foresearch_limit)])
    short_pnl = numpy.array([(data.loc[data.index[
            j + window + 1], 'Target'] / data.loc[
            data.index[j + window + 1 + i], 'Target']) * ((1 - fee_rate) / (
                1 + fee_rate)) - 1 for i in range(foresearch_limit)])  # note: improve base selection

    # sorted_long = numpy.argsort(long_pnl)
    # ix_long = (long_pnl[sorted_long] > target_yield).nonzero()[0]
    ix_long = (long_pnl > target_yield).nonzero()[0]
    d_long = (ix_long[0] + 1) / future_window if len(ix_long) > 0 else foresearch_limit / future_window

    # sorted_short = numpy.argsort(short_pnl)
    # ix_short = (short_pnl[sorted_short] > target_yield).nonzero()[0]
    ix_short = (short_pnl > target_yield).nonzero()[0]
    d_short = (ix_short[0] + 1) / future_window if len(ix_short) > 0 else foresearch_limit / future_window

    data.loc[data.index[j + window], 'd_long'] = d_long
    data.loc[data.index[j + window], 'd_short'] = d_short

    if (data.loc[data.index[j + window], 'Target'] < data.loc[data.index[j + window], 'bot_thresh']):
        data.loc[data.index[j + window], 'y'] = d_long
    elif data.loc[data.index[j + window], 'Target'] > data.loc[data.index[j + window], 'top_thresh']:
        data.loc[data.index[j + window], 'y'] = d_short
    else:
        data.loc[data.index[j + window], 'y'] = numpy.nan

data, c_features = make_close_features(data, data_mash_4h, data_mash_daily)

x_ixs, y_ix = [data.columns.values.tolist().index(c) for c in c_features], data.columns.values.tolist().index('y')

ghoul_mask = ~data['y'].isna() * (numpy.array(range(data.shape[0])) >= window) * (numpy.array(range(data.shape[0])) < (data.shape[0] - foresearch_limit))

thresh = int(data.loc[ghoul_mask, :].values.shape[0] * 0.5)
thresh_ix = data.index[ghoul_mask].values[thresh]

ix_train = numpy.array(range(data.shape[0])) < data.index.values.tolist().index(thresh_ix)
ix_test = numpy.array(range(data.shape[0])) >= data.index.values.tolist().index(thresh_ix)

ghoul_train = ghoul_mask * ix_train
ghoul_test = ghoul_mask * ix_test

joint = []
for j in numpy.arange(start=0, stop=(data.shape[0] - window)):
    joint.append(data.values[numpy.newaxis, j:j+window, :])
joint = numpy.concatenate(joint, axis=0)
joint = torch.tensor(joint, dtype=torch.float32)

train, val = joint[ghoul_train, :, :], joint[ghoul_test, :, :]

class LDV(nn.Module):
    def __init__(self, n_one, n_two, d=(0.1, 0.1), epochs=1_000):

        self._n_one = n_one
        self._n_two = n_two

        self.linear1 = nn.Linear
        self.batch1 = nn.BatchNorm1d(self._n_one)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(p=d[0])

        self.linear2 = nn.Linear(self._n_one, self._n_two)
        self.act2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(p=d[0])

        self.linear3 = nn.Linear(self._n_two, 1)
        self.act3 = nn.LogSigmoid()

        self.layers = None
        self.optimiser = None
        self.train_loss = None
        self.validation_loss = None
        self.epochs = epochs

        super().__init__()

    def fit(self, train, val, x_ixs, y_ix, loss_function, lr=0.001, weight_decay=0.01):

        in_features = sum(x_ixs)

        self.linear1 = self.linear1(in_features, self._n_one)
        self.layers = nn.ModuleList([self.linear1, self.batch1, self.act1, self.drop1, self.linear2, self.act2, self.drop2, self.linear3, self.act3])

        self.optimiser = torch.optim.Adamax(self.parameters(), lr=lr, weight_decay=weight_decay)

        self.train_loss = []
        self.validation_loss = []

        for i in range(self.epochs):
            i += 1
            for phase in ['train', 'validate']:

                if phase == 'train':
                    y_pred = self(train[:, :, x_ixs])
                    single_loss = loss_function(y_pred, train[:, :, y_ix])
                else:
                    y_pred = self(val[:, :, x_ixs])
                    single_loss = loss_function(y_pred, val[:, :, y_ix])

                self.optimiser.zero_grad()

                if phase == 'train':
                    train_lost = single_loss.item()
                    self.train_loss.append(train_lost)
                    single_loss.backward()
                    self.optimiser.step()
                else:
                    validation_lost = single_loss.item()
                    self.validation_loss.append(validation_lost)

            if i % 25 == 1:
                print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost,
                                                                                             validation_lost))
        print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost, validation_lost))

    def predict(self, sample, x_ixs):

        output = self(sample[:, :, x_ixs])
        result = output.detach().numpy()

        return result

def ded_loss(prob, y):

    # prob corr

    pass

    # averaging

    prob *

model = LDV(n_one=32, n_two=16)
model.fit(train=train, val=val, x_ixs=x_ixs, y_ix=y_ix, loss_function=ded_loss)
y_hat_train = model.predict(sample=train, x_ixs=x_ixs)
y_hat_val = model.predict(sample=val, x_ixs=x_ixs)
