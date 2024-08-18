import numpy
import pandas

from neura import WrappedNN

import torch
from torch import nn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from futures_runner_overdrive_features import make_features


data = pandas.read_csv('./runned_data.csv')

fee_rate = 0.0004

window = 30
future_window = 7
future_q = 0.1
foresearch_limit = 100

target_yield = 0.000404
# target_yield = 0

data['d_long'] = numpy.nan
data['d_short'] = numpy.nan
data['y'] = numpy.nan

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
        data.loc[data.index[j + window], 'y'] = d_long < 1
    elif data.loc[data.index[j + window], 'Target'] > data.loc[data.index[j + window], 'top_thresh']:
        data.loc[data.index[j + window], 'y'] = d_short < 1
    else:
        data.loc[data.index[j + window], 'y'] = numpy.nan

data, x_features = make_features(data, window=window)

# data = data.iloc[window:-foresearch_limit].copy()
ghoul_mask = ~data['y'].isna() * (numpy.array(range(data.shape[0])) >= window) * (numpy.array(range(data.shape[0])) < (data.shape[0] - foresearch_limit))

# data = data.loc[ghoul_mask, :].copy()

mxses = {}
measures = {}
perfs = {}
for k in range(5):

    print(k)

    # raise Exception()
    start = int(data.loc[ghoul_mask, :].values.shape[0] * (0.0 + 0.1 * k))
    start_ix = data.index[ghoul_mask].values[start]

    thresh = int(data.loc[ghoul_mask, :].values.shape[0] * (0.5 + 0.1 * k))
    thresh_ix = data.index[ghoul_mask].values[thresh]

    end = int(data.loc[ghoul_mask, :].values.shape[0] * (0.5 + 0.1 * (k + 1))) - 1
    end_ix = data.index[ghoul_mask].values[end]

    ix_train = (numpy.array(range(data.shape[0])) >= data.index.values.tolist().index(start_ix)) * \
               (numpy.array(range(data.shape[0])) < data.index.values.tolist().index(thresh_ix))
    ix_test = (numpy.array(range(data.shape[0])) >= data.index.values.tolist().index(thresh_ix)) *\
              (numpy.array(range(data.shape[0])) < data.index.values.tolist().index(end_ix))

    ghoul_train = ghoul_mask * ix_train
    ghoul_test = ghoul_mask * ix_test

    x_train = data.loc[ghoul_train, x_features].values
    y_train = data.loc[ghoul_train, 'y'].values.astype(dtype=int)

    x_test = data.loc[ghoul_test, x_features].values
    y_test = data.loc[ghoul_test, 'y'].values.astype(dtype=int)

    '''
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    nn_kwargs = {'layers': [nn.Linear, nn.Linear, nn.Linear],
                 'layers_dimensions': [20, 2, 2],
                 'layers_kwargs': [{}, {}, {}],
                 'batchnorms': [None, nn.BatchNorm1d, None],  # nn.BatchNorm1d
                 'activators': [None, nn.LeakyReLU, nn.Softmax],
                 'interdrops': [0.0, 0.0, 0.0],
                 'optimiser': torch.optim.Adamax,  # Adamax / AdamW / SGD
                 'optimiser_kwargs': {'lr': 0.001,
                                      # 'weight_decay': 0.01,
                                      # 'momentum': 0.9,
                                      # 'nesterov': True,
                                      },
    
                 'loss_function': loss_finder_over,
                 'epochs': 30_000,    # 14_000 # 20_000
                 #  'device': device,
                 }
    
    
    model = WrappedNN(**nn_kwargs)
    
    model.fit(X_train=x_train, Y_train=y_train, X_val=x_test, Y_val=y_test)
    
    y_hat_train_ = model.predict(X=x_train)
    y_hat_test_ = model.predict(X=x_test)
    
    y_hat_train = y_hat_train_[:, 0]
    y_hat_test = y_hat_test_[:, 0]
    '''

    # '''

    # m_kwargs = {'n_estimators': 100, 'max_depth': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': None,}
    m_kwargs = {'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': None,}

    model = RandomForestClassifier(**m_kwargs)

    model.fit(X=x_train, y=y_train)

    y_hat_train_ = model.predict(X=x_train)
    y_hat_test_ = model.predict(X=x_test)

    y_hat_train = y_hat_train_
    y_hat_test = y_hat_test_
    # '''

    as_train = accuracy_score(y_true=y_train, y_pred=y_hat_train)
    as_test = accuracy_score(y_true=y_test, y_pred=y_hat_test)
    bs_train = balanced_accuracy_score(y_true=y_train, y_pred=y_hat_train)
    bs_test = balanced_accuracy_score(y_true=y_test, y_pred=y_hat_test)
    ps_train = precision_score(y_true=y_train, y_pred=y_hat_train)
    ps_test = precision_score(y_true=y_test, y_pred=y_hat_test)
    rs_train = recall_score(y_true=y_train, y_pred=y_hat_train)
    rs_test = recall_score(y_true=y_test, y_pred=y_hat_test)
    fs_train = f1_score(y_true=y_train, y_pred=y_hat_train)
    fs_test = f1_score(y_true=y_test, y_pred=y_hat_test)
    cm_train = confusion_matrix(y_true=y_train, y_pred=y_hat_train)
    cm_test = confusion_matrix(y_true=y_test, y_pred=y_hat_test)

    data.loc[ghoul_train, 'filter'] = y_hat_train
    data.loc[ghoul_test, 'filter'] = y_hat_test

    for j in numpy.arange(start=0, stop=(data.shape[0] - window - future_window - 1)):

        values = data['Target'].values[j:(j + window)]

        real_future_price_long = numpy.quantile(a=data['Target'].values[(j + window + 1):(j + window + future_window + 1)],
                                                q=(1 - future_q))
        real_future_price_short = numpy.quantile(a=data['Target'].values[(j + window + 1):(j + window + future_window + 1)],
                                                 q=future_q)

        if (data.loc[data.index[j + window], 'Target'] < data.loc[data.index[j + window], 'bot_thresh']) and \
                (data.loc[data.index[j + window], 'filter']) == 1:
            data.loc[data.index[j + window], 'f_mask'] = -1

            data.loc[data.index[j + window], 'f_est_pnl'] = (values.mean() / data.loc[
                data.index[j + window], 'Target']) * ((1 - fee_rate) / (1 + fee_rate)) - 1
            data.loc[data.index[j + window], 'f_est_fees'] = fee_rate + (
                    values.mean() / data.loc[data.index[j + window], 'Target']) * fee_rate

            data.loc[data.index[j + window], 'f_real_pnl'] = (real_future_price_long / data.loc[
                data.index[j + window], 'Target']) * ((1 - fee_rate) / (1 + fee_rate)) - 1
            data.loc[data.index[j + window], 'f_real_fees'] = fee_rate + (
                    real_future_price_long / data.loc[data.index[j + window], 'Target']) * fee_rate

        elif data.loc[data.index[j + window], 'Target'] > data.loc[data.index[j + window], 'top_thresh'] and \
                (data.loc[data.index[j + window], 'filter']) == 1:

            data.loc[data.index[j + window], 'f_mask'] = 1

            data.loc[data.index[j + window], 'f_est_pnl'] = (data.loc[
                                                                     data.index[
                                                                         j + window], 'Target'] / values.mean()) * (
                                                                        (1 - fee_rate) / (
                                                                        1 + fee_rate)) - 1  # note: improve base selection
            data.loc[data.index[j + window], 'f_est_fees'] = fee_rate + (
                    data.loc[data.index[j + window], 'Target'] / values.mean()) * fee_rate

            data.loc[data.index[j + window], 'f_real_pnl'] = (data.loc[data.index[
                j + window], 'Target'] / real_future_price_short) * ((1 - fee_rate) / (
                    1 + fee_rate)) - 1  # note: improve base selection
            data.loc[data.index[j + window], 'f_real_fees'] = fee_rate + (
                    data.loc[data.index[j + window], 'Target'] / real_future_price_short) * fee_rate
        else:
            data.loc[data.index[j + window], 'f_mask'] = 0

            data.loc[data.index[j + window], 'f_est_pnl'] = numpy.nan
            data.loc[data.index[j + window], 'f_est_fees'] = numpy.nan

            data.loc[data.index[j + window], 'f_real_pnl'] = numpy.nan
            data.loc[data.index[j + window], 'f_real_fees'] = numpy.nan

    perf = data[['filter', 'mask', 'f_real_pnl', 'real_pnl']].copy()

    perf_train = perf.loc[ghoul_train, :].copy()
    perf_test = perf.loc[ghoul_test, :].copy()

    # perf.loc[numpy.isin(perf['mask'].values, [-1, 1]), 'real_pnl'].describe()
    # perf.loc[~perf['filter'].isna(), 'f_real_pnl'].describe()

    # perf_train.loc[numpy.isin(perf_train['mask'].values, [-1, 1]), 'real_pnl'].describe()
    # perf_train['f_real_pnl'].describe()

    # perf_test.loc[numpy.isin(perf_test['mask'].values, [-1, 1]), 'real_pnl'].describe()
    # perf_test['f_real_pnl'].describe()

    mxses[k] = {'train': cm_train, 'test': cm_test}
    measures[k] = {'fold': ['train', 'test', 'train', 'test'], 'measure': ['bs', 'bs', 'fs', 'fs'], 'value': [bs_train, bs_test, fs_train, fs_test]}
    perfs[k] = {'source': perf_test.loc[numpy.isin(perf_test['mask'].values, [-1, 1]), 'real_pnl'].describe(),
                'test': perf_test['f_real_pnl'].describe()}
