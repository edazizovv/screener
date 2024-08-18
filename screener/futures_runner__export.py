#


#
import numpy
import pandas
from scipy.stats import t
from matplotlib import pyplot


#


#
d = './data/futures/pct.csv'
data = pandas.read_csv(d)
data['date'] = pandas.to_datetime(data['date'])
data = data.set_index('date')
data['Target'] = data['Target'].astype(dtype=float)

data['mask'] = numpy.nan

inter_m = 3
# a = 0.01

fee_rate = 0.0004

window = 30
future_window = 7
future_q = 0.1

for j in numpy.arange(start=0, stop=(data.shape[0] - window - future_window - 1)):
    values = data['Target'].values[j:(j + window)]

    # """
    q25 = numpy.quantile(a=values, q=0.25)
    q75 = numpy.quantile(a=values, q=0.75)
    bot_thresh = q25 - inter_m * (q75 - q25)
    top_thresh = q75 + inter_m * (q75 - q25)
    # """
    """
    df = values.shape[0] - 1
    tq = t.ppf(q=(1 - a / 2), df=df)
    bot_thresh = values.mean() - tq * (values.std(ddof=1) / (values.shape[0] ** 0.5))
    top_thresh = values.mean() + tq * (values.std(ddof=1) / (values.shape[0] ** 0.5))
    """
    """
    bot_thresh = values.mean() - inter_m * values.std(ddof=1)
    top_thresh = values.mean() + inter_m * values.std(ddof=1)
    """

    data.loc[data.index[j + window], 'bot_thresh'] = bot_thresh
    data.loc[data.index[j + window], 'top_thresh'] = top_thresh

    real_future_price_long = numpy.quantile(a=data['Target'].values[(j + window + 1):(j + window + future_window + 1)],
                                            q=(1 - future_q))
    real_future_price_short = numpy.quantile(a=data['Target'].values[(j + window + 1):(j + window + future_window + 1)],
                                             q=future_q)

    if data.loc[data.index[j + window], 'Target'] < bot_thresh:
        data.loc[data.index[j + window], 'mask'] = -1

        data.loc[data.index[j + window], 'est_pnl'] = (values.mean() / data.loc[data.index[j + window], 'Target']) * ((1 - fee_rate) / (1 + fee_rate)) - 1
        data.loc[data.index[j + window], 'est_fees'] = fee_rate + (values.mean() / data.loc[data.index[j + window], 'Target']) * fee_rate

        data.loc[data.index[j + window], 'real_pnl'] = (real_future_price_long / data.loc[data.index[j + window], 'Target']) * ((1 - fee_rate) / (1 + fee_rate)) - 1
        data.loc[data.index[j + window], 'real_fees'] = fee_rate + (real_future_price_long / data.loc[data.index[j + window], 'Target']) * fee_rate
    elif data.loc[data.index[j + window], 'Target'] > top_thresh:
        data.loc[data.index[j + window], 'mask'] = 1

        data.loc[data.index[j + window], 'est_pnl'] = (data.loc[data.index[j + window], 'Target'] / values.mean()) * ((1 - fee_rate) / (1 + fee_rate)) - 1    # note: improve base selection
        data.loc[data.index[j + window], 'est_fees'] = fee_rate + (data.loc[data.index[j + window], 'Target'] / values.mean()) * fee_rate

        data.loc[data.index[j + window], 'real_pnl'] = (data.loc[data.index[j + window], 'Target'] / real_future_price_short) * ((1 - fee_rate) / (1 + fee_rate)) - 1    # note: improve base selection
        data.loc[data.index[j + window], 'real_fees'] = fee_rate + (data.loc[data.index[j + window], 'Target'] / real_future_price_short) * fee_rate
    else:
        data.loc[data.index[j + window], 'mask'] = 0

        data.loc[data.index[j + window], 'est_pnl'] = numpy.nan
        data.loc[data.index[j + window], 'est_fees'] = numpy.nan

        data.loc[data.index[j + window], 'real_pnl'] = numpy.nan
        data.loc[data.index[j + window], 'real_fees'] = numpy.nan

data.to_csv('./runned_data.csv')
