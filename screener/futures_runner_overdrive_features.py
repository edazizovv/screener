#


#
import numpy
from scipy.stats import norm, skew, kurtosis

#


#
def q_05(array):
    return numpy.quantile(array, q=0.05)


def q_25(array):
    return numpy.quantile(array, q=0.25)


def q_50(array):
    return numpy.quantile(array, q=0.50)


def q_75(array):
    return numpy.quantile(array, q=0.75)


def q_95(array):
    return numpy.quantile(array, q=0.95)


def gauss_q_05(row, loc_name, scale_name):
    return norm.ppf(0.05, loc=row[loc_name], scale=row[scale_name])


def gauss_q_25(row, loc_name, scale_name):
    return norm.ppf(0.25, loc=row[loc_name], scale=row[scale_name])


def gauss_q_50(row, loc_name, scale_name):
    return norm.ppf(0.50, loc=row[loc_name], scale=row[scale_name])


def gauss_q_75(row, loc_name, scale_name):
    return norm.ppf(0.75, loc=row[loc_name], scale=row[scale_name])


def gauss_q_95(row, loc_name, scale_name):
    return norm.ppf(0.95, loc=row[loc_name], scale=row[scale_name])


def if_rated(row, numerator_name, denominator_name, base_name, conservative):
    rate = (row[numerator_name] - row[base_name]) / (row[denominator_name] - row[base_name])
    result = rate if rate > 0 else conservative
    return result


def make_features(data, window, full=False):

    x_features = []

    # usual lags

    for k in range(window):
        data['x_{0}'.format(k + 1)] = data['Target'].shift(periods=k + 1)
        x_features.append('x_{0}'.format(k + 1))

    # usual pcts

    data['Target_pct'] = data['Target'] / data['x_1'] - 1
    x_features.append('Target_pct')

    for k in range(window-1):
        data['x_{0}_pct'.format(k + 2)] = data['x_{0}'.format(k + 2)] / data['x_{0}'.format(k + 1)] - 1
        x_features.append('x_{0}_pct'.format(k + 2))

    add_features = []

    if window > 20:

        # horizon 20 base temp

        data['ewm_20'] = data['Target'].ewm(span=20).mean()
        data['q_05_20'] = data['Target'].rolling(20).apply(q_05, raw=True)
        data['q_25_20'] = data['Target'].rolling(20).apply(q_25, raw=True)
        data['q_50_20'] = data['Target'].rolling(20).apply(q_50, raw=True)
        data['q_75_20'] = data['Target'].rolling(20).apply(q_75, raw=True)
        data['q_95_20'] = data['Target'].rolling(20).apply(q_95, raw=True)
        data['mean_20'] = data['Target'].rolling(20).mean()
        data['std_20'] = data['Target'].rolling(20).std()
        data['max_20'] = data['Target'].rolling(20).max()
        data['min_20'] = data['Target'].rolling(20).min()
        data['q_norm_05_20'] = data[['mean_20', 'std_20']].apply(func=gauss_q_05, args=('mean_20', 'std_20'), axis=1)
        data['q_norm_25_20'] = data[['mean_20', 'std_20']].apply(func=gauss_q_25, args=('mean_20', 'std_20'), axis=1)
        data['q_norm_50_20'] = data[['mean_20', 'std_20']].apply(func=gauss_q_50, args=('mean_20', 'std_20'), axis=1)
        data['q_norm_75_20'] = data[['mean_20', 'std_20']].apply(func=gauss_q_75, args=('mean_20', 'std_20'), axis=1)
        data['q_norm_95_20'] = data[['mean_20', 'std_20']].apply(func=gauss_q_95, args=('mean_20', 'std_20'), axis=1)
        data['q_pct_05_20'] = data['Target_pct'].rolling(20).apply(q_05, raw=True)
        data['q_pct_25_20'] = data['Target_pct'].rolling(20).apply(q_25, raw=True)
        data['q_pct_50_20'] = data['Target_pct'].rolling(20).apply(q_50, raw=True)
        data['q_pct_75_20'] = data['Target_pct'].rolling(20).apply(q_75, raw=True)
        data['q_pct_95_20'] = data['Target_pct'].rolling(20).apply(q_95, raw=True)
        data['mean_pct_20'] = data['Target_pct'].rolling(20).mean()
        data['std_pct_20'] = data['Target_pct'].rolling(20).std()
        data['q_pct_norm_05_20'] = data[['mean_pct_20', 'std_pct_20']].apply(func=gauss_q_05, args=('mean_pct_20', 'std_pct_20'), axis=1)
        data['q_pct_norm_25_20'] = data[['mean_pct_20', 'std_pct_20']].apply(func=gauss_q_25, args=('mean_pct_20', 'std_pct_20'), axis=1)
        data['q_pct_norm_50_20'] = data[['mean_pct_20', 'std_pct_20']].apply(func=gauss_q_50, args=('mean_pct_20', 'std_pct_20'), axis=1)
        data['q_pct_norm_75_20'] = data[['mean_pct_20', 'std_pct_20']].apply(func=gauss_q_75, args=('mean_pct_20', 'std_pct_20'), axis=1)
        data['q_pct_norm_95_20'] = data[['mean_pct_20', 'std_pct_20']].apply(func=gauss_q_95, args=('mean_pct_20', 'std_pct_20'), axis=1)

        # horizon 20 base

        data['x00'] = data['Target'] / data['ewm_20']
        data['x01'] = data['Target_pct'].rolling(20).apply(skew, raw=True)
        data['x02'] = data['Target_pct'].rolling(20).apply(kurtosis, raw=True)
        data['x03'] = data[['q_pct_05_20', 'q_pct_norm_05_20', 'mean_pct_20']].apply(func=if_rated, args=('q_pct_05_20', 'q_pct_norm_05_20', 'mean_pct_20', -1000), axis=1)
        data['x04'] = data[['q_pct_95_20', 'q_pct_norm_95_20', 'mean_pct_20']].apply(func=if_rated,args=('q_pct_95_20', 'q_pct_norm_95_20', 'mean_pct_20', 1000), axis=1)
        data['x05'] = data[['q_05_20', 'q_norm_05_20', 'mean_20']].apply(func=if_rated, args=('q_05_20', 'q_norm_05_20', 'mean_20', -1000), axis=1)
        data['x06'] = data[['q_95_20', 'q_norm_95_20', 'mean_20']].apply(func=if_rated,args=('q_95_20', 'q_norm_95_20', 'mean_20', 1000), axis=1)
        data['x07'] = data['q_50_20'] / data['mean_20']
        data['x08'] = data['Target'] / data['q_05_20']
        data['x09'] = data['Target'] / data['q_95_20']
        data['x10'] = data['Target'] / data['q_50_20']
        data['x11'] = data['Target'] / data['mean_20']
        data['x12'] = data['std_20'] / data['mean_20']
        data['x13'] = (data['max_20'] - data['min_20']) / data['mean_20']
        data['x14'] = (data['q_75_20'] - data['q_25_20']) / data['mean_20']

        data['x01'] = data['x01'] - 1
        data['x03'] = data['x03'] - 1
        data['x04'] = data['x04'] - 1
        data['x05'] = data['x05'] - 1
        data['x06'] = data['x06'] - 1
        data['x07'] = data['x07'] - 1
        data['x08'] = data['x08'] - 1
        data['x09'] = data['x09'] - 1
        data['x10'] = data['x10'] - 1
        data['x11'] = data['x11'] - 1

        add_features += ['x{0:02d}'.format(j) for j in range(15)]

    x_features = x_features + add_features if full else add_features

    return data, x_features


def make_close_features(data, data_mash_4h, data_mash_daily):

    # enrich

    data_mash_4h = data_mash_4h[['H4_smma200', 'H4_ema200']].copy()
    data_mash_daily = data_mash_daily[['d1_ema200']].copy()

    data['y'] = data['date'].dt.year
    data['m'] = data['date'].dt.month
    data['d'] = data['date'].dt.day
    data['H'] = data['date'].dt.hour

    data_mash_4h['y'] = data_mash_4h['date'].dt.year
    data_mash_4h['m'] = data_mash_4h['date'].dt.month
    data_mash_4h['d'] = data_mash_4h['date'].dt.day
    data_mash_4h['H'] = data_mash_4h['date'].dt.hour

    data_mash_daily['y'] = data_mash_daily['date'].dt.year
    data_mash_daily['m'] = data_mash_daily['date'].dt.month
    data_mash_daily['d'] = data_mash_daily['date'].dt.day
    data_mash_daily['H'] = data_mash_daily['date'].dt.hour

    data = data.merge(right=data_mash_4h, how='left', left_on=['y', 'm', 'd', 'H'], right_on=['y', 'm', 'd', 'H'])
    data = data.merge(right=data_mash_daily, how='left', left_on=['y', 'm', 'd'], right_on=['y', 'm', 'd'])

    data[['H4_smma200', 'H4_ema200', 'd1_ema200']] = data[['H4_smma200', 'H4_ema200', 'd1_ema200']].fillna(method='ffill')

    # build rules

    def differentiator(x, names, thresh=0.04):
        if ((x[names[0]] / x[names[1]]) - 1) >= thresh:
            return +1
        elif ((x[names[0]] / x[names[1]]) - 1) >= -thresh:
            return 0
        else:
            return -1

    # build

    data['ft__mf_vs_lf'] = data[['H4_smma200', 'd1_ema200']].apply(func=differentiator, args=('H4_smma200', 'd1_ema200'), axis=1)
    data['ft__hf_vs_mf'] = data[['H4_ema200', 'H4_smma200']].apply(func=differentiator, args=('H4_ema200', 'H4_smma200'), axis=1)

    c_features = ['ft__mf_vs_lf', 'ft__hf_vs_mf']

    return data, c_features
