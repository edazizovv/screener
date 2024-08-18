#
import json
import datetime

#
import urllib
import urllib3
import numpy
import pandas


#
"""
def bb(symbol, n=100):
    http = urllib3.PoolManager()
    r = http.request('GET',
                     'https://api.binance.com/fapi/v1/exchangeInfo'
                     )
    result = json.loads(r.data)
    result = result['result']
    data_bid = pandas.DataFrame(data={
        'bid_price': [x[0] for x in result['bids']], 'bid_amount': [x[1] for x in result['bids']],
        }).astype(dtype=float)
    data_ask = pandas.DataFrame(data={
        'ask_price': [x[0] for x in result['asks']], 'ask_amount': [x[1] for x in result['asks']],
        }).astype(dtype=float)
    return data_bid, data_ask
"""


# step 1
# get all current quarter futures
# """
http = urllib3.PoolManager()
r = http.request('GET',
                 'https://fapi.binance.com/fapi/v1/exchangeInfo'
                 )
result = json.loads(r.data)
cqf = [x for x in result['symbols'] if x['contractType'] == 'CURRENT_QUARTER']

selected_tickers = []
for j in range(len(cqf)):
    dt = datetime.datetime.fromtimestamp(cqf[0]['deliveryDate'] // 1000)
    diff = (dt - datetime.datetime.now()).days
    if diff <= 14:
        print("Less than 2 weeks remain! Ticker {0} skipped".format(cqf[j]['symbol']))
    else:
        selected_tickers.append(cqf[j]['symbol'])
# """

# step 2
# get candles
"""
horizon_days = 50

# 1m / 5m / 1h / 1d / 1w / 1M

# symbol = 'ETHUSDT_230331'
symbol = 'BTCUSDT_230331'
interval = '1h'
start_time = str(int((datetime.datetime.now() - datetime.timedelta(days=horizon_days)).timestamp() * 1000))
end_time = str(int(datetime.datetime.now().timestamp() * 1000))
limit = 1500
http = urllib3.PoolManager()
r = http.request('GET',
                 'https://fapi.binance.com/fapi/v1/klines?symbol={0}&interval={1}&startTime={2}&endTime={3}&limit={4}'.format(
                     symbol, interval, start_time, end_time, limit
                 )
                 )
result = json.loads(r.data_quarter)
data_qf = pandas.DataFrame(data=result, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                                'Quote asset', 'Number of trades', 'Taker buy base asset volume',
                                                'Taker buy quote asset volume', 'Ignore'])
data_qf['Open time'] = data_qf['Open time'].apply(func=lambda x: datetime.datetime.fromtimestamp(x / 1000))
data_qf['Close time'] = data_qf['Close time'].apply(func=lambda x: datetime.datetime.fromtimestamp(x / 1000))


# symbol = 'ETHUSDT'
symbol = 'BTCUSDT'
interval = '1h'
start_time = str(int((datetime.datetime.now() - datetime.timedelta(days=horizon_days)).timestamp() * 1000))
end_time = str(int(datetime.datetime.now().timestamp() * 1000))
limit = 1500
http = urllib3.PoolManager()
r = http.request('GET',
                 'https://fapi.binance.com/fapi/v1/klines?symbol={0}&interval={1}&startTime={2}&endTime={3}&limit={4}'.format(
                     symbol, interval, start_time, end_time, limit
                 )
                 )
result = json.loads(r.data_quarter)
data_pt = pandas.DataFrame(data=result, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                                'Quote asset', 'Number of trades', 'Taker buy base asset volume',
                                                'Taker buy quote asset volume', 'Ignore'])
data_pt['Open time'] = data_pt['Open time'].apply(func=lambda x: datetime.datetime.fromtimestamp(x / 1000))
data_pt['Close time'] = data_pt['Close time'].apply(func=lambda x: datetime.datetime.fromtimestamp(x / 1000))

ein = data_pt[['Close time', 'Close']]
ein = ein.set_index('Close time')

zwei = data_qf[['Close time', 'Close']]
zwei = zwei.set_index('Close time')

ein['Close'] = ein['Close'].astype(dtype=float)
zwei['Close'] = zwei['Close'].astype(dtype=float)

dry = zwei / ein - 1
dry = dry.dropna()
z = dry[dry.index >= '2023-01-01'].copy()


def q25(x):
    return numpy.quantile(a=x, q=0.25)


def q75(x):
    return numpy.quantile(a=x, q=0.75)


# TODO: просто меняем агрегационный интервал в день на интервал в неделю
z.index = pandas.to_datetime(z.index)
z['D'] = z.index.day
zz = z.groupby(by='D')['Close'].agg(func=[q25, q75])
zz['low_bound'] = zz['q25'] - 1.5 * (zz['q75'] - zz['q25'])
zz['up_bound'] = zz['q75'] + 1.5 * (zz['q75'] - zz['q25'])
zz = zz[['low_bound', 'up_bound']].copy()

z['DOWN'] = z.apply(func=lambda x: x['Close'] < zz.to_dict()['low_bound'][x['D']], axis=1)
z['UP'] = z.apply(func=lambda x: x['Close'] > zz.to_dict()['up_bound'][x['D']], axis=1)
"""