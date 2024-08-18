#
import os
import zipfile
import datetime


#
import pandas


#


#

d = './data/futures/raw/'
hub = {}
for file in os.listdir(d):

    with zipfile.ZipFile(d + file, 'r') as zip_ref:
        zip_ref.extractall(d)

    name = file[:file.index('.zip')]
    sliced = pandas.read_csv(d + name + '.csv',
                             header=None,
                             names=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                    'Quote asset volume', 'Number of trades',
                                    'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    os.remove(d + name + '.csv')

    if sliced.values[0, 0] == 'open_time':
        print(file)
        sliced = sliced.iloc[1:, :].copy()

    sliced = sliced[['Close time', 'Close']].copy()
    sliced = sliced.rename(columns={'Close time': 'date'})
    sliced['date'] = sliced['date'].astype(dtype='int64')
    sliced['date'] = sliced['date'].apply(func=lambda x: datetime.datetime.fromtimestamp(x / 1000))
    sliced = sliced.set_index('date')
    sliced['Close'] = sliced['Close'].astype(dtype=float)

    key = file[:file.index('-1h')]
    if key not in hub.keys():
        hub[key] = [sliced]
    else:
        hub[key].append(sliced)


g = './data/futures/semi/'
for key in hub.keys():
    hub[key] = pandas.concat(hub[key], axis=0, ignore_index=False)
    hub[key].to_csv(g + key + '.csv', index=True)
