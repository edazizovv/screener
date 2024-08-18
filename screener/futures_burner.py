#
import os
import datetime


#
import pandas
import seaborn
from matplotlib import pyplot


#


#
d = './data/futures/semi/'
_hub = {}
for file in os.listdir(d):
    if '_' in file:
        sliced = pandas.read_csv(d + file)
        sliced['date'] = pandas.to_datetime(sliced['date'])
        sliced = sliced.set_index('date')

        name = file[:file.index('.csv')]
        sliced['contract'] = name

        _hub[name] = sliced

hub = {}
mx_dt = [_hub[key].index.max() for key in _hub.keys()]
for mx in sorted(mx_dt):
    ix = mx_dt.index(mx)
    key = list(_hub.keys())[ix]
    hub[key] = _hub[key]

end_cutoff_days = 3
data_quarter = []
for key in hub.keys():
    sliced = hub[key].copy()
    sliced = sliced[sliced.index < (sliced.index.max() - datetime.timedelta(days=end_cutoff_days))].copy()
    if len(data_quarter) > 0:
        sliced = sliced[sliced.index > data_quarter[-1].index.max()].copy()
    data_quarter.append(sliced)

data_quarter = pandas.concat(data_quarter, axis=0, ignore_index=False)

data_perpetual = pandas.read_csv(d + 'BTCUSDT.csv')
data_perpetual['date'] = pandas.to_datetime(data_perpetual['date'])
data_perpetual = data_perpetual.set_index('date')

start_dt = max([data_quarter.index.min(), data_perpetual.index.min()])
end_dt = min([data_quarter.index.max(), data_perpetual.index.max()])

data_quarter = data_quarter[(data_quarter.index >= start_dt) * (data_quarter.index <= end_dt)].copy()
data_perpetual = data_perpetual[(data_perpetual.index >= start_dt) * (data_perpetual.index <= end_dt)].copy()

pct = data_quarter.copy()
pct['Target'] = pct['Close'] / data_perpetual['Close']
pct = pct.rename(columns={'Close': 'Q'})

contracts = pct['contract'].value_counts().index
palette = seaborn.color_palette()

fig, ax = pyplot.subplots(1, 1)

for j in range(len(contracts)):
    c = contracts[j]
    p = palette[j]

    s = pct[pct['contract'] == c].copy()
    s.plot(ax=ax, color=p, legend=False)

pct.to_csv('./data/futures/pct.csv', index=True)
