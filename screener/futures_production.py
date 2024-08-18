#
import os
import json
import asyncio
import datetime


#
import pytz
import numpy
import pandas
import urllib3
from dotenv import load_dotenv
from telethon import TelegramClient, events
from sqlalchemy import create_engine, text
from apscheduler.schedulers.background import BlockingScheduler

#


#
utc = pytz.utc

load_dotenv()
APP_ID = int(os.getenv('APP_ID'))
APP_HASH = os.getenv('APP_HASH')
BOT_TOKEN = os.getenv('BOT_TOKEN')
DB_USER = os.getenv('DB_USER')
DB_PW = os.getenv('DB_PW')
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_STORE_TABLE = os.getenv('DB_STORE_TABLE')
DB_REPORT_TABLE = os.getenv('DB_REPORT_TABLE')

bot = TelegramClient(
    'bot',
    APP_ID,
    APP_HASH
).start(bot_token=BOT_TOKEN)

conn = create_engine("postgresql+psycopg2://{0}:{1}@{2}/{3}".format(
    DB_USER, DB_PW, DB_HOST, DB_NAME
)).connect()

service_signature = '@futures_test'

message_form = """
Signal spotted:

Target: {target}
Status: {direction}

Current quotes: {quote}
Current time: {time}

{signature}
"""


class Storage:

    def __init__(self):
        self.users = []
        self.subscribed_for = {}

    def new_user(self, user_id):
        self.users.append(user_id)
        self.subscribed_for[user_id] = []


s = Storage()


def get_actual_symbols():

    http = urllib3.PoolManager()
    r = http.request('GET',
                     'https://fapi.binance.com/fapi/v1/exchangeInfo'
                     )
    result = json.loads(r.data)
    cqf = [x for x in result['symbols'] if x['contractType'] == 'CURRENT_QUARTER']

    # for now filtering to keep only BTC
    cqf = [x for x in cqf if 'BTC' in x['symbol']]

    selected_tickers = []
    for j in range(len(cqf)):
        dt = utc.localize(datetime.datetime.fromtimestamp(cqf[0]['deliveryDate'] // 1000))
        diff = (dt - utc.localize(datetime.datetime.utcnow())).days
        if diff <= 14:
            print("Less than 2 weeks remain! Ticker {0} skipped".format(cqf[j]['symbol']))
        else:
            selected_tickers.append(cqf[j]['symbol'])

    return selected_tickers


# @sched.scheduled_job('interval', id='nato-bomber', hours=1)
def calculate_parameters():

    # quarter select

    selected_tickers = get_actual_symbols()

    # tz fix

    # query = "SET TIME ZONE 'UTC';"
    # conn.execute(text(query))

    # merge data

    symbol_perpetual = 'BTCUSDT'

    query = """
    SELECT *
    FROM {0}
    WHERE "date" > '{1}'
    AND ticker = '{2}'
    ;
    """.format(DB_STORE_TABLE, utc.localize((datetime.datetime.utcnow() - datetime.timedelta(days=2))).isoformat(), symbol_perpetual)
    data = pandas.read_sql(sql=text(query), con=conn)
    data['date'] = pandas.to_datetime(data['date'])
    data = data.rename(columns={'close': 'perpetual'})

    result = {}
    for symbol in selected_tickers:

        query = """
        SELECT *
        FROM {0}
        WHERE "date" > '{1}'
        AND ticker = '{2}'
        ;
        """.format(DB_STORE_TABLE, utc.localize((datetime.datetime.utcnow() - datetime.timedelta(days=2))).isoformat(), symbol)
        sliced = pandas.read_sql(sql=text(query), con=conn)
        sliced['date'] = pandas.to_datetime(sliced['date'])
        sliced = sliced.rename(columns={'close': 'quarter'})

        sliced = data.merge(right=sliced, left_on='date', right_on='date', how='left')

        # sliced = sliced.dropna(axis=1)
        sliced = sliced.sort_values(by='date', ascending=True)

        window = 30

        # print(symbol)
        # print(sliced.ffill().iloc[-window-1:-1, :])

        sliced = sliced.ffill().iloc[-window-1:-1, :].copy()

        if sliced['quarter'].isna().any() or sliced['perpetual'].isna().any():
            print("Missing values detected, skipping {0}".format(symbol))
        else:
            # calculate

            inter_m = 1.5
            sliced['rate'] = sliced['quarter'] / sliced['perpetual']

            #
            values = sliced['rate'].values

            # """
            q25 = numpy.quantile(a=values, q=0.25)
            q75 = numpy.quantile(a=values, q=0.75)
            bot_thresh = q25 - inter_m * (q75 - q25)
            top_thresh = q75 + inter_m * (q75 - q25)

            # print(bot_thresh)
            # print(top_thresh)

            result[symbol] = {'bot_thresh': bot_thresh, 'top_thresh': top_thresh}

    with open("./parameters.json", "w") as outfile:
        json.dump(result, outfile)


def update_data():

    interval = '1h'
    limit = 100

    # perpetual call

    symbol_perpetual = 'BTCUSDT'

    query = """
    SELECT *
    FROM {0}
    WHERE "date" > '{1}'
    AND ticker = '{2}'
    ;
    """.format(DB_STORE_TABLE, utc.localize((datetime.datetime.utcnow() - datetime.timedelta(days=2))).isoformat(), symbol_perpetual)
    data = pandas.read_sql(sql=text(query), con=conn)
    data['date'] = pandas.to_datetime(data['date'])
    max_date = data['date'].max()

    http = urllib3.PoolManager()
    r = http.request('GET',
                     'https://fapi.binance.com/fapi/v1/klines?symbol={0}&interval={1}&limit={2}'.format(
                         symbol_perpetual, interval, limit))
    result = json.loads(r.data)
    data_perpetual = pandas.DataFrame(result)
    data_perpetual = data_perpetual.iloc[:, [0, 4]]
    data_perpetual = pandas.DataFrame(data=data_perpetual.values, columns=['date', 'close'])
    data_perpetual['date'] = data_perpetual['date'].apply(func=lambda x: utc.localize(datetime.datetime.fromtimestamp(x / 1000)))
    data_perpetual['ticker'] = symbol_perpetual

    if not pandas.isna(max_date):
        data_perpetual = data_perpetual[data_perpetual['date'] > max_date].copy()

    # print(data_perpetual)
    data_perpetual.to_sql(name=DB_STORE_TABLE, con=conn, index=False, if_exists='append')
    conn.commit()

    # quarter select

    selected_tickers = get_actual_symbols()

    # quarter call

    for symbol in selected_tickers:

        query = """
        SELECT *
        FROM {0}
        WHERE "date" > '{1}'
        AND ticker = '{2}'
        ;
        """.format(DB_STORE_TABLE, utc.localize((datetime.datetime.utcnow() - datetime.timedelta(days=2))).isoformat(), symbol)
        sliced = pandas.read_sql(sql=text(query), con=conn)
        sliced['date'] = pandas.to_datetime(sliced['date'])
        max_date = sliced['date'].max()

        http = urllib3.PoolManager()
        r = http.request('GET',
                         'https://fapi.binance.com/fapi/v1/klines?symbol={0}&interval={1}&limit={2}'.format(
                             symbol, interval, limit))
        result = json.loads(r.data)
        data_quarter = pandas.DataFrame(result)
        data_quarter = data_quarter.iloc[:, [0, 4]]
        data_quarter = pandas.DataFrame(data=data_quarter.values, columns=['date', 'close'])
        data_quarter['date'] = data_quarter['date'].apply(func=lambda x: utc.localize(datetime.datetime.fromtimestamp(x / 1000)))
        data_quarter['ticker'] = symbol

        if not pandas.isna(max_date):
            data_quarter = data_quarter[data_quarter['date'] > max_date].copy()

        # print(symbol)
        # print(data_quarter)
        data_quarter.to_sql(name=DB_STORE_TABLE, con=conn, index=False, if_exists='append')
        conn.commit()


def check_signal():

    with open("./parameters.json", "r") as outfile:
        parameters = json.load(outfile)

    # interval = '5m'
    interval = '1h'
    limit = 1

    # perpetual call

    symbol_perpetual = 'BTCUSDT'

    http = urllib3.PoolManager()
    r = http.request('GET',
                     'https://fapi.binance.com/fapi/v1/klines?symbol={0}&interval={1}&limit={2}'.format(
                         symbol_perpetual, interval, limit))
    result = json.loads(r.data)
    data_perpetual = pandas.DataFrame(result)
    data_perpetual = data_perpetual.iloc[:, [0, 4]]
    data_perpetual = pandas.DataFrame(data=data_perpetual.values, columns=['date', 'close'])
    data_perpetual['date'] = data_perpetual['date'].apply(func=lambda x: utc.localize(datetime.datetime.fromtimestamp(x / 1000)))
    data_perpetual['ticker'] = symbol_perpetual
    # print(data_perpetual)

    date_reported_perpetual = data_perpetual['date'].values[0]
    quote_perpetual = float(data_perpetual.iloc[0, 1])

    # quarter select

    selected_tickers = get_actual_symbols()

    # quarter call

    report = {}
    quoted = {}

    for symbol in selected_tickers:

        http = urllib3.PoolManager()
        r = http.request('GET',
                         'https://fapi.binance.com/fapi/v1/klines?symbol={0}&interval={1}&limit={2}'.format(
                             symbol, interval, limit))
        result = json.loads(r.data)
        data_quarter = pandas.DataFrame(result)
        data_quarter = data_quarter.iloc[:, [0, 4]]
        data_quarter = pandas.DataFrame(data=data_quarter.values, columns=['date', 'close'])
        data_quarter['date'] = data_quarter['date'].apply(func=lambda x: utc.localize(datetime.datetime.fromtimestamp(x / 1000)))
        data_quarter['ticker'] = symbol

        date_reported_quarter = data_quarter['date'].values[0]
        quote_quarter = float(data_quarter.iloc[0, 1])

        if date_reported_perpetual != date_reported_quarter:
            print("DATE MISMATCH: {0} to {1}".format(symbol, symbol_perpetual))
            print(data_perpetual)
            print(data_quarter)

        else:

            report_date = date_reported_perpetual
            rate = quote_quarter / quote_perpetual

            if rate < parameters[symbol]['bot_thresh']:
                relation = 'BELOW'
                report[symbol] = {'relation': relation, 'quotes': '{0} quarter / {1} perpetual'.format(quote_quarter, quote_perpetual)}
                quoted[symbol] = 'quarter: {0:.4f}; perpetual: {1:.4f}'.format(quote_quarter, quote_perpetual)  # {'quarter': quote_quarter, 'perpetual': quote_perpetual}
            elif rate > parameters[symbol]['top_thresh']:
                relation = 'ABOVE'
                report[symbol] = {'relation': relation, 'quotes': '{0} quarter / {1} perpetual'.format(quote_quarter, quote_perpetual)}
                quoted[symbol] = 'quarter: {0:.4f}; perpetual: {1:.4f}'.format(quote_quarter, quote_perpetual)  # {'quarter': quote_quarter, 'perpetual': quote_perpetual}
            else:
                relation = 'NO'

            reported = pandas.DataFrame(data={'timestamp': [report_date], 'rate': [rate],
                                              'symbol_perpetual': [symbol_perpetual], 'symbol_quarter': [symbol],
                                              'quote_perpetual': [quote_perpetual], 'quote_quarter': [quote_quarter],
                                              'rate_bot_thresh': [parameters[symbol]['bot_thresh']],
                                              'rate_top_thresh': [parameters[symbol]['top_thresh']],
                                              'relation': relation})
            reported.to_sql(name=DB_REPORT_TABLE, con=conn, index=False, if_exists='append')
            conn.commit()

    return report, quoted


async def once():

    update_data()
    print('updated')
    calculate_parameters()
    print('calculated')
    results, quoted = check_signal()
    print('checked')
    for key in results.keys():

        for user in s.users:
            print('USER: {0}'.format(user))

            if key in s.subscribed_for[user]:

                direction = results[key]
                message = message_form.format(target=key,
                                              direction=direction,
                                              quote=quoted[key],
                                              time=utc.localize(datetime.datetime.utcnow()).isoformat(),
                                              signature=service_signature)
                await bot.send_message(user,
                                       message,
                                       )
    print('sent')
    print('===================================')


@bot.on(events.NewMessage(pattern="/start"))
async def start(event):

    print("Hi mate")

    user_id = event.sender_id

    s.new_user(user_id=user_id)


@bot.on(events.NewMessage(pattern="/sub"))
async def sub(event):

    sub_list = event.text
    sub_list = sub_list.split(' ')
    sub_list = sub_list[1:]

    user_id = event.sender_id
    s.subscribed_for[user_id] = [x for x in s.subscribed_for[user_id] if x not in sub_list] + sub_list


@bot.on(events.NewMessage(pattern="/unsub"))
async def unsub(event):

    user_id = event.sender_id
    s.subscribed_for[user_id] = []


@bot.on(events.NewMessage(pattern="/show"))
async def show(event):

    user_id = event.sender_id

    message = str(s.subscribed_for[user_id])

    await bot.send_message(user_id,
                           message,
                           )


@bot.on(events.NewMessage(pattern="/list"))
async def list_(event):

    user_id = event.sender_id

    message = str(get_actual_symbols())

    await bot.send_message(user_id,
                           message,
                           )


"""
async def check_nein():
    while True:
        print(datetime.datetime.now())
        # await asyncio.sleep(10)
        now = datetime.datetime.now()
        target = now + datetime.timedelta(hours=1)
        bot.loop.run_until_complete(run_at(datetime.datetime(target.year, target.month, target.day, target.hour, 5, 0),
                                           once()))
"""

async def wait_until(dt):
    # sleep until the specified datetime
    now = datetime.datetime.now()
    await asyncio.sleep((dt - now).total_seconds())


async def run_at(dt, coro):
    await wait_until(dt)
    return await coro


"""
import time
async def runner():
    while True:
        print(datetime.datetime.now())
        await asyncio.sleep(10)
        now = datetime.datetime.now()
        target = now + datetime.timedelta(minutes=1)
        bot.loop.create_task(run_at(datetime.datetime(target.year, target.month, target.day, target.hour, target.minute, 0),
                             once()))
        # bot.loop.create_task(once())


def main():
    # asyncio.run(once())
    bot.loop.run_until_complete(runner())
    # asyncio.run(once())
"""


async def main():
    while True:
        print(datetime.datetime.now())
        # await asyncio.sleep(10)
        now = datetime.datetime.now()
        target = now + datetime.timedelta(hours=1)
        # target = now + datetime.timedelta(minutes=10)
        '''
        bot.loop.run_until_complete(run_at(datetime.datetime(target.year, target.month, target.day, target.hour, 5, 0),
                                    once()))
        '''
        await run_at(datetime.datetime(target.year, target.month, target.day, target.hour, 5, 0),
                                    once())
        # bot.loop.create_task(once())

# sched.start()

# bot.loop.subprocess_exec()
# bot.loop.create_task()

def maim():
    print('hi there')
    bot.loop.create_task(main())
    print('task created')
    bot.run_until_disconnected()
    print('wooohoo')


if __name__ == '__main__':
    # main()
    maim()

# main()

# sched.start()

# update_data()
# calculate_parameters()
# result = check_signal()
