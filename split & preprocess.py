import pandas as pd
import numpy as np
import gc
import urllib
import requests
import time
import datetime
import heapq
import math
import sklearn
import lightgbm
from sklearn import metrics
import os
from multiprocessing import Process
import multiprocessing
from sklearn.externals import joblib

# 去除来源为weibo、抽屉的新闻
def wash(tdata):
    for i in range(0, 4):
        tdata['url_part_'+str(i)] = tdata['url'].apply(lambda x: str(str(x).split('.')[i]) if len(str(x).split('.')) > i else '')
    tdata = tdata.loc[(tdata.url_part_1 != 'weibo') & (tdata.url_part_2 != 'weibo') & (tdata.url_part_0 != 'https://weibo')]
    tdata = tdata.loc[(tdata.url_part_1 != 'chouti') & (tdata.url_part_1 != '')]
    del (tdata['url_part_0'])
    del (tdata['url_part_1'])
    del (tdata['url_part_2'])
    del (tdata['url_part_3'])
    del tdata['category']
    gc.collect()
    print('wash done')
    return tdata


# timestamp转datetime
def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value/1000)
    dt = time.strftime(format, value)
    return dt


# 互动时间和新闻发布时间时间差
def time_delta(row, t1, t2):
    t1 = row[t1]
    t2 = row[t2]
    t1 = datetime.datetime.strptime(t1, "%Y-%m-%d %H:%M:%S")
    t2 = datetime.datetime.strptime(t2, "%Y-%m-%d %H:%M:%S")
    diff_hour = (t1 - t2).seconds
    diff_hour = diff_hour/3600
    return diff_hour


# 重命名列，时间预处理: timestamp 转 datetime，取出日期、小时
def time_shift(tdata):
    tdata.rename(columns={'refresh-time': 'refresh_timestamp', 'publish-time': 'publish_timestamp', 'is-click': 'is_click'}, inplace=True)
    tdata['refresh_time'] = tdata['refresh_timestamp'].apply(timestamp_datetime)
    tdata['publish_time'] = tdata['publish_timestamp'].apply(timestamp_datetime)

    tdata['diff_hour'] = tdata.apply(time_delta, axis=1, t1='refresh_time', t2='publish_time')

    tdata['refresh_time'] = pd.to_datetime(tdata['refresh_time'])
    tdata['publish_time'] = pd.to_datetime(tdata['publish_time'])
    tdata['refresh_date'] = tdata['refresh_time'].dt.date
    tdata['refresh_day'] = tdata['refresh_time'].dt.day
    tdata['refresh_hour'] = tdata['refresh_time'].dt.hour

    print('time shift done')
    return tdata


def subsample(data, fraction):
    data_label_1 = data.loc[data.is_click == 1].sample(frac=fraction)
    data_label_0 = data.loc[data.is_click == 0].sample(frac=fraction)
    data = pd.concat([data_label_0, data_label_1])
    data = data.sample(frac=1)
    return data


def split(startdate, enddate):
    data = pd.read_csv(dir + '/history.csv', index_col=False)
    if startdate < enddate:
        data = data.loc[(data.refresh_day >= startdate) & (data.refresh_day <= enddate)]
        data.to_csv(dir + '/data_for_train.csv')
    else:
        L = list(range(startdate, enddate+32))
        for i in range(len(L)):
            if L[i] > 31:
                L[i] -= 31
            if i == 0:
                tdata = data.loc[data.refresh_day == L[i]]
            else:
                tdata = pd.concat([tdata, data.loc[data.refresh_day == L[i]]])
        tdata.to_csv(dir + '/data_for_train.csv', index=False)


def preprocess():
    batch = 0
    if not os.path.exists(dir + '/history.csv'):
        for df in pd.read_csv('./data/raw/data.csv', index_col=False, chunksize=100000):
            batch += 1
            df = wash(df)
            df = time_shift(df)
            df['is_click'] = df['is_click'].apply(lambda x: 1 if x == 'Y' else 0)
            gc.collect()
            print('writing chunk %d...' % batch)
            if batch == 1:
                df.to_csv(dir + '/history.csv', index=False)
            else:
                df.to_csv(dir + '/history.csv', index=False, header=False, mode='a')
    print('preprocess done')


if __name__ == '__main__':
    year = datetime.datetime.now().year
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day
    dir = './version_'+str(year)+'-'+str(month)+'-'+str(day)
    if not os.path.exists(dir):
        os.makedirs(dir)
    preprocess()
    split(30, 3)