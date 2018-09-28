# coding: utf-8
import pandas as pd
import numpy as np
import gc
import time
import datetime
import heapq
from sklearn.externals import joblib
from flask import request
import flask
import urllib
import requests
import json
from flask_apscheduler import APScheduler
import os


def convert(row, r_dict):
    key = int(row['link_id'])
    r_dict[key] = row['predict']
    return r_dict


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
    # tdata['refresh_timestamp'] = int(round(time.time() * 1000))
    tdata['refresh_time'] = tdata['refresh_timestamp'].apply(timestamp_datetime)
    tdata['publish_time'] = tdata['publish_timestamp'].apply(timestamp_datetime)

    tdata['diff_hour'] = tdata.apply(time_delta, axis=1, t1='refresh_time', t2='publish_time')

    # tdata['refresh_time'] = pd.to_datetime(tdata['refresh_time'])
    # tdata['publish_time'] = pd.to_datetime(tdata['publish_time'])
    # tdata['refresh_date'] = tdata['refresh_time'].dt.date
    # tdata['refresh_day'] = tdata['refresh_time'].dt.day
    # tdata['refresh_hour'] = tdata['refresh_time'].dt.hour

    # print('time shift done')
    return tdata


# 获取新闻w2v vector
def request_w2v_vector(tdata):
    headers = {'Server': 'Tengine',
               'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36 Edge/15.15063'}
    url = list(tdata['url'])
    for i in range(len(url)):
        url[i] = urllib.pathname2url(url[i])
    para = {'urlStr': url}
    code = -1
    i = 0
    while (code != 200) & (i <= 4):
        r = requests.post('http://ai.chouti.com/news/feature', data=para, headers=headers)
        code = r.status_code
        i += 1
        if i > 1:
            print('timeout retrying...')
    wv = r.json()['data']
    filling_vector = [-100 for x in range(0, 200)]
    for i in range(len(wv)):
        if wv[i] is None:
            wv[i] = filling_vector
    wv = pd.DataFrame(wv)
    pca = joblib.load('./pca_model.m')
    wv = pca.transform(wv)
    print('vector get')
    # wv = list_to_frame(wv)
    tdata = tdata.reset_index(drop=True)
    tdata = pd.concat([tdata, wv], axis=1)
    r.close()
    time.sleep(1)
    return tdata


# 获取新闻category
def request_category(tdata):
    headers = {'Server': 'Tengine',
               'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36 Edge/15.15063'}
    url = list(tdata['url'])
    for i in range(len(url)):
        url[i] = urllib.pathname2url(url[i])
    para = {'urlStr': url}
    code = -1
    i = 0
    while (code != 200) & (i <= 4):
        r = requests.post('http://ai.chouti.com/news/category', data=para, headers=headers)
        code = r.status_code
        i += 1
        if i > 1:
            print('timeout retrying...')
    category = pd.Series(r.json()).reset_index()
    category.columns = ['url', 'category']
    tdata = tdata.reset_index(drop=True)
    tdata = pd.merge(tdata, category, how='left', on=['url'])
    print('category get')
    r.close()
    time.sleep(1)
    return tdata


# 获取新闻和用户兴趣点的相似度list
def request_correlation(tdata):
    headers = {'Server': 'Tengine',
               'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36 Edge/15.15063'}
    tdata['post'] = tdata.apply(lambda row: str(row['device_id']) + ',' + str(urllib.pathname2url(row['url'])), axis=1)
    url = '|'.join(list(tdata['post']))
    para = {'str': url}
    code = -1
    i = 0
    while (code != 200) & (i <= 4):
        r = requests.post('http://ai.chouti.com/news/recommend/user_url_similar', data=para, headers=headers)
        code = r.status_code
        i += 1
        if i > 1:
            print('timeout retrying...')
    corr = pd.Series(r.json()['data'])
    print('correlation get')
    tdata = tdata.reset_index(drop=True)
    tdata = pd.concat([tdata, corr], axis=1)
    tdata = tdata.rename(columns={0: 'corr'})
    del tdata['post']
    r.close()
    time.sleep(1)
    return tdata


# def get_click_features(row, history):
#     history = history.loc[history.url == row.url]
#     history = history.loc[(history.refresh_timestamp < int(row['refresh_timestamp'])) & (history.refresh_timestamp >= row['refresh_timestamp'] - 3600000*25)]
#     for i in [1, 3, 6, 12, 24]:
#         tdata = history.loc[history.refresh_timestamp >= row['refresh_timestamp'] - 3600000*int(i)]
#         row['click_count_in_'+str(i)] = len(tdata)
#         row['click_ratio_in_'+str(i)] = len(tdata.loc[tdata.is_click == 1])/(len(tdata)+0.00001)
#         # rank
#     # print('click features done')
#     return row


def top_k_corr(row):
    if row['corr'] != []:
        temp = heapq.nlargest(5, row['corr'])
        temp = pd.Series(temp)
        temp_index = []
        for j in range(len(temp)):
            temp_index.append('cosine_with_top'+str(j+1))
        temp.index = temp_index
        temp['cosine_top_5_avg'] = temp.mean()
        row = pd.concat([row, temp])
        row = row.fillna(-100)
        # print('corr get')
        return row
    else:
        row['corr'] = -100
        return row


def get_cat_ratio(row):
    x = row['cat_'+str(row['category'])]
    return x


def to_dict(row):
    return row.to_dict()

####################################


def preprocess(df):
    # df = wash(df)
    # del df['category']
    df = time_shift(df)
    print('preprocess done')
    return df


def make_features(df, features):
    # 预处理
    df = preprocess(df)
    # 兴趣点和新闻相关性特征
    # df = request_correlation(df)
    df['cosine_all_avg'] = df['corr'].apply(lambda x: sum(x) / len(x) if x != [] else -100)
    df = df.apply(top_k_corr, axis=1)
    del df['corr']
    # 用户历史浏览新闻的类别分布特征
    df = pd.merge(df, features, on=['device_id'], how='left')
    # df = request_category(df)
    # df['category'] = df['category'].fillna('other')
    # df['news_cat_ratio'] = df.apply(get_cat_ratio, axis=1)
    # # 新闻的word2vec向量特征
    # df = request_w2v_vector(df)

    return df


def read_data():  #########################
    global cat_features, gbm, pca
    cat_features = pd.read_csv('./data/category_features_online.csv', index_col=False)
    gbm = joblib.load('./lightgbm_model.m')
    pca = joblib.load('./pca_model.m')
    print('category features updated at ' + str(datetime.datetime.now()))
    # print(cat_features.head(1))


class Config(object):
    # Scheduler config
    JOBS = [
        {
            'id': 'job1',
            'func': '__main__:read_data',
            'args': None,
            # 'trigger': 'interval',
            # 'seconds': 20,
            'trigger': 'cron',
            'hour': 4,
            'minute': 00,
        }
            ]


server = flask.Flask(__name__) # 创建一个flask对象


@server.route('/predict', methods=['post'])
def predict():
    device_id = request.form.get('device_id')
    data = request.form.get('news_data')
    data = pd.DataFrame(json.loads(data))
    data.rename(columns={'link': 'link_id', 'p_time': 'publish_timestamp', 'click_24': 'click_ratio_in_24', 'cat': 'category'}, inplace=True)
    data['device_id'] = device_id
    data['refresh_timestamp'] = int(round(time.time() * 1000))

    data = make_features(data, cat_features)
    # data['vector'] = data['vector'].apply(lambda x: eval(x))
    # print(data.columns)
    vector = pd.DataFrame(list(data['vector']))
    vector = pca.transform(vector)
    data = pd.concat([data, pd.DataFrame(vector)], axis=1)
    del data['vector']

    used_features = [c for c in data if
                     c not in ['device_id', 'link_id', 'is_click', 'category', 'publish_time', 'publish_timestamp', 'refresh_date',
                               'refresh_day',  'refresh_time', 'refresh_hour', 'refresh_timestamp', 'category', 'click_ratio_in_1',
                     ]]
    data['predict'] = gbm.predict_proba(data[used_features])[:, 1]
    r_dict = {}
    result = data.apply(convert, r_dict=r_dict, axis=1)
    return json.dumps(result.iloc[0])


if __name__ == '__main__':
    cat_features = pd.read_csv('./data/category_features_online.csv', index_col=False)
    gbm = joblib.load('./lightgbm_model.m')
    pca = joblib.load('./pca_model.m')
    print('category features loaded')
    scheduler = APScheduler()
    server.config.from_object(Config())
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        scheduler.init_app(server)
    # trigger schduler
    scheduler.start()
    server.run(host='0.0.0.0', port=8000, debug=True)
