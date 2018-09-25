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
    # tdata['refresh_timestamp'] = int(round(time.time() * 1000))
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


# 获取新闻w2v vector
def request_w2v_vector(tdata):
    url = list(tdata['url'])
    for i in range(len(url)):
        url[i] = urllib.parse.quote(url[i])
    para = {'urlStr': url}
    code = -1
    i = 0
    while (code != 200) & (i <= 4):
        r = requests.post('http://ai.chouti.com/news/feature', data=para)
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
    return tdata


# 获取新闻category
def request_category(tdata):
    url = list(tdata['url'])
    for i in range(len(url)):
        url[i] = urllib.parse.quote(url[i])
    para = {'urlStr': url}
    code = -1
    i = 0
    while (code != 200) & (i <= 4):
        r = requests.post('http://ai.chouti.com/news/category', data=para)
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
    return tdata


# 获取新闻和用户兴趣点的相似度list
def request_correlation(tdata):
    tdata['post'] = tdata.apply(lambda row: str(row['device_id']) + ',' + str(urllib.parse.quote(row['url'])), axis=1)
    url = '|'.join(list(tdata['post']))
    para = {'str': url}
    code = -1
    i = 0
    while (code != 200) & (i <= 4):
        r = requests.post('http://ai.chouti.com/news/recommend/user_url_similar', data=para)
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
    return tdata


def get_click_features(row, history):
    history = history.loc[history.url == row.url]
    history = history.loc[(history.refresh_timestamp < int(row['refresh_timestamp'])) & (history.refresh_timestamp >= row['refresh_timestamp'] - 3600000*25)]
    for i in [1, 3, 6, 12, 24]:
        tdata = history.loc[history.refresh_timestamp >= row['refresh_timestamp'] - 3600000*int(i)]
        row['click_count_in_'+str(i)] = len(tdata)
        row['click_ratio_in_'+str(i)] = len(tdata.loc[tdata.is_click == 1])/(len(tdata)+0.00001)
        # rank
    # print('click features done')
    return row


def top_k_corr(row):
    if row['corr'] != []:
        temp = heapq.nlargest(5, row['corr'])
        temp = pd.Series(temp)
        temp_index = []
        for j in range(len(temp)):
            temp_index.append('cosine_with_top'+str(j+1))
        temp.index = temp_index
        temp['top_5_avg_cosine'] = temp.mean()
        row = pd.concat([row, temp])
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
    ############################################
    del df['category']
    df = time_shift(df)
    print('preprocess done')
    return df


def get_category_features(df):
    # category features
    t_history = history.loc[history.device_id.isin(df['device_id'])]
    result = pd.DataFrame()
    L = list(df['refresh_date'].unique())
    for i in range(len(L)):
        tt_history = t_history.loc[
            (t_history.refresh_date < str(L[i])) & (t_history.refresh_date >= str(L[i] - datetime.timedelta(days=7)))]
        temp = tt_history.groupby(['device_id'])[['cat_art', 'cat_car', 'cat_edu', 'cat_ent', 'cat_finance', 'cat_game',
                                                  'cat_gj', 'cat_other', 'cat_party', 'cat_sh', 'cat_sport', 'cat_tech',
                                                  'cat_war', 'cat_weather']].mean().reset_index()
        temp['refresh_date'] = L[i]

        if i == 0:
            result = temp
        else:
            result = pd.concat([result, temp])
        print('%s category features done' % L[i])
    df = pd.merge(df, result, on=['device_id', 'refresh_date'], how='left')

    df = request_category(df)
    df['category'] = df['category'].fillna('other')
    df['news_cat_ratio'] = df.apply(get_cat_ratio, axis=1)
    return df


def make_features(df, history):
    df = preprocess(df)
    ############################################
    # df = df.apply(get_click_features, axis=1, history=history)
    df = request_correlation(df)
    df['avg_cosine_center'] = df['corr'].apply(lambda x: sum(x) / len(x) if x != [] else -100)
    df = df.apply(top_k_corr, axis=1)
    del df['corr']

    df = get_category_features(df)
    df = request_w2v_vector(df)

    return df


def read_data():  #########################
    #category = pd.read_csv('./data/news_category.csv', index_col=False)
    category2 = pd.read_csv('./data/news_category2.csv', index_col=False)
    category3 = pd.read_csv('./data/news_category3.csv', index_col=False)
    category = pd.concat([category2, category3])
    category = category.drop_duplicates(['url']).reset_index(drop=True)
    category = category.fillna('other')
    del (category['link_id'])
    del category2, category3
    gc.collect()

    # 取出历史点击情况
    #data = pd.read_csv('./data/raw_data.csv', index_col=False)
    data2 = pd.read_csv('./data/raw_data2.csv', index_col=False)
    data3 = pd.read_csv('./data/raw_data3.csv', index_col=False)

    #data = data.loc[data.is_click == 1]
    data2 = data2.loc[data2.is_click == 1]
    data3 = data3.loc[data3.is_click == 1]

    #del (data['category'])
    del (data2['category'])
    del (data3['category'])
    data = pd.concat([data2, data3])
    data = data.drop_duplicates().reset_index(drop=True)
    data = pd.merge(data, category, how='left', on=['url'])
    data = data[['device_id', 'refresh_timestamp', 'refresh_date', 'refresh_day', 'url', 'category', 'is_click']]
    data = pd.concat([data, pd.get_dummies(data['category'], prefix='cat')], axis=1)
    return data


server = flask.Flask(__name__) # 创建一个flask对象


@server.route('/predict', methods=['post'])
def predict():

    ###############################
    id = request.form.get('device_id')
    data = request.form.get('news_data')
    data = pd.DataFrame(json.loads(data))
    data['device_id'] = id
    print(data.columns)
    data = make_features(data, history)

    used_features = [c for c in data if
                     c not in ['device_id', 'link_id', 'is_click', 'url', 'category', 'publish_time', 'publish_timestamp',
                               'refresh_day', 'refresh_time', 'refresh_date', 'refresh_hour', 'refresh_timestamp', 'category',
                               # 'cosine_with_top1', 'cosine_with_top2', 'cosine_with_top3', 'cosine_with_top4',
                               # 'cosine_with_top5',
                               'euclidean_with_top1', 'euclidean_with_top2', 'euclidean_with_top3', 'euclidean_with_top4',
                               'euclidean_with_top5',
                               'manhattan_with_top1', 'manhattan_with_top2', 'manhattan_with_top3', 'manhattan_with_top4',
                               'manhattan_with_top5',
                               'top_5_avg_euclidean', 'top_5_avg_manhattan',  # 'top_5_avg_cosine',
                               'avg_euc_center', 'avg_man_center',  # 'avg_cosine_center',
                               'click_ratio_in_6', 'click_ratio_in_3', 'click_ratio_in_12',
                               # 'click_ratio_in_1','click_ratio_in_24',
                               'click_count_in_1', 'click_count_in_3', 'click_count_in_6', 'click_count_in_12',
                               'click_count_in_24',

                               ]]
    model = joblib.load('./lightgbm_model.m')
    data['predict'] = model.predict_proba(data[used_features])[:, 1]
    data['result'] = data[['link_id', 'predict']].apply(to_dict, axis=1)
    return json.dumps(list(data['result']))
    # result = 'hello'
    # return result


history = read_data() ################################
server.run(port=8000, debug=True)
