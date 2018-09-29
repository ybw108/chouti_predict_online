# coding: utf-8
import schedule
import time
import datetime
import pandas as pd
import numpy as np
import os
import gc
import multiprocessing
import urllib
import requests
import warnings

warnings.filterwarnings("ignore")


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


def preprocess_mp(df):
    df = df.loc[df['is-click'] == 'Y']   # 统计点击新闻类别分布时只需要取用户点击过的新闻
    df = wash(df)
    df = time_shift(df)
    df['is_click'] = df['is_click'].apply(lambda x: 1 if x == 'Y' else 0)
    if 'category' in df.columns:
        del df['category']
    return df


def preprocess():
    batch = 0
    for df in pd.read_csv('./data/export.csv', index_col=False, chunksize=2000000, error_bad_lines=False):
        batch += 1
        p = multiprocessing.Pool(processes=4)
        split_dfs = np.array_split(df, 4)
        pool_results = p.map(preprocess_mp, split_dfs)
        p.close()
        p.join()
        parts = pd.concat(pool_results, axis=0)

        print('writing chunk %d...' % batch)
        if batch == 1:
            parts.to_csv('./data/history.csv', index=False)
        else:
            parts.to_csv('./data/history.csv', index=False, header=False, mode='a')


def get_category(tdata):
    headers = {'Server': 'Tengine',
               'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36 Edge/15.15063'}
    url = list(tdata['url'])
    for i in range(len(url)):
        url[i] = urllib.parse.quote(url[i])
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


def job():
    starttime = datetime.datetime.now()
    # 更新raw data 取数据中出现的最晚时间往前倒7天
    print('begin refreshing export.csv...')
    batch = 0
    for data in pd.read_csv('./data/export.csv', index_col=False, chunksize=500000, error_bad_lines=False):
        batch += 1
        latest = data['refresh-time'].max()
        data = data.loc[data['refresh-time'] >= latest - 3600000 * 24 * 7]
        if batch == 1:
            data.to_csv('./data/export2.csv', index=False)
        else:
            data.to_csv('./data/export2.csv', index=False, header=False, mode='a')
        print('chunk %d done' % batch)
    os.remove('./data/export.csv')
    os.rename('./data/export2.csv', './data/export.csv')
    # 预处理：转换时间、更改列名...
    print('begin preprocessing...')
    preprocess()
    gc.collect()
    endtime = datetime.datetime.now()
    print('preprocess done, time cost: ' + str((endtime - starttime).seconds / 60))

    # 请求接口获取数据中出现过的新闻类别
    history = pd.read_csv('./data/history.csv')
    starttime = datetime.datetime.now()
    news_list = history.drop_duplicates(['url'], keep='first')
    news_list = news_list[['url']]
    news_list = np.array_split(news_list, 50)
    result = pd.DataFrame()
    for i in range(len(news_list)):
        df = get_category(news_list[i])
        print('chunk %d done' % i)
        if i == 0:
            result = df
        else:
            result = pd.concat([result, df])
        del df
        gc.collect()
    del news_list
    gc.collect()
    result = result.fillna('other')
    result = result.fillna('other')
    # result.to_csv('./data/cat.csv', index=False)
    history = pd.merge(history, result, on=['url'], how='left')
    endtime = datetime.datetime.now()
    print('category vector get, time cost: ' + str((endtime - starttime).seconds / 60))

    col_name = pd.read_csv('./data/col_name.csv', index_col=False)
    col_name = eval(col_name[1])
    history = history[['device_id', 'refresh_timestamp', 'refresh_date', 'url', 'category']]
    history = pd.concat([history, pd.get_dummies(history['category'], prefix='cat')], axis=1)
    # 如果出现 训练集中的部分类别 在近7天没有出现的情况，就把近7天这些类别的对应分布填0 防后面报错
    for i in col_name:
        if i not in list(history.columns[5:]):
            history[i] = 0
    # 7天内出现的所有用户的点击历史
    temp = history.groupby(['device_id'])[col_name].mean().reset_index()

    if not os.path.exists('./data/category_features_online.csv'):
        temp.to_csv('./data/category_features_online.csv', index=False)
    else:
        temp.to_csv('./data/category_features_online_tmp.csv', index=False)
        os.remove('./data/category_features_online.csv')
        os.rename('./data/category_features_online_tmp.csv', './data/category_features_online.csv')
    os.remove('./data/history.csv')
    endtime = datetime.datetime.now()
    del temp, history
    gc.collect()
    print('category features for online done at ' + str(endtime))


if __name__ == '__main__':
    job()
    # schedule.every(20).minutes.do(job)
    schedule.every().day.at("03:00").do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)