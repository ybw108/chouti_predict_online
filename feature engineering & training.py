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
    tdata['refresh_day'] = tdata['refresh_time'].dt.day
    tdata['refresh_hour'] = tdata['refresh_time'].dt.hour

    print('time shift done')
    return tdata


# 获取新闻w2v vector
def get_w2v_vector(tdata):
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
    print('vector get')
    # wv = list_to_frame(wv)
    tdata = tdata.reset_index(drop=True)
    tdata = pd.concat([tdata, wv], axis=1)
    return tdata


# 获取新闻category
def get_category(tdata):
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
def get_correlation(tdata):
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
    # wv = list_to_frame(wv)
    tdata = tdata.reset_index(drop=True)
    tdata = pd.concat([tdata, corr], axis=1)
    tdata = tdata.rename(columns={0: 'corr'})
    return tdata[['device_id', 'url', 'corr']]


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


def news_click_features(data, history, name):
    data = data.apply(get_click_features, axis=1, history=history)
    data = data[['url', 'refresh_day', 'refresh_hour', 'click_count_in_1', 'click_ratio_in_1',
                 'click_count_in_3', 'click_ratio_in_3', 'click_count_in_6', 'click_ratio_in_6',
                 'click_count_in_12', 'click_ratio_in_12', 'click_count_in_24', 'click_ratio_in_24',
                 ]]
    data.to_csv(dir+'/'+str(name)+'.csv', index=False)
    print('%s done' % name)
    gc.collect()


def top_k_corr(row):
    if row['corr'] != '[]':
        temp = heapq.nlargest(5, eval(row['corr']))
        temp = pd.Series(temp)
        temp_index = []
        for j in range(len(temp)):
            temp_index.append('cosine_with_top'+str(j+1))
        temp.index = temp_index
        row = pd.concat([row, temp])
        # print('corr get')
        return row
    else:
        row['corr'] = -100
        return row

###################################################

#
# def preprocessing():
#     batch = 0
#     if not os.path.exists(dir + '/raw_data.csv'):
#         for df in pd.read_csv('./data/data.csv', index_col=False, chunksize=100000):
#             batch += 1
#             df = wash(df)
#             df = time_shift(df)
#             df['is_click'] = df['is_click'].apply(lambda x: 1 if x == 'Y' else 0)
#             gc.collect()
#             print('writing chunk %d...' % batch)
#             if batch == 1:
#                 df.to_csv(dir+'/raw_data.csv', index=False)
#             else:
#                 df.to_csv(dir+'/raw_data.csv', index=False, header=False, mode='a')
#     print('preprocess done')


def get_w2v_category_correlation():
    # get w2v vector, category and correlation
    data = pd.read_csv(dir + '/data_for_train.csv', index_col=False)
    data = data.drop_duplicates(['url'], keep='first')
    data = data[['url']]
    data.to_csv(dir+'/news_list.csv', index=False)

    # vector & category
    if not os.path.exists(dir + '/news_vector.csv'):
        batch = 0
        wrong_batch_list = []
        for df in pd.read_csv(dir + '/news_list.csv', index_col=False, chunksize=500):
            if batch >= 0:
                try:
                    batch += 1
                    df = get_w2v_vector(pd.DataFrame(df))
                    if batch == 1:
                        df.to_csv(dir+'/news_vector.csv', index=False)
                    else:
                        df.to_csv(dir+'/news_vector.csv', index=False, header=False, mode='a')
                    print('chunk %d done.' % batch)

                except KeyError as reason:
                    print("Error: " + str(reason) + " wrong batch is " + str(batch))
                    wrong_batch_list.append(batch)
                    print(datetime.datetime.now())
                    continue
                except IndexError as reason:
                    print("Error: " + str(reason) + " wrong batch is " + str(batch))
                    wrong_batch_list.append(batch)
                    print(datetime.datetime.now())
                    continue
                except ValueError as reason:
                    print("Error: " + str(reason) + " wrong batch is " + str(batch))
                    wrong_batch_list.append(batch)
                    print(datetime.datetime.now())
                    continue
                except UnicodeDecodeError as reason:
                    print("Error: " + str(reason) + " wrong batch is " + str(batch))
                    wrong_batch_list.append(batch)
                    print(datetime.datetime.now())
                    continue
            else:
                batch += 1
        print('w2v vector done')
        print(wrong_batch_list)

    if not os.path.exists(dir + '/news_category.csv'):
        batch = 0
        wrong_batch_list = []
        for df in pd.read_csv(dir + '/news_list.csv', index_col=False, chunksize=500):
            if batch >= 0:
                try:
                    batch += 1
                    df = get_category(pd.DataFrame(df))
                    if batch == 1:
                        df.to_csv(dir+'/news_category.csv', index=False)
                    else:
                        df.to_csv(dir+'/news_category.csv', index=False, header=False, mode='a')
                    print('chunk %d done.' % batch)

                except KeyError as reason:
                    print("Error: " + str(reason) + " wrong batch is " + str(batch))
                    wrong_batch_list.append(batch)
                    print(datetime.datetime.now())
                    continue
                except IndexError as reason:
                    print("Error: " + str(reason) + " wrong batch is " + str(batch))
                    wrong_batch_list.append(batch)
                    print(datetime.datetime.now())
                    continue
                except ValueError as reason:
                    print("Error: " + str(reason) + " wrong batch is " + str(batch))
                    wrong_batch_list.append(batch)
                    print(datetime.datetime.now())
                    continue
                except UnicodeDecodeError as reason:
                    print("Error: " + str(reason) + " wrong batch is " + str(batch))
                    wrong_batch_list.append(batch)
                    print(datetime.datetime.now())
                    continue
            else:
                batch += 1
        print('category done')
        print(wrong_batch_list)

    if not os.path.exists(dir + '/news_corr.csv'):
        batch = 0
        wrong_batch_list = []
        for df in pd.read_csv(dir + '/raw_data.csv', index_col=False, chunksize=300):
            if batch >= 0:
                try:
                    batch += 1
                    df = get_correlation(df)
                    if batch == 1:
                        df.to_csv(dir+'/news_corr.csv', index=False, encoding='utf-8')
                    else:
                        df.to_csv(dir+'/news_corr.csv', index=False, header=False, mode='a', encoding='utf-8')
                    print('chunk %d done.' % batch)

                except KeyError as reason:
                    print("Error: " + str(reason) + " wrong batch is " + str(batch))
                    wrong_batch_list.append(batch)
                    continue
                except IndexError as reason:
                    print("Error: " + str(reason) + " wrong batch is " + str(batch))
                    wrong_batch_list.append(batch)
                    continue
                except ValueError as reason:
                    print("Error: " + str(reason) + " wrong batch is " + str(batch))
                    wrong_batch_list.append(batch)
                    continue
                except UnicodeDecodeError as reason:
                    print("Error: " + str(reason) + " wrong batch is " + str(batch))
                    wrong_batch_list.append(batch)
                    continue
            else:
                batch += 1
        print('correlation done')
        print(wrong_batch_list)
    os.remove(dir+'/news_list.csv')
    print('w2v,category,correlation done')


def category_features():
    if not os.path.exists(dir + '/category_features.csv'):
        data = pd.read_csv(dir+'/data_for_train.csv', index_col=False)
        category = pd.read_csv(dir+'/news_category.csv', index_col=False)
        category = category[['url', 'category']]
        category = category.fillna('other')

        # 取出历史点击情况
        history = pd.read_csv(dir+'/history.csv', index_col=False)
        history = history.loc[history.is_click == 1]
        history = pd.merge(history, category, how='left', on=['url'])
        history = history[['device_id', 'refresh_timestamp', 'refresh_date', 'url', 'category']]
        history = pd.concat([history, pd.get_dummies(history['category'], prefix='cat')], axis=1)
        # print('read done')

        cat_data = data.drop_duplicates(['device_id', 'refresh_date'])  # 取用户-日期
        # 7月10日之前在训练集里出现的的点击历史
        t_history = history.loc[history.device_id.isin(cat_data['device_id'])]
        result = pd.DataFrame()
        L = list(cat_data['refresh_date'].unique())
        for i in range(len(L)):
            tt_history = t_history.loc[(t_history.refresh_date < L[i]) & (t_history.refresh_date >= str(datetime.datetime.strptime(L[i], '%Y-%m-%d') - datetime.timedelta(days=7)))]
            temp = tt_history.groupby(['device_id'])[['cat_art', 'cat_car', 'cat_edu', 'cat_ent', 'cat_finance', 'cat_game',
                                                      'cat_gj', 'cat_other', 'cat_party', 'cat_sh', 'cat_sport', 'cat_tech',
                                                      'cat_war', 'cat_weather']].mean().reset_index()
            temp['refresh_date'] = L[i]

            if i == 0:
                result = temp
            else:
                result = pd.concat([result, temp])
            print('%s done' % L[i])
        cat_data = pd.merge(cat_data, result, on=['device_id', 'refresh_date'], how='left')

        cat_data = cat_data[
            ['device_id', 'refresh_day', 'cat_art', 'cat_car', 'cat_edu', 'cat_ent', 'cat_finance', 'cat_game',
             'cat_gj', 'cat_other', 'cat_party', 'cat_sh', 'cat_sport', 'cat_tech',
             'cat_war', 'cat_weather']]
        cat_data.to_csv(dir+'/category_features.csv', index=False)
    print('category features done')


def click_features():
    if not os.path.exists(dir + '/click_features.csv'):
        history = pd.read_csv(dir + '/history.csv', index_col=False)
        data = pd.read_csv(dir + '/data_for_train.csv', index_col=False)
        data = data.drop_duplicates(['url', 'refresh_day', 'refresh_hour'], keep='first')  # 近似
        #### 需要改动
        print('read done')

        news_click_features(data, history, 'click_features')
    # p1 = Process(target=news_click_features, args=(data1, history, 'click_features_1'))
    # p2 = Process(target=news_click_features, args=(data2, history, 'click_features_2'))
    # p3 = Process(target=news_click_features, args=(data3, history, 'click_features_3'))
    # p4 = Process(target=news_click_features, args=(data4, history, 'click_features_4'))
    #
    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    #
    # print("The number of CPU is:" + str(multiprocessing.cpu_count()))
    # for p in multiprocessing.active_children():
    #     print("child p.name: " + p.name + "\tp.id: " + str(p.pid))
    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()
    print('click features done')


def correlation_features():
    if not os.path.exists(dir + '/corr_features.csv'):
        news_corr = pd.read_csv(dir + '/news_corr.csv', index_col=False)
        news_corr['avg_cosine_center'] = news_corr['corr'].apply(lambda x: sum(eval(x))/len(eval(x)) if x != '[]' else -100)
        news_corr = news_corr.apply(top_k_corr, axis=1)
        del news_corr['corr']
        news_corr.to_csv(dir + '/corr_features.csv', index=False)
        del news_corr
        gc.collect()
    print('correlation features done')


def merge_features():
    train = pd.read_csv(dir + '/data_for_train.csv', index_col=False)
    click_features = pd.read_csv(dir + '/click_features.csv', index_col=False)

    category_features = pd.read_csv(dir + '/category_features.csv', index_col=False)
    # 先fill merge的时候还是会出现空值（没有历史的那部分用户）
    # category_features = category_features.fillna(1/14)

    corr_features = pd.read_csv(dir + '/corr_features.csv', index_col=False)
    corr_features = corr_features.fillna(-100)

    news_vector = pd.read_csv(dir + '/news_vector.csv', index_col=False)
    print('read done')
    train = pd.merge(train, click_features, how='left', on=['url', 'refresh_day', 'refresh_hour'])
    train = pd.merge(train, category_features, how='left', on=['device_id', 'refresh_day'])
    train = train.fillna(1/14)
    train = pd.merge(train, corr_features, how='left', on=['device_id', 'url'])
    # label = train['is_click']
    # used_features = [c for c in train if c not in ignore]
    # train = train[used_features]
    train = pd.merge(train, news_vector, how='left', on=['url'])
    del train['url']
    # train.to_csv(dir+'/train_set.csv', index=False)
    print('merge done')
    return train


def best_cutoff_search(train, used_features):
    date = train['refresh_date'].unique()
    date.sort()
    date = date[-1]
    valid = train.loc[train.refresh_date == date]
    train = train.loc[train.refresh_date != date]

    gbm = lightgbm.LGBMClassifier(objective='binary', n_estimators=2000, seed=2018,
                                  learning_rate=0.05,
                                  # colsample_bytree=0.7,
                                  # subsample=0.7,
                                  max_depth=-1,
                                  # num_leaves=95,
                                  # reg_lambda=8,
                                  # max_bin=1000,
                                  # min_child_samples=12,
                                  )
    print('begin searching best cutoff...')
    model = gbm.fit(train[used_features], train['is_click'], eval_set=[(valid[used_features], valid['is_click'])],
                    eval_metric='auc', early_stopping_rounds=100, verbose=False)
    valid['predict'] = gbm.predict_proba(valid[used_features], num_iteration=model.best_iteration_)[:, 1]
    L = [i/100.0 for i in range(30, 75, 5)]
    f1_dict = {}
    for i in L:
        valid['predict_label'] = valid['predict'].apply(lambda x: 1 if x >= i else 0)
        f1_score = metrics.f1_score(valid['is_click'], valid['predict_label'])
        print('cutoff: ' + str(i) + ', f1_score: ' + str(f1_score))
        f1_dict[i] = f1_score

    best_offline = max(zip(f1_dict.values(), f1_dict.keys()))
    print('best_f1: ', best_offline[0])
    print('best_f1_cutoff: ', best_offline[1])
    return model.best_iteration_


def online_model(train, label, iterations):
    gbm = lightgbm.LGBMClassifier(objective='binary', n_estimators=int(iterations), seed=2018,
                                  learning_rate=0.05,
                                  # colsample_bytree=0.7,
                                  # subsample=0.7,
                                  max_depth=-1,
                                  # num_leaves=95,
                                  # reg_lambda=8,
                                  # max_bin=1000,
                                  # min_child_samples=12,
                                  )
    print('begin training...')
    model = gbm.fit(train, label)
    joblib.dump(model, dir + '/lightgbm_model.m')
    print('model get')
    return


if __name__ == '__main__':
    year = datetime.datetime.now().year
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day
    dir = './version_'+str(year)+'-'+str(month)+'-'+str(day)
    if not os.path.exists(dir):
        os.makedirs(dir)

    get_w2v_category_correlation()
    category_features()
    click_features()
    correlation_features()

    ignored_features = ['device_id', 'link_id', 'is_click', 'category', 'publish_time', 'publish_timestamp', 'refresh_date',
                        'refresh_day',  'refresh_time', 'refresh_hour', 'refresh_timestamp', 'category',
                        ]
    train = merge_features()
    used_features = [c for c in train if c not in ignored_features]

    iterations = best_cutoff_search(train, used_features)
    online_model(train[used_features], train['is_click'], iterations)

