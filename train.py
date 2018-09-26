# coding: utf-8
import pandas as pd
import numpy as np
import gc
import urllib
import requests
import time
import math
import datetime
import heapq
import lightgbm
from sklearn import metrics
from sklearn.decomposition import PCA
import os
from multiprocessing import Process
import multiprocessing
from sklearn.externals import joblib
from functools import partial
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
    headers = {'Server': 'Tengine',
               'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36 Edge/15.15063'}
    url = list(tdata['url'])
    for i in range(len(url)):
        url[i] = urllib.parse.quote(url[i])
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
    print('vector get')
    # wv = list_to_frame(wv)
    tdata = tdata.reset_index(drop=True)
    tdata = pd.concat([tdata, wv], axis=1)
    return tdata


# 获取新闻category
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
    return tdata


# 获取新闻和用户兴趣点的相似度list
def get_interest(tdata):
    headers = {'Server': 'Tengine',
               'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36 Edge/15.15063'}
    device = list(tdata['device_id'])
    para = {'deviceIds': device}
    code = -1
    i = 0
    while (code != 200) & (i <= 4):
        r = requests.post('http://ai.chouti.com/news/recommend/userFeature', data=para, headers=headers)
        code = r.status_code
        i += 1
        if i > 1:
            print('timeout retrying...')
    interest = pd.Series(r.json()['data'])
    tdata = tdata.reset_index(drop=True)
    tdata = pd.concat([tdata, interest], axis=1)
    tdata.columns = ['device_id', 'interest']
    tdata = tdata.fillna(-100)
    print('interest get')
    r.close()
    time.sleep(1)
    return tdata


def get_click_features(row, history):
    history = history.loc[history.url == row.url]
    history = history.loc[history.refresh_timestamp < int(row['refresh_timestamp'])]

    history = history.loc[history.refresh_timestamp >= row['refresh_timestamp'] - 3600000*24]
    row['click_ratio_in_24'] = len(history.loc[history.is_click == 1]) / (len(history) + 0.00001)

    history = history.loc[history.refresh_timestamp >= row['refresh_timestamp'] - 3600000]
    row['click_ratio_in_1'] = len(history.loc[history.is_click == 1]) / (len(history) + 0.00001)
    # for i in [1, 24]: # [1, 3, 6, 12, 24]:
    #     tdata = history.loc[history.refresh_timestamp >= row['refresh_timestamp'] - 3600000*int(i)]
    #     # row['click_count_in_'+str(i)] = len(tdata)
    #     row['click_ratio_in_'+str(i)] = len(tdata.loc[tdata.is_click == 1])/(len(tdata)+0.00001)
    #     # rank
    print('click features done')
    return row


def click_features_mp(df, history):
    df = df.apply(get_click_features, axis=1, history=history)
    df = df[['url', 'refresh_day', 'refresh_hour', 'click_ratio_in_1', 'click_ratio_in_24']]
    return df


def top_k_corr(row, user, news):
    news_vector = news.loc[news.url == row['url']].iloc[0, 1:]
    # news_vector = np.array(news_vector)
    center_list = user.loc[user.device_id == row['device_id'], ['interest']].iloc[0, 0]
    if center_list != -100:
        center_list = eval(center_list)
    else:
        center_list = -100
    cosine_list = []
    if (news_vector['0'] != -100) & (center_list != -100):
        news_vector = np.array(news_vector)
        for i in center_list:
            i = np.array(i)
            cosine_list.append(i.dot(news_vector)/(math.sqrt((i**2).sum()) * math.sqrt((news_vector**2).sum())))
        temp = heapq.nlargest(5, cosine_list)
        temp = pd.Series(temp)
        temp_index = []
        for j in range(len(temp)):
            temp_index.append('cosine_with_top' + str(j + 1))
        temp.index = temp_index
        row = pd.concat([row, temp])
        row['cosine_top_5_avg'] = sum(temp)/len(temp)
        row['cosine_all_avg'] = sum(cosine_list)/len(cosine_list)
    print('correlation get')
    row = row.fillna(-100)
    return row


def correlation_features_mp(df, user, news):
    df = df.apply(top_k_corr, user=user, news=news, axis=1)
    df = df[['device_id', 'url', 'refresh_timestamp',  'cosine_with_top1', 'cosine_with_top2', 'cosine_with_top3', 'cosine_with_top4',
             'cosine_with_top5', 'cosine_top_5_avg', 'cosine_all_avg']]
    return df

###################################################


def get_w2v_category_correlation():
    # get w2v vector, category and correlation

    data = pd.read_csv(dir + '/data_for_train.csv', index_col=False)
    print('number of instances: ' + str(len(data)))
    user = data.drop_duplicates(['device_id'], keep='first')
    user = user[['device_id']]
    print('number of users: ' + str(len(user)))
    data = data.drop_duplicates(['url'], keep='first')
    data = data[['url']]
    print('number of news: ' + str(len(data)))
    data.to_csv(dir+'/news_list.csv', index=False)
    user.to_csv(dir+'/user_list.csv', index=False)

    # vector & category
    starttime = datetime.datetime.now()
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
        endtime = datetime.datetime.now()
        print('w2v vector get, time cost: ' + str((endtime - starttime).seconds / 60))
        print(wrong_batch_list)

    starttime = datetime.datetime.now()
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
        endtime = datetime.datetime.now()
        print('category vector get, time cost: ' + str((endtime - starttime).seconds / 60))
        print(wrong_batch_list)

    starttime = datetime.datetime.now()
    if not os.path.exists(dir + '/user_interest.csv'):
        batch = 0
        wrong_batch_list = []
        for df in pd.read_csv(dir + '/user_list.csv', index_col=False, chunksize=500):
            if batch >= 0:
                try:
                    batch += 1
                    df = get_interest(df)
                    if batch == 1:
                        df.to_csv(dir+'/user_interest2.csv', index=False, encoding='utf-8')
                    else:
                        df.to_csv(dir+'/user_interest2.csv', index=False, header=False, mode='a', encoding='utf-8')
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
        endtime = datetime.datetime.now()
        print('correlation get, time cost: ' + str((endtime - starttime).seconds / 60))
        print(wrong_batch_list)
    os.remove(dir+'/news_list.csv')
    print('w2v,category,correlation done')


def category_features():
    starttime = datetime.datetime.now()
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
    endtime = datetime.datetime.now()
    print('category features done, time cost: ' + str((endtime - starttime).seconds / 60))


def click_features():
    starttime = datetime.datetime.now()
    if not os.path.exists(dir + '/click_features.csv'):
        data = pd.read_csv(dir + '/data_for_train.csv', index_col=False)
        data = data.drop_duplicates(['url', 'refresh_day', 'refresh_hour'], keep='first')  # 近似

        # data = click_features_mp(data, history)
        # data.to_csv(dir + '/click_features.csv', index=False)

        split_dfs = np.array_split(data, 4)
        del data
        gc.collect()
        history = pd.read_csv(dir + '/history.csv', index_col=False)
        L = list(history['refresh_date'].unique())
        L.sort()
        history = history.loc[history.refresh_date.isin(L[-8:])]
        history = history[['url', 'refresh_timestamp', 'is_click']]
        print('read done')

        p = multiprocessing.Pool(processes=4)
        pool_results = p.map(partial(click_features_mp, history=history), split_dfs)
        p.close()
        p.join()

        # merging parts processed by different processes
        parts = pd.concat(pool_results, axis=0)

        # merging newly calculated parts to big_df
        parts.to_csv(dir + '/click_features.csv', index=False)
    endtime = datetime.datetime.now()
    print('click features done, time cost: ' + str((endtime - starttime).seconds/60))


def correlation_features():
    starttime = datetime.datetime.now()
    if not os.path.exists(dir + '/corr_features.csv'):
        data = pd.read_csv(dir + '/data_for_train.csv', index_col=False)
        data = data[['device_id', 'url', 'refresh_timestamp']]
        user_interest = pd.read_csv(dir + '/user_interest.csv', index_col=False)
        news_vector = pd.read_csv(dir + '/news_vector.csv', index_col=False)

        # pool_results = correlation_features_mp(data, user_interest, news_vector)
        p = multiprocessing.Pool(processes=4)
        split_dfs = np.array_split(data, 4)
        pool_results = p.map(partial(correlation_features_mp, user=user_interest, news=news_vector), split_dfs)
        p.close()
        p.join()

        # merging parts processed by different processes
        parts = pd.concat(pool_results, axis=0)

        # merging newly calculated parts to big_df
        parts = parts.fillna(-100)
        parts.to_csv(dir + '/corr_features.csv', index=False)

        gc.collect()
    endtime = datetime.datetime.now()
    print('correlation features done, time cost: ' + str((endtime - starttime).seconds / 60))


def merge_features():
    train = pd.read_csv(dir + '/data_for_train.csv', index_col=False)
    click_features = pd.read_csv(dir + '/click_features.csv', index_col=False)
    category_features = pd.read_csv(dir + '/category_features.csv', index_col=False)

    # corr_features = pd.read_csv(dir + '/corr_features.csv', index_col=False)
    # corr_features = corr_features.fillna(-100)

    # 对 news_vector PCA降维
    news_vector = pd.read_csv(dir + '/news_vector.csv', index_col=False)
    x = news_vector.iloc[:, 1:]
    pca = PCA(n_components=100)
    x_pca = pca.fit_transform(x)
    joblib.dump(pca, dir + '/pca_model.m')
    x_pca = x_pca.astype(np.float16)
    vector = pd.concat([news_vector['url'], pd.DataFrame(x_pca)], axis=1)

    del news_vector
    gc.collect()

    print('read done')
    train = pd.merge(train, click_features, how='left', on=['url', 'refresh_day', 'refresh_hour'])
    train = pd.merge(train, category_features, how='left', on=['device_id', 'refresh_day'])
    train = train.fillna(1/14)
    train = pd.merge(train, vector, how='left', on=['url'])
    del train['url']
    # train.to_csv(dir+'/train_set.csv', index=False)
    print('merge done')
    return train


# 抽样训练
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
                    eval_metric='auc', early_stopping_rounds=50, verbose=False)
    valid['predict'] = gbm.predict_proba(valid[used_features], num_iteration=model.best_iteration_)[:, 1]
    L = [i/100.0 for i in range(0, 75, 5)]
    accuracy_dict = {}
    for i in L:
        pred = valid['predict'].apply(lambda x: 1 if x >= i else 0)
        acc_score = metrics.accuracy_score(valid['is_click'], pred)
        print('cutoff: ' + str(i) + ', accuracy: ' + str(acc_score))
        accuracy_dict[i] = acc_score

    best_offline = max(zip(accuracy_dict.values(), accuracy_dict.keys()))
    print('best_accuracy: ', best_offline[0])
    print('best_accuracy_cutoff: ', best_offline[1])
    return model.best_iteration_


def online_model(train, used_features, iterations):
    starttime = datetime.datetime.now()
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
    model = gbm.fit(train[used_features], train['is_click'])
    joblib.dump(model, dir + '/lightgbm_model.m')
    endtime = datetime.datetime.now()
    print('model get, time cost: ' + str((endtime - starttime).seconds / 60))
    return

# 增量训练
# def best_cutoff_search(train, used_features):
#     starttime = datetime.datetime.now()
#     date = train['refresh_date'].unique()
#     date.sort()
#     date = date[-1]
#     valid = train.loc[train.refresh_date == date]
#     train = train.loc[train.refresh_date != date]
#     train = np.array_split(train, 4)
#     gbm = None
#     iterations = []
#     params = {
#         'objective': 'binary',
#         'learning_rate': 0.05,
#         'max_depth': -1,
#         'seed': 2018,
#         'metric': 'auc',
#     }
#     batch = 0
#     print('begin searching best cutoff...')
#
#     for df in train:
#         # 区分特征x和结果Y
#
#         # 创建lgb的数据集
#         lgb_train = lightgbm.Dataset(df[used_features], df['is_click'].values)
#         lgb_eval = lightgbm.Dataset(valid[used_features], valid['is_click'].values, reference=lgb_train)
#
#         # 增量训练模型
#         # 通过 init_model 和 keep_training_booster 两个参数实现增量训练
#         gbm = lightgbm.train(params,
#                              lgb_train,
#                              num_boost_round=2000,
#                              valid_sets=lgb_eval,
#                              early_stopping_rounds=100,
#                              init_model=gbm,  # 如果gbm不为None，那么就是在上次的基础上接着训练
#                              verbose_eval=True,
#                              keep_training_booster=True)  # 增量训练
#
#         batch += 1
#         print('batch %d done' % batch)
#         iterations.append(gbm.best_iteration)
#     valid['predict'] = gbm.predict(valid[used_features], num_iteration=gbm.best_iteration)
#     L = [i/100.0 for i in range(0, 75, 5)]
#     f1_dict = {}
#     for i in L:
#         pred = valid['predict'].apply(lambda x: 1 if x >= i else 0)
#         f1_score = metrics.f1_score(valid['is_click'], pred)
#         print('cutoff: ' + str(i) + ', f1_score: ' + str(f1_score))
#
#         f1_dict[i] = f1_score
#
#     best_offline = max(zip(f1_dict.values(), f1_dict.keys()))
#     print('best_f1: ', best_offline[0])
#     print('best_f1_cutoff: ', best_offline[1])
#     print('best_iterations: ' + str(gbm.best_iteration))
#     endtime = datetime.datetime.now()
#     print('time cost: ' + str((endtime - starttime).seconds / 60))
#     return iterations


# def online_model(train, used_features, iterations):
#     starttime = datetime.datetime.now()
#     train = np.array_split(train, 4)
#     gbm = None
#     params = {
#                 'objective': 'binary',
#                 'learning_rate': 0.05,
#                 'max_depth': -1,
#                 'seed': 2018,
#                 'metric': 'auc',
#              }
#     batch = 0
#     print('begin training...')
#     for df in train:
#         # 区分特征x和结果Y
#         train_X = df[used_features]
#         train_Y = df['is_click']
#
#         # 创建lgb的数据集
#         lgb_train = lightgbm.Dataset(train_X, train_Y.values)
#         del train_X, train_Y
#         gc.collect()
#
#         if batch == 0:
#             boost_round = iterations[0]
#         else:
#             boost_round = iterations[batch] - iterations[batch-1]
#         # 增量训练模型
#         # 通过 init_model 和 keep_training_booster 两个参数实现增量训练
#         gbm = lightgbm.train(params,
#                              lgb_train,
#                              num_boost_round=boost_round,
#                              init_model=gbm,  # 如果gbm不为None，那么就是在上次的基础上接着训练
#                              verbose_eval=False,
#                              keep_training_booster=True)  # 增量训练
#
#
#         batch += 1
#         print('batch %d done' %batch)
#     joblib.dump(gbm, dir + '/lightgbm_model.m')
#     endtime = datetime.datetime.now()
#     print('model get, time cost: ' + str((endtime - starttime).seconds / 60))
#     return


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
    # train = train.sample(frac=0.6)

    used_features = [c for c in train if c not in ignored_features]
    iterations = best_cutoff_search(train, used_features)
    print(train[used_features].info())
    online_model(train, used_features, iterations)

