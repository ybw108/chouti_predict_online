import schedule
import time
import datetime
import pandas as pd
import os


def job():
    starttime = datetime.datetime.now()
    data = pd.read_csv(dir + '/data_for_train.csv', index_col=False)
    category = pd.read_csv(dir + '/news_category.csv', index_col=False)
    category = category[['url', 'category']]
    category = category.fillna('other')

    # 取出历史点击情况
    history = pd.read_csv(dir + '/history.csv', index_col=False)
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
        tt_history = t_history.loc[(t_history.refresh_date < L[i]) & (t_history.refresh_date >= str(
            datetime.datetime.strptime(L[i], '%Y-%m-%d') - datetime.timedelta(days=7)))]
        temp = tt_history.groupby(['device_id'])[
            ['cat_art', 'cat_car', 'cat_edu', 'cat_ent', 'cat_finance', 'cat_game',
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
    cat_data.to_csv(dir + '/category_features.csv', index=False)
    endtime = datetime.datetime.now()
    print('category features done, time cost: ' + str((endtime - starttime).seconds / 60))


if __name__ == '__main__':
    job()
    schedule.every().day.at("04:00").do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)