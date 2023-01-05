#coding=utf-8
'''
特征提取---注释
'''

import gc
import pickle
import gensim
import pandas as pd
from gensim.models import Word2Vec
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


def get_agg_features(dfs, f1, f2, agg, log):
    '''
        df
        字段1：
        字段2：
        agg 计算方式
        log:数据
    '''
    # 判定特殊情况
    if type(f1) == str:
        f1 = [f1]
    if agg != 'size':
        # 把需要聚合的两列拿出来
        data = log[f1+[f2]]
    else:
        # size的话只需要一列就行
        data = log[f1]
    print('f1:', f1, ' ', 'f2:', f2)
    f_name = '_'.join(f1) + "_" + f2 + "_" + agg

    # 聚合操作
    if agg == "size":
        tmp = pd.DataFrame(data.groupby(f1).size()).reset_index()
    elif agg == "count":
        tmp = pd.DataFrame(data.groupby(f1)[f2].count()).reset_index()
    elif agg == "mean":
        tmp = pd.DataFrame(data.groupby(f1)[f2].mean()).reset_index()
    elif agg == "unique":
        tmp = pd.DataFrame(data.groupby(f1)[f2].nunique()).reset_index()
    elif agg == "max":
        tmp = pd.DataFrame(data.groupby(f1)[f2].max()).reset_index()
    elif agg == "min":
        tmp = pd.DataFrame(data.groupby(f1)[f2].min()).reset_index()
    elif agg == "sum":
        tmp = pd.DataFrame(data.groupby(f1)[f2].sum()).reset_index()
    elif agg == "std":
        tmp = pd.DataFrame(data.groupby(f1)[f2].std()).reset_index()
    elif agg == "median":
        tmp = pd.DataFrame(data.groupby(f1)[f2].median()).reset_index()
    else:
        raise Exception("agg error")

    # 赋值聚合特征
    for df in dfs:
        try:
            del df[f_name]
        except:
            pass
        tmp.columns = f1+[f_name]
        df[f_name] = df.merge(tmp, on=f1, how='left')[f_name]
    del tmp
    del data
    gc.collect()
    return [f_name]



def sequence_text(dfs, f1, f2, log):
    '''
    根据两个字段，生成序列特征，比如user_id下的ad_id序列
    '''
    f_name = 'sequence_text_'+f1+'_'+f2
    print(f_name)

    # --------遍历log，获得用户的点击序列，该序列为f2(某个物品id+fold)--------
    dic, items = {}, []
    for item in log[[f1, f2]].values:
        try:
            dic[item[0]].append(str(item[1]))
        except:
            dic[item[0]] = [str(item[1])]
    for key in dic:
        items.append([key, ' '.join(dic[key])])


    # 赋值序列特征
    temp = pd.DataFrame(items)
    temp.columns = [f1, f_name]
    temp = temp.drop_duplicates(f1)
    for df in dfs:
        try:
            del df[f_name]
        except:
            pass
        temp.columns = [f1]+[f_name]
        df[f_name] = df.merge(temp, on=f1, how='left')[f_name]
    gc.collect() 
    del temp
    del items
    del dic
    return [f_name]



def kfold(train_df, test_df, log_data, pivot):
    '''
    train_df: 每个用户一条数据
    log_data: 用户行为数据
    pivot: id字段
    '''

    # log_data = log
    # pivot = 'creative_id'

    # 先对log做kflod统计, 统计每条记录中pivot特征的性别年龄分布
    kfold_features = ['age_{}'.format(i) for i in range(10)] + ['gender_{}'.format(i) for i in range(2)]
    # 将【"12个label字段"，'user_id', pivot, 'fold'】拿出来
    log = log_data[kfold_features + ['user_id', pivot, 'fold']]
    tmps = []

    # 将01234 交叉求出每个label target的平均数，拼接起来
    for fold in range(6): # 012345
        # 求出大部分舍1后的fold的平均值，比如性别1的平均值，年龄的平均值
        tmp = pd.DataFrame(log[(log['fold'] != fold) & (log['fold'] != 5)].groupby(pivot)[kfold_features].mean()).reset_index()
        tmp.columns = [pivot]+kfold_features
        # 舍1后的特征，当作其余一个fold的特征
        tmp['fold'] = fold
        tmps.append(tmp)
    tmp = pd.concat(tmps, axis=0).reset_index()

    # Index(['user_id', 'creative_id', 'fold', 'index', 'age_0', 'age_1', 'age_2',
    #        'age_3', 'age_4', 'age_5', 'age_6', 'age_7', 'age_8', 'age_9',
    #        'gender_0', 'gender_1'],
    #       dtype='object')
    tmp = log[['user_id', pivot, 'fold']].merge(tmp, on=[pivot, 'fold'], how='left')

    del log
    del tmps
    gc.collect()


    # 获得用户点击的所有记录的平均性别年龄分布
    # ！！！注意点：根据log,将用户进行交叉target,这时候，每个用户是N条数据，比如'篮球'在年龄'10-20'为0.8,'口红'为0.2
    # 用户可能即点'篮球'又点'口红'再求平均，则这个用户年龄'10-20'此特征为0.5;
    tmp_mean = pd.DataFrame(tmp.groupby('user_id')[kfold_features].mean()).reset_index()
    tmp_mean.columns = ['user_id'] + [f+'_'+pivot+'_mean' for f in kfold_features]
    for df in [train_df, test_df]:
        temp = df.merge(tmp_mean, on='user_id', how='left')
        temp = temp.fillna(-1)
        for f1 in [f+'_'+pivot+'_mean' for f in kfold_features]:
            df[f1] = temp[f1]
        del temp
        gc.collect()

    del tmp
    del tmp_mean
    gc.collect()



def kfold_sequence(train_df, test_df, log_data, pivot):
    '''

    '''
    # 先对log做kflod统计，统计每条记录中pivot特征的性别年龄分布
    kfold_features = ['age_{}'.format(i) for i in range(10)]+['gender_{}'.format(i) for i in range(2)]
    log = log_data[kfold_features + [pivot, 'fold', 'user_id']]
    tmps = []
    for fold in range(6):
        # 不包含本fold及测试数据，统计某个字段下的label均值
        tmp = pd.DataFrame(log[(log['fold'] != fold) & (log['fold'] != 5)].groupby(pivot)[kfold_features].mean()).reset_index()
        tmp.columns = [pivot] + kfold_features
        tmp['fold'] = fold
        tmps.append(tmp)
    tmp = pd.concat(tmps, axis=0).reset_index()
    tmp = log[[pivot, 'fold', 'user_id']].merge(tmp, on=[pivot, 'fold'], how='left')
    tmp = tmp.fillna(-1)
    # tmp.columns:
    # 'industry', 'fold', 'user_id', 'index', 'age_0', 'age_1', 'age_2',
    #        'age_3', 'age_4', 'age_5', 'age_6', 'age_7', 'age_8', 'age_9',
    #        'gender_0', 'gender_1'],


    # ！！！不同点：
    # 将id与fold进行拼接
    tmp[pivot+'_fold'] = tmp[pivot]*10 + tmp['fold']
    del log
    del tmps
    gc.collect()

    # 获得用户点击记录的年龄性别分布序列
    tmp[pivot+'_fold'] = tmp[pivot+'_fold'].astype(int)
    # 将用户对不同的id 点击生成序列，只不过这个序列id加上fold;
    kfold_sequence_features = sequence_text([train_df, test_df], 'user_id', pivot+'_fold', tmp)
    # ！！！重点：同一个id+fold一个就行，因为都是一样的（数据量减少非常多）
    tmp = tmp.drop_duplicates([pivot+'_fold']).reset_index(drop=True)


    # 对每条记录年龄性别分布进行标准化
    kfold_features = ['age_{}'.format(i) for i in range(10)]+['gender_{}'.format(i) for i in range(2)]
    ss = StandardScaler()
    ss.fit(tmp[kfold_features])
    tmp[kfold_features] = ss.transform(tmp[kfold_features])
    for f in kfold_features:
        tmp[f] = tmp[f].apply(lambda x: round(x, 4))

    # 将每条记录年龄性别分布转成w2v形式的文件
    with open('data/sequence_text_user_id_'+pivot+'_fold'+".{}d".format(12),'w') as f:
        f.write(str(len(tmp))+' '+'12'+'\n')
        for item in tmp[[pivot+'_fold'] + kfold_features].values:
            f.write(' '.join([str(int(item[0]))]+[str(x) for x in item[1:]])+'\n') 
    tmp = gensim.models.KeyedVectors.load_word2vec_format('data/sequence_text_user_id_'+pivot+'_fold'+".{}d".format(12), binary=False)
    pickle.dump(tmp, open('data/sequence_text_user_id_'+pivot+'_fold'+".{}d".format(12), 'wb'))
    del tmp
    gc.collect()  
    return kfold_sequence_features




if __name__ == "__main__":

    # 读取数据
    root_path = '/Users/youshu_/Python_Workspace/Tencent2020_Rank1st/'
    click_log = pd.read_pickle(root_path + 'data/click.pkl',).sample(frac=0.01)
    train_df = pd.read_pickle(root_path + 'data/train_user.pkl').sample(frac=1)
    test_df = pd.read_pickle(root_path + 'data/test_user.pkl').sample(frac=1)

    # click_log = pd.read_pickle('data/click.pkl')
    # train_df = pd.read_pickle('data/train_user.pkl')
    # test_df = pd.read_pickle('data/test_user.pkl')

    # (63668283, 23) (900000, 3) (1000000, 3)
    print(click_log.shape, train_df.shape, test_df.shape)

    # Index(['time', 'user_id', 'creative_id', 'click_times', 'ad_id', 'product_id',
    #        'product_category', 'advertiser_id', 'industry', 'age', 'gender',
    #        'age_0', 'age_1', 'age_2', 'age_3', 'age_4', 'age_5', 'age_6', 'age_7',
    #        'age_8', 'age_9', 'gender_0', 'gender_1'],
    #       dtype='object')
    print(train_df.columns)



    ################################################################################
    # ---------------1：获取聚合特征---------------
    # 注意：将train / test合并算了，也就是最后一天每个用户
    print("Extracting aggregate feature...")
    agg_features = []
    # 1：每个用户总条数
    agg_features += get_agg_features([train_df, test_df], 'user_id', '', 'size', click_log)
    # 2：每个用户ad_id不同数
    agg_features += get_agg_features([train_df, test_df], 'user_id', 'ad_id', 'unique', click_log)
    # 3：每个用户creative_id不同数
    agg_features += get_agg_features([train_df, test_df], 'user_id', 'creative_id', 'unique', click_log)
    # 4：每个用户advertiser_id不同数
    agg_features += get_agg_features([train_df, test_df], 'user_id', 'advertiser_id', 'unique', click_log)
    # 5：每个用户industry不同数
    agg_features += get_agg_features([train_df, test_df], 'user_id', 'industry', 'unique', click_log)
    # 6：每个用户product_id不同数
    agg_features += get_agg_features([train_df, test_df], 'user_id', 'product_id', 'unique', click_log)
    # 7：每个用户time不同数
    agg_features += get_agg_features([train_df, test_df], 'user_id', 'time', 'unique', click_log)

    # 8：每个用户click_times总数
    agg_features += get_agg_features([train_df, test_df], 'user_id', 'click_times', 'sum', click_log)
    # 9：每个用户click_times平均数
    agg_features += get_agg_features([train_df, test_df], 'user_id', 'click_times', 'mean', click_log)
    # 10：每个用户click_times方差
    agg_features += get_agg_features([train_df, test_df], 'user_id', 'click_times', 'std', click_log)

    # train_df:
    #         user_id  age  ...  user_id_click_times_mean  user_id_click_times_std
    # 0             1    3  ...                  1.076923                 0.277350
    # 1             2    9  ...                  1.022222                 0.149071
    # 2             3    6  ...                  1.000000                 0.000000
    train_df[agg_features] = train_df[agg_features].fillna(-1)
    test_df[agg_features] = test_df[agg_features].fillna(-1)
    print("Extracting aggregate feature done!")
    print("List aggregate feature names:")
    # ['user_id__size',
    # 'user_id_ad_id_unique',
    # 'user_id_creative_id_unique',
    # 'user_id_advertiser_id_unique',
    # 'user_id_industry_unique',
    # 'user_id_product_id_unique',
    # 'user_id_time_unique',
    # 'user_id_click_times_sum',
    # 'user_id_click_times_mean',
    # 'user_id_click_times_std']
    print(agg_features)



    ################################################################################
    #---------------2：获取序列特征，用户点击的id序列---------------
    print("Extracting sequence feature...")
    text_features = []
    # 1：ad_id序列
    text_features += sequence_text([train_df, test_df], 'user_id', 'ad_id', click_log)
    # 2：creative_id序列
    text_features += sequence_text([train_df, test_df], 'user_id', 'creative_id', click_log)
    # 3：advertiser_id序列
    text_features += sequence_text([train_df, test_df], 'user_id', 'advertiser_id', click_log)
    # 4：product_id序列
    text_features += sequence_text([train_df, test_df], 'user_id', 'product_id', click_log)
    # 5：industry序列
    text_features += sequence_text([train_df, test_df], 'user_id', 'industry', click_log)
    # 6：product_category序列
    text_features += sequence_text([train_df, test_df], 'user_id', 'product_category', click_log)
    # 7: time序列
    text_features += sequence_text([train_df, test_df], 'user_id', 'time', click_log)
    # 8：click_times序列
    text_features += sequence_text([train_df, test_df], 'user_id', 'click_times', click_log)
    print("Extracting sequence feature done!")
    print("List sequence feature names:")   
    print(text_features)
    # ['sequence_text_user_id_ad_id',
    #  'sequence_text_user_id_creative_id',
    #  'sequence_text_user_id_advertiser_id',
    #  'sequence_text_user_id_product_id',
    #  'sequence_text_user_id_industry',
    #  'sequence_text_user_id_product_category',
    #  'sequence_text_user_id_time',
    #  'sequence_text_user_id_click_times']



    ################################################################################
    #-------------3:获取K折统计特征，求出用户点击的所有记录的年龄性别平均分布---------
    # 赋值index,训练集为0-4，测试集为5
    print("Extracting Kflod feature...")
    log = click_log.drop_duplicates(['user_id', 'creative_id']).reset_index(drop=True)
    del click_log
    gc.collect()
    log['cont'] = 1

    # 将数据分为01234分，加上fold特征
    train_df['fold'] = train_df.index % 5
    # 测试为第5分
    test_df['fold'] = 5

    # 将log上附加fold字段
    df = train_df.append(test_df)[['user_id', 'fold']].reset_index(drop=True)
    log = log.merge(df, on='user_id', how='left')
    del df
    gc.collect()

    # 获取用户点击某特征的年龄性别平均分布
    for pivot in ['creative_id', 'ad_id', 'product_id', 'advertiser_id', 'industry']:
        print("Kfold", pivot)
        kfold(train_df, test_df, log, pivot)
    del log
    gc.collect()       
    print("Extracting Kflod feature done!")

    a = train_df[['user_id','fold']].merge(test_df[['user_id','fold']],on='user_id',how='left')
    list(set(a['fold_y'].values))


    ################################################################################
    #-------------4:获取K折序列特征,求出用户点击的每一条记录的年龄性别分布
    # 赋值index,训练集为0-4，测试集为5
    print("Extracting Kflod sequence feature...")
    click_log = pd.read_pickle(root_path+'data/click.pkl').sample(frac=0.01)
    # click_log = pd.read_pickle('data/click.pkl')
    log = click_log.reset_index(drop=True)
    del click_log
    gc.collect()
    log['cont'] = 1
    train_df['fold'] = train_df.index % 5
    train_df['fold'] = train_df['fold'].astype(int)
    test_df['fold'] = 5
    df = train_df.append(test_df)[['user_id', 'fold']].reset_index(drop=True)
    log = log.merge(df, on='user_id', how='left')


    # 获取用户点击某特征的年龄性别分布序列
    kfold_sequence_features = []
    for pivot in ['creative_id', 'ad_id', 'product_id', 'advertiser_id', 'industry']:
        print("Kfold sequence", pivot)
        kfold_sequence_features += kfold_sequence(train_df, test_df, log, pivot)
    del log
    gc.collect()        
    print("Extracting Kfold sequence feature done!")
    print("List Kfold sequence feature names:")   
    print(kfold_sequence_features)

    ################################################################################
    print("Extract features done! saving data...")
    train_df.to_pickle('data/train_user.pkl')
    test_df.to_pickle('data/test_user.pkl')
    # train_df:
    # ['user_id', 'age', 'gender',
    #
    #        'user_id__size', 'user_id_ad_id_unique',
    #        'user_id_creative_id_unique', 'user_id_advertiser_id_unique',
    #        'user_id_industry_unique', 'user_id_product_id_unique',
    #        'user_id_time_unique', 'user_id_click_times_sum',
    #        'user_id_click_times_mean', 'user_id_click_times_std',

    #        'sequence_text_user_id_ad_id', 'sequence_text_user_id_creative_id',
    #        'sequence_text_user_id_advertiser_id',
    #        'sequence_text_user_id_product_id', 'sequence_text_user_id_industry',
    #        'sequence_text_user_id_product_category', 'sequence_text_user_id_time',
    #        'sequence_text_user_id_click_times',
    #
    #        'fold',
    #
    #        'age_0_creative_id_mean',
    #        'age_1_creative_id_mean', 'age_2_creative_id_mean',
    #        'age_3_creative_id_mean', 'age_4_creative_id_mean',
    #        'age_5_creative_id_mean', 'age_6_creative_id_mean',
    #        'age_7_creative_id_mean', 'age_8_creative_id_mean',
    #        'age_9_creative_id_mean', 'gender_0_creative_id_mean',
    #        'gender_1_creative_id_mean', 'age_0_ad_id_mean', 'age_1_ad_id_mean',
    #        'age_2_ad_id_mean', 'age_3_ad_id_mean', 'age_4_ad_id_mean',
    #        'age_5_ad_id_mean', 'age_6_ad_id_mean', 'age_7_ad_id_mean',
    #        'age_8_ad_id_mean', 'age_9_ad_id_mean', 'gender_0_ad_id_mean',
    #        'gender_1_ad_id_mean', 'age_0_product_id_mean', 'age_1_product_id_mean',
    #        'age_2_product_id_mean', 'age_3_product_id_mean',
    #        'age_4_product_id_mean', 'age_5_product_id_mean',
    #        'age_6_product_id_mean', 'age_7_product_id_mean',
    #        'age_8_product_id_mean', 'age_9_product_id_mean',
    #        'gender_0_product_id_mean', 'gender_1_product_id_mean',
    #        'age_0_advertiser_id_mean', 'age_1_advertiser_id_mean',
    #        'age_2_advertiser_id_mean', 'age_3_advertiser_id_mean',
    #        'age_4_advertiser_id_mean', 'age_5_advertiser_id_mean',
    #        'age_6_advertiser_id_mean', 'age_7_advertiser_id_mean',
    #        'age_8_advertiser_id_mean', 'age_9_advertiser_id_mean',
    #        'gender_0_advertiser_id_mean', 'gender_1_advertiser_id_mean',
    #        'age_0_industry_mean', 'age_1_industry_mean', 'age_2_industry_mean',
    #        'age_3_industry_mean', 'age_4_industry_mean', 'age_5_industry_mean',
    #        'age_6_industry_mean', 'age_7_industry_mean', 'age_8_industry_mean',
    #        'age_9_industry_mean', 'gender_0_industry_mean',
    #        'gender_1_industry_mean',
    #
    #        'sequence_text_user_id_creative_id_fold',
    #        'sequence_text_user_id_ad_id_fold',
    #        'sequence_text_user_id_product_id_fold',
    #        'sequence_text_user_id_advertiser_id_fold',
    #        'sequence_text_user_id_industry_fold']
