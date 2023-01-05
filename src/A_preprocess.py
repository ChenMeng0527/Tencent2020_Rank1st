# coding=utf-8
'''
代码注释，将训练测试数据进行合并，最大数据集
@20221028
'''

import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/youshu_/Python_Workspace/Tencent2020_Rank1st')
root_path = '/Users/youshu_/Python_Workspace/Tencent2020_Rank1st/'

def merge_files():
    '''
    合并点击记录
    '''

    # -----点击数据合并------
    print("merge click files...")
    # [30082771 rows x 4 columns]
    click_df = pd.read_csv(root_path+"data/train_preliminary/click_log.csv")
    # 可以去掉减少数据
    # click_df=click_df.append(pd.read_csv("data/train_semi_final/click_log.csv"))
    # [63668283 rows x 4 columns]
    click_df = click_df.append(pd.read_csv(root_path+"data/test/click_log.csv"))
    click_df = click_df.sort_values(by=["time"]).drop_duplicates()


    # -----合并广告信息------
    print("merge ad files...")
    ad_df = pd.read_csv(root_path+"data/train_preliminary/ad.csv")
    # 可以去掉减少数据
    # ad_df=ad_df.append(pd.read_csv("data/train_semi_final/ad.csv"))
    ad_df = ad_df.append(pd.read_csv(root_path+"data/test/ad.csv"))
    # [3412772 rows x 6 columns]
    ad_df = ad_df.drop_duplicates()
    

    # -----合并用户信息------
    print("merge user files...")
    # [900000 rows x 3 columns]
    train_user = pd.read_csv(root_path+"data/train_preliminary/user.csv")
    # 可以去掉减少数据
    # train_user=train_user.append(pd.read_csv("data/train_semi_final/user.csv"))
    train_user = train_user.reset_index(drop=True)
    train_user['age'] = train_user['age']-1
    train_user['gender'] = train_user['gender']-1

    # 将测试点击数据的用户去重复，user_id取出来，label都设置为-1
    # [33585512 rows x 4 columns] 去重user_id后 [1000000 rows x 4 columns]
    test_user = pd.read_csv(root_path+"data/test/click_log.csv").drop_duplicates('user_id')[['user_id']].reset_index(drop=True)
    test_user = test_user.sort_values(by='user_id').reset_index(drop=True)
    test_user['age'] =- 1
    test_user['gender'] =- 1


    # 合并点击，广告，用户信息,并添加label-onehot列
    print("merge all files...")
    # [63668283 rows x 11 columns]
    click_df = click_df.merge(ad_df, on="creative_id", how='left')
    click_df = click_df.merge(train_user, on="user_id", how='left')
    click_df = click_df.fillna(-1)
    click_df = click_df.replace("\\N", -1)
    for f in click_df:
        click_df[f] = click_df[f].astype(int)
    for i in range(10):
        click_df['age_{}'.format(i)] = (click_df['age']==i).astype(np.int16)
    for i in range(2):
        click_df['gender_{}'.format(i)] = (click_df['gender']==i).astype(np.int16)
    
    
    return click_df, train_user, test_user


if __name__ == "__main__":
    # click_df：
    #           time  user_id  creative_id  ...  age_9  gender_0  gender_1
    # 0            1  3915646        22854  ...      0         0         0
    # 1            1  3920282        22182  ...      0         0         0
    #
    # train_user：
    #         user_id  age  gender
    # 0             1    3       0
    # 1             2    9       0
    # 2             3    6       1
    # 3             4    4       0
    #
    # test_user：
    #         user_id  age  gender
    # 0       3000001   -1      -1
    # 1       3000002   -1      -1
    # 2       3000003   -1      -1

    click_df, train_user, test_user = merge_files()
    # 保存预处理文件
    print("preprocess done! saving data...")
    click_df.to_pickle(root_path + "data/click.pkl")
    train_user.to_pickle(root_path + "data/train_user.pkl")
    test_user.to_pickle(root_path + "data/test_user.pkl")

import pickle
print(pd.read_pickle(root_path + "data/click.pkl"))