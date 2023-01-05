import os
import gc
import torch
import logging
import argparse
import models.ctrNet as ctrNet
import pickle
import gensim
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.D_data_loader import TextDataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
base_path="data"



# ------1:定义浮点数特征------
# 统计特征
dense_features=['user_id__size',

                'user_id_ad_id_unique',
                'user_id_creative_id_unique',
                'user_id_advertiser_id_unique',
                'user_id_industry_unique',
                'user_id_product_id_unique',
                'user_id_time_unique',

                'user_id_click_times_sum',
                'user_id_click_times_mean',
                'user_id_click_times_std']


# ------2:kfold特征------
for l in ['age_{}'.format(i) for i in range(10)]+['gender_{}'.format(i) for i in range(2)]:
    for f in ['creative_id', 'ad_id', 'product_id', 'advertiser_id', 'industry']:
        dense_features.append(l+'_'+f+'_mean')

# ['user_id__size',
#  'user_id_ad_id_unique',
#  'user_id_creative_id_unique',
#  'user_id_advertiser_id_unique',
#  'user_id_industry_unique',
#  'user_id_product_id_unique',
#  'user_id_time_unique',
#  'user_id_click_times_sum',
#  'user_id_click_times_mean',
#  'user_id_click_times_std',

#  'age_0_creative_id_mean',
#  'age_0_ad_id_mean',
#  'age_0_product_id_mean',
#  'age_0_advertiser_id_mean',
#  'age_0_industry_mean',
#  'age_1_creative_id_mean',
#  'age_1_ad_id_mean',
#  'age_1_product_id_mean',
#  'age_1_advertiser_id_mean',
#  'age_1_industry_mean',
#  'age_2_creative_id_mean',
#  'age_2_ad_id_mean',
#  'age_2_product_id_mean',
#  'age_2_advertiser_id_mean',
#  'age_2_industry_mean',
#  'age_3_creative_id_mean',
#  'age_3_ad_id_mean',
#  'age_3_product_id_mean',
#  'age_3_advertiser_id_mean',
#  'age_3_industry_mean',
#  'age_4_creative_id_mean',
#  'age_4_ad_id_mean',
#  'age_4_product_id_mean',
#  'age_4_advertiser_id_mean',
#  'age_4_industry_mean',
#  'age_5_creative_id_mean',
#  'age_5_ad_id_mean',
#  'age_5_product_id_mean',
#  'age_5_advertiser_id_mean',
#  'age_5_industry_mean',
#  'age_6_creative_id_mean',
#  'age_6_ad_id_mean',
#  'age_6_product_id_mean',
#  'age_6_advertiser_id_mean',
#  'age_6_industry_mean',
#  'age_7_creative_id_mean',
#  'age_7_ad_id_mean',
#  'age_7_product_id_mean',
#  'age_7_advertiser_id_mean',
#  'age_7_industry_mean',
#  'age_8_creative_id_mean',
#  'age_8_ad_id_mean',
#  'age_8_product_id_mean',
#  'age_8_advertiser_id_mean',
#  'age_8_industry_mean',
#  'age_9_creative_id_mean',
#  'age_9_ad_id_mean',
#  'age_9_product_id_mean',
#  'age_9_advertiser_id_mean',
#  'age_9_industry_mean',
#  'gender_0_creative_id_mean',
#  'gender_0_ad_id_mean',
#  'gender_0_product_id_mean',
#  'gender_0_advertiser_id_mean',
#  'gender_0_industry_mean',
#  'gender_1_creative_id_mean',
#  'gender_1_ad_id_mean',
#  'gender_1_product_id_mean',
#  'gender_1_advertiser_id_mean',
#  'gender_1_industry_mean']



# ------3:定义用户点击的序列特征------
text_features = [
                [base_path+"/sequence_text_user_id_product_id.128d", 'sequence_text_user_id_product_id', 128],
                [base_path+"/sequence_text_user_id_ad_id.128d", 'sequence_text_user_id_ad_id', 128],
                [base_path+"/sequence_text_user_id_creative_id.128d", 'sequence_text_user_id_creative_id', 128],
                [base_path+"/sequence_text_user_id_advertiser_id.128d", 'sequence_text_user_id_advertiser_id', 128],
                [base_path+"/sequence_text_user_id_industry.128d", 'sequence_text_user_id_industry', 128],
                [base_path+"/sequence_text_user_id_product_category.128d", 'sequence_text_user_id_product_category', 128],
                [base_path+"/sequence_text_user_id_time.128d", 'sequence_text_user_id_time', 128],
                [base_path+"/sequence_text_user_id_click_times.128d", 'sequence_text_user_id_click_times', 128],
                ]

# ------4:定义用户点击的人工构造序列特征------
text_features_1 = [
                    [base_path+"/sequence_text_user_id_creative_id_fold.12d", 'sequence_text_user_id_creative_id_fold', 12],
                    [base_path+"/sequence_text_user_id_ad_id_fold.12d", 'sequence_text_user_id_ad_id_fold', 12],
                    [base_path+"/sequence_text_user_id_product_id_fold.12d", 'sequence_text_user_id_product_id_fold', 12],
                    [base_path+"/sequence_text_user_id_advertiser_id_fold.12d", 'sequence_text_user_id_advertiser_id_fold', 12],
                    [base_path+"/sequence_text_user_id_industry_fold.12d", 'sequence_text_user_id_industry_fold', 12],
                    ]


if __name__ == "__main__":

    # ----------------------logging----------------------
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


    # ----------------------设置参数----------------------
    # 1：变换参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--index', type=int, default=0) # savemodel文件夹编号
    parser.add_argument('--train_batch_size', type=int, default=512)
    parser.add_argument('--max_len_text', type=int, default=128)
    parser.add_argument('--num_hidden_layers', type=int, default=6)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.2)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--vocab_size_v1', type=int, default=500000)
    parser.add_argument('--vocab_dim_v1', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--display_steps', type=int, default=100)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--eval_batch_size', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--num_label', type=int, default=20)
    args = parser.parse_args()
    
    # 2：写死参数
    # 8个行为序列总维度：8*128
    args.hidden_size = sum([x[-1] for x in text_features])
    logger.info("Argument %s", args)

    # 预训练模型vocab
    args.vocab = pickle.load(open(os.path.join(args.pretrained_model_path, "vocab.pkl"), 'rb'))

    # vocab个数
    args.vocab_size_v1 = len(args.vocab)

    # 128维度--行为4维度数组
    args.text_features = text_features
    # 8*128
    args.text_dim = sum([x[-1] for x in text_features])

    # 12维度--行为4维度数组
    args.text_features_1 = text_features_1
    # 4*12
    args.text_dim_1 = sum([x[-1] for x in text_features_1])

    # 稠密特征列
    args.dense_features = dense_features

    args.linear_layer_size = [1024, 512]


    args.output_dir = "saved_models/index_{}".format(args.index)


    # 3：读取word2vector模型
    args.embeddings_tables={}
    for x in args.text_features:
        if x[0] not in args.embeddings_tables:
            try:
                # gensim读取
                args.embeddings_tables[x[0]] = gensim.models.KeyedVectors.load_word2vec_format(x[0], binary=False)
            except:
                # pickle读取
                args.embeddings_tables[x[0]] = pickle.load(open(x[0], 'rb'))

    args.embeddings_tables_1={}
    for x in args.text_features_1:
        if x[0] not in args.embeddings_tables_1:
            try:
                args.embeddings_tables_1[x[0]] = gensim.models.KeyedVectors.load_word2vec_format(x[0], binary=False)
            except:
                args.embeddings_tables_1[x[0]] = pickle.load(open(x[0], 'rb'))



    #-------------------label设置为age*2 + gender，20个   + 稠密统计特征进行归一化-------------------
    # label = age*2 + gender
    train_df = pd.read_pickle('data/train_user.pkl')
    train_df['label'] = train_df['age']*2 + train_df['gender']
    test_df = pd.read_pickle('data/test_user.pkl')
    test_df['label'] = test_df['age']*2 + test_df['gender']

    # 特征进行归一化
    df = train_df[args.dense_features].append(test_df[args.dense_features])
    ss = StandardScaler()
    ss.fit(df[args.dense_features])
    train_df[args.dense_features] = ss.transform(train_df[args.dense_features])
    test_df[args.dense_features] = ss.transform(test_df[args.dense_features])

    test_dataset = TextDataset(args, test_df)



    #-------------------建立模型------------
    skf = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    model = ctrNet.ctrNet(args)
    # 训练模型
    # 将训练集分为训练/验证
    for i, (train_index, test_index) in enumerate(skf.split(train_df, train_df['label'])):
        if i != args.index:
            continue
        logger.info("Index: %s", args.index)
        train_dataset = TextDataset(args, train_df.iloc[train_index])
        dev_dataset = TextDataset(args, train_df.iloc[test_index])
        model.train(train_dataset, dev_dataset)
        dev_df = train_df.iloc[test_index]
    
    # 输出结果
    accs=[]
    for f,num in [('age',10),('gender',2)]:
        # 加载模型
        model.reload(f)
        if f=="age":
            dev_preds = model.infer(dev_dataset)[0]
        else:
            dev_preds = model.infer(dev_dataset)[1]
        for j in range(num):
            dev_df['{}_{}'.format(f,j)] = np.round(dev_preds[:,j],4)
        acc = model.eval(dev_df[f].values, dev_preds)['eval_acc']
        accs.append(acc)
        if f=="age":
            test_preds=model.infer(test_dataset)[0]
        else:
            test_preds=model.infer(test_dataset)[1]

        logger.info("Test %s %s",f,np.mean(test_preds,0))
        logger.info("ACC %s %s",f,round(acc,5))

        out_fs = ['user_id','age','gender','predict_{}'.format(f)]
        out_fs += ['{}_{}'.format(f,i) for i in range(num)]
        for i in range(num):
            test_df['{}_{}'.format(f,i)]=np.round(test_preds[:,i],4)
        test_df['predict_{}'.format(f)]=np.argmax(test_preds,-1)+1
        try:
            os.system("mkdir submission")
        except:
            pass

        test_df[out_fs].to_csv('submission/submission_test_{}_{}_{}.csv'.format(f,args.index,round(acc,5)),index=False)
        
    logger.info("  best_acc = %s",round(sum(accs),4))