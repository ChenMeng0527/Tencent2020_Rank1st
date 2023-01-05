# coding=utf-8

'''
pytporch数据处理---注释
最终ctrNet模型的数据处理
'''

import logging
import torch
import numpy as np
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class TextDataset(Dataset):
    def __init__(self, args, df):

        # 20个类别
        self.label = df['label'].values

        # 8个id的点击列表
        # array([['41136 129', '9391 23742'],
        #        ['-1 -1', '14668 9058'],
        #        ['-1 -1 1567', '7809 8385 16170'],
        #        ...,
        #        ['26858 -1', '31963 14176'],
        #        ['-1 21937', '6974 13005'],
        self.text_features = df[[x[1] for x in args.text_features]].values

        # 4个k_fold的点击列表
        # [['90964 552594 510184', '-6 -6 -6'],[],[],[],
        #        ['248943 146813 90583', '3193 63 63'],[],[],[],
        #        ['229182 13792', '3192 2612'],[],[],[],
        self.text_features_1 = df[[x[1] for x in args.text_features_1]].values


        # [[ 2.        ,  2.        ,  2.        , ...,  2.        , 1.        ,  0.        ],
        #  [ 2.        ,  2.        ,  2.        , ...,  2.        , 1.        ,  0.        ],
        #  [ 3.        ,  3.        ,  3.        , ...,  3.        , 1.        ,  0.        ],
        self.dense_features = df[args.dense_features].values

        # 两个embedding
        self.embeddings_tables = []
        for x in args.text_features:
            self.embeddings_tables.append(args.embeddings_tables[x[0]] if x[0] is not None else None)
        self.embeddings_tables_1 = []
        for x in args.text_features_1:
            self.embeddings_tables_1.append(args.embeddings_tables_1[x[0]] if x[0] is not None else None)            

        self.args = args
        self.df = df


    def __len__(self):
        return len(self.label)


    def __getitem__(self, i):  
        # 标签信息
        label = self.label[i]

        # --------------------------------------------
        # BERT的输入特征
        if len(self.args.text_features) == 0:
            text_features = 0
            text_masks = 0
            text_ids = 0

        else:

            # --------针对id行为：把text_features/text_masks/text_ids进行填补-----------

            # text_features = [max_len_text, 8*128]
            text_features = np.zeros((self.args.max_len_text, self.args.text_dim))
            # text_masks = [max_len_text]
            text_masks = np.zeros(self.args.max_len_text)
            # text_ids = [max_len_text,8]
            text_ids = np.zeros((self.args.max_len_text, len(self.args.text_features)), dtype=np.int64)

            begin_dim = 0

            for idx, (embed_table, x) in enumerate(zip(self.embeddings_tables, self.text_features[i])):
                # 128/128的往后加
                end_dim = begin_dim + self.args.text_features[idx][-1]
                # x为行为序列
                for w_idx, word in enumerate(x.split()[:self.args.max_len_text]):
                    # 把这个行为下的id对应的embedding写入text_features中
                    text_features[w_idx, begin_dim:end_dim] = embed_table[word]
                    text_masks[w_idx] = 1
                    try:
                        # 对应的id索引填入
                        text_ids[w_idx, idx] = self.args.vocab[self.args.text_features[idx][1], word]
                    except:
                        #
                        text_ids[w_idx, idx] = self.args.vocab['unk']
                begin_dim = end_dim



        # # --------针对fold_id行为：把text_features/text_masks进行填补(没有text_ids)-----------
        # decoder的输入特征
        if len(self.args.text_features_1) == 0:
            text_features_1 = 0
            text_masks_1 = 0

        else:
            # text_features_1:[max_len_text,4*12]
            text_features_1 = np.zeros((self.args.max_len_text, self.args.text_dim_1))
            # text_masks_1:[max_len_text]
            text_masks_1 = np.zeros(self.args.max_len_text)

            begin_dim = 0
            # 遍历每个embedding ,及对应的行为序列
            for idx, (embed_table, x) in enumerate(zip(self.embeddings_tables_1, self.text_features_1[i])):
                end_dim = begin_dim + self.args.text_features_1[idx][-1]

                # 如果有预训练的embedding
                if embed_table is not None:
                    for w_idx, word in enumerate(x.split()[:self.args.max_len_text]):
                        text_features_1[w_idx, begin_dim:end_dim] = embed_table[word]
                        text_masks_1[w_idx] = 1

                # 如果没有预训练的embedding
                # embedding中放入的为行为的id
                else:
                    # 行为序列
                    for w_idx, v in enumerate(x[:self.args.max_len_text]):
                        text_features_1[w_idx, begin_dim:end_dim] = v
                        text_masks_1[w_idx] = 1
                begin_dim = end_dim


        # 浮点数特征
        if len(self.args.dense_features) == 0:
            dense_features = 0
        else:
            dense_features = self.dense_features[i]

        return (
                # 20分类的label
                torch.tensor(label),

                # 10个统计特征+target统计特征
                torch.tensor(dense_features),

                # 8个行为序列用embedding进行填满[max_len_text, 8*128]
                torch.tensor(text_features),
                # 8个行为序列用vocab进行填满[max_len_text,8]
                torch.tensor(text_ids),
                # 8个行为序列用mask进行填满[max_len_text]
                torch.tensor(text_masks),

                # 4个合并id序列[max_len_text,4*12]
                torch.tensor(text_features_1),
                # [max_len_text]
                torch.tensor(text_masks_1),            
               )



