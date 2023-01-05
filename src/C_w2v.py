# coding=utf-8
'''

'''
import os
import pandas as pd
from gensim.models import Word2Vec
import pickle
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def w2v(dfs, f, L=128):
    print("w2v", f)
    sentences = []
    for df in dfs:
        for line in df[f].values:
            try:
                sentences.append(line.split())
            except:
                pass
    print("Sentence Num {}".format(len(sentences)))
    w2v = Word2Vec(sentences, size=L, window=8, min_count=1, sg=1, workers=32, iter=10)
    print("save w2v to {}".format(os.path.join('data', f+".{}d".format(L))))
    pickle.dump(w2v, open(os.path.join('data', f+".{}d".format(L)), 'wb'))


if __name__ == "__main__":
    train_df = pd.read_pickle('data/train_user.pkl')
    test_df = pd.read_pickle('data/test_user.pkl')
    # ['user_id', 'age', 'gender', 'user_id__size', 'user_id_ad_id_unique',
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
    #        'sequence_text_user_id_industry_fold'],


    # 训练word2vector，维度为128
    w2v([train_df, test_df], 'sequence_text_user_id_ad_id', L=128)
    w2v([train_df, test_df], 'sequence_text_user_id_creative_id', L=128)
    w2v([train_df, test_df], 'sequence_text_user_id_advertiser_id', L=128)
    w2v([train_df, test_df], 'sequence_text_user_id_product_id', L=128)
    w2v([train_df, test_df], 'sequence_text_user_id_industry', L=128)
    w2v([train_df, test_df], 'sequence_text_user_id_product_category', L=128)

    w2v([train_df, test_df], 'sequence_text_user_id_time', L=128)
    w2v([train_df, test_df], 'sequence_text_user_id_click_times', L=128)



