#coding:utf-8
__author__='lt'

import os
import gc
import pickle
import random
import xgboost as xgb
import numpy as np
import pandas as pd
import config as cf
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

'''
使用xgb模型或者IV值的方式迭代的加载变换和交叉后的特征进行特征排序筛选
'''

def Feat_Filter(y):
    # f_y = file('./y.pkl', 'rb')
    # y=pickle.load(f_y)
    #os.remove('./y.pkl')

    path = os.path.abspath(cf.path) + '/pickledata'
    fileNames = os.listdir(path)
    file_num = len(fileNames)

    start = 0
    feat_sub = 0
    #分批从磁盘中加载特征到内存,采用模型或者IV值对交换特征进行排序,并留下topN个特征与下一次加载的特征进行对比
    while(True):
        if file_num>30:
            for i in xrange(start*30,(start+1)*30):
                f = file(path+'/'+fileNames[i],'rb')
                tmp = pickle.load(f)
                try:
                    feat_sub = np.hstack((feat_sub,tmp))
                except:
                    feat_sub = tmp
                f.close()
                os.remove(path+'/'+fileNames[i])
        else:
            for i in xrange(start*30, len(fileNames)):
                # print fileNames[i]
                f = file(path + '/' + fileNames[i], 'rb')
                tmp = pickle.load(f)
                try:
                    feat_sub = np.hstack((feat_sub, tmp))
                except:
                    feat_sub = tmp
                f.close()
                os.remove(path + '/' + fileNames[i])
        print "===============第{}次特征加载完毕,共有特征{}个===============".format(start+1,feat_sub.shape[1])

        if cf.filter_type=='model':
            feat_sub = model_filter(feat_sub,y)
            print "===============第{}次模型筛选特征完毕,保留前{}个特征===========".format(start+1,feat_sub.shape[1])
        elif cf.filter_type=='IV':
            if cf.model_type=='lin':
                print "回归问题无法用IV对特征进行筛选,默认采用模型筛选"
                feat_sub = model_filter(feat_sub,y)
                print "===============第{}次模型筛选特征完毕,保留前{}个特征===========".format(start + 1, feat_sub.shape[1])
            else:
                feat_sub = IV_filter(feat_sub,y)
                print "===============第{}次IV值筛选特征完毕,保留前{}个特征===========".format(start + 1, feat_sub.shape[1])

        start = start+1
        file_num = file_num-30
        if file_num <=0:
            break

    os.rmdir(path)
    # gc.collect()
    return feat_sub

def model_filter(X,Y):
    '''
    模型筛选特征
    :param X:
    :param Y:
    :return:
    '''

    df_X = pd.DataFrame(np.mat(X[1:,:], dtype=float))
    df_Y = Y
    train_X, val_X, train_Y, val_Y = train_test_split(df_X, df_Y, test_size=0.2, random_state=3)
    dval = xgb.DMatrix(val_X, label=val_Y)
    dtrain = xgb.DMatrix(train_X, label=train_Y)

    #模型参数
    random_seed = range(10, 200, 10)
    gamma = [i/100.0 for i in xrange(1,11,1)]
    max_depth = [4,5,6, 7, 8]
    lambd = [0.1, 1, 2,0.01,0.05]
    subsample = [i/10.0 for i in range(5, 10, 1)]
    colsample_bytree = [i/100.0 for i in range(20, 45, 5)]
    min_child_weight = [i for i in range(1, 10, 1)]
    random.shuffle(random_seed)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)

    global feat_trian
    feat_trian = 0
    for i in range(cf.xgb_num):
        model_train(random_seed[i%10], gamma[i%len(gamma)], max_depth[i % len(max_depth)],
                    lambd[i % len(lambd)],subsample[i%len(subsample)],colsample_bytree[i% len(colsample_bytree)],
                    min_child_weight[i % len(min_child_weight)], dtrain, dval)

    try:
        #模型筛选的特征数
        head_feat = int(X.shape[1]*cf.topk_ratio)
        f_top = feat_trian.sort_values(by='weight', ascending=False).head(head_feat)
        df = pd.DataFrame(X)
        df.columns = [int(col) for col in df.columns]
        f_top.index = [int(col2) for col2 in f_top.index]
        f_top_df = df.loc[:, list(f_top.index)]
        f_top_mat = np.mat(f_top_df)
        return f_top_mat
    except Exception as e:
        print e
        return np.mat([])


def model_train(random_seed, gamma, max_depth, lambd,subsample,colsample_bytree,min_child_weight, dtrain, dval):

    if cf.model_type=='lin':
        params = {
            'booster': 'gbtree',
            'objective': 'reg:linear',
            # 'scale_pos_weight': float(len(y)-sum(y))/float(sum(y)),
            'eval_metric': 'mae',
            'gamma': gamma,
            'silent': 1,
            'max_depth': max_depth,
            'lambda': lambd,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'eta': 0.04,
            'seed': random_seed,
            'nthread': 5
        }

    if cf.model_type=='log':
        params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'scale_pos_weight': 484130 / 7183,
            'eval_metric': 'auc',
            'gamma': gamma,
            'silent': 1,
            'max_depth': max_depth,
            'lambda': lambd,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'eta': 0.016,
            'nthread': 5
        }

    watchlist = [(dtrain,'train'),(dval,'val')]
    xgb_model = xgb.train(params,dtrain,num_boost_round=1000,early_stopping_rounds=200,verbose_eval=50,evals=watchlist,evals_result={})

    feat_r = pd.DataFrame.from_dict(xgb_model.get_score(importance_type='gain'),orient='index')
    feat_r.columns = ['weight']
    try:
        feat_trian = feat_trian + feat_r
    except:
        global feat_trian
        feat_trian = feat_r

def IV_filter(X,Y):

    #判断label是连续型还是二分类
    label = pd.Series(np.array(Y.T)[0])
    if (label.unique().size>2):
        label = label.apply(lambda x: '1' if x>cf.label_thr else '0')
    Y = np.vstack((np.mat(['label']),np.mat(label).T))
    data_mat = np.hstack((X,Y))
    data_df = pd.DataFrame(data_mat[1:,:],columns=data_mat[0,:])

    iv_list = []
    for col in data_df.columns[:-1]:
        iv_list.append(calc_woe_iv(data_df.loc[:,[col,'label']],col))
    iv_df = pd.DataFrame(iv_list)
    head_feat = int(X.shape[1] * cf.topk_ratio)
    iv_top = iv_df.sort_values(by=0,ascending=False).head(head_feat)
    feat_top = X[:,list(iv_top.index)]

    return feat_top

def calc_woe_iv(df_col,col_name):

    df_sub = df_col.copy()
    #默认特征都为连续数值型
    total_good = len(df_sub[df_sub.label=='0'])
    total_bad = len(df_sub[df_sub.label=='1'])
    total_count = df_sub.shape[0]
    rank_col = 'rk_'+col_name
    #分箱
    df_sub[rank_col] = df_sub[col_name].rank(method='max')/(total_count/cf.bin_num)
    df_sub.fillna(-999,inplace=True)
    df_sub[rank_col] = df_sub[rank_col].apply(lambda x: int(x) if x>0 else -1)
    #统计分组
    grouping_data = []
    for gvar,gdata in df_sub.groupby(rank_col):
        g_info ={}
        g_info['g_name'] = gvar
        g_info['good_num'] = len(gdata[gdata.label=='0'])
        g_info['good_ratio'] =(1.0*g_info['good_num']/total_good)
        g_info['bad_num'] = len(gdata[gdata.label == '1'])
        g_info['bad_ratio'] = (1.0*g_info['bad_num']/total_bad)
        if g_info['good_num']>0 and g_info['bad_num']>0:
            g_info['woe'] = np.math.log(1.0*g_info['good_ratio']/g_info['bad_ratio'])
        elif g_info['good_num']==0:
            g_info['woe'] = -1
        else:
            g_info['woe']=1
        g_info['iv'] = 1.0*(g_info['good_ratio']-g_info['bad_ratio']) * g_info['woe']
        grouping_data.append(g_info)

    g_columns = ['g_name','good_num','good_ratio','bad_num','bad_ratio','woe','iv']
    g_df = pd.DataFrame(grouping_data,columns=g_columns)
    iv_sum = g_df.iv.sum()

    return iv_sum



if __name__=='__main__':
    r =  Feat_Filter()





