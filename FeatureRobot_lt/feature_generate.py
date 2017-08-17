#coding:utf-8
__author__='lt'

import numpy as np
import config as cf
import pandas as pd
import xgboost as xgb
import re
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

'''
使用LDA和GBDT生成新特征
'''

def feat_LDA(X,Y):
    model, score_1 = model_xgb(X,Y)
    X = np.nan_to_num(np.mat(X[1:,:],dtype=float))
    lda =LDA(n_components=10)
    lda.fit(X,Y)
    X_lda = lda.transform(X)
    X_head = np.mat(["LDA{}".format(i) for i in xrange(X_lda.shape[1])])
    X_trans = np.vstack((X_head,X_lda))
    X_final = np.hstack((X,X_trans))

    model,score_2 = model_xgb(X_final,Y)
    print "增加LDA特征后,效果提升%.2f%%" %(100*(score_2-score_1)/score_1)
    return X_trans

def feat_GBDT(X,Y):
    model,m_score = model_xgb(X,Y)
    #获取每棵数的叶子节点编号,并对所有叶子节点进行重新编号
    #---------------
    trees = model.get_dump()
    leafs = []
    index = 0
    for i in xrange(len(trees)):
        leaf = {}
        nodes = re.findall("(\d+):leaf",trees[i])
        for node in nodes:
            leaf[node] = index
            index =index+1
        leafs.append(leaf)
    #-----------若要进行0-1编码,这部分代码才有用

    #直接把模型输出的样本在每棵树的叶子节点编号作为特征,不进行0-1编码.可以节约空间
    sample_leaf = model.predict(xgb.DMatrix(X[1:,:]),pred_leaf = True)
    feat_new = np.mat(sample_leaf)
    head = ["GBDT{}".format(i) for i in xrange(feat_new.shape[1])]
    feat_new = np.vstack((head,feat_new))
    X_final = np.hstack((X,feat_new))
    model,m_score2 = model_xgb(X_final,Y)
    print "增加GBDT特征后,效果提升%.2f%%" % (100 * (m_score2 - m_score) / m_score)
    return feat_new

def model_xgb(X, Y):
    '''

    :param X: dataframe
    :param Y: series
    :return:
    '''
    X_df = pd.DataFrame(X[1:,:],columns=np.array(X[0,:])[0])
    train_X,val_X,train_Y,val_Y = train_test_split(X_df,Y,test_size=0.2)
    dtrain = xgb.DMatrix(train_X,label=train_Y)
    dval = xgb.DMatrix(val_X,label=val_Y)

    if cf.model_type=='lin':
        params = {
            'booster': 'gbtree',
            'objective': 'reg:linear',
            # 'scale_pos_weight': float(len(y)-sum(y))/float(sum(y)),
            'eval_metric': 'mae',
            'gamma': 0.01,
            'silent': 1,
            'max_depth': 5,
            'lambda': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.2,
            'min_child_weight': 0.1,
            'eta': 0.04,
            'seed': 3,
            'nthread': 5
        }

    if cf.model_type=='log':
        params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'scale_pos_weight': 484130 / 7183,
            'eval_metric': 'auc',
            'gamma': 0.01,
            'silent': 1,
            'max_depth': 5,
            'lambda': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.2,
            'min_child_weight': 0.1,
            'eta': 0.016,
            'nthread': 5
        }

    evals_result = {}
    watchlist = [(dtrain,'train'),(dval,'val')]
    xgb_model = xgb.train(params,dtrain,num_boost_round=1000,early_stopping_rounds=200,verbose_eval=50,evals=watchlist,evals_result=evals_result)

    result = 0.0
    if cf.model_type=='log':
        result = max(evals_result['val']['auc'])
    elif cf.model_type=='lin':
        result = min(evals_result['val']['mae'])
    # top_num = X.shape[1]*cf.topk_ratio
    # feat_weight = pd.DataFrame.from_dict(xgb_model.get_fscore(),orient='index')
    # feat_top = feat_weight.sort_values(by=0,ascending=False).head(top_num)
    return xgb_model,result

