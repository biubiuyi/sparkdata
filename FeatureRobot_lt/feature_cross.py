#coding:utf-8
__author__='lt'

import os
import config as cf
import pandas as pd
from multiprocessing import Pool
import numpy as np
import minepy
import pickle
from model_disply import model_See
from feature_filter import Feat_Filter
from feature_generate import *
import warnings
warnings.filterwarnings("ignore")

'''
包括特征预处理和特征变换交叉
'''

#===================特征预处理:离散特征数值化,特征标准化====================
def dummy_data(dummy_col, df):
    if cf.dummy_type == "dummy":  # one-hot 编码
        df_new = pd.DataFrame()
        for col in dummy_col:
            df_new = df_new.concat(pd.get_dummies(df[col]))
        df = df.drop(dummy_col, axis=1)
        df = df.concat(df_new)
    if cf.dummy_type == "factorize":  # 特征属性值枚举
        for col in dummy_col:
            df[col] = pd.factorize(df.col)[0]
    return df

def normalize(norma_col, df):
    df_sub = df.loc[:,norma_col]
    df_sub = df_sub.apply(lambda x:(x-np.min(x)) / (np.max(x) - np.min(x)))
    df = df.drop(norma_col,axis=1)
    df =df.concat(df_sub)
    return df

# ==============================特征筛选:空值率,方差,相关性=================================
def null_filter(df):
    print "1.过滤空值较多的特征列"
    null_num = df.shape[0] * cf.null_ratio
    for col in df.columns:
        if sum(df[col].isnull()) > null_num:
            df.drop(col, axis=1, inplace=True)
            # 也可以过滤空值较多的行
    return df

def std_filter(df):
    print "2.过滤方差较小的特征列"
    for col in df.columns:
        try:
            std_c = np.nanstd(df[col])
            if std_c < cf.std_num:
                df.drop(col, axis=1, inplace=True)
        except:
            pass

def corr_func(X1, X2, corr_type=None):
    if corr_type == None:
        corr_type = cf.corr_type

    X1 = pd.Series(np.array(X1.reshape(-1, 1)).T[0])
    X2 = pd.Series(np.array(X2.reshape(-1, 1)).T[0])
    if corr_type == 'MIC':
        mine = minepy.MINE(alpha=0.6, c=15, est="mic_approx")
        mine.compute_score(X1, X2)
        corr = mine.mic()
    if corr_type == 'pearson':
        corr = X1.corr(X2)
    if corr_type == 'spearman':
        corr = X1.corr(X2, method="spearman")
    if corr_type == 'kendall':
        corr = X1.corr(X2, method="kendall")
    return abs(corr)

# ==============================特征变换和交叉,剔除高相关性特征=================================
def f2f_remove(feature_c, feat_other=None):
    global feature_final
    feature_final = 0

    block = cf.block
    pool = Pool(processes=cf.process_num)
    if feature_c.shape[1] < block*2:
        r = pool.apply_async(remove_feature, (feature_c, feature_c, 0, 0), callback=pickle_data2)
    else:
        block_size = feature_c.shape[1] / block
        for i in xrange(block):
            if i == block - 1:
                r = pool.apply_async(remove_feature, (feature_c[:, block_size * i:], feature_c, i, block_size),
                                     callback=pickle_data2)
            else:
                r = pool.apply_async(remove_feature,
                                     (feature_c[:, block_size * i:block_size * (i + 1)], feature_c, i, block_size),
                                     callback=pickle_data2)
    pool.close()
    pool.join()

    try:
        if r.successful() and feature_final.shape[1]>0:
            # print 'f2f_removed feature shape:', feature_final.shape
            return feature_final
    except:
        return np.mat([])


def remove_feature(f_sub, f_all, index, block_size):
    n = 0
    start = index * block_size
    for i in xrange(f_sub.shape[1]):
        for j in xrange(start + i + 1, f_all.shape[1]):
            x1 = np.mat(f_sub[1:, i], dtype=float)
            x2 = np.mat(f_all[1:, j], dtype=float)
            if corr_func(x1, x2, ) >= cf.corr_f2f:
                break
            try:
                n = np.hstack((n, f_sub[:, i]))
            except:
                n = f_sub[:, i]
    return n

# 中间数据合并
def pickle_data2(result):
    try:
        result.shape
        try:
            global feature_final
            feature_final = np.hstack((feature_final, result))
        except:  # 合并之前feature_final还未初始化为矩阵
            feature_final = result
    except:  # 返回值不是矩阵
        pass

# 中间数据序列化到磁盘
def pickle_data1(cross_feat):
    if cross_feat.shape[1] > 1: #还有符合条件的交叉特征存在且大于1
        block = cross_feat[0, 0]
        feature_c = cross_feat[:, 1:]  # 去掉第一列块号

        global operater
        sts=operater
        if operater == "/":
            sts = "d"
        elif operater == "1/x":
            sts = "l"
        elif operater == "1/x+1/y":
            sts = "m"
        fileName = path_tmp + "/" + str(block) + sts + ".pkl"
        # 判断该文件是否已经存在
        while (True):
            if os.path.exists(fileName):
                sts = sts + "1"
                fileName = path_tmp + "/" + str(block) + str(sts) + ".pkl"
            else:
                break

        if feature_c.shape[1]>1:
            # 去除特征交叉后高相关的特征
            feature_c = f2f_remove(feature_c)
        if feature_c.shape[1] > 0:
            f = file(fileName, "wb")
            pickle.dump(feature_c, f)
            f.close()
    else:
        pass

def feature_cross(X, op):
    global operater
    operater = op
    print operater

    # 设置numpy计算错误的处理类型,除计算操作出现错误时,抛出异常
    old_err_state = np.seterr(divide='raise')
    ignored_states = np.seterr(**old_err_state)  # 除以0为0

    block = cf.block
    pool = Pool(processes=cf.process_num)
    if X.shape[1] < block*2:
        result = pool.apply_async(cross_fun, (X, op, 0, 0), callback=pickle_data1)
    else:
        size = X.shape[1] / block
        for index in xrange(block):
            result = pool.apply_async(cross_fun, (X, op, index, size), callback=pickle_data1)
    pool.close()
    pool.join()
    if result.successful():
        print "feature cross successfully"
    else:
        print "feature cross failed"


def cross_fun(X, op, index, size):
    if index == cf.block - 1:
        X_sub = X[:, index * size:]
    else:
        X_sub = X[:, index * size:(index + 1) * size]
    start = index * size
    cross_feat = np.ones((X_sub.shape[0], 1)) * index  # 交叉特征的第一列为块序号

    if op in ["1/x", "log", "fft"]:
        for col in xrange(X_sub.shape[1]):
            cross_feat = cross_cal(X_sub[:, col], X_sub[:, col], op, cross_feat)
    else:
        for col1 in xrange(X_sub.shape[1]):
            for col2 in xrange(start + col1 + 1, X.shape[1]):
                cross_feat = cross_cal(X_sub[:, col1], X[:, col2], op, cross_feat)

    return cross_feat


def cross_cal(X1, X2, op, cross_feat):
    try:
        x1 = np.mat(X1[1:, :], dtype='float')
        x2 = np.mat(X2[1:, :], dtype='float')
        h1 = str(X1[0, 0])
        h2 = str(X2[0, 0])

        if op == "*":
            a = np.multiply(x1, x2)  # x*y
            head = np.mat(["{} * {}".format(h1, h2)])  # 保存header
        if op == "+":
            a = x1 + x2  # x+y
            head = np.mat(["{} + {}".format(h1, h2)])  # 保存header
        if op == "-":
            a = x1 - x2  # x-y
            head = np.mat(["{} - {}".format(h1, h2)])  # 保存header
        if op == "/":
            with np.errstate(divide='ignore', invalid='ignore'):
                a = np.true_divide(x1, x2)  # x/y
                a[a == np.inf] = -999
                head = np.mat(["{}/{}".format(h1, h2)])  # 保存header
        if op == "1/x":
            oney = np.mat(np.ones(X1.shape[0]-1)).T
            with np.errstate(divide='ignore', invalid='ignore'):  # 1/x
                a = np.true_divide(oney, x1)
                a[a == np.inf] = -999
            head = np.mat(["1/{}".format(h1)])  # 保存header
        if op == "1/x+1/y":  # 1/x+1/y
            oney = np.mat(np.ones(X1.shape[0]-1)).T
            with np.errstate(divide='ignore', invalid='ignore'):
                a = np.true_divide(oney, x1) + np.true_divide(oney, x2)
                a[a == np.inf] = -999
            head = np.mat(["1/{} + 1/{}".format(h1, h2)])  # 保存header
        if op == "x^2+y^2":  # x^2+y^2
            a = np.multiply(x1, x1) + np.multiply(x2, x2)
            head = np.mat(["{}^2 + {}^2".format(h1, h2)])
        if op == "log":
            a = np.log(x1)
            a[a == -np.inf] = 0
            head = np.mat([" log({})".format(h1)])
        if op=="fft":
            a = np.nan_to_num(x1)
            a = np.fft.fft(a)
            head = np.mat([" fft({})".format(h1)])

        # 变换后特征与原特征相关性比较
        corr2label = corr_func(a, label)
        corr2x1 = corr_func(a, x1)
        x12label = corr_func(x1, label)
        if corr2x1 < cf.corr_f2f and corr2label > x12label and corr2label > cf.corr_f2label:
            a = np.vstack((head, a))  # 合并列名
            cross_feat = np.hstack((cross_feat, a))  # 与原特征合并
    except Exception as e:
        print "cross fail:{} and {} comput {}!".format(h1, h2, op)
        print e
        return cross_feat
    return cross_feat

# ==============================特征机器人主程序=================================
def feature_robot(X_df, y, dummy_col=[],norma_col = []):
    global label  # 样本标签
    label = y

    # 新建缓存数据文件夹
    global path_tmp
    path_tmp = os.path.abspath(cf.path) + '/pickledata'
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)

    if len(dummy_col) > 0:
        print "特征预处理中,请稍后....."
        X_df = dummy_data(dummy_col, X_df)

    if cf.null_ratio < 1:
        print "特征筛选中,请稍后....."
        X_df = null_filter(X_df)

    if cf.std_num > 0:
        X_df = std_filter(X_df)

    if len(norma_col)>0:
        print "特征标准化中,请稍后....."
        X_df = normalize(norma_col,X_df)

    print "======================开始进行特征变换，请稍后......============================="
    funs = cf.funcs
    head = np.mat(X_df.columns)
    X = np.mat(X_df)
    X = np.vstack((head, X))
    del X_df#删除废弃变量
    for op in funs:
        feature_cross(X, op)

    print "==============特征变换完成，开始进行交叉特征筛选，请稍后......================="
    # f = file('./y.pkl', 'wb')
    # pickle.dump(y, f) #y值缓存
    feat_filered = Feat_Filter(y)
    print feature_final.shape[1]

    print "==============正在剔除高相关性特征,请稍后......================="
    feat_filered = f2f_remove(feat_filered)
    X_all = np.hstack((X,feat_filered))
    del feat_filered
    del X
    # X_all = f2f_remove(X_all)
    print "==============正在验证新增特征,请稍后.....================"
    feat_lda = feat_LDA(X_all,y)
    feat_gbdt = feat_GBDT(X_all,y)
    print "是否保留新增特征？输入“yes”保留，“no”放弃"
    flag = raw_input()
    if flag=='yes' or flag=='y' or flag=='Y' or flag=='YES':
        X_all = np.hstack((X_all,feat_lda))
        X_all = np.hstack((X_all, feat_gbdt))
        print "新增LDA和GBDT特征后,一个有{}个特征".format(X_all.shape[1])

    print "===============特征模型评估中,请稍后.....================"
    model, m_score = model_xgb(X_all,y)
    y_pre = model.predict(xgb.DMatrix(X_all[1:,:]))

    show_obj = model_See(model,y_pre,y,m_score)
    show_obj.show()





if __name__=='__main__':
    print "====================正在读取文件，请稍后......==========================="
    path1 = "/home/sf/work/data/zhima_score_weibo_text_describe.csv"  # 微博特征
    feat = pd.read_csv(path1, index_col=0)
    feat = feat.dropna(subset=['user_id', 'sum_reposts'])

    label = feat['score']
    X = feat.iloc[:, 1:]
    feature_robot(X, label)