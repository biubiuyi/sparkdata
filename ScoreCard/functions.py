#coding:utf-8


import pandas as pd
import numpy as np
import os

"""
此代码快包含了所有公共函数
    变量值计算函数
    变量统计分析函数
    变量预处理函数
"""

#读取目录下所有文件名（二层目录）
def VisitDir(dirname):
    """
    读取二级目录下所有文件名
    :param dirname: string
    文件目录
    :return: list
     目录下所有文件名
    """
    files = [dirname]
    dirs = os.listdir(dirname)
    for d in dirs:
        d = os.path.join(dirname,d)
        l = [os.path.join(d,f) for f in os.listdir(d)]
        files.extend(l)
    return files

#==============================变量值计算函数================================
#时间间隔均值，方差(单位:天)
def time_gap(times):
    """
    :param times:pandas.Series
    交易订单的时间列表
    """
    try:
        times = pd.to_datetime(pd.Series(times).drop_duplicates())
        if len(times)==1:
            mean = np.nan
            std = np.nan
        else:
            times = times.sort_values()
            gaps = (times -times.shift()).dropna()
            gaps = gaps.map(lambda x: x/np.timedelta64(1,'D')) #时间间隔换算为天
            if gaps.shape[0]==1:
                mean = gaps.iloc[0]
                std = 0
            else:
                mean = gaps.mean()
                std =gaps.std()
        return mean,std
    except:
        return np.nan,np.nan

#交易稳定性：近三个月交易量变化均值
def trade_stab_all(df):
    """
    :param df: pandas.DataFrame
    样本变量值
    """
    if df.shape[0]>1:
        gap = (df[u'销售数量（吨）'].shift()-df[u'销售数量（吨）'])
        tmp = gap/df[u'销售数量（吨）']
        return tmp.mean()
    else:
        return 0

#合作时间
def coo_months(sf):
    """
    :param sf: pandas.Series
    样本订单的时间列表
    """
    sf = pd.to_datetime(sf)
    sf = sf.sort_values(ascending=False).dropna()
    month = (sf.iloc[0].year-sf.iloc[-1].year)*12 + (sf.iloc[0].month-sf.iloc[-1].month)+1
    return month

#最近交易时间（XX个月之前）
def recent_month(sf,t_now):
    """
    :param sf: pandas.Series
    样本订单的时间列表
    :param t_now: datetime
    当前样本订单的时间
    """
    sf =pd.to_datetime(sf)
    if sf.shape[0]>1:
        sf = sf.sort_values(ascending=False).dropna()
        month = (t_now.year-sf.iloc[0].year)*12 + (t_now.month - sf.iloc[0].month)
    else:
        month=np.nan
    return month

#最近连续交易月数
def recent_last_month(times):
    """
    :param times: pandas.Series
    样本订单的时间列表
    """
    time = pd.to_datetime(times)
    if time.shape[0]>1:
        time = time.sort_values(ascending=False).tolist()
        month = 0
        befor = time[0]
        for t in time[1:]:
            gap = (befor.year-t.year)*12 + (befor.month -t.month)
            if gap==1:
                month +=1
            elif gap>1:
                break
            befor = t
        return month+1
    else:
        return 1

#采购月数：有订单记录的月份数
def purchase_months(sf):
    """
    :param sf: pandas.Series
    样本订单的时间列表
    """
    times = sf.dropna().apply(lambda x:str(x.year)+str(x.month))
    times_set =set(times)
    return len(times_set)


#==============================变量统计分析函数===============================
#空值率
def null_ratio(df):
    """
    :param df: pandas.DataFrame
    样本的所有候选变量值矩阵
    """
    dic ={}
    for col in df.columns:
        r = df[df[col].isnull()].shape[0]*1.0/df.shape[0]
        dic[col]=r
    return dic

#计算所有变量IV值
def calc_ivs(df, label_thr=None):
    """
    :param df: pandas.DataFrame
    样本候选变量值矩阵
    :param label_thr: float
    若label列是连续型数值，则根据指定的阈值label_thr对label进行离散化
    """
    if df['label'].unique().size > 2:  # label为连续型变量
        df['label'] = df['label'].apply(lambda x: 1 if x > label_thr else 0)
    cols = df.drop('label', axis=1).columns
    ivs = {}
    for col in cols:
        gf = calc_woe(df[[col, 'label']], col)
        ivs[col] = gf.iv.sum()
    return ivs

# 计算某个变量的woe,iv值
def calc_woe(df, col, bin_num=3):
    """
    :param df: pandas.DataFrame
    样本候选变量值矩阵
    :param col: string
    指定计算IV值的变量名
    :param bin_num: int
    指定连续型变量分段数
    """
    df_sub = df.copy()
    if 'label' not in df_sub.columns:
        print 'label not in the columns'
        return
    total_good = len(df_sub[df_sub.label == 0])
    total_bad = len(df_sub[df_sub.label == 1])
    total_count = df_sub.shape[0]
    rank_col = col
    #连续型变量，则需要先进行分段编号处理
    if df_sub[col].unique().size > 6:
        rank_col = col + '_rank'
        df_sub[rank_col] = df_sub[col].rank(method='max') / (total_count / bin_num * 1.0)
        df_sub.loc[df_sub[df_sub[col].isnull()].index, rank_col] = np.nan  # 空值复原
        df_sub[rank_col] = df_sub[rank_col].apply(lambda x: int(x) if x > 0 else np.nan)
        tmp = df_sub[df_sub[rank_col] == bin_num]
        if tmp.shape[0] < (total_count / (bin_num * 2)) and tmp.shape[0] > 0:  # 最后一组与前一组合并
            df_sub[rank_col].loc[tmp.index] = bin_num - 1

    # 分组统计
    grouping_data = []
    for gname, gdata in df_sub.groupby(rank_col):
        g_info = {}
        g_info['name'] = gname
        g_info['good_num'] = len(gdata[gdata['label'] == 0])
        g_info['good_ratio'] = g_info['good_num'] * 1.0 / total_good
        g_info['bad_num'] = len(gdata[gdata['label'] == 1])
        g_info['bad_ratio'] = g_info['bad_num'] * 1.0 / total_bad

        if g_info['good_num'] > 0 and g_info['bad_num'] > 0:
            g_info['woe'] = np.math.log(1.0 * g_info['bad_ratio'] / g_info['good_ratio'])
        elif g_info['good_num'] == 0:
            g_info['woe'] = -1
        else:
            g_info['woe'] = 1
        g_info['iv'] = 1.0 * (g_info['bad_ratio'] - g_info['good_ratio']) * g_info['woe']
        grouping_data.append(g_info)

    g_df = pd.DataFrame(grouping_data, columns=g_info.keys())
    return g_df


#==============================变量预处理函数==================================
#woe值字典
def woe_to_dict(df):
    """
    将指定列转换为字典格式
    :param df: pandas.DataFrame
    以列格式存储的数据
    :return:
    转换后的数据字典
    """
    dict ={}
    for i in xrange(df.shape[0]):
        dict[df['name'].iloc[i]] = df['woe'].iloc[i]
    return dict

#变量归一化
def normalize(sf):
    """
    :param sf: pandas.Series
     指定归一化处理的变量
    """
    gap = sf.max()-sf.min()
    return (sf-sf.min())/gap






