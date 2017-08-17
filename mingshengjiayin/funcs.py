#coding:utf-8

import pandas as pd
import os
import numpy as np
import re

#读取目录下左右文件名（二层目录）
def VisitDir(dirname):
    files = [dirname]
    dirs = os.listdir(dirname)
    for d in dirs:
        d = os.path.join(dirname,d)
        l = [os.path.join(d,f) for f in os.listdir(d)]
        files.extend(l)
    return files

#归一化
def normalize(sf):
    gap = sf.max()-sf.min()
    return (sf-sf.min())/gap

#空值率
def null_ratio(df):
    dic ={}
    for col in df.columns:
        r = df[df[col].isnull()].shape[0]*1.0/df.shape[0]
        dic[col]=r
    return dic

#方差
def std(df):
    dic ={}
    for col in df.columns:
        r = df[col].std()
        dic[col]=r
    return dic

#合作时间
def coo_months(sf):
    sf = pd.to_datetime(sf)
    sf = sf.sort_values(ascending=False).dropna()
    month = (sf.iloc[0].year-sf.iloc[-1].year)*12 + (sf.iloc[0].month-sf.iloc[-1].month)+1
    return month

#最近交易时间（XX个月之前）
def recent_month(sf,t_now):
    sf =pd.to_datetime(sf)
    if sf.shape[0]>1:
        sf = sf.sort_values(ascending=False).dropna()
        month = (t_now.year-sf.iloc[1].year)*12 + (t_now.month - sf.iloc[1].month)
    else:
        month=np.nan
    return month

#采购月数
def purchase_months(sf):
    times = sf.dropna().apply(lambda x:str(x.year)+str(x.month))
    times_set =set(times)
    return len(times_set)

#时间间隔均值，方差(单位:天)
def time_gap(times):
    try:
        times = pd.to_datetime(pd.Series(times).drop_duplicates())
        if len(times)==1:
            mean = np.nan
            std = np.nan
        else:
            times = times.sort_values()
            gaps = (times -times.shift()).dropna()
            gaps = gaps.map(lambda x: x/np.timedelta64(1,'D'))
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
def trade_stab(sf):
    if sf.shape[0]>1:
        sf = sf.sort_values(by=u'创建时间',ascending=False)
        days = sf[u'创建时间'].iloc[0].day +91
        stop =sf[u'创建时间'].iloc[0]- pd.to_timedelta('%d days'%days)
        sf = sf[sf[u'创建时间']>stop]        
        gap = (sf[u'销售数量（吨）'].shift()-sf[u'销售数量（吨）'])
        tmp = gap/sf[u'销售数量（吨）']
        return tmp.mean()
    else:
        return 0

#最近连续交易月数
def recent_last_month(times):
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

#计算IV值        
def calc_iv(df,label_thr=None,box_num=3):
    if df['label'].unique().size>2: #label为连续型变量
        df['label'] = df['label'].apply(lambda x:1 if x>label_thr else 0)      
    cols = df.drop('label',axis=1).columns
    ivs ={}
    for col in cols:
        iv = calc_woe_iv(df[[col,'label']],col,box_num)
        ivs[col]=iv
    return ivs

#计算woe和IV值，等量
def calc_woe_iv(df_sub,col,box_num):
    df_sub = df_sub.copy()
    total_good = len(df_sub[df_sub.label==0])
    total_bad = len(df_sub[df_sub.label==1])
    total_count = df_sub.shape[0]
    rank_col =col #离散变量，默认分组
    if df_sub[col].unique().size>6: #连续型特征，分箱:4
        rank_col = col+'_rank'
        df_sub[rank_col] = df_sub[col].rank(method='max')/(total_count/box_num)
        df_sub.fillna(-999,inplace=True)
        #分组
        df_sub[rank_col] =df_sub[rank_col].apply(lambda x:int(x) if x>0 else -1)
    
    grouping_data =[]
    for gname,gdata in df_sub.groupby(rank_col):
        g_info ={}
        g_info['name']= gname
        g_info['good_num'] = len(gdata[gdata['label']==0])
        g_info['good_ratio'] = g_info['good_num']*1.0/total_good
        g_info['bad_num'] = len(gdata[gdata['label']==1])
        g_info['bad_ratio'] = g_info['bad_num']*1.0/total_bad
        
        if g_info['good_num']>0 and g_info['bad_num']>0:
            g_info['woe'] =np.math.log(1.0*g_info['good_ratio']/g_info['bad_ratio'])
        elif g_info['good_num'] ==0:
            g_info['woe'] =-1
        else:
            g_info['woe']=1
        g_info['iv'] =1.0*(g_info['good_ratio']-g_info['bad_ratio'])*g_info['woe']
        grouping_data.append(g_info)
        
    g_df = pd.DataFrame(grouping_data,columns = g_info.keys())
    iv_sum =g_df.iv.sum()
    return iv_sum   


    
    