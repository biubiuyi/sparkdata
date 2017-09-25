#coding:utf-8
__author__='lt'

import pandas as pd
import os
import re
from functions import *
import numpy as np
from pandas.tseries.offsets import DateOffset
import sys
reload(sys)
sys.setdefaultencoding('utf8')


"""
此代码模块包括了样本所有变量的预处理和计算代码：
    feature_company_info： 样本企业基础信息变量整理
    feature_season_trade： 提取样本季度交易数据变量
    feature_season_loan： 提取样本季度逾期情况变量
    feature_history_trade： 提取样本历史交易数据变量
    feature_history_loan： 提取样本历史逾期情况变量
注：在提取变量之前，需要对数据的准确性进行人工核查，并对字段取值进行统一
    用到pandas，numpy工具包
"""


indexs_season = ['订单总数','订单时间间隔均值（天）','订单时间间隔方差（天）','交易总量','交易量最大值','交易量最小值','交易量方差',
                 '交易量均值','交易额总量','交易额最大值','交易额最小值','交易额方差','交易额均值','交易稳定性',
                 '账期订单占比','账期订单交易量均值','账期订单交易额均值']

indexs_history = ['订单总数','合作时间（月）','最近交易（X月之前）','最近连续交易月数','采购断档总月数','订单时间间隔均值（天）',
                  '订单时间间隔方差（天）','交易总量','交易量最大值','交易量最小值','交易量方差','交易量均值','交易额总量',
                  '交易额最大值','交易额最小值','交易额方差','交易额均值','交易稳定性','供应商数量','采购产品种类数量',
                  '账期订单占比','账期订单交易量均值','账期订单交易额均值','账期合作时间（月）','最近一笔账期订单（X月之前）']

#==========================企业基本信息变量==============================
def feature_company_info(company_info_path):
    """
    企业基本信息变量处理：变量拆分，非结构变量处理
    Parameters
    ----------
    company_info_path: string
    企业基本信息表(excel)，包含每个企业的所有原始基础信息数据
    return:pandas.DataFrame
    返回处理过后的企业基础信息数据

    """
    data = pd.read_excel(company_info_path)
    dic = {'no': 0, u'无': 0, 'yes': 1, u'有': 1, u'否': 0, u'是': 1, 1: 1, 0: 0, 'A': 1, 'B': 0, '自有': 1, '租赁': 0}
    dic2 = {u'出口': 4, u'市政工程': 4, u'大中企业': 4, u'中小企业': 2, u'中小微企业': 2, u'自营': 1}
    dic3 = {u'正常': 1, u'欠息': 2, u'关注': 3, u'不良': 4}

    # 拆分
    data['厂房面积'] = data[u'厂房面积'].apply(lambda x: re.findall('(\d+)', x)[0])
    data['场地归属'] = data[u'厂房面积'].apply(lambda x: '租赁' if u'租赁' in x else '自有')

    data_new = pd.DataFrame()
    data_new['注册地址'] = data[u'注册地址'].map(dic)
    data_new['注册资本变更'] = pd.get_dummies(data[u'注册资本变更'])
    data_new['法人变更'] = data[u'法人变更'].map(dic)
    data_new['分支机构'] = data[u'分支机构'].map(dic)
    data_new['商标专利'] = data[u'商标专利'].map(dic)
    data_new['场地归属'] = data['场地归属'].map(dic)
    data_new['有无贷款'] = data[u'有无贷款'].map(dic)
    data_new['企业征信'] = data[u'企业征信'].map(dic3)
    data_new['涉诉信息'] = data[u'涉诉信息']
    data_new['下游客户情况'] = data[u'下游客户情况'].map(dic2)

    data_new['注册时间'] = (data[u'注册时间'])
    data_new['实收资本'] = (data[u'实收资本'])
    data_new['员工人数'] = (data[u'员工人数'])
    tmp = (data['厂房面积'].astype(float))
    data_new['厂房面积'] = tmp
    data_new['资产负债率'] = (data[u'资产负债率'])
    data_new['流动比率'] = (data[u'流动比率'])
    data_new['净资产'] = (data[u'净资产'])
    data_new['货币资金'] = (data[u'货币资金'])
    data_new['存货周转天数'] = (data[u'存货周转天数'])
    data_new['应收账款周转天数'] = (data[u'应收账款周转天数'])
    data_new['主营业务利润率'] = (data[u'主营业务利润率'])
    data_new['净资产增长率'] = (data[u'净资产增长率'])
    data_new['销售增长率'] = (data[u'销售增长率'])
    data_new['当地经济环境'] = (data[u'当地经济环境'])
    data_new['撮合单数'] = (data[u'撮合单数'])

    return data_new


#===========================交易记录变量================================
def stop_time(start):
    #返回start日期对应的前三个月的日期
    stop = start - DateOffset(months=3, days=start.day)
    return stop

def read_all_loan_record(zq_dir):
    """
    合并所有企业的账期订单记录，并对非结构化数据进行处理
    :param zq_dir: string
    存储账期记录数据文件的路径
    :return: pandas.DataFrame
    返回处理后的账期记录数据
    """
    files = os.listdir(zq_dir)
    loan_all = pd.DataFrame()
    for f in files:
        record = pd.read_excel(os.path.join(zq_dir, f))
        loan_all = loan_all.append(record.iloc[1:, :], ignore_index=True)
    loan_all['zq_day'] = loan_all[u'账期'].apply(lambda x: re.findall('\d+', x)[0])
    loan_all['yq_day'] = loan_all[u'逾期'].apply(lambda x: re.findall('^(-\d+|\d+)', x)[0])
    loan_all['zq_day'] = loan_all['zq_day'].astype(int)
    loan_all['yq_day'] = loan_all['yq_day'].apply(lambda x: int(x))
    loan_all = loan_all.drop_duplicates() #去重
    return loan_all


#==========================季度交易数据变量==============================
def feature_season_trade(zq_dir,jy_dir):
    """
    统计每个企业每个账期订单样本一个季度的交易数据变量值
    :param zq_dir: string
    存储账期记录数据的文件（excel）路径
    :param jy_dir: string
    存储交易记录数据的文件（excel）路径
    :return:pandas.DataFrame
    返回每个订单样本季度交易数据变量值
    """

    #读取所有账期记录和交易记录excel表文件名
    zq_files = os.listdir(zq_dir)
    files = VisitDir(jy_dir)
    files = [f for f in files if '交易记录' in f]

    samples = pd.DataFrame()
    for f in zq_files:
        record = pd.read_excel(os.path.join(zq_dir, f))
        record = record.iloc[1:, :]  # 去掉第一行标题
        for f1 in files:
            if f1.split('/')[-2] == f.split('.')[0]: #账期记录和交易记录文件属于同一个公司
                trade = pd.read_excel(f1)  # 交易数据
                trade = trade.dropna(thresh=5)
                df = company_season_feature(record, trade)
                if samples.shape[0] == 0:
                    samples = df
                else:
                    samples = samples.merge(df, left_index=True, right_index=True, how='left')
                continue

    samples = samples.T
    return samples

def company_season_feature(record,trade):
    """
    针对企业的每笔账期订单，计算其起算日之前一个季度的交易数据对应变量的值
    :param record: pandas.DataFrame
    企业的所有账期订单记录
    :param trade: pandas.DataFrame
    企业的所有交易记录
    :return: pandas.DataFrame
    该企业每笔账期订单对应的季度交易数据变量值
    """
    df = pd.DataFrame(index=indexs_season)
    tmps = zip(record[u'应收单号'],record[u'起算日期'])
    for t in tmps:
        trade_sub = trade[trade[u'创建时间']<t[1]]  #账期订单起算日之前的交易记录
        if trade_sub.shape[0]>1:#有历史交易记录
            fs = season_feature(trade_sub.iloc[1:,:])
        else:#无历史交易记录
            fs = [np.nan]*len(indexs_season)
        df[t[0]]=fs
    return df

def season_feature(trade):
    """
    计算季度交易数据变量的值
    :param trade: pandas.DataFrame
    账期订单下单之前企业的所有交易记录
    :return: list
    变量对应的值
    """
    feature = []
    stop = stop_time(trade[u'创建时间'].iloc[0])
    trade_3m = trade[trade[u'创建时间'] > stop] #一个季度的交易记录
    if trade_3m.shape[0] > 0:
        num = trade_3m.shape[0]  # 订单数
        feature.append(num)
        tim_st = time_gap(trade_3m[u'创建时间'])
        feature.append(tim_st[0])  # 时间间隔均值
        feature.append(tim_st[1])  # 时间间隔方差

        t_num = trade_3m[u'销售数量（吨）'].agg(['sum', 'max', 'min', 'std', 'mean'])
        t_money = trade_3m[u'销售金额（元）'].agg(['sum', 'max', 'min', 'std', 'mean'])
        feature.extend(t_num)
        feature.extend(t_money)
        #交易稳定性
        feature.append(trade_stab_all(trade_3m))
    else:  # 近三个月无订单
        feature.extend([0] * 14)

    zhangqi = trade_3m[trade_3m[u'是否账期'] == u'是'] #近三个月的账期订单
    if zhangqi.shape[0] > 0:
        feature.append(zhangqi.shape[0] * 1.0 / trade_3m.shape[0])
        feature.append(zhangqi[u'销售数量（吨）'].mean())
        feature.append(zhangqi[u'销售金额（元）'].mean())
    else:
        feature.append(0)
        feature.append(0)
        feature.append(0)
    return feature

def feature_season_loan(zq_dir):
    """
    统计客户一个季度的逾期数据变量值
    :param zq_dir: string
    存储账期记录数据的文件路径
    :return: pandas.DataFrame
    返回增加统计字段后的账期记录数据
    """
    loan_all = read_all_loan_record(zq_dir)

    yq_ratio = []
    tq_ratio = []
    yq_avg = []
    loan_all[u'起算日期'] = pd.to_datetime(loan_all[u'起算日期'])
    for i in xrange(loan_all.shape[0]):
        name = loan_all.iat[i, 1] #客户名
        time = loan_all.iat[i, 10] #账期订单起算日期
        stop = stop_time(time)
        loan_sub = loan_all[(loan_all[u'客户'] == name) & (loan_all[u'起算日期'] < time) & (loan_all[u'起算日期'] > stop)]
        if loan_sub.shape[0] > 0: #一个季度的账期订单
            yq_ratio.append(loan_sub[loan_sub['yq_day'] > 0].shape[0] * 1.0 / loan_sub.shape[0])
            tq_ratio.append(loan_sub[loan_sub['yq_day'] < 0].shape[0] * 1.0 / loan_sub.shape[0])
            if loan_sub[loan_sub['yq_day'] > 0].shape[0] > 0:
                yq_avg.append(loan_sub[loan_sub['yq_day'] > 0]['yq_day'].mean())
            else:
                yq_avg.append(0)
        else:
            yq_ratio.append(np.nan)
            tq_ratio.append(np.nan)
            yq_avg.append(np.nan)

    loan_all[u'季度逾期占比'] = yq_ratio
    loan_all[u'季度提前还款占比'] = tq_ratio
    loan_all[u'季度平均逾期天数'] = yq_avg
    return loan_all


#==========================历史交易数据变量==============================
def feature_history_trade(zq_dir,jy_dir):
    """
    统计每个企业每个账期订单样本历史交易数据变量值
    :param
    zq_dir: string
    存储账期记录数据的文件路径
    :param
    jy_dir: string
    存储交易记录数据的文件路径
    :return:pandas.DataFrame
    返回每个订单样本历史交易数据变量值
    """

    #读取所有账期记录和交易记录excel表文件名
    zq_files = os.listdir(zq_dir)
    files = VisitDir(jy_dir)
    files = [f for f in files if '交易记录' in f]

    samples = pd.DataFrame()
    for f in zq_files:
        record = pd.read_excel(os.path.join(zq_dir, f))
        record = record.iloc[1:, :]  # 去掉第一行标题
        for f1 in files:
            if f1.split('/')[-2] == f.split('.')[0]: #账期记录和交易记录文件属于同一个公司
                trade = pd.read_excel(f1)  # 交易数据
                trade = trade.dropna(thresh=5)
                df = company_history_feature(record, trade)
                if samples.shape[0] == 0:
                    samples = df
                else:
                    samples = samples.merge(df, left_index=True, right_index=True, how='left')
                continue

    samples = samples.T
    samples['采购频率（订单总数/合作月数）'] = samples['订单总数'] * 1.0 / samples['合作时间（月）']
    samples['采购聚集程度（订单总数/供应商数）'] = samples['订单总数'] * 1.0 / samples['供应商数量']
    return samples

def company_history_feature(record,trade):
    """
       针对企业的每笔账期订单，计算其起算日之前所有的交易数据对应变量的值
       :param record: pandas.DataFrame
       企业的所有账期订单记录
       :param trade: pandas.DataFrame
       企业的所有交易记录
       :return: pandas.DataFrame
       该企业每笔账期订单对应的历史交易数据变量值
    """
    df = pd.DataFrame(index=indexs_history)
    tmps = zip(record[u'应收单号'],record[u'起算日期'])
    for t in tmps:
        trade_sub = trade[trade[u'创建时间']<t[1]]
        if trade_sub.shape[0]>1:
            fs = history_feature(trade_sub.iloc[1:,:],t[1])
        else:
            fs = [np.nan]*len(indexs_history)
        df[t[0]]=fs
    return df

def history_feature(trade,t_now):
    """
       计算季度交易数据变量的值
       :param trade: pandas.DataFrame
       账期订单下单之前企业的所有交易记录
       :param t_now: datetime
       账期订单的起算日期
       :return: list
       变量对应的值
    """

    feature = []
    num = trade.shape[0]  # 订单数
    coo_month = coo_months(trade[u'创建时间'])  # 合作时间（月）
    feature.append(num)
    feature.append(coo_month)
    feature.append(recent_month(trade[u'创建时间'], t_now)) #最近交易（X月之前）
    feature.append(recent_last_month(trade[u'创建时间'])) #最近连续交易月数

    purchase_month = purchase_months(trade[u'创建时间'])
    gap_months = coo_month - purchase_month  # 采购断档总月数
    feature.append(gap_months)

    tim_st = time_gap(trade[u'创建时间'])
    feature.append(tim_st[0])  # 时间间隔均值
    feature.append(tim_st[1])  # 时间间隔方差

    t_num = trade[u'销售数量（吨）'].agg(['sum', 'max', 'min', 'std', 'mean'])
    t_money = trade[u'销售金额（元）'].agg(['sum', 'max', 'min', 'std', 'mean'])
    feature.extend(t_num)
    feature.extend(t_money)

    feature.append(trade_stab_all(trade)) #交易稳定性
    feature.append(trade[u'货物'].apply(lambda x: x.split(' ')[1]).value_counts().shape[0]) #供应商数量
    feature.append(trade[u'货物'].apply(lambda x: x.split(' ')[0]).value_counts().shape[0]) #采购产品种类数量

    zhangqi = trade[trade[u'是否账期'] == u'是']
    if zhangqi.shape[0] > 0:
        feature.append(zhangqi.shape[0] * 1.0 / num)
        feature.append(zhangqi[u'销售数量（吨）'].mean())
        feature.append(zhangqi[u'销售金额（元）'].mean())
        feature.append(coo_months(zhangqi[u'创建时间'])) #账期合作时间（月）
        feature.append(recent_month(zhangqi[u'创建时间'], t_now)) #最近一笔账期订单（X月之前）
    else:
        feature.append(0)
        feature.append(0)
        feature.append(0)
        feature.append(0)
        feature.append(np.nan)
    return feature

def feature_history_loan(zq_dir):
    """
    统计客户所有的逾期数据变量值
    :param zq_dir: string
    存储账期记录数据的文件路径
    :return: pandas.DataFrame
    返回增加统计字段后的账期记录数据
    """
    loan_all = read_all_loan_record(zq_dir)
    loan_all['4天以内占比'] = 0
    loan_all['5-10天占比'] = 0
    loan_all['11天以上占比'] = 0
    loan_all[u'逾期占比'] = 0
    loan_all[u'提前还款占比'] = 0

    yq_ratio = []
    tq_ratio = []
    yq_day_ratio = []
    yq_avg = []
    loan_all[u'起算日期'] = pd.to_datetime(loan_all[u'起算日期'])
    for i in xrange(loan_all.shape[0]):
        name = loan_all.iat[i, 1]
        time = loan_all.iat[i, 10]
        loan_sub = loan_all[(loan_all[u'客户'] == name) & (loan_all[u'起算日期'] < time)]
        if loan_sub.shape[0] > 0:
            tmps = yq_sts(loan_sub['yq_day'])
            yq_day_ratio.append(tmps)
            yq_ratio.append(loan_sub[loan_sub['yq_day'] > 0].shape[0] * 1.0 / loan_sub.shape[0])
            tq_ratio.append(loan_sub[loan_sub['yq_day'] < 0].shape[0] * 1.0 / loan_sub.shape[0])
            if loan_sub[loan_sub['yq_day'] > 0].shape[0] > 0:
                yq_avg.append(loan_sub[loan_sub['yq_day'] > 0]['yq_day'].mean())
            else:
                yq_avg.append(0)
        else: #无历史账期订单
            yq_day_ratio.append((np.nan, np.nan, np.nan))
            yq_ratio.append(np.nan)
            tq_ratio.append(np.nan)
            yq_avg.append(np.nan)

    loan_all['4天以内占比'] = [x[0] for x in yq_day_ratio]
    loan_all['5-10天占比'] = [x[1] for x in yq_day_ratio]
    loan_all['11天以上占比'] = [x[2] for x in yq_day_ratio]
    loan_all[u'逾期占比'] = yq_ratio
    loan_all[u'提前还款占比'] = tq_ratio
    loan_all[u'平均逾期天数'] = yq_avg
    return loan_all

def yq_sts(list):
    """
    统计不同逾期时间段订单数占比
    :param list: list
    逾期天数列表
    :return:
    各个逾期时间段的订单数占比
    """
    yq_4 = 0
    yq_10 = 0
    yq_11 = 0
    zq_num = len(list)
    for i in list:
        if i > 0 and i <= 4:
            yq_4 += 1
        elif i > 4 and i <= 10:
            yq_10 += 1
        elif i > 10:
            yq_11 += 1

    return yq_4 * 1.0 / zq_num, yq_10 * 1.0 / zq_num, yq_11 * 1.0 / zq_num



