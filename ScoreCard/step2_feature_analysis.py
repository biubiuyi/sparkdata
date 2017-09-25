#coding:utf-8
__author__='lt'

import pandas as pd
from step1_feature_generate import *
from sklearn.decomposition import FactorAnalysis,PCA
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf8')

"""
此代码块包含了样本候选变量的探索性统计分析代码：
    company_loan_statistic： 分时间段统计了客户所有账期订单的逾期情况
    statistic_feature： 统计了所有候选变量的空值率，取值分布，IV值，与label的相关性
    feature_correlation： 统计了候选变量间的相关性，并筛选出了相关性大于0.8的变量对
    pca,fa: 分别是采用主成分分析（PCA）和因子分析（FA）方法对所有候选变量间的多重共线性进行分析
    discrete_feature： 统计了离散变量所有取值类别对应的样本逾期情况
    bins_num,bins_value： 对连续变量进行了等量和等距分段，并统计变量每个取值分段对应的样本逾期情况
用到pandas，sklearn，numpy工具包
"""

#============================客户逾期情况统计===================================
def company_loan_statistic(zq_dir):
    """
    统计客户所有订单的逾期情况
    :param zq_dir: string
    账期记录文件存储路径
    :return: pandas.DataFrame
    客户逾期情况统计结果
    """
    loan_all = read_all_loan_record(zq_dir)
    loan_all = loan_all[loan_all[u'收款状态'] == u'已收款']

    stats = pd.DataFrame()
    tmps = loan_all.groupby(by=u'客户')['yq_day'].apply(yq_sts)
    stats['4天以内'] = tmps.apply(lambda x: x[0])
    stats['5-10天'] = tmps.apply(lambda x: x[1])
    stats['11-15天'] = tmps.apply(lambda x: x[2])
    stats['16-30天'] = tmps.apply(lambda x: x[3])
    stats['30天以上'] = tmps.apply(lambda x: x[4])
    stats['逾期合计'] = stats['4天以内'] + stats['5-10天'] + stats['11-15天'] + stats['16-30天'] + stats['30天以上']
    stats['提前还款'] = tmps.apply(lambda x: x[5])
    stats['实际应收笔数'] = tmps.apply(lambda x: x[6])
    stats['逾期占比'] = 1.0 * stats['逾期合计'] / stats['实际应收笔数']
    stats['提前还款占比'] = 1.0 * stats['提前还款'] / stats['实际应收笔数']
    stats['平均账期天数'] = loan_all.groupby(by=u'客户')['zq_day'].mean()
    stats['平均逾期天数'] = loan_all[loan_all['yq_day'] > 0].groupby(by=u'客户')['yq_day'].mean()
    return stats

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
    yq_15 = 0
    yq_30 = 0
    yq_b = 0
    zq_num = len(list)
    ad_d = 0
    for i in list:
        if i < 0:
            ad_d += 1
        elif i > 0 and i <= 4:
            yq_4 += 1
        elif i > 4 and i <= 10:
            yq_10 += 1
        elif i > 10 and i <= 15:
            yq_15 += 1
        elif i > 15 and i <= 30:
            yq_30 += 1
        elif i > 30:
            yq_b += 1

    return yq_4, yq_10, yq_15, yq_30, yq_b, ad_d, zq_num


#============================候选变量统计======================================
def statistic_feature(zq_dir,jy_dir,company_info_path):
    """
    调用step1_feature_generate模块包中的变量计算函数，提取所有变量并合并，增加交叉变量
    :param zq_dir: string
    存储账期记录数据的文件（excel）路径
    :param jy_dir: string
    存储交易记录数据的文件（excel）路径
    :param company_info_path: string
    企业基本信息表(excel)，包含每个企业的所有原始基础信息数据
    :return: pandas.DataFrame
    返回样本所有候选变量初始值以及变量统计分析结果
    """
    feature_all = merge_all_feature(zq_dir,jy_dir,company_info_path)
    #样本按时间排序
    feature_all = feature_all.sort_values(by='起算日期')
    #打标签
    del_col = ['客户','收款状态','yq_day','起算日期']
    feature_all = label_zq(feature_all,del_col)
    #统计变量分布，空值率，IV值，与label的相关系数
    feature_sts = statistic(feature_all)
    return feature_sts,feature_all

def merge_all_feature(zq_dir,jy_dir,company_info_path):
    """
    调用step1_feature_generate模块包中的变量计算函数，提取所有变量并合并，增加交叉变量
    :param zq_dir: string
    存储账期记录数据的文件（excel）路径
    :param jy_dir: string
    存储交易记录数据的文件（excel）路径
    :param company_info_path: string
    企业基本信息表(excel)，包含每个企业的所有原始基础信息数据
    :return: pandas.DataFrame
    返回样本合并后的所有候选变量值
    """

    #企业基本信息变量
    company_info = feature_company_info(company_info_path)
    company_info.columns = [col.encode('utf-8') for col in company_info.columns] #excel汉字编码与环境编码不一致，修正
    # 季度变量
    season_trade = feature_season_trade(zq_dir,jy_dir)
    season_loan = feature_season_loan(zq_dir)
    season_loan.columns = [col.encode('utf-8') for col in season_loan.columns]
    # 历史变量
    history_trade = feature_history_trade(zq_dir,jy_dir)
    history_loan = feature_history_loan(zq_dir)
    history_loan.columns = [col.encode('utf-8') for col in history_loan.columns]

    #合并
    tmp = season_loan[['客户', '起算日期', '应收单号', '收款状态', 'zq_day', 'yq_day', '季度逾期占比', '季度提前还款占比', '季度平均逾期天数']]
    feature_all = season_trade.merge(tmp, left_index=True, right_on='应收单号', how='left')
    feature_all = company_info.merge(feature_all, left_index=True, right_on='客户', how='right')
    # 历史变量合并
    tmp2 = history_trade.merge(history_loan[['4天以内占比', '5-10天占比', '11天以上占比', '逾期占比', '提前还款占比', '平均逾期天数', '应收单号']],
                          left_index=True, right_on='应收单号', how='left')
    tmp2.columns = pd.Series(tmp2.columns).apply(lambda x: '历史_' + x if x != '应收单号' else x)
    feature_all = feature_all.merge(tmp2, on='应收单号', how='left')
    feature_all = feature_all[feature_all['收款状态'] == '已收款'] #筛选有效样本

    #交叉变量
    feature_all['订单时间间隔均值变化'] = feature_all['订单时间间隔均值（天）'] / feature_all['历史_订单时间间隔均值（天）']
    feature_all['交易额均值变化'] = feature_all['交易额均值'] / feature_all['历史_交易额均值']
    feature_all['交易稳定性变化'] = feature_all['交易稳定性'] / feature_all['历史_交易稳定性']
    feature_all['账期订单交易量均值变化'] = feature_all['账期订单交易量均值'] / feature_all['历史_账期订单交易量均值']
    feature_all['账期订单交易额均值变化'] = feature_all['账期订单交易额均值'] / feature_all['历史_账期订单交易额均值']
    feature_all['逾期占比变化'] = feature_all['季度逾期占比'] / feature_all['历史_逾期占比']
    feature_all['提前还款占比变化'] = feature_all['季度提前还款占比'] / feature_all['历史_提前还款占比']
    feature_all['平均逾期天数变化'] = feature_all['季度平均逾期天数'] / feature_all['历史_平均逾期天数']
    return feature_all

def label_zq(feature_all, del_col):
    """
    给所有样本加好坏标注（label），好坏样本划分标准：
    账期<=7天，逾期大于5天为坏客户
    账期8-15天，逾期大于4天为坏客户
    账期>15天，逾期大于7天为坏客户
    :param feature_all: pandas.DataFrame
    样本所有变量值
    :param del_col: pandas.DataFrame
    需要剔除的合并辅助变量
    :return: pandas.DataFrame
    返回打上样本标签的样本集
    """

    #分账期打标签
    zq_7 = feature_all[feature_all.zq_day <= 7]
    zq_15 = feature_all[(feature_all.zq_day > 7) & (feature_all.zq_day <= 15)]
    zq_30 = feature_all[feature_all.zq_day > 15]
    zq_7.loc[:, 'label'] = zq_7['yq_day'].apply(lambda x: 1 if x > 4 else 0)
    zq_15.loc[:, 'label'] = zq_15['yq_day'].apply(lambda x: 1 if x > 3 else 0)
    zq_30.loc[:, 'label'] = zq_30['yq_day'].apply(lambda x: 1 if x > 6 else 0)

    #合并
    zq_all = zq_7.append(zq_15)
    zq_all = zq_all.append(zq_30)
    zq_all = zq_all.set_index('应收单号') #样本唯一标识
    zq_all = zq_all.drop(del_col, axis=1) #删除合并辅助列
    return zq_all

def statistic(df):
    """
    统计变量分布，空值率，IV值，与label的相关系数
    :param df: pandas.DataFrame
    样本所有候选变量值
    :return: pandas.DataFrame
    候选变量统计结果
    """
    # mean,std,min,max，25%,50%,75%
    features_st = df.describe().T
    #空值率
    features_st['null_ratio'] = pd.DataFrame.from_dict([null_ratio(df)]).T
    # ivs
    features_st['ivs'] = pd.DataFrame.from_dict([calc_ivs(df)]).T
    # pearson
    features_st['pearson_label'] = df.corr()['label']
    return features_st


#============================变量相关性与多重共线性分析===========================
def feature_correlation():
    """
    计算候选变量间两两相关性，相关性高于0.8时保存输出
    :return: pandas.DataFrame
    变量间相关性高于0.8的变量记录保存
    """
    feature_all = pd.read_csv('./data/samples_all.csv',index_col=0) #读取样本集合，第一列是样本标识号

    #相关性计算，数值型变量
    cols = [col for col in feature_all.columns if feature_all[col].dtypes == 'float64']
    result = []
    for i in xrange(len(cols)):
        for j in xrange(i + 1, len(cols)):
            corrs = feature_all[cols[i]].corr(feature_all[cols[j]])
            if np.abs(corrs) > 0.8:
                result.append([cols[i], cols[j], corrs])
    result = pd.DataFrame(result)
    return result

def pca(n_com):
    """
    使用sklearn工具包的PCA降维方法分析所有候选变量的共线性，计算指定因子数的方差贡献率总和
    :param n_com: int
    因子数（降维后样本的变量数）
    :return data_pca: array
    降维后的样本变量值矩阵
    :return sum: float
    降维后所有因子的方差贡献率总和
    """
    feature_all = pd.read_csv('./data/samples_all.csv', index_col=0)  # 读取样本集合，第一列是样本标识号
    data = feature_all.fillna(0) #缺失值填充
    pca = PCA(n_components=n_com)
    #pca降维
    data_pca = pca.fit_transform(data.iloc[:, :-1])
    #降维后每个因子的方差贡献率
    vars = pca.explained_variance_ratio_
    #降维后所有因子的方差贡献率总和
    sum = pd.DataFrame(vars).sum()[0]
    return data_pca,sum

def fa(n_com):
    """
    使用sklearn工具包的FactorAnalysis方法分析所有候选变量的共线性，输出指定因子数的因子成分矩阵
    :param n_com: int
    因子数（降维后样本的变量数）
    :return: array
    降维后的样本变量值矩阵
    """
    feature_all = pd.read_csv('./data/samples_all.csv', index_col=0)  # 读取样本集合，第一列是样本标识号
    data = feature_all.fillna(0)
    fa = FactorAnalysis(n_components=n_com)
    #降维
    data_fa = fa.fit_transform(data.iloc[:, :-1], y=data.iloc[:, -1])

    #因子成分矩阵
    factor = fa.components_
    factor_df = pd.DataFrame(factor,columns=feature_all.iloc[:, :-1].columns)
    return factor_df


#=============================单变量统计分析=====================================
def discrete_feature(df,col):
    """
    离散变量取值对应的逾期情况统计
    :param df: pandas.DataFrame
    样本所有候选变量值矩阵
    :param col: string
    指定统计逾期情况的变量名
    :return: pandas.DataFrame
    指定变量的统计结果
    """
    result = pd.DataFrame()
    result['逾期订单数'] = df[[col,'label']].groupby(by=col)['label'].sum()
    result['订单总数'] = df[col].value_counts()
    result['逾期订单占比'] = result['逾期订单数']*1.0/result['订单总数']
    return result

def bins_num(df,col,bin_num):
    """
    连续变量按照相同的样本数进行分段，然后统计每个取值分段对应的逾期情况.
    :param df:pandas.DataFrame
    样本所有候选变量值矩阵
    :param col: string
    指定统计逾期情况的变量名
    :param bin_num: int
    指定变量分段数
    :return result: pandas.DataFrame
    指定变量的统计结果
    :return df[rank_col]: pandas.Series
    指定变量对应的分段号
    """
    total_count =df.shape[0]
    #变量值从小到大排序，并按照排序序号将变量值分到对应的段数
    rank_col = col+'_rank'
    df[rank_col] = df[col].rank(method='max')/(total_count/bin_num*1.0)
    df.loc[df[df[col].isnull()].index,rank_col] =np.nan #空值的分段数为空
    df[rank_col] =df[rank_col].apply(lambda x:int(x) if x>0 else np.nan)
    #最后一段如果样本数较少则与前一段合并
    tmp = df[df[rank_col]==bin_num]
    if tmp.shape[0] > 0:
        df.loc[tmp.index,rank_col]=bin_num-1

    #统计
    result =pd.DataFrame()
    gp = df.groupby(by=rank_col)
    result['分段上下界'] = gp[col].apply(lambda x:(x.min(),x.max()))
    result['逾期样本数'] = gp['label'].sum()
    result['样本总数'] = gp[col].count()
    gf =calc_woe(df,rank_col)
    result['woe'] =gf['woe']
    return result,df[rank_col]

def bins_value(df,col,bin_num):
    """
        连续变量按照相同的取值区间距离进行分段，然后统计每个取值分段对应的逾期情况.
        :param df:pandas.DataFrame
        样本所有候选变量值矩阵
        :param col: string
        指定统计逾期情况的变量名
        :param bin_num: int
        指定变量分段数
        :return result: pandas.DataFrame
        指定变量的统计结果
        :return df[rank_col]: pandas.Series
        指定变量对应的分段号
    """
    #变量取值范围
    total_gap = df[col].max()-df[col].min()
    #变量按照取值进行分段编号
    rank_col = col+'_rank'
    bins = [df[col].min()+(total_gap*1.0/bin_num)*i for i in xrange(0,bin_num)]
    df[rank_col] =df[col].apply(lambda x:cut_bin(x,bins))

    #统计
    result = pd.DataFrame()
    gp = df.groupby(by=rank_col)
    result['分段上下界'] = gp[col].apply(lambda x: (x.min(), x.max()))
    result['逾期样本数'] = gp['label'].sum()
    result['样本总数'] = gp[col].count()
    gf = calc_woe(df, rank_col)
    result['woe'] = gf['woe']
    return result, df[rank_col]

def cut_bin(x,list_bin):
    """
    判断输入x在list_bin的那个取值区间，并返回取值区间编号
    :param x: float
    需要判断的值
    :param list_bin: list
    区间取值列表
    :return:
    取值区间编号
    """
    for i in xrange(1,len(list_bin)):
        if x>list_bin[i-1] and x<=list_bin[i]:
            return i
    return len(list_bin)



#=============================代码测试=====================================
if __name__=='__main__':
    zq_dir = './data/loan_list/'
    jy_dir = './data/client_data/'
    company_info_path = './data/feature_checked.xlsx'
    #变量提取代码测试
    feature_sts, feature_all = statistic_feature(zq_dir,jy_dir,company_info_path)
    feature_sts.to_csv('./data/feature_statistic.csv')  # 保存变量统计分析结果
    feature_all.to_csv('./data/samples_all.csv')  # 保存样本所有候选变量初始值

    #变量统计分析代码测试
    stats = company_loan_statistic(zq_dir)
    stats.to_csv('./data/loan_statistic.csv')  # 保存

    result = feature_correlation()
    result.to_csv('./data/feature_pearson.csv', index=None)

    factor_df = fa(13)
    factor_df.to_csv('./data/feature_collinearity.csv')
