#coding:utf-8
__author__='lt'

import pandas as pd
from step3_mdoel_validation import feature_pretreatment_woe,ks_value
from sklearn.externals import joblib
import numpy as np

"""
此代码块包含模型转信用分值的代码：
    score_card： 输出所有样本的违约概率与信用分值
    calc_score： 采用概率公式直接计算样本信用分值
    calc_score_by_card： 采用评分卡方式计算信用分值
    bin_statistic: 所有样本的信用分值分组统计
注：采用概率公式和采用评分卡方式计算得出的样本信用分值相同
评分卡最终输出需要手动整理完成，最后以excel形式交付
"""

#最优变量集合
cols_filter = ['实收资本', '注册地址', '场地归属', '注册时间', '下游客户情况', '涉诉信息', '有无贷款', '企业征信', '净资产', '流动比率',
               '资产负债率', '主营业务利润率', '应收账款周转天数', '存货周转天数', '季度逾期占比', '季度平均逾期天数', '历史_逾期占比',
               '历史_提前还款占比', '季度提前还款占比', '历史_平均逾期天数', '平均逾期天数变化', '逾期占比变化', '订单总数', '交易额总量',
               '历史_交易稳定性', '交易量最小值', '交易总量', '交易额最小值', '交易额方差', '提前还款占比变化', '历史_账期订单交易量均值',
               '历史_交易额均值', '历史_账期订单交易额均值', '交易额均值', '交易稳定性', '历史_最近连续交易月数', '历史_账期合作时间（月）',
               '订单时间间隔均值变化', '账期订单交易量均值变化', '账期订单交易额均值变化','label']

#=======================评分卡分值转换设定值======================
#样本违约比正常概率比为base_ratio时,指定的信用分值
base_score = 600
#样本违约比正常概率比
base_ratio = 1/9.0
#当样本的违约比正常概率比降低一倍时，对应增加的信用分数
add_score = 50
#计算转换公式常数A，B
B = add_score / np.log(2)
A = base_score + B * np.log(base_ratio)
print B,A

#======================评分卡构建================================
def score_card(model_name):
    """
    最优模型预测所有样本的分值
    :param model_name: string
    评分卡模型存储文件名
    :return: pandas.DataFrame
    样本违约概率与信用分值
    """
    #从本地加载最优模型
    model = joblib.load('./data/model/%s.m'%model_name)

    #所有样本
    samples = feature_pretreatment_woe()
    samples = samples[cols_filter]
    samples.to_csv('./scorecard_data.csv')

    result = pd.DataFrame(index=samples.index)
    #模型预测所有样本的违约概率
    y_pre = model.predict_proba(samples.iloc[:,:-1])
    y_pre = [x[1] for x in y_pre]
    result['sample_1_pro'] = y_pre
    ks = ks_value(samples.iloc[:,-1],y_pre)
    print '模型全局ks值与auc值',ks

    # 1)模型对所有样本评分预测
    result['sample_score'] = result['sample_1_pro'].apply(lambda p: calc_score(p))
    # 2)模型预测分值（评分卡）
    # weight = pd.Series(index=samples.iloc[:,:-1].columns,data=list(model.coef_[0]))
    # bias = model.intercept_[0]
    # result['sample_score'] = samples.iloc[:,:-1].apply(lambda x:calc_score_by_card(weight,x,bias),axis=1)

    #样本真实标签
    result['label'] = samples['label']
    return result

def calc_score(p):
    """
    采用概率公式直接计算样本信用分值
    :param p: float
    样本违约概率
    :return: int
    根据分值转换公式计算得出的样本信用分值
    """
    odds = np.log(p/(1-p))
    score = A - B * odds
    score =int(score)
    return score

def calc_score_by_card(weight,sample,bias):
    """
    采用评分卡方式计算信用分值
    :param weight: pandas.Series
    模型中样本变量对应的变量权重值
    :param sample: pandas.Series
    样本每个变量对应的woe取值
    :param bias: float
    模型偏置
    :return: int
    根据评分卡分值计算方式得出的样本信用分值
    """
    #样本变量值乘以样本变量对应的权重
    sf = weight * pd.Series(sample)
    feature_score = sf.sum()
    # 信用分值
    score = A - B * (bias + feature_score)
    return int(score)


#======================样本分值分段统计===========================
def bin_statistic(result):
    """
    统计样本的信用分值分布
    :param result: pandas.DataFrame
    所有样本的信用分值
    :return: pandas.DataFrame
    返回分值分布统计结果
    """

    # 样本等量分组统计，默认分5组
    bin_num = pd.DataFrame()
    #分组编号
    result['rank_score'] = result['sample_score'].rank(method='max') / (result.shape[0] / 5.0)
    result['rank_score'] = result['rank_score'].apply(lambda x: int(x) if x > 0 and x < 5 else 4)
    #统计
    bin_num['等级界点'] = result.groupby(by='rank_score')['sample_score'].apply(lambda x: (x.min(), x.max()))
    bin_num['样本数'] = result['rank_score'].value_counts()
    bin_num['违约样本数'] = result.groupby(by='rank_score')['label'].sum()

    # 等距统计
    bin_value = pd.DataFrame()
    #分值区间
    result['score_bin'] = pd.cut(result['sample_score'], [x for x in xrange(190, 800, 100)])
    bin_value['样本数'] = result['score_bin'].value_counts()
    bin_value['违约样本数'] = result.groupby(by='score_bin')['label'].sum()

    return bin_num,bin_value


if __name__=='__main__':

    result =score_card('model_4')
    bin_num, bin_value = bin_statistic(result)
    print bin_num
    print bin_value