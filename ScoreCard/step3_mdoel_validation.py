#coding:utf-8
__author__='lt'

import pandas as pd
from functions import *
from step2_feature_analysis import bins_num
import xgboost as xgb
from sklearn.cross_validation import train_test_split,StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import sys
reload(sys)
sys.setdefaultencoding('utf8')

"""
此代码块包含最优模型寻找过程中的所有备选模型训练代码：
    feature_pretreatment_value： 变量预处理方式一
    feature_pretreatment_woe： 变量预处理方式二（适用于评分卡构建模型）
    model_GBDT： 梯度提升决策树GBDT模型训练代码，调用xgboost工具包
    model_SVM： 支持向量机SVM模型训练代码,调用sklearn机器学习工具包
    model_NB： 朴素贝叶斯Naive Bayes模型训练代码,调用sklearn机器学习工具包
    model_KNN： K近邻KNN模型训练代码,调用sklearn机器学习工具包
    model_NN： 神经网络Neural Network模型训练代码,调用sklearn机器学习工具包
    model_DT： 决策树DT模型训练代码,调用sklearn机器学习工具包
    model_LR： 逻辑回归Logistic Regression模型训练代码,调用sklearn机器学习工具包
    ks_value： 模型KS统计量计算代码
    psi_value： 模型PSI值计算代码
    model_train： 所有模型训练代码
    GBDT_CV： 梯度提升决策树GBDT模型5折交叉验证代码
    model_CV： 其他模型5折交叉验证代码
用到pandas，sklearn，matplotlib工具包
"""


#=============================样本变量预处理==========================
def feature_pretreatment_value():
    """
    变量第一种预处理方式：
    所有离散变量进行woe值编码
    所有变量最大值大于10的连续变量取值归一化处理
    所有变量缺失值由变量均值填充
    :return:
    返回预处理后的样本变量值
    """
    samples = pd.read_csv('./data/samples_all.csv',index_col=0) # 读取样本集合，第一列是样本标识号
    cols = samples.columns

    # 离散变量woe值编码
    woe_col = [col for col in cols if samples[col].unique().size < 10] #变量取值个数小于10个视为离散变量
    for col in woe_col:
        if col != 'label':
            woe_df = calc_woe(samples[[col, 'label']], col)
            woe_dic = woe_to_dict(woe_df)
            samples[col] = samples[col].map(woe_dic)

    # 连续变量归一化
    rate_col = [x for x in cols if samples[x].max() > 10]
    for col in rate_col:
        samples[col] = normalize(samples[col])

    # 缺失值均值填充
    for col in cols:
        samples[col] = samples[col].fillna(samples[col].mean())
    return samples

def feature_pretreatment_woe():
    """
    变量第二种预处理方式（用于构建评分卡）：
    所有离散变量进行woe值编码
    所有连续变量分段，然后进行woe编码
    所有变量缺失值由变量均值填充
    :return:
    返回预处理后的样本变量值
    """

    samples = pd.read_csv('./data/samples_all.csv', index_col=0)  # 读取样本集合，第一列是样本标识号
    cols = samples.columns

    #连续变量分段编号
    bin_col = [col for col in cols if samples[col].unique().size > 10]
    for col in bin_col:
        tmp, sf = bins_num(samples[[col, 'label']], col, 3) #分3段，返回变量取值对应的分段号
        samples[col] = sf

    #所有变量woe值编码，缺失值填充
    for col in samples.columns:
        if col != 'label':
            woe_df = calc_woe(samples[[col, 'label']], col) #计算woe值
            woe_dic = woe_to_dict(woe_df)
            samples[col] = samples[col].map(woe_dic)
            samples[col] = samples[col].fillna(samples[col].mean()) #缺失值均值填充
    return samples


#=============================对比验证模型===============================
def model_GBDT(train,test):
    """
    梯度提升决策树GBDT模型训练，调用xgboost工具包
    :param train: pandas.DataFrame
    用于模型训练的样本
    :return:
    返回训练好的GBDT模型
    """
    # 数据训练集和测试集格式转换
    dtrain = xgb.DMatrix(train.iloc[:, :-1], label=train.iloc[:, -1])
    dval = xgb.DMatrix(test.iloc[:, :-1], label=test.iloc[:, -1])

    #模型参数字典
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'scale_pos_weight': 1100 / 200,  # 类别权重
        'eval_metric': 'auc',
        'gamma': 0.01,
        'silent': 1,
        'max_depth': 5,
        'lambda': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 0.1,
        'eta': 0.016,
        'nthread': 5
    }

    #模型训练
    evals_result = {}
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    xgb_model = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=200, verbose_eval=False,
                          evals=watchlist, evals_result=evals_result)
    return xgb_model

def model_SVM(train_X,train_Y):
    """
    支持向量机SVM模型训练,调用sklearn机器学习工具包
    :param train_X: pandas.DataFrame
    用于模型训练的样本变量值矩阵
    :param train_Y: pandas.DataFrame
    用于模型训练的样本标签
    :return:
    训练好的模型
    """
    m_svm = svm.SVC(class_weight='balanced',probability=True)
    m_svm.fit(train_X,train_Y)
    return m_svm

def model_NB(train_X,train_Y):
    """
    朴素贝叶斯Naive Bayes模型训练,调用sklearn机器学习工具包
    :param train_X: pandas.DataFrame
    用于模型训练的样本变量值矩阵
    :param train_Y: pandas.DataFrame
    用于模型训练的样本标签
    :return:
    训练好的模型
    """
    nb = GaussianNB() #高斯先验模型
    nb.fit(train_X,train_Y)
    return nb

def model_KNN(train_X,train_Y):
    """
    K近邻KNN模型训练,调用sklearn机器学习工具包
    :param train_X: pandas.DataFrame
    用于模型训练的样本变量值矩阵
    :param train_Y: pandas.DataFrame
    用于模型训练的样本标签
    :return:
    训练好的模型
    """
    knn = KNeighborsClassifier(n_neighbors=10,algorithm='auto', leaf_size=30, p=2)
    knn.fit(train_X,train_Y)
    return knn

def model_NN(train_X,train_Y):
    """
    神经网络Neural Network模型训练,调用sklearn机器学习工具包
    :param train_X: pandas.DataFrame
    用于模型训练的样本变量值矩阵
    :param train_Y: pandas.DataFrame
    用于模型训练的样本标签
    :return:
    训练好的模型
    """
    nn = MLPClassifier(activation='logistic',alpha=0.0001,learning_rate='constant',solver='adam')
    nn.fit(train_X,train_Y)
    return nn

def model_DT(train_X,train_Y):
    """
    决策树DT模型训练,调用sklearn机器学习工具包
    :param train_X: pandas.DataFrame
    用于模型训练的样本变量值矩阵
    :param train_Y: pandas.DataFrame
    用于模型训练的样本标签
    :return:
    训练好的模型
    """
    dt = DecisionTreeClassifier(class_weight='balanced',max_depth=8,criterion='gini')
    dt.fit(train_X,train_Y)
    return dt

def model_LR(train_X,train_Y):
    """
    逻辑回归Logistic Regression模型训练,调用sklearn机器学习工具包
    :param train_X: pandas.DataFrame
    用于模型训练的样本变量值矩阵
    :param train_Y: pandas.DataFrame
    用于模型训练的样本标签
    :return:
    训练好的模型
    """
    lg = LogisticRegressionCV(class_weight='balanced',max_iter=100,penalty='l2',solver='lbfgs')
    lg.fit(train_X,train_Y)
    return lg


#=============================模型评估(KS,PSI)===============================
def ks_value(y_true, y_pre):
    """
    模型的KS统计量
    :param y_true: array
    测试样本真实标签值
    :param y_pre: array
    模型预测的样本违约概率
    :return ks_value:float
    返回模型的ks统计量
    :return ks_value:float
    返回模型的auc值
    """
    fpr, tpr, t = roc_curve(y_true, y_pre)
    gap = tpr - fpr
    ks_value = np.max(gap)
    auc_value = auc(fpr, tpr)
    return ks_value, auc_value

def psi_value(model,y_pro):
    """
    计算训练后的模型在5个验证集上的PSI值
    :param model: model class
    训练好的的模型
    :param y_pro: array
    模型预测测试集样本的违约概率列表
    :return: pandas.DataFrame
    返回模型在5个验证样本集上的PSI值
    """
    PSI = pd.DataFrame()
    #统计测试集每个违约概率段的样本数占比
    t = pd.cut(y_pro,[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]).value_counts()
    PSI['test'] = t.apply(lambda x:x*1.0/t.sum() if x>0 else 0.00001)
    # 统计模型在5个验证集上每个违约概率段的样本数占比
    for i in xrange(5):
        #读取验证集样本
        val_data =pd.read_csv('./data/valid/val_data_%d.csv'%i,index_col=0)
        try:
            y_pre = model.predict_proba(val_data.iloc[:,:-1])
            #提取样本类别为坏样本的概率
            y_pre = [x[1] for x in y_pre]
        except:
            test_ma = xgb.DMatrix(val_data.iloc[:,:-1], val_data.iloc[:,-1])
            y_pre = model.predict(test_ma)
        y_pre_bin = pd.cut(y_pre,[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]).value_counts()
        PSI['valid%d'%i] = y_pre_bin.apply(lambda x:x*1.0/y_pre_bin.sum() if x>0 else 0.00001)
        # ks =ks_value(val_data.iloc[:,-1],y_pre)
        # print '模型在验证集上的ks统计量：%0.3f'%ks
    #计算PSI值
    psi = compute_psi(PSI)
    return psi

def compute_psi(PSI):
    """
    计算PSI值：psi=sum(test%-valid%)*ln(test%/valid%))
    :param PSI: pandas.DataFrame
    模型在测试集和5个验证集上的违约概率段样本数占比
    :return: pandas.Sereis
    模型在5个验证集上的PSI值
    """
    psi_df =pd.DataFrame()
    for col in PSI.columns[1:]:
        #test%-valid%
        PSI[col+'-t'] = PSI[col]-PSI['test']
        #ln(test%/valid%)
        PSI[col+'ln_t'] = np.log(PSI[col]*1.0/PSI['test'])
        psi_df[col+'_psi']=PSI[col+'-t'] *PSI[col+'ln_t']
    psi = psi_df.sum()
    return psi

def ks_plot(y_true, y_pre, model_name=None):
    """
    可视化KS曲线图
    :param y_true: array
    测试样本真实标签值
    :param y_pre: array
    模型预测的样本违约概率
    :param model_name: string
    保存KS曲线图的文件名
    """
    plt.figure()
    fpr, tpr, t = roc_curve(y_true, y_pre)
    plt.plot(fpr, label='False Positive Rate')
    plt.plot(tpr, label='Ture Positive Rate')

    gap = tpr - fpr
    ks_value = np.max(gap)
    ks_index = np.where(gap == ks_value)
    y = [fpr[ks_index], tpr[ks_index]]
    x = [int(ks_index[0]), int(ks_index[0])]

    plt.plot(x, y, label='KS_max=%.2f' % ks_value)
    plt.title("%s KS graph\n th=%.3f" % (model_name, t[ks_index]))
    plt.legend(loc="mid right")
    if model_name:
        plt.savefig('./%s_%.2f.png' % (model_name, ks_value))
    plt.show()

def auc_plot(y_true, y_pre, model_name=None):
    """
    可视化ROC曲线图
    :param y_true: array
    测试样本真实标签值
    :param y_pre: array
    模型预测的样本违约概率
    :param model_name: string
    保存KS曲线图的文件名
    """
    plt.figure()
    fpr, tpr, th = roc_curve(y_true, y_pre)
    auc_value = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label="ROC fold (AUC=%.2f)" %auc_value)
    plt.xlim([-0.05, 1.05])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("%s ROC curve" % model_name)
    plt.legend(loc='lower right')
    if model_name:
        plt.savefig('./%s_%.2f.png' % (model_name, auc_value))
    plt.show()


#=============================模型训练（5折交叉验证）===============================
def model_train(model_name,cols_filter,model_type='ScoreCard'):
    """
    模型5折交叉验证训练，根据模型的ks值和psi值选择最优模型
    :param model_name: model class
    用于训练的模型
    :param cols_filter: list
    指定加入模型训练的样本变量列表
    :param model_type: string
    指定训练的模型类别，若为ScoreCard，则表示训练好的模型将用于构建评分卡，因此变量需做离散化处理，且模型只能是LR
    若为其他值，则表示训练好的模型不用做构建评分卡，但可作为评分卡的辅助交叉验证模型
    :return: ks:list, psi:pandas.DataFrame
    返回模型的两个评估指标：ks和PSI值
    """
    #根据模型最终用途选择变量预处理方式
    if model_type == 'ScoreCard':
        samples = feature_pretreatment_woe()
        samples = samples[cols_filter]
    else:
        samples = feature_pretreatment_value()
        samples = samples[cols_filter]

    #保存验证集
    save_valid_data(samples)
    #5折交叉验证样本集划分
    skf = StratifiedKFold(samples.iloc[:,-1], n_folds=5)
    #模型训练
    if model_name == model_GBDT:
        ks,psi = GBDT_CV(samples,skf)
    else:
        ks,psi = model_CV(model_name,samples,skf)

    return ks,psi

def GBDT_CV(samples,skf):
    """
    采用5折交叉验证的方式进行GBDT最优模型训练
    :param samples: pandas.DataFrame
    用于模型训练的样本集
    :param skf: StratifiedKFold class
    样本划分集合（将样本均分为5等份，每份中的好坏样本占比等于总体好坏样本占比）
    :return ks_list: list
    返回5个GBDT模型的ks值和auc值
    :return psi_df: pandas.DataFrame
    返回5个GBDT模型在5个验证集上的psi值
    """
    ks_list = []
    psi_df = pd.DataFrame()
    count = 0
    for train,test in skf:
        train_data = samples.iloc[train]
        test_data = samples.iloc[test]
        test_ma=xgb.DMatrix(test_data.iloc[:,:-1],test_data.iloc[:,-1])
        gbdt = model_GBDT(train_data,test_data)
        y_pro = gbdt.predict(test_ma)
        # 评估模型效果
        ks = ks_value(test_data.iloc[:,-1],y_pro)
        ks_list.append(ks)
        psi_df['model_%d'%count] = psi_value(gbdt,y_pro)
        #保存模型
        joblib.dump(gbdt, "./data/model/GBDT_%d.m" %count)
        count = count + 1
    return ks_list,psi_df

def model_CV(model_name,samples,skf):
    """
    采用5折交叉验证的方式进行最优模型训练
    :param model_name: model class
    用于训练的模型
    :param samples: pandas.DataFrame
    用于模型训练的样本集
    :param skf: StratifiedKFold class
    样本划分集合（将样本均分为5等份，每份中的好坏样本占比等于总体好坏样本占比）
     :return ks_list: list
    返回5个模型的ks值和auc值
    :return psi_df: pandas.DataFrame
    返回5个模型在5个验证集上的psi值
    """
    ks_list = []
    psi_df = pd.DataFrame()
    count = 0
    for train,test in skf:
        train_data = samples.iloc[train]
        test_data = samples.iloc[test]
        model = model_name(train_data.iloc[:,:-1],train_data.iloc[:,-1])
        y_pro = model.predict_proba(test_data.iloc[:,:-1])
        #提取样本类别为坏样本的概率
        y_pro = [x[1] for x in y_pro]
        #评估模型效果
        ks = ks_value(test_data.iloc[:,-1],y_pro)
        ks_list.append(ks)
        psi_df['model_%d' % count] = psi_value(model, y_pro)
        #保存模型
        joblib.dump(model, "./data/model/model_%d.m" % count)
        count = count + 1
    return ks_list, psi_df

def save_valid_data(samples):
    """
    从给定的样本集合中5次随机抽取40%作为模型验证集，用于计算模型PSI值
    :param samples: pandas.DataFrame
    用于验证集提取的样本集合
    """
    for i in xrange(5):
        train_X,val_X,train_Y,val_Y =train_test_split(samples.iloc[:,:-1],samples.iloc[:,-1],test_size=0.4)
        val_X['label']=val_Y
        val_X.to_csv('./data/valid/val_data_%d.csv'%i)


if __name__=='__main__':
    #最优评分卡构建模型
    cols_filter = ['实收资本', '注册地址', '场地归属', '注册时间', '下游客户情况', '涉诉信息', '有无贷款', '企业征信', '净资产', '流动比率',
                   '资产负债率', '主营业务利润率', '应收账款周转天数', '存货周转天数', '季度逾期占比', '季度平均逾期天数', '历史_逾期占比',
                   '历史_提前还款占比', '季度提前还款占比', '历史_平均逾期天数', '平均逾期天数变化', '逾期占比变化', '订单总数', '交易额总量',
                   '历史_交易稳定性', '交易量最小值', '交易总量', '交易额最小值', '交易额方差', '提前还款占比变化', '历史_账期订单交易量均值',
                   '历史_交易额均值', '历史_账期订单交易额均值', '交易额均值', '交易稳定性', '历史_最近连续交易月数', '历史_账期合作时间（月）',
                   '订单时间间隔均值变化', '账期订单交易量均值变化', '账期订单交易额均值变化','label']

    ks,psi = model_train(model_GBDT,cols_filter)
    print 'GBDT'
    print ks, '\n', psi
    ks, psi = model_train(model_SVM, cols_filter)
    print 'SVM'
    print ks, '\n', psi
    ks, psi = model_train(model_NB, cols_filter)
    print 'NB'
    print ks,'\n', psi
    ks, psi = model_train(model_KNN, cols_filter)
    print 'KNN'
    print ks,'\n', psi
    ks, psi = model_train(model_NN, cols_filter)
    print 'NN'
    print ks, '\n', psi
    ks, psi = model_train(model_DT, cols_filter)
    print 'DT'
    print ks, '\n', psi
    ks, psi = model_train(model_LR, cols_filter)
    print 'LR'
    print ks, '\n', psi
