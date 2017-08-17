#coding:utf-8
#created on 2017-06-08
__author__='lt'


#Transformation and cross function:['+', '-', '*', '/','1/x','1/x+1/y','x^2+y^2','log','fft']
funcs=['+', '-', '*', '/','1/x','1/x+1/y','x^2+y^2','log','fft']


#==特征预处理
#离散变量数值化,数值化方法["dummy","factorize"]
dummy_type = "dummy"
#特征归一化,可选方法:[]
normalize_type = ""

#==特征筛选
#空值率,大于该值剔除
null_ratio = 0.2
#特征方差,小于该值剔除
std_num = 0.0
#特征与样本标签相关性绝对值,小于该值剔除
corr_f2label = 0.001
#特征之间相关性绝对值,大于该值则剔除其中一个特征
corr_f2f = 0.8


#相关性计算方法,可选['MIC','pearson','kendall','spearman','IV']
corr_type = 'pearson'
#特征分块多进程处理的块数
block = 4
#进程池进程数量,默认值为CPU核数,4核
process_num = 4
#将回归模型样本标签离散化的离散阈值(便于计算IV值)
label_thr = 600
#交叉特征筛选的类型,使用xgb模型筛选为model,使用IV值筛选为IV
filter_type='model'
#训练模型类型,分类为log,回归为lin
model_type='lin'
#用于特征排序的xgboost模型个数
xgb_num = 2
#计算IV值时对连续型特征分箱的数量
bin_num = 5
#筛选保留特征的特征比例
topk_ratio = 0.5


#缓存数据路径
path='../tmp'
