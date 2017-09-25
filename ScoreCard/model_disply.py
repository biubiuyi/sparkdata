#coding:utf-8
__author__='lt'


from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
对模型训练结果进行可视化
'''

class model_See():

    def __init__(self,model,y_pre,y_true,model_type='log'):
        self.model = model
        self.y_pre = y_pre
        self.y_true = y_true
        self.model_type = model_type

    def ks_plot(self):
        fpr,tpr,t = roc_curve(self.y_true,self.y_pre)
        plt.plot(fpr,label='FPR')
        plt.plot(tpr,label='TPR')

        gap = fpr-tpr
        ks_index = np.where(gap == np.max(gap))
        y = [fpr[ks_index], tpr[ks_index]]
        x = [int(ks_index[0]), int(ks_index[0])]
        ks_value = np.max(gap)
        plt.plot(x,y)
        plt.title("KS graph\nKS=%.2f" % ks_value)
        plt.legend(loc="mid right")
        plt.show()

    def auc_plot(self):
        fpr,tpr,th = roc_curve(self.y_true, self.y_pre)
        auc_value = auc(fpr,tpr)
        plt.plot(fpr,tpr,lw=1,label="ROC fold (AUC=%.2f)" % (auc_value))
        plt.xlim([-0.05,1.05])
        plt.ylim([0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend(loc='lower right')
        plt.show()

    def feature_sort(self):
        features = pd.DataFrame.from_dict(self.model.get_fscore())
        feat_top = features.sort_values(by=0,ascending=False).head(20)
        feat_top.plot(kind='barh')
        plt.show()

    def show(self):
        if self.model_type=='log':
            print "auc value of model:%.3f" % (self.score)
            self.auc_plot()
            self.ks_plot()
            self.feature_sort()
        elif self.model_type=='lin':
            print "mae value of model:%.3f" %(self.score)
            self.feature_sort()
