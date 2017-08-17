#coding:utf-8
__author__='lt'

import json
import csv
import pandas as pd
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def goods_data():
    '''
    数据格式化为每个商品一条数据
    :return:
    '''
    file = open('./data/orders_17w.json','r')
    print file.readline()

    f_out = csv.writer(open('./data/goods_17w.csv','w'))
    cols = ['shopNick','shopName', 'shopId', 'ordNum', 'quantity']
    f_out.writerow(['uid', 'shopNick','shopName', 'shopId', 'ordNum', 'quantity','goodsSize', 'goodsName', 'promotionPrice', 'originalPrice', 'goodsNum'])

    for line in file:
        d = json.loads(line.replace('￥',''))
        orders = d['orders']
        for order in orders:
            l1 = []
            l1.append(d['uid'])
            l1.extend([order[i] for i in cols])
            goods = order['goodsInfo']
            # print order.keys()
            for g in goods:
                l2 = l1[:]
                f= g.values()
                l2.extend(f)
                f_out.writerow(l2)
    print 'goods ok!'


def orders_data():
    '''
    uid-order
    :return:
    '''
    file = open('./data/orders_17w.json','r')
    f_out = csv.writer(open('./data/usr_orders_17w.csv','w'))
    f_out.writerow(['uid','huabeiAmount','huabeiTotal', 'actualFee', 'ordNum', 'postFee', 'shopNick', 'shopId',
                    'ordTime', 'orderStatus', 'orderTimeout','quantity', 'shopName'])
    for line in file:
        # d = json.loads(file.readline().replace('￥', ''))
        d = json.loads(line.replace('￥',''))
        orders = d['orders']
        for order in orders:
            l1 = []
            l1.append(d['uid'])
            l1.append(d['huabeiAmount'])
            l1.append(d['huabeiTotal'])
            l1.extend([order[i] for i in order.keys() if i!='goodsInfo'])
            f_out.writerow(l1)
    print 'orders ok!'



def read_csv():
    data = pd.read_csv('./data/tb_order.csv', sep=',',nrows=10000)


goods_data()
# orders_data()