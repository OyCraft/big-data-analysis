# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 15:09:09 2021

@author: Ouyang
"""

import math
import copy
import numpy as np
import pandas as pd

def LR(xx, yy, txx, w = 0.1, e = 0.0001, gamma = 0.005, maxI = 5000, b = True):
    x = copy.deepcopy(xx)           #深拷贝，防止传递引用
    y = copy.deepcopy(yy)
    tx = copy.deepcopy(txx)
    
    if b==True:                     #b为真表示包含常数项
        x.insert(0, "b", 1)
        tx.insert(0, "b", 1)
    n = np.shape(x)[0]              #训练集观测数
    p = np.shape(x)[1]              #变量总数
    
    #初始权重处理
    if (type(w)==int) or (type(w)==float):
      w = np.zeros(p) + w 
    else:
        w = w.astype(np.float64)
        if len(w)<p:
            wcopy = w
            for i in range(len(wcopy) + 1, p + 1):
                mod = i % len(wcopy)
                if mod==0:
                    w = np.append(w, wcopy[len(wcopy) - 1])
                else:
                    w = np.append(w, wcopy[mod - 1])
        elif len(w)>p:
            w = w[0:p]
            
    #主循环部分，迭代产生权重
    convFlag = False                #convFlag指示是否收敛
    for k in range(0, maxI + 1):    #k表示迭代次数
        nw = copy.deepcopy(w)
        if k%100==0: 
            print("迭代：" + str(k))
        
        #核心计算部分，为了加快速度全部换成了矩阵运算
        z = pd.Series(np.dot(x, w))
        diff = np.dot(z.apply(lambda x: (1 / (1 + math.exp(-x)))) - y, x)
        nw = w - gamma * (1/n) * diff
        
        #比较新的权重和旧权重的差异
        a = np.linalg.norm(nw - w) / np.linalg.norm(w)
        w = nw
        if a<e:
            convFlag = True         #提示成功收敛
            break
    if convFlag==False:
        print("警告：权重未能在最大迭代次数内收敛")
    
    #回判及预测部分
    tn = np.shape(tx)[0]            #测试集观测数
    result1 = np.zeros(n)           #初始化回判结果
    result2 = np.zeros(tn)          #初始化测试结果
    z1 = pd.Series(np.dot(x, w))
    result1 = z1.apply(lambda x: (1 / (1 + math.exp(-x))))
    z2 = pd.Series(np.dot(tx, w))
    result2 = z2.apply(lambda x: (1 / (1 + math.exp(-x))))
        
    return result1, result2, w
 
    
#下面使用mushroom数据集进行测试
def load_process_data(y_name='class'):
    
    # data_precess
    data = pd.read_csv("C:\\Users\\Ouyang\\Desktop\\One\\Workspace\\github repo\\mushrooms.csv")
    c1,c2 = data['class'].value_counts() # 4208:3916
    data_rebuild = data
    #所有特征的one hot 处理
    for f_name in data.columns[1:np.shape(data)[1]]:
        f_name_Df = pd.get_dummies(data_rebuild[f_name] , prefix=f_name )
        data_rebuild = pd.concat([data_rebuild,f_name_Df],axis=1)
        data_rebuild.drop(f_name,axis=1,inplace=True)
       
    train_data, test_data = data_rebuild[0:int(len(data_rebuild) * 0.8)], data_rebuild[int(len(data_rebuild) * 0.8):]
    train_x, train_y = train_data, train_data.pop(y_name)
    test_x, test_y = test_data, test_data.pop(y_name)
    train_y = train_y.replace('e',0)
    train_y = train_y.replace('p',1)
    test_y = test_y.replace('p',1)
    test_y = test_y.replace('e',0)
    return train_x, train_y,test_x, test_y


#主程序
train_x, train_y,test_x, test_y = load_process_data()

r1, r2, w = LR(train_x, train_y, test_x, w = 0.1, gamma = 0.05)

train_comp = pd.concat([r1, train_y], 1)
r2.index = test_y.index
test_comp = pd.concat([r2, test_y], 1)

r22 = r2>0.5                                            #不妨以0.5为决策边界
COR = sum(r22==test_y) / len(r22)                       #正确率
TPR = sum((r22==test_y) & (r22==1)) / sum(test_y==1)    #真阳性率
TFR = sum((r22==test_y) & (r22==0)) / sum(test_y==0)    #真阴性率
print("测试观测数：" + str(len(r22)) + "    阳性个数：" + str(sum(test_y==1)) + \
      "\n综合正确率：" + str(COR) + "\n真阳性率：" + str(TPR) + "\n真阴性率：" + \
      str(TFR) + "\n阳性代表蘑菇品种有毒，阴性代表蘑菇品种无毒")