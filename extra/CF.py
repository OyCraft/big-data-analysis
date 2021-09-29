# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:50:43 2021

@author: Ouyang
"""

import copy
import numpy as np
import pandas as pd
import genMatrix as gM

def simMat(m, itemCF = False, simMethod = "cosine"):
    mat = copy.deepcopy(m)
    
    #为了方便计算相似度，把无打分的位置记作0
    mat[mat==-1] = 0
    
    #若为itemCF，只需转置矩阵即可
    if itemCF==True:
        mat = mat.T
    
    #初始化相似度矩阵   
    nRow = np.shape(mat)[0]
    nCol = np.shape(mat)[1] 
    
    #求行列打分均值（去掉未打分的格子），并找出从未打分的用户或商品
    noDataRow = [""]
    noDataCol = [""]
    mRow = pd.Series([0.0] * nRow, index = mat.index)  #储存行均值（浮点型）
    for i in range(0, nRow):
        ite = mat.iloc[i]
        if sum(ite!=0)==0:      #从未打分
            noDataRow.append(mat.index[i])
        else:
            #下面的判断是防止出现所有评分相等，减去均值等于0的情况。此时不应算均值
            if len(ite.unique())>2 or (len(ite.unique())==2 & sum(ite==0)==0):
                mRow[i] = np.mean(ite[ite!=0])
            
    mCol = pd.Series([0.0] * nCol, index = mat.columns)   #储存列均值
    for j in range(0, nCol):
        ite = mat.iloc[:, j]    #当前列
        if sum(ite!=0)==0:      #从未被打分
            noDataCol.append(mat.columns[j])
        else:
            if len(ite.unique())>2 or (len(ite.unique())==2 & sum(ite==0)==0):
                mCol[j] = np.mean(ite[ite!=0])
    
    #删除无评分的行列     
    del noDataRow[0]
    del noDataCol[0]
    mat = mat.drop(index = noDataRow)
    mat = mat.drop(columns = noDataCol)
    mRow = mRow.drop(index = noDataRow)
    mCol = mCol.drop(index = noDataCol)
        
    #计算有评分的相似度矩阵（这是个对称矩阵，且对角元素无意义，设为0）
    nRow = np.shape(mat)[0]
    nCol = np.shape(mat)[1] 
    simMat = np.zeros([nRow, nRow])
    for i in range(0, nRow):
        simMat[i, i] = 0
        
        for j in range(i + 1, nRow):
            a = np.array(mat.iloc[i])
            b = np.array(mat.iloc[j])
            
            #下面使用相关系数打分，注意没有打分（等于0）的格子不需要纠偏
            if simMethod=="pearson_same":
                a[a!=0] = a[a!=0] - np.array([mRow[i]] * sum(a!=0))
                b[b!=0] = b[b!=0] - np.array([mRow[j]] * sum(b!=0))
            elif simMethod=="pearson_alter":
                a[a!=0] = a[a!=0] - mCol[a!=0]
                b[b!=0] = b[b!=0] - mCol[b!=0]
            
            #计算向量余弦值
            simMat[i, j] = sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b))      
            simMat[j, i] = simMat[i, j]
    
    #继承mat的用户名和商品名      
    simMat = pd.DataFrame(simMat, index = mat.index, columns = mat.index)
    
    if itemCF==True:
        return simMat, noDataCol, noDataRow
    else:
        return simMat, noDataRow, noDataCol


def CF(m, uID, top = 5, rec = 5, itemCF = False, simMethod = "cosine",
       simBound = 0, positive = 0):
    [sMat, ndU, ndI] = simMat(m, itemCF, simMethod)
    mat = copy.deepcopy(m)
    mat[mat==-1] = 0
    
    #初始化推荐数据框，每行代表一个用户，列从左至右依次代表推荐物品
    recMat = pd.DataFrame(np.full([len(uID), rec], ""), index = uID,
                          columns = range(0, rec))
    
    if itemCF==False:
        for i in range(0, len(uID)):
            #如果用户没有购买任何商品，提示无数据
            if uID[i] in ndU:
                recMat.loc[uID[i], 0] = "no user data!"
            else:
                #寻找最相似的top个用户
                simU = [""] * top
                simRow = copy.deepcopy(sMat.loc[uID[i]])
                for j in range(0, top):
                    if simRow.max()>simBound:   #相似度至少要大于simBound
                        simU[j] = simRow.idxmax()   #标记用户名称
                        simRow[simRow.idxmax()] = simBound
                
                if simU[0]=="":     #此时无法提供任何推荐
                    recMat.loc[uID[i], 0] = "no similar user"
                else:
                    #基于最相似的用户，预测目标用户所有未购买商品的评分
                    new = copy.deepcopy(mat.loc[uID[i]])       #new是新的打分
                    new[new!=0] = -2    #已经买过的商品不再打分
                    
                    #j遍历所有打分格子，k遍历最相似的用户
                    simRow = copy.deepcopy(sMat.loc[uID[i]])
                    for j in range(0, len(new)):
                        if new[j]==0:
                            up = 0
                            down = 0
                            
                            #尝试推荐rec个商品
                            for k in range(0, top):
                                if simU[k]!="":
                                    up = up + simRow[simU[k]] * \
                                         mat.loc[simU[k]][j]
                                    down = down + simRow[simU[k]]
                            new[j] = up / down
                            
                    #尝试推荐rec个最高评分的商品
                    for j in range(0, rec):
                        if new.max()>simBound:
                            recMat.loc[uID[i], j] = new.idxmax() + " " + \
                                                    str(new.max())
                            new[new.idxmax()] = -2
                        else:
                            recMat.loc[uID[i], j] = "lower than simBound"
                            break
                                               
    else:
        for i in range(0, len(uID)):           
            if uID[i] in ndU:
                recMat.loc[uID[i], 0] = "no user data!"
            else:
                #寻找用户i的正反馈商品表
                matRow = copy.deepcopy(mat.loc[uID[i]])
                pos = matRow[matRow>positive].index
                
                if len(pos)==0:     #如果没有正反馈商品，则无法提供任何推荐
                    recMat.loc[uID[i], 0] = "no positive feedback"
                else:
                    #计算用户i所有未购买商品与正反馈商品的相似度
                    new = copy.deepcopy(matRow)
                    new[new!=0] = -2
                    
                    #j遍历所有打分格子，k遍历所有正反馈商品
                    for j in range(0, len(new)):
                        if new[j]==0 and (new.index[j] not in ndI):
                            for k in range(0, len(pos)):
                                new[j] = new[j] + matRow[pos[k]] * \
                                         sMat.at[new.index[j], pos[k]]
                    
                    #找出相似度最高的rec个商品
                    for j in range(0, rec):
                        if new.max()>simBound:
                            recMat.at[uID[i], j] = new.idxmax() + " " + \
                                                   str(new.max())
                            new[new.idxmax()] = simBound
                        else:
                            recMat.at[uID[i], j] = "lower than simBound"
                            break
                            
    return recMat, sMat
   

             
mat = gM.genMatrix(10, 10, [1, 10])

uID = mat.index[[2, 4, 9]]

[recMat, sMat] = CF(mat, uID, itemCF = True, simMethod = "pearson_same", 
                 positive = 5)

#mat要求：用户和商品名称不重复，无打分项设定为-1，打分区间大于0   

 
        
    
    
    


