# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 21:08:57 2018

@author: 123
"""
import numpy as np

def GM11(x0):
    #先判断原数列能否作为灰色与猜测的数列使用
    '''
    ramda=np.array([x0[k-1]/x0[k] for k in np.arange(1,len(x0))])
    ramda1=np.exp(-2/(len(x0)+1))
    ramda2=np.exp(-2/(len(x0)+2))
    '''
    x1=x0.cumsum() #x1为x0的1次累加生成序列
    z1=np.array([0.5*x1[:len(x1)-1]+0.5*x1[1:]]).reshape(-1,1) #z1位x1的均值生成序列
    Y=np.array([x0[1:]]).reshape(-1,1)
    B1=np.array(-z1)
    B2=np.ones((len(Y),1))
    B=np.hstack((B1,B2))
    [[a],[b]]=np.dot(np.dot(np.linalg.inv(np.dot(B.T,B)),B.T),Y)
    func=lambda k:(x0[0]-b/a)*np.exp(-a*k)-(x0[0]-b/a)*np.exp(-a*(k-1))
    return func

if __name__ == "__main__":
    x0=np.array([71.1,72.4,72.4,72.1,71.4,72.0,71.6])
    ramda=np.array([x0[k-1]/x0[k] for k in np.arange(1,len(x0))])

    model=GM11(x0)

    x0_pred=[]
    x0_pred.append(x0[0])
    for i in np.arange(1,len(x0)):
        x0_pred.append(model(i))
    
    print(x0_pred)