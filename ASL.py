# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 23:58:05 2019

@author: 蒋平
"""
import torch
import numpy as np
import time
def Weight(nclass=7, fdims=512):
    lr = 0.1
    y=torch.randn(nclass,fdims)
    y.requires_grad = True
    y.data = y.data/y.data.norm(dim=1).view([nclass,1])
    for ep in range(20000):        
        if ep>2000 and ep % 100 ==0:
            lr = lr*0.90
        loss = 0    
        for i in range(nclass-1):
            for j in range(i+1,nclass):
                loss += (y[i].matmul(y[j])-1.0/(1-nclass)).pow(2)
        loss.backward()
        y.data -= lr*y.grad
        y.data = y.data/y.data.norm(dim=1).view([nclass,1])
        y.grad.zero_()
    return y.data

#st = time.time()
#x =  Weight(nclass=7, fdims=512)
#print(time.time()-st)