# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 17:26:46 2022

@author: Erlend Johansen
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import os



import fractalGeneration
from dataGeneration import getFolderName


def plotEigenvectors(order,detail,stencil):
    folderName=getFolderName(stencil)
    
    valsFileName=os.path.join(folderName,f"eigenVals_order{order}_detail{detail}.npy")
    vals=np.load(valsFileName)
    
    vecsFileName=os.path.join(folderName,f"eigenVecs_order{order}_detail{detail}.npy")
    vecs=np.load(vecsFileName)
    
    insideMatrixFileName=os.path.join("fractalData",f"inside_order{order}_detail{detail}.npy")
    insideMatrix=np.load(insideMatrixFileName)
    
    for i in range(np.shape(vecs)[1]):
        plt.figure(i)
        print(vals[i])
        plt.imshow(fractalGeneration.recreateFractal(vecs[:,i],insideMatrix))    

    
    plt.matshow(insideMatrix)
    
    
def plotEigensolverTimes(stencil):
    folderName=getFolderName(stencil)
    eigensolverTimesFileName=os.path.join(folderName,"eigensolverTimes.npy")
    eigensolverTimes=np.load(eigensolverTimesFileName)
    
    numberOfInsidePointsFileName=os.path.join("fractalData","numberOfInsidePoints.npy")
    numberOfInsidePoints=np.load(numberOfInsidePointsFileName)
    
    fig,ax=plt.subplots()
    for order in range(20):
        data_x=[]
        data_y=[]
        for detail in range(20):
            if numberOfInsidePoints[order,detail] and eigensolverTimes[order,detail]:
                data_x.append(numberOfInsidePoints[order,detail])
                data_y.append(eigensolverTimes[order,detail])
        if(len(data_x)):
            ax.scatter(data_x,data_y,label=f"Order {order}")
            ax.set_xscale("log")
            ax.set_yscale("log")
    ax.legend()
    
    
#def plotInsidePoints():