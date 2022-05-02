# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:28:52 2022

@author: Erlend Johansen
"""

import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt


import dataPresentation
import dataGeneration


def recalculateValues(order,detail,k,stencil):
    recalculate=True
    print(f"\nRecalculate order {order} detail {detail}")
    dataGeneration.corners(order, detail, recalculate)
    dataGeneration.boundaryMatrix(order, detail, recalculate)
    dataGeneration.insidePointsRays(order, detail, recalculate)
    dataGeneration.stencilMatrix(order, detail, recalculate, stencil)
    dataGeneration.eigensolutions(order, detail, k, recalculate, stencil)

if __name__=="__main__":    
    
    stencils=["L5p","L9p"]
    
    detail=1
    k=10
    
    recalculateValues(2,1,10,stencils[0])
    recalculate=False
    start=time.time()
    for stencil in stencils:
        orders=range(2,7+1)
        for order in orders:
            print(f"\nOrder {order} detail {detail}")
            dataGeneration.corners(order, detail, recalculate)
            dataGeneration.boundaryMatrix(order, detail, recalculate)
            dataGeneration.insidePointsRays(order, detail, recalculate)
        orders=range(2,4+1)
        for order in orders:
            dataGeneration.stencilMatrix(order, detail, recalculate, stencil)
            dataGeneration.eigensolutions(order, detail, k, recalculate, stencil)
    end=time.time()
    print(f"Data calculation time elapsed: {end-start}\n")
    
    order,detail=3,1
    stencil=stencils[1]
    dataGeneration.stencilMatrix(order=order, detail=detail, recalculate=True, stencil=stencil)
    dataGeneration.eigensolutions(order=order, detail=detail, k=1000, recalculate=True, stencil=stencil)
    
    """
    folderName=dataGeneration.getFolderName(stencils[0])
    stencilMatrixFileName=os.path.join(folderName,f"stencilMatrix_order{order}_detail{detail}.npy")
    stencilMatrix=np.load(stencilMatrixFileName,allow_pickle=True).item()
    print(f"{stencilMatrix.data.nbytes + stencilMatrix.indptr.nbytes + stencilMatrix.indices.nbytes} bytes")
    """
    
    """
    tic=time.time()
    dataGeneration.calculateNumberOfInsidePoints()
    toc=time.time()
    print("Inside points time:",toc-tic)
    """
    """
    numberOfInsidePointsFileName=os.path.join("fractalData","numberOfInsidePoints.npy")
    numberOfInsidePoints=np.load(numberOfInsidePointsFileName)
    print(numberOfInsidePoints[:10,1])    
    """
    """
    orders=range(2,7+1)
    dataPresentation.plotInsidePointPerformance(orders, recalculate=False)
    """
    
    
    """
    dataPresentation.plotFractalBoundary(2, 1)
    dataPresentation.plotFractalInside(2, 1)
    dataPresentation.plotFractalBoundary(3, 1)
    dataPresentation.plotFractalInside(3, 1)
    """
    """
    dataPresentation.printEigenValues(orders=[4], stencils=stencils)
    """
    """ 
    order=4
    detail=1
    
    
    
    for stencil in stencils[1:]:
        dataPresentation.plotEigenvectors(order, detail,stencil)
        #dataPresentation.plotEigensolverTimes(stencil)
    
    
    dataPresentation.plotLocalizedState(order, detail, stencils[1])
    """
    
    
    
    
    