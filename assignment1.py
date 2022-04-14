# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:28:52 2022

@author: Erlend Johansen
"""

import numpy as np
import os
import time

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
    stencil=stencils[0]
    orders=[2,3]
    details=[1,2,4,8]
    k=10
    recalculate=False
    start=time.time()
    for order in orders:
        for detail in details:
            print(f"\nOrder {order} detail {detail}")
            dataGeneration.corners(order, detail, recalculate)
            dataGeneration.boundaryMatrix(order, detail, recalculate)
            dataGeneration.insidePointsRays(order, detail, recalculate)
            dataGeneration.stencilMatrix(order, detail, recalculate, stencil)
            dataGeneration.eigensolutions(order, detail, k, recalculate, stencil)
    dataGeneration.calculateNumberOfInsidePoints()
    end=time.time()
    print(f"Data calculation time elapsed: {end-start}\n")
           
    recalculateValues(1,2,10,stencil)
    recalculateValues(1,4,10,stencil)
    recalculateValues(1,8,10,stencil)
    
            
    order=3
    detail=2       
    
    #dataPresentation.plotEigenvectors(order, detail,stencil)
    
    
    dataPresentation.plotEigensolverTimes(stencil)
    
    
    
    
    
    