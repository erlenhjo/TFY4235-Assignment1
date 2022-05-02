# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 17:54:24 2022

@author: Erlend Johansen
"""


#from numba import njit
import numpy as np
import scipy.sparse.linalg as lin
import time
import os
import sys

import fractalGeneration
import matrixGeneration

def corners(order,detail,recalculate):
    print("Calculate corners:")
    
    dataFileName=os.path.join("fractalData",f"corners_order{order}_detail{detail}.npy")
    timeFileName=os.path.join("fractalData","calculateCornersTimes.npy")
    
    if (not recalculate and os.path.exists(dataFileName)):
        print(f"Order {order} detail {detail} already calculated.")
        
    else:
        start = time.time()
        corners=fractalGeneration.generatePath(order)
        if(detail>1):
            corners=fractalGeneration.addDetailToPath(corners, detail)
        end = time.time()
        
        
        print("Time elapsed = %s" % (end - start))
        data=corners
        
        times=np.load(timeFileName)
        times[order,detail]=end-start
        np.save(timeFileName,times)
        np.save(dataFileName,data)
        
def boundaryMatrix(order,detail,recalculate):
    print("Generate boundary matrix:")
    
    dataFileName=os.path.join("fractalData",f"boundaryMatrix_order{order}_detail{detail}.npy")
    timeFileName=os.path.join("fractalData","generateBoundaryMatrixTimes.npy")
    
    if (not recalculate and os.path.exists(dataFileName)):
        print(f"Order {order} detail {detail} already calculated.")
        
    else:
        cornersFileName=os.path.join("fractalData",f"corners_order{order}_detail{detail}.npy")
        corners=np.load(cornersFileName)
        
        start = time.time()
        boundaryMatrix=fractalGeneration.generateBoundaryMatrix(corners)
        end = time.time()
        
        print("Time elapsed = %s" % (end - start))
        
        data=boundaryMatrix
        
        times=np.load(timeFileName)
        times[order,detail]=end-start
        np.save(timeFileName,times)
        np.save(dataFileName,data)
    


def insidePointsRays(order,detail,recalculate):
    print("Find inside points (ray):")
    
    dataFileName=os.path.join("fractalData",f"inside_order{order}_detail{detail}.npy")
    timeFileName=os.path.join("fractalData","insideRayTimes.npy")
    
    if (not recalculate and os.path.exists(dataFileName)):
        print(f"Order {order} detail {detail} already calculated.")
        
    else:
        cornersFileName=os.path.join("fractalData",f"corners_order{order}_detail{detail}.npy")
        corners=np.load(cornersFileName)
        boundaryMatrixFileName=os.path.join("fractalData",f"boundaryMatrix_order{order}_detail{detail}.npy")
        boundaryMatrix=np.load(boundaryMatrixFileName)
        
        start = time.time()
        insideMatrix=fractalGeneration.findInsidePointsRays(corners, boundaryMatrix)
        end = time.time()
        
        print("Time elapsed = %s" % (end - start))
        
        data=insideMatrix
        
        times=np.load(timeFileName)
        times[order,detail]=end-start
        np.save(timeFileName,times)
        np.save(dataFileName,data)


def insidePointsWinding(order,detail,recalculate):
    print("Find inside points (winding):")
    
    dataFileName=os.path.join("fractalData",f"inside_order{order}_detail{detail}.npy")
    timeFileName=os.path.join("fractalData","insideWindingTimes.npy")
    
    if (not recalculate and os.path.exists(dataFileName)):
        print(f"Order {order} detail {detail} already calculated.")
        
    else:
        cornersFileName=os.path.join("fractalData",f"corners_order{order}_detail{detail}.npy")
        corners=np.load(cornersFileName)
        boundaryMatrixFileName=os.path.join("fractalData",f"boundaryMatrix_order{order}_detail{detail}.npy")
        boundaryMatrix=np.load(boundaryMatrixFileName)
        
        start = time.time()
        insideMatrix=fractalGeneration.findInsidePointsWinding(corners,boundaryMatrix)
        end = time.time()
        
        print("Time elapsed = %s" % (end - start))
        
        data=insideMatrix
        
        times=np.load(timeFileName)
        times[order,detail]=end-start
        np.save(timeFileName,times)
        np.save(dataFileName,data)
    

def stencilMatrix(order,detail,recalculate,stencil):
    print(f"Stencil matrix generator {stencil}:")
    folderName=getFolderName(stencil)
    
    dataFileName=os.path.join(folderName,f"stencilMatrix_order{order}_detail{detail}.npy")
    timeFileName=os.path.join(folderName,"matrixGenerationTimes.npy")
    
    if (not recalculate and os.path.exists(dataFileName)):
        print(f"Order {order} detail {detail} already calculated.")
        
    else:
        insideMatrixFileName=os.path.join("fractalData",f"inside_order{order}_detail{detail}.npy")
        insideMatrix=np.load(insideMatrixFileName)
        
        if stencil=="L5p":
            start = time.time()
            stencilMatrix=matrixGeneration.generate5pLaplacian(insideMatrix)
            end = time.time()
        elif stencil=="L9p":
            start = time.time()
            stencilMatrix=matrixGeneration.generate9pLaplacian(insideMatrix)
            end = time.time()
        else:
            raise("Not valid stencil")
        
        
        print("Time elapsed = %s" % (end - start))

        data=stencilMatrix
        
        times=np.load(timeFileName)
        times[order,detail]=end-start
        np.save(timeFileName,times)
        np.save(dataFileName,data,allow_pickle=True)
        

def eigensolutions(order,detail,k,recalculate,stencil):
    print(f"Eigensolver {stencil}:")
    
    folderName=getFolderName(stencil)
    
    valsFileName=os.path.join(folderName,f"eigenVals_order{order}_detail{detail}.npy")
    vecsFileName=os.path.join(folderName,f"eigenVecs_order{order}_detail{detail}.npy")
    timeFileName=os.path.join(folderName,"eigensolverTimes.npy")
    
    if (not recalculate and os.path.exists(valsFileName) and os.path.exists(vecsFileName)):
        print(f"Order {order} detail {detail} already calculated.")
        
    else:
        stencilMatrixFileName=os.path.join(folderName,f"stencilMatrix_order{order}_detail{detail}.npy")
        stencilMatrix=np.load(stencilMatrixFileName,allow_pickle=True).item()
        
        if stencil[0]=="L":
            prefactor=-1
        elif stencil[0]=="B":
            prefactor=1
        else:
            raise("Not valid stencil")
        
        
        start = time.time()
        #vals,vecs=lin.eigsh(prefactor*stencilMatrix,k=k,return_eigenvectors=True,which="SM")
        vals,vecs=lin.eigsh(prefactor*stencilMatrix,k=k,return_eigenvectors=True,sigma=0,which="LM")
        end = time.time()
        
        print("Time elapsed = %s" % (end - start))
        
        times=np.load(timeFileName)
        times[order,detail]=end-start
        np.save(timeFileName,times)
        np.save(valsFileName,vals)
        np.save(vecsFileName,vecs)

def getFolderName(stencil):
    if stencil=="L5p":
        folderName="laplacianData_5p"
    elif stencil=="L9p":
        folderName="laplacianData_9p"
    else:
        raise("Not valid stencil")
        
    return folderName

def calculateNumberOfInsidePoints():
    dataFileName=os.path.join("fractalData","numberOfInsidePoints.npy")
    numberOfInsidePoints=np.zeros((20,20))

    for order in range(20):
        for detail in range(20):
            insideMatrixFileName=os.path.join("fractalData",f"inside_order{order}_detail{detail}.npy")
            if os.path.exists(insideMatrixFileName):
                insideMatrix=np.load(insideMatrixFileName)
                numberOfInsidePoints[order,detail]=sum(insideMatrix.flatten())
            
    np.save(dataFileName,numberOfInsidePoints)

"""
lacking biharmonic generation and eigensolver
"""
