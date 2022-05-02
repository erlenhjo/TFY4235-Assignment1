# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 17:26:46 2022

@author: Erlend Johansen
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import matplotlib
from scipy.stats import linregress

import fractalGeneration
from dataGeneration import getFolderName
import dataGeneration
import matrixGeneration


def plotEigenvectors(order,detail,stencil):
    folderName=getFolderName(stencil)
    
    valsFileName=os.path.join(folderName,f"eigenVals_order{order}_detail{detail}.npy")
    vals=np.load(valsFileName)
    
    vecsFileName=os.path.join(folderName,f"eigenVecs_order{order}_detail{detail}.npy")
    vecs=np.load(vecsFileName)
    
    insideMatrixFileName=os.path.join("fractalData",f"inside_order{order}_detail{detail}.npy")
    insideMatrix=np.load(insideMatrixFileName)
    
    boundaryFileName=os.path.join("fractalData",f"boundaryMatrix_order{order}_detail{detail}.npy")
    boundaryMatrix=np.load(boundaryFileName)
    
    cornersFileName=os.path.join("fractalData",f"corners_order{order}_detail{detail}.npy")
    corners=np.load(cornersFileName)
    
    fig,ax=plt.subplots(2,5,figsize=(2,5))
    z_max=np.max(np.abs(vecs))
    normalization=matplotlib.colors.Normalize(vmin=-z_max,vmax=z_max)

    for i in range(10):
        ax[i//5,i%5].imshow(fractalGeneration.recreateFractal(vecs[:,i],insideMatrix),norm=normalization)    
        ax[i//5,i%5].plot(corners[1],corners[0],c="r",linewidth=1)
        ax[i//5,i%5].axis("off")
        
def plotLocalizedState(order,detail,stencil):
    folderName=getFolderName(stencil)
        
    vecsFileName=os.path.join(folderName,f"eigenVecs_order{order}_detail{detail}.npy")
    vecs=np.load(vecsFileName)
    
    insideMatrixFileName=os.path.join("fractalData",f"inside_order{order}_detail{detail}.npy")
    insideMatrix=np.load(insideMatrixFileName)
    
    boundaryFileName=os.path.join("fractalData",f"boundaryMatrix_order{order}_detail{detail}.npy")
    boundaryMatrix=np.load(boundaryFileName)
    
    cornersFileName=os.path.join("fractalData",f"corners_order{order}_detail{detail}.npy")
    corners=np.load(cornersFileName)
    
    
    z_max=np.max(np.abs(vecs))
    normalization=matplotlib.colors.Normalize(vmin=-z_max,vmax=z_max)

    state2=fractalGeneration.recreateFractal(vecs[:,1],insideMatrix)
    state3=fractalGeneration.recreateFractal(vecs[:,2],insideMatrix)
    state4=fractalGeneration.recreateFractal(vecs[:,3],insideMatrix)
    state5=fractalGeneration.recreateFractal(vecs[:,4],insideMatrix)
    combinations=[[-1, -1, -1, 1],[-1, -1, 1, -1],[1, 1, -1, 1],[1, 1, 1, -1],[1, -1, 1, 1],[1, -1, -1, -1]]
    for nmij in combinations:
        n,m,i,j=nmij[0],nmij[1],nmij[2],nmij[3]
        localized=n*state2+m*state3+i*state4+j*state5
                    
        fig,ax=plt.subplots()
        ax.imshow(localized*0.5*(-1),norm=normalization)    
        ax.plot(corners[1],corners[0],c="r",linewidth=1)
        ax.axis("off")
                    

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
        for detail in [1]:
            if numberOfInsidePoints[order,detail] and eigensolverTimes[order,detail]:
                data_x.append(numberOfInsidePoints[order,detail])
                data_y.append(eigensolverTimes[order,detail])
        if(len(data_x)):
            ax.scatter(data_x,data_y,label=f"Order {order}")
            ax.set_xscale("log")
            ax.set_yscale("log")
    ax.legend()
    
def plotInsidePointPerformance(orders,recalculate):
    detail=1
    if recalculate:
        dataGeneration.insidePointsRays(2, 1, True)
        dataGeneration.insidePointsWinding(2, 1, True)
        for order in orders:
            dataGeneration.insidePointsWinding(order, detail, recalculate=True)
            dataGeneration.insidePointsRays(order, detail, recalculate=True)
    timeFileName=os.path.join("fractalData","insideWindingTimes.npy")
    windingTimes=np.load(timeFileName)
    timeFileName=os.path.join("fractalData","insideRayTimes.npy")
    rayTimes=np.load(timeFileName)
    
    fig,ax=plt.subplots()
    ax.scatter(orders,rayTimes[orders,1],marker="x",label="Ray casting")
    ax.scatter(orders,windingTimes[orders,1],marker="x",label="Winding number")
    ax.set_ylabel("Runtime [s]")
    ax.set_xlabel("Fractal level")
    ax.set_yscale("log")
    ax.legend()
    
def plotFractalBoundary(order,detail):
    cornersFileName=os.path.join("fractalData",f"corners_order{order}_detail{detail}.npy")
    corners=np.load(cornersFileName)
    
    
    fig,ax=plt.subplots(figsize=(1,1))
    ax.plot(corners[0],corners[1],marker=",")
    ax.axis("off")
    
def plotFractalInside(order, detail):
    boundaryFileName=os.path.join("fractalData",f"boundaryMatrix_order{order}_detail{detail}.npy")
    boundaryMatrix=np.load(boundaryFileName)
    insideMatrixFileName=os.path.join("fractalData",f"inside_order{order}_detail{detail}.npy")
    insideMatrix=np.load(insideMatrixFileName)
    
    fig,ax=plt.subplots()
    ax.matshow(insideMatrix+2*boundaryMatrix)
    ax.axis("off")

def plotOrder2Laplacian():
    order,detail=2,1
    insideMatrixFileName=os.path.join("fractalData",f"inside_order{order}_detail{detail}.npy")
    insideMatrix=np.load(insideMatrixFileName)
    laplacian=matrixGeneration.generate5pLaplacian(insideMatrix)
    fig,ax=plt.subplots()
    ax.matshow(laplacian.todense())
    ax.axis("off")

def plotOrder1_2_Laplacian():
    order,detail=2,2
    
    insideMatrixFileName=os.path.join("fractalData",f"inside_order{order}_detail{detail}.npy")
    insideMatrix=np.load(insideMatrixFileName)
    laplacian=matrixGeneration.generate5pLaplacian(insideMatrix)
    fig,ax=plt.subplots()
    ax.matshow(laplacian.todense())
    ax.axis("off")

def printEigenValues(orders,stencils):
    detail=1
    for stencil in stencils:
        for order in orders:
            l_scaling=4**order
            folderName=dataGeneration.getFolderName(stencil)
            valsFileName=os.path.join(folderName,f"eigenVals_order{order}_detail{detail}.npy")
            
            vals=np.load(valsFileName)
            print(stencil,":",order)
            
            omegas=np.sqrt(vals)*l_scaling
            latex_string=""
            for omega in omegas[:10]:
                latex_string+=f"&{omega:.2f}"
            print(latex_string)
        
def N_of_omega(omega,omegas):
    return omegas[omegas<=omega].size
        
def plotDOS(order):
    detail=1
    stencil="L5p"
    h=1/(4**order)
    folderName=dataGeneration.getFolderName(stencil)
    valsFileName=os.path.join(folderName,f"eigenVals_order{order}_detail{detail}.npy")
    
    vals=np.load(valsFileName)
    print(stencil,":",order)
    omegas=np.sqrt(vals)/h
    print(omegas)
    omegas=omegas#[:100]
    A=1
    N=[N_of_omega(omega, omegas) for omega in omegas]
    fig,ax=plt.subplots()
    ax.plot(omegas,A/(4*np.pi)*omegas**2)
    ax.plot(omegas,N)
    deltaN=A/(4*np.pi)*omegas**2-N
    ax.plot(omegas,deltaN)
    
    
    regress=linregress(np.log(omegas),np.log(deltaN))
    fig,ax=plt.subplots()
    ax.plot(np.log(omegas),np.log(deltaN))
    ax.plot(np.log(omegas), regress.slope*np.log(omegas),"--")
    print("Slope:", regress.slope)
    
    


if __name__=="__main__":
    
    plotDOS()
    
    
    