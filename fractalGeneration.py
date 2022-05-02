# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:26:15 2022

@author: Erlend Johansen
"""

from numba import njit
import numpy as np


rotate=np.array([[0,-1],[1,0]],dtype=np.int32)

template=np.array([[0,1,1,2,2,2,3,3,4],[0,0,1,1,0,-1,-1,0,0]],dtype=np.int32)
directions=np.array([[1,0,1,0,0,1,0,1],[0,1,0,-1,-1,0,1,0]],dtype=np.int32)



def mutate_line(template,directions):
    a=np.shape(template)[1]-1
    new_template_x=np.zeros(a*8+1,dtype=np.int32)
    new_template_y=np.zeros(a*8+1,dtype=np.int32)
    for n in range(8):
        if directions[0][n]==1:   #if the original template goes to the right
            new_template_x[a*n:a*(n+1)+1]=np.copy(template[0])+template[0][n*a//8]*4
            new_template_y[a*n:a*(n+1)+1]=np.copy(template[1])+template[1][n*a//8]*4
                
        elif directions[1][n]==1: #if the original template goes up
            new_template_x[a*n:a*(n+1)+1]=np.dot(rotate,np.copy(template))[0]+template[0][n*a//8]*4
            new_template_y[a*n:a*(n+1)+1]=np.dot(rotate,np.copy(template))[1]+template[1][n*a//8]*4  

        else: #if the original template goes down
            new_template_x[a*n:a*(n+1)+1]=-np.dot(rotate,np.copy(template))[0]+template[0][n*a//8]*4
            new_template_y[a*n:a*(n+1)+1]=-np.dot(rotate,np.copy(template))[1]+template[1][n*a//8]*4
            
    return np.array([new_template_x,new_template_y],dtype=np.int32)



def generatePath(order):
    global directions,template
    
    scale_factor=4**order  #length of square side
    line_segment=np.copy(template)

    for n in range(1,order):
        line_segment=mutate_line(line_segment, directions)
    
    q=8**order        #number of points in a mutated line including one endpoint

    path_x=np.zeros(4*8**order+1,dtype=np.int32)
    path_y=np.zeros(4*8**order+1,dtype=np.int32)
    
    path_x[0:q+1]=np.copy(line_segment)[0]
    path_y[0:q+1]=np.copy(line_segment)[1]
    
    path_x[q:2*q+1]=np.dot(rotate,np.copy(line_segment))[0]+scale_factor
    path_y[q:2*q+1]=np.dot(rotate,np.copy(line_segment))[1]
    
    path_x[2*q:3*q+1]=-np.copy(line_segment)[0]+scale_factor
    path_y[2*q:3*q+1]=-np.copy(line_segment)[1]+scale_factor
    
    path_x[3*q:4*q+1]=-np.dot(rotate,np.copy(line_segment))[0]
    path_y[3*q:4*q+1]=-np.dot(rotate,np.copy(line_segment))[1]+scale_factor
    
    
    

    offset=sum([4**n for n in range(0,order)])
    path_x+=offset
    path_y+=offset
    #originalSidelength=4**order
    #center=originalSidelength//2+offset    

    return np.array([path_x,path_y])


def addDetailToPath(path,detail):
    N=np.shape(path)[1]
    path*=detail
    new_path_x=np.zeros((N-1)*detail+1,dtype=np.int32)
    new_path_y=np.zeros((N-1)*detail+1,dtype=np.int32)
    for n in range(N-1):
        for i in range(detail):
            new_path_x[n*detail+i]=((detail-i)*path[0,n]+i*path[0,n+1])//detail
            new_path_y[n*detail+i]=((detail-i)*path[1,n]+i*path[1,n+1])//detail
            
    new_path_x[-1]=path[0,-1]
    new_path_y[-1]=path[1,-1]
    return np.array([new_path_x,new_path_y])

#@njit(cache=True)
def generateBoundaryMatrix(path):
    size=max(path[0])+1
    boundaryMatrix=np.zeros((size,size),dtype=bool)  
    for i in path.transpose():
        boundaryMatrix[size-1-i[1],i[0]]=True
        
    return boundaryMatrix

@njit(cache=True)
def findInsidePointsWinding(path,boundary):
    size=max(path[0])+1
    inside=np.zeros((size,size),dtype=np.int8)
    wasUnder=False
    wasOver=False
    
    for n in range(np.shape(path)[1]-1):
        
        cur_y=size-1-path[1,n]
        cur_x=path[0,n]
    
        if path[1,n]==path[1,n+1]:
            continue
    
        elif (path[1,n]<path[1,n+1]):
            if wasUnder:
                for i in range(cur_x):
                    inside[cur_y,i]+=1 
            wasUnder=True
            wasOver=False
        elif (path[1,n]>path[1,n+1]):
            if wasOver:
                for i in range(cur_x):
                    inside[cur_y,i]-=1 
            wasOver=True
            wasUnder=False
    
    return inside-inside*boundary

@njit(cache=True)
def findInsidePointsRays(path,boundary):
    size=max(path[0])+1
    inside=np.zeros((size,size),dtype=np.int8)
    wasUnder=False
    wasOver=False
    
    for n in range(np.shape(path)[1]-1):
        
        cur_y=size-1-path[1,n]
        cur_x=path[0,n]
    
        if path[1,n]==path[1,n+1]:
            continue
    
        elif (path[1,n]<path[1,n+1]):
            if wasUnder:
                for i in range(cur_x):
                    inside[cur_y,i]+=1
            wasUnder=True
            wasOver=False
        elif (path[1,n]>path[1,n+1]):
            if wasOver:
                for i in range(cur_x):
                    inside[cur_y,i]+=1 
            wasOver=True
            wasUnder=False
    
    return np.mod(inside,2)-np.mod(inside,2)*boundary

@njit(cache=True)
def recreateFractal(eigenvector,inside):
    N=np.shape(inside)[0]
    x_0=inside.flatten()
    eigenFractal=np.zeros((N,N))
    for i,n in enumerate(np.argwhere(x_0==True)):
        eigenFractal[n[0]//N,n[0]%N]=eigenvector[i]
    return eigenFractal


"""
def getInsideAndBoundaryMatrix(order,detail):
    path=generate_path(order)
    path=add_detail_to_path(path,detail)
    size=max(path[0])+1
    boundary=np.zeros((size,size),dtype=bool)  
    for i in path.transpose():
        boundary[size-1-i[1],i[0]]=True

    inside=findInsidePoints(path,boundary).astype(bool)
    
    return inside,boundary

if __name__=="__main__":
    inside,boundary=getInsideAndBoundaryMatrix(3, 2)
    plt.matshow(inside)

"""

