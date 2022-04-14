# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:26:32 2022

@author: Erlend Johansen
"""
from numba import njit
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix



def generate5pLaplacian(inside):
    # using five point stencil
    # [[0  1  0]
    #  [1 -4  1]
    #  [0  1  0]]
    
    N=np.shape(inside)[0]
    x_0=inside.flatten()
    number_of_inside_points=sum(x_0)
    laplacian=lil_matrix((number_of_inside_points,number_of_inside_points))
    index=0
    for n in range(1,N-1):     #for each y-row in inside, due to boundary n=0 and n=N-1 can been omitted
                               #meaning that there will be no out of range/negative indexing
        #the coordinates are given as arrays eventhough there is only one value, 
        #so indexing [:,0] gives a normal array of x coordinates
        prev_y=np.argwhere(inside[n-1,:]==True)[:,0]
        cur_y=np.argwhere(inside[n,:]==True)[:,0]
        next_y=np.argwhere(inside[n+1,:]==True)[:,0]
        
        for x in cur_y:   
            
            
            if x in prev_y:
                offset=np.argwhere(cur_y==x)[0]+len(prev_y)-np.argwhere(prev_y==x)[0]
                laplacian[index,index-offset]=1
            if x-1 in cur_y:
                laplacian[index,index-1]=1
            
            laplacian[index,index]=-4
            
            if x+1 in cur_y:
                laplacian[index,index+1]=1
            if x in next_y:
                offset=len(cur_y)-np.argwhere(cur_y==x)[0]+np.argwhere(next_y==x)[0]
                laplacian[index,index+offset]=1
            
            index+=1

    return csr_matrix(laplacian)
"""
def generate9pLaplacian(inside):
    
    #using nine point stencil
    #[[1   4  1]
    # [4 -20  4]
    # [1   4  1]]
    
    
    N=np.shape(inside)[0]
    x_0=inside.flatten()
    number_of_inside_points=sum(x_0)
    laplacian=lil_matrix((number_of_inside_points,number_of_inside_points))
    index=0
    for n in range(1,N-1):     #for each y-row in inside, due to boundary n=0 and n=N-1 can been omitted
                               #meaning that there will be no out of range/negative indexing
        #the coordinates are given as arrays eventhough there is only one value, 
        #so indexing [:,0] gives a normal array of x coordinates
        prev_y=np.argwhere(inside[n-1,:]==True)[:,0]
        cur_y=np.argwhere(inside[n,:]==True)[:,0]
        next_y=np.argwhere(inside[n+1,:]==True)[:,0]
        
        for x in cur_y:   
            
            
            if x in prev_y:
                offset=np.argwhere(cur_y==x)[0]+len(prev_y)-np.argwhere(prev_y==x)[0]
                laplacian[index,index-offset]=1
            if x-1 in cur_y:
                laplacian[index,index-1]=1
            
            laplacian[index,index]=-4
            
            if x+1 in cur_y:
                laplacian[index,index+1]=1
            if x in next_y:
                offset=len(cur_y)-np.argwhere(cur_y==x)[0]+np.argwhere(next_y==x)[0]
                laplacian[index,index+offset]=1
            
            index+=1

    return csr_matrix(laplacian)
"""

 

def generateReducedSparseBiharmonic(inside):
    """
    using thirteen point stencil
    [[0  0  1  0 0]
     [0  2 -8  2 0]
     [1 -8 20 -8 1]
     [0  2 -8  2 0]
     [0  0  1  0 0]]
    """
    N=np.shape(inside)[0]
    x_0=inside.flatten()
    number_of_inside_points=sum(x_0)
    biharmonic=lil_matrix((number_of_inside_points,number_of_inside_points))
    index=0
    
    #n=1 case handle separatly. Means more code, but fewer checks in the for-loop
    n=1
    prev_y=np.argwhere(inside[n-1,:]==True)[:,0]
    cur_y=np.argwhere(inside[n,:]==True)[:,0]
    next_y=np.argwhere(inside[n+1,:]==True)[:,0]
    next_next_y=np.argwhere(inside[n+2,:]==True)[:,0]
    #the coordinates are given as arrays eventhough there is only one value, 
    #so indexing [:,0] gives a normal array of x coordinates
    
    
    for x in cur_y:   
                    
        if x-1 in prev_y:
            offset=np.argwhere(cur_y==x)[0]+len(prev_y)-np.argwhere(prev_y==x-1)[0]
            biharmonic[index,index-offset]=2
        if x in prev_y:
            offset=np.argwhere(cur_y==x)[0]+len(prev_y)-np.argwhere(prev_y==x)[0]
            biharmonic[index,index-offset]=-8
        if x+1 in prev_y:
            offset=np.argwhere(cur_y==x)[0]+len(prev_y)-np.argwhere(prev_y==x+1)[0]
            biharmonic[index,index-offset]=2
        
        if x-2 in cur_y:
            biharmonic[index,index-2]=1
        if x-1 in cur_y:
            biharmonic[index,index-1]=-8
        
        biharmonic[index,index]=20
        
        if x+1 in cur_y:
            biharmonic[index,index+1]=-8
        if x+2 in cur_y:
            biharmonic[index,index+2]=1
         
        if x-1 in next_y:
            offset=len(cur_y)-np.argwhere(cur_y==x)[0]+np.argwhere(next_y==x-1)[0]
            biharmonic[index,index+offset]=2
        if x in next_y:
            offset=len(cur_y)-np.argwhere(cur_y==x)[0]+np.argwhere(next_y==x)[0]
            biharmonic[index,index+offset]=-8
        if x+1 in next_y:
            offset=len(cur_y)-np.argwhere(cur_y==x)[0]+np.argwhere(next_y==x+1)[0]
            biharmonic[index,index+offset]=2
        
        if x in next_next_y:
            offset=len(cur_y)-np.argwhere(cur_y==x)[0]+\
                len(next_y)+np.argwhere(next_next_y==x)[0]
            biharmonic[index,index+offset]=1
        
        index+=1
    
    
    
    
    
    for n in range(2,N-2):      #n=1 and n=N-2 is near boundary so that the indexing would be out of range
                                #n=0 and n=N-1 are on boundary and therefore not of interest as the boundary is zero
        
        prev_prev_y=np.argwhere(inside[n-2,:]==True)[:,0]
        prev_y=np.argwhere(inside[n-1,:]==True)[:,0]
        cur_y=np.argwhere(inside[n,:]==True)[:,0]
        next_y=np.argwhere(inside[n+1,:]==True)[:,0]
        next_next_y=np.argwhere(inside[n+2,:]==True)[:,0]
        
        for x in cur_y:   
            
            if x in prev_prev_y:
                offset=np.argwhere(cur_y==x)[0]+len(prev_y)\
                    +len(prev_prev_y)-np.argwhere(prev_prev_y==x)[0]
                biharmonic[index,index-offset]=1
                
            if x-1 in prev_y:
                offset=np.argwhere(cur_y==x)[0]+len(prev_y)-np.argwhere(prev_y==x-1)[0]
                biharmonic[index,index-offset]=2
            if x in prev_y:
                offset=np.argwhere(cur_y==x)[0]+len(prev_y)-np.argwhere(prev_y==x)[0]
                biharmonic[index,index-offset]=-8
            if x+1 in prev_y:
                offset=np.argwhere(cur_y==x)[0]+len(prev_y)-np.argwhere(prev_y==x+1)[0]
                biharmonic[index,index-offset]=2
            
            if x-2 in cur_y:
                biharmonic[index,index-2]=1
            if x-1 in cur_y:
                biharmonic[index,index-1]=-8
            
            biharmonic[index,index]=20
            
            if x+1 in cur_y:
                biharmonic[index,index+1]=-8
            if x+2 in cur_y:
                biharmonic[index,index+2]=1
             
            if x-1 in next_y:
                offset=len(cur_y)-np.argwhere(cur_y==x)[0]+np.argwhere(next_y==x-1)[0]
                biharmonic[index,index+offset]=2
            if x in next_y:
                offset=len(cur_y)-np.argwhere(cur_y==x)[0]+np.argwhere(next_y==x)[0]
                biharmonic[index,index+offset]=-8
            if x+1 in next_y:
                offset=len(cur_y)-np.argwhere(cur_y==x)[0]+np.argwhere(next_y==x+1)[0]
                biharmonic[index,index+offset]=2
            
            if x in next_next_y:
                offset=len(cur_y)-np.argwhere(cur_y==x)[0]+\
                    len(next_y)+np.argwhere(next_next_y==x)[0]
                biharmonic[index,index+offset]=1
            
            index+=1
        
    #n=N-2 case
    n=N-2
    prev_prev_y=np.argwhere(inside[n-2,:]==True)[:,0]
    prev_y=np.argwhere(inside[n-1,:]==True)[:,0]
    cur_y=np.argwhere(inside[n,:]==True)[:,0]
    next_y=np.argwhere(inside[n+1,:]==True)[:,0]
    
    for x in cur_y:   
        
        if x in prev_prev_y:
            offset=np.argwhere(cur_y==x)[0]+len(prev_y)\
                +len(prev_prev_y)-np.argwhere(prev_prev_y==x)[0]
            biharmonic[index,index-offset]=1
            
        if x-1 in prev_y:
            offset=np.argwhere(cur_y==x)[0]+len(prev_y)-np.argwhere(prev_y==x-1)[0]
            biharmonic[index,index-offset]=2
        if x in prev_y:
            offset=np.argwhere(cur_y==x)[0]+len(prev_y)-np.argwhere(prev_y==x)[0]
            biharmonic[index,index-offset]=-8
        if x+1 in prev_y:
            offset=np.argwhere(cur_y==x)[0]+len(prev_y)-np.argwhere(prev_y==x+1)[0]
            biharmonic[index,index-offset]=2
        
        if x-2 in cur_y:
            biharmonic[index,index-2]=1
        if x-1 in cur_y:
            biharmonic[index,index-1]=-8
        
        biharmonic[index,index]=20
        
        if x+1 in cur_y:
            biharmonic[index,index+1]=-8
        if x+2 in cur_y:
            biharmonic[index,index+2]=1
        
        print(next_y)
        if x-1 in next_y:
            offset=len(cur_y)-np.argwhere(cur_y==x)[0]+np.argwhere(next_y==x-1)[0]
            biharmonic[index,index+offset]=2
        if x in next_y:
            offset=len(cur_y)-np.argwhere(cur_y==x)[0]+np.argwhere(next_y==x)[0]
            biharmonic[index,index+offset]=-8
        if x+1 in next_y:
            offset=len(cur_y)-np.argwhere(cur_y==x)[0]+np.argwhere(next_y==x+1)[0]
            biharmonic[index,index+offset]=2
        
        index+=1
    

    return csr_matrix(biharmonic)



"""
@njit(cache=True)
def generateSimpleLaplacian(inside):
    N=np.shape(inside)[0]
    laplacian=np.zeros((N**2,N**2),dtype=np.single)
    x_0=inside.flatten()
    for m in np.argwhere(x_0==True):
        n=m[0]
        laplacian[n,n-N]=1
        laplacian[n,n-1]=1
        laplacian[n,n]=-4
        laplacian[n,n+N]=1
        laplacian[n,n+1]=1
    return laplacian

@njit(cache=True)
def generateReducedLaplacian(inside):
    N=np.shape(inside)[0]
    laplacian=generateSimpleLaplacian(inside)
    x_0=inside.flatten()
    for n in np.argwhere(x_0==False):
        laplacian[:,n[0]]=0    
    
    nonzero_rows=[i for i in range(N**2) if np.any(laplacian[i,:])]
    rowReducedLaplacian=np.empty((len(nonzero_rows),N**2),dtype=np.single)
    for n,row_index in enumerate(nonzero_rows):
        rowReducedLaplacian[n,:]=laplacian[row_index,:]
        
    nonzero_coloumns=[i for i in range(N**2) if np.any(laplacian[:,i])]
    reducedLaplacian=np.empty((len(nonzero_rows),len(nonzero_coloumns)),dtype=np.single)
    for n,coloumn_index in enumerate(nonzero_coloumns):
        reducedLaplacian[:,n]=rowReducedLaplacian[:,coloumn_index]
    
    return reducedLaplacian

"""