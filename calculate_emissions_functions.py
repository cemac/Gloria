# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

import pandas as pd
import numpy as np
import time
import scipy.linalg as sla

try:
    from memory_profiler import profile
except ImportError:
    # use dummy profile
    from profile import *
    pass

@profile
def make_x(Z, Y, verbose):
    
    x = np.sum(Z, 1)+np.sum(Y, 1)
    x[x == 0] = 0.000000001
    if verbose:
        print("DBG: X shape is ", x.shape)
    return x

@profile
# equivalent function of make_x but does it as components
def make_x_comp_new(S, U, Y, verbose):
    # components of what was x
    sumS=np.sum(S,1)
    sumS[sumS == 0] = 0.000000001 # this is x1
    sumU=np.sum(U,1) 
    sumY=np.sum(Y,1)
    sumUY=sumU+sumY
    sumUY[sumUY==0] = 0.000000001 # this is x2

    if verbose:
        print("DBG: sumS, sumUY shape is ", sumS.shape, sumUY.shape)

    return sumS, sumUY

@profile
def make_L(Z, x, verbose, do_timing):
    
    bigX = np.zeros(shape = (len(Z)))    
    # bigX is as big as Z yet another wasteful storage (delete it? or chose other method)
    bigX = np.tile(np.transpose(x), (len(Z), 1))
    # MRI A is doubling the storage requirement (Z) and then L also adds same again
    A = np.divide(Z, bigX)
    #np.save('A_old.npy', A)    
    I_minus_A=np.identity(len(Z))-A
    if do_timing:
        time0=time.time()
    L = np.linalg.inv(I_minus_A)
    #L = sla.inv(np.identity(len(Z))-A) Marks new bit
    if do_timing:
        time1=time.time()
        print('TIME: inverting I-A matrix', time1-time0)

    #np.save('L_old.npy', L)
    if verbose:
        print("DBG: bigX shape is ", bigX.shape)
        print("DBG: A shape is ", A.shape)
        print("DBG: L shape is ", L.shape)

    return L, I_minus_A

@profile
# equivalent of make_L but does it as components
def make_L_comp_new(S, U, sumS, sumUY, verbose, do_timing):

    bigSumS = np.tile(np.transpose(sumS), (S.shape[0],1))
    bigSumUY = np.tile(np.transpose(sumUY), (U.shape[0],1))

    # use elementwise divide as was done in make_L to get A
    scaledS=np.divide(S,bigSumUY)
    scaledU=np.divide(U,bigSumS)
    #np.save('scaledS.npy', scaledS)
    #np.save('scaledU.npy', scaledU)

    # in equation [I-A]X=D where I-A top is  [I,-scaledS] and I-A bottom is [-scaledU, I], Dtop is e1 and Dbottom=0
    # assume X is [X1,X2] from which we get
    # 1. X1-scaledS.X2 = e1 and
    # 2. -scaledU.X1 + X2 = 0
    #   
    # from 2. we get 3. X2=scaledU.X1
    # insert into 1. X1 - scaledS.(scaledU.X1) = e1
    # (I-scaledS.scaledU).X1 = e1
    # so X1 = inv(I-scaledS.scaledU).e1
    # then use X2=scaledU.X1

    I=np.identity(S.shape[0])
    if do_timing:
        time0=time.time()
    L=np.linalg.inv(I-np.matmul(scaledS,scaledU))
    # use sci version - faster? Test this on big data on machine with multiple cores
    #L = sla.inv(I-np.matmul(scaledS,scaledU))
    if do_timing:
         time1=time.time()
         print('TIME: inverting matrix', time1-time0)
    #np.save('L_new.npy', L)
    if verbose:
        print('scaledU and U shape', scaledU.shape, scaledS.shape)
        print("DBG: L shape is ", L.shape)

    return L, scaledU

@profile
def make_e(stressor, x):
    # MRI not used in this model for some reason
    e = np.zeros(shape = (1, np.size(x)))
    e[0, 0:np.size(stressor)] = np.transpose(stressor)
    e = e/x

@profile
def make_Z_from_S_U(S, U, verbose):
    # MRI this makes Z a numpy array and fills with zeroes
    Z = np.zeros(shape = (np.size(S, 0)+np.size(U, 0), np.size(S, 1)+np.size(U, 1)))
    
    Z[np.size(S, 0):, 0:np.size(U, 1)] = U
    Z[0:np.size(S, 0), np.size(U, 1):] = S
    if verbose:
        print("DBG: make Z from S and U", Z.size, Z.shape )

    return Z

@profile
def indirect_footprint_SUT(S, U, Y, stressor, use_Le, verbose, do_timing):
    # make column names
    s_cols = S.columns.tolist()
    u_cols = U.columns.tolist()
    su_idx = pd.MultiIndex.from_arrays([[x[0] for x in s_cols] + [x[0] for x in u_cols],
                                        [x[1] for x in s_cols] + [x[1] for x in u_cols]])
    y_cols = Y.columns

    # calculate emissions
    if do_timing:
        time0=time.time()
    Z = make_Z_from_S_U(S, U,verbose)
    if do_timing:
        time1=time.time()
        print('TIME: make_Z_from_S_U', time1-time0)
    # clear memory
    del S, U
    
    bigY = np.zeros(shape = [np.size(Y, 0)*2, np.size(Y, 1)])
    
    footprint = np.zeros(shape = bigY.shape).T
    footprint_Le = np.zeros(shape = bigY.shape).T

    bigY[np.size(Y, 0):np.size(Y, 0)*2, 0:] = Y 
    x = make_x(Z, bigY,verbose)
    if do_timing:
        time0=time.time()
    L,I_minus_A = make_L(Z, x, verbose, do_timing)
    if do_timing:
        time1=time.time()
        print('TIME: make_L', time1-time0)

    #np.save('L_old.npy', L)
    bigstressor = np.zeros(shape = [np.size(Y, 0)*2, 1])
    bigstressor[:np.size(Y, 0), 0] = np.array(stressor)
    e = np.sum(bigstressor, 1)/x
    #np.save('e_old.npy', e)

    if use_Le:
        Le=np.dot(L,e)
        #np.save('Le_old.npy', Le)
        dot_prod=Le
        dot_prod_str='Le'
    else:
        eL = np.dot(e, L)
        #np.save('eL_old.npy', eL)
        dot_prod=eL
        dot_prod_str='eL'

    if verbose:
        print('DBG: bigY shape', bigY.shape)
        print("DBG: e shape is ", e.shape, "big_stressor is ", bigstressor.shape)
        print("DBG: "+dot_prod_str+" shape is ", dot_prod.shape)
        # check it works the other way
        exp_e=np.dot(I_minus_A,dot_prod)
        diff=abs(exp_e-e)
        ix=np.where(diff>0.00001)
        print(len(ix[0]), 'diffs exp_e and e using', dot_prod_str)

    if do_timing:
        time0=time.time()
    for a in range(np.size(Y, 1)):
        footprint[a] = np.dot(dot_prod, np.diag(bigY[:, a]))
    
    old_shape=footprint.shape
    footprint = pd.DataFrame(footprint, index=y_cols, columns=su_idx)
    footprint = footprint[u_cols]
    if do_timing:
         time1=time.time()
         print('TIME: make footprint', time1-time0)
    if verbose:
         print('DBG: full,u_cols footprint shape is',old_shape, footprint.shape)
 
    return footprint

@profile
def indirect_footprint_SUT_new(S, U, Y, stressor,verbose, do_timing):
    # calculate emissions
    sumS, sumUY=make_x_comp_new(S,U,Y,verbose)

    # stressor has 1 row also may be different indexing which messes up np.divide so just look at array
    stress=stressor.to_numpy()[0,:]
    e1=np.divide(stress, sumS) # lower part of e is 0 as bigstressor only had stressor in top part
    e2=0
    if verbose:
        print('DBG: e1 shape', e1.shape)
    #np.save('e1_new.npy', e1)

    if do_timing:
        time0=time.time()
    L, scaledU=make_L_comp_new(S, U, sumS, sumUY, verbose, do_timing)
    if do_timing:
        time1=time.time()
        print('TIME: make_L_comp_new', time1-time0)

    # It should be L dot e not e dot L - in fact I have shown it does not work in reverse if you use e dot L
    X1=np.dot(L,e1)
    X2=np.dot(scaledU, X1)
    Le=np.zeros(len(X1)*2)
    Le[:len(X1)]=X1
    Le[len(X1):]=X2
    if verbose:
        # check it works the other way
        exp_e1=X1-np.dot(scaledS, X2)
        exp_e2=X2-np.dot(scaledU,X1)
        diff=exp_e1-e1
        ix=np.where(abs(diff)>0.000001)
        if len(ix[0])>0:
            print('L.e calc not reversible!')

        print('DBG: X1, X2 and Le shape', X1.shape, X2.shape, Le.shape)
    #np.save('Le_new.npy', Le)
    
    # then for each column in Y the code used to take the diagonal of bigY to find the dot product with eL
    # as bigY was 0 in the top half only the bottom half of eL would have been valid
    # therefore we only need to use the X2 part
    
    footprint = np.zeros(shape = Y.shape).T
    Y2=Y.to_numpy()
    if do_timing:
        time0=time.time()
    for a in range(np.size(Y2, 1)):
        footprint[a] = np.dot(X2, np.diag(Y2[:, a]))

    y_cols = Y.columns
    u_cols=U.columns
    footprint = pd.DataFrame(footprint, index=y_cols, columns=u_cols)
    if do_timing:
         time1=time.time()
         print('TIME: make footprint', time1-time0)
    if verbose:
        print('DBG: footprint shape is',footprint.shape)

    return footprint
