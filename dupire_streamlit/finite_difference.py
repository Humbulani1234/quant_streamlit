import numpy as np
import pandas as pd
from scipy.sparse import diags, linalg
import matplotlib.pyplot as plt
from math import *
import warnings
import decimal
from datetime import datetime
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
import scipy
import time

import volatility_function

# get_ipython().run_line_magic('matplotlib', 'inline')

# =========
# Settings
# =========

pd.set_option("display.width",100000)
pd.set_option("display.max_columns",50000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.float_format", lambda x: "%.3f" %x)
warnings.filterwarnings("ignore")

#decimal.getcontext().prec = 4

# ===================================================================================================================================

# ====================================
# Parameters and arrays intialization
# ====================================

# ===========
# Parameters
# ===========

T = .4166667 # convert monthly scale into real numbers between (0-1), where T = half-year.
N = 5 # number of periods until half-year
S_max = 8000
M = 20
dt = T/N
ds = S_max/M
# r = float(0.1)
# q = float(0)
# K = float(50)
# S0 = 4000

# ===============================
# Creating time and price arrays
# ===============================

t_array = np.arange(0, T+ 0.08333, dt)
s_array = np.arange(0, S_max + 5, ds) 

# ======================================
# FINITE DIFFERENCE METHOD OPTION PRICE
# ======================================

# =======================================
# 2D Finite Difference Call option price
# =======================================

def Finite_Difference_Function_2D(beta_1, K,S0, r):
    
    '''Finite difference algorithm option price'''
    
    #beta_1 = po

    #Volatility matrix
    
    volatility = volatility_function.Volatility_Function(beta_1)
    
    # Matrices of constants

    '''Matrix of constant -- a'''
    
    a = np.zeros((M+1,N+1))
    
    for j in range(M+1):
        for i in range(N+1):
            
            a[j,i] = 0.5*dt*(r*j - volatility[j,i]**2*j**2)

    '''Matrix of constant -- b'''

    b = np.zeros((M+1,N+1))
    
    for j in range(M+1):
        for i in range(N+1):
            
            b[j,i] = 1 + dt*(r + volatility[j,i]**2*j**2)
   
    '''Matrix of constant -- c'''

    c = np.zeros((M+1,N+1))
    
    for j in range(M+1):
        for i in range(N+1):
            
            c[j,i] = -1/2*dt*(r*j + volatility[j,i]**2*j**2)
      
    # Sparse matrix of option prices

    F = np.zeros((M+1,N+1))

    for j in range(0,M+1):

        F[j,N] = max(j*ds - K, 0)

    F[0,:] = 0
    F[M,:] = S_max - K
    
    '''modified F'''

    F_modified = np.delete(F,[0,M],0)
    
    # Matrix of modified constants for sparse matrix construction

    '''modified a'''

    a_modified = np.delete(a,[0,1,M], 0)
    
    '''modified b'''

    b_modified = np.delete(b,[0,M],0)
    
    '''modified c'''

    c_modified = np.delete(c,[0,M,M-1],0)
    
    #list of sparse matrices for each time slice

    Y = []

    for i in range(0, N+1):

        Y.append(diags([a_modified[:,i], b_modified[:,i], c_modified[:,i]], [-1, 0, 1], format="csc"))

    for i in range(1,N+1):

            b = np.where(F_modified[0:M-1,-i]==F_modified[0:M-1,-i][0],(F_modified[0:M-1,-i][0]-a[1,-(i+1)]*F[0,-(i+1)])\
                                                                                                        , F_modified[0:M-1,-i])
            p = np.where(b == b[-1], (b[-1] - c[M-1,-(i+1)]*F[M,-(i+1)]), b)

            F_modified[0:M-1,-(i+1)] = np.linalg.solve(Y[-(i+1)].toarray(), p)

    F_modified_df = pd.DataFrame(F_modified)
    F_df = pd.DataFrame(F)
    final_price_df = pd.concat([F_df.iloc[[0]],F_modified_df])._append(F_df.iloc[-1])
    final_price_df.set_index(s_array, inplace=True)
    
    price = final_price_df.loc[S0][0]                          
    
    return price 

a = Finite_Difference_Function_2D(10,50, 4000,0.1) 
print(a)