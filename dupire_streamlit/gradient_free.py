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

T = .4166667
N = 5
S_max = 8000
M = 20
dt = T/N
ds = S_max/M
r = float(0.1)
q = float(0)
K = float(50)
S0 = 4000

# ===============================
# Creating time and price arrays
# ===============================

t_array = np.arange(0, T+ 0.08333, dt)
s_array = np.arange(0, S_max + 5, ds) 


# ===============================================
# Gradient free optimization method - Python API
# ===============================================

def FDM_Error_Function_args(po, *args):
    
    '''Local volatility Surface Error Function for calibration of parameters'''
    
    arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10 = args
    
    global i
    
    beta_1 = po
    se =[] 
    min_MSE = 100
    
    for row, option in options_df.iterrows():
        
        T = (option["MATURITY"].date() - option["DATE"].date()).days/365
        K=option["STRIKE"]
        model_value = Finite_Difference_Function_2D(po,arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10)
        se.append((model_value - option["PRICE"])**2)
    
    MSE = 1/2*(sum(se))    
    i+=1
    
    return MSE

# ======================
# Nelder Mead Algorithm
# ======================

if __name__ == "__main__":
    
    i = 0 
    bnds = (0, 1000)
    xr=5
    #args = (t_array, s_array, K, T, S_max, S0, ds, dt, r, N, M)
    
    Opt_Nelder = optimize.minimize(FDM_Error_Function, xr, args=() ,method = "Nelder-Mead", options = {"maxiter":500}, bounds=bnds) 

    print(Opt_Nelder)   

# ==================================
# Differentinal evolution Algorithm
# ==================================

if __name__ == "__main__":
    
    i = 0 
    bnds = (0, 1000)
    xr=5
    #args = (t_array, s_array, K, T, S_max, S0, ds, dt, r, N, M)
    #initial = np.ones((20,1))

    Diff = scipy.optimize.differential_evolution(FDM_Error_Function, bounds=[bnds], args=(), strategy='best1exp', popsize=20)

    print(Diff)     

