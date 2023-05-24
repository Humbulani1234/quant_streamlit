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

import finite_difference
import data_options

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


# =============================================================
# Least Squares FDM Error function for parameters optimization
# =============================================================


i=0
min_MSE=2000

def FDM_Error_Function(beta_1, K,S0, r):
    
    '''Local volatility Surface Error Function for calibration of parameters'''
    
    global i
    global min_MSE

    # i = 0
    # min_MSE = 2000.0
    
    se =[] 
    
    for row, option in data_options.options_df.iterrows():
        
        T = (option["MATURITY"].date() - option["DATE"].date()).days/365
        K=option["STRIKE"]
        model_value = finite_difference.Finite_Difference_Function_2D(beta_1, K,S0, r)
        se.append((model_value - option["PRICE"])**2)
    
    MSE = 1/2*(sum(se))
    min_MSE = min(MSE, min_MSE)
    
    if i % 2 == 0:
    
        print(i, MSE, min_MSE)

    i += 1 
    
    return MSE  

# ==================================================================
# Gradient based optimization methods - Python API - BFGS Algorithm
# ==================================================================
 
def opt(func, xr, method=None):

    #bnds = (0, float("inf"))
    args = (K,S0, r)

    BFGS = optimize.minimize(func, xr, args=args, method=None, options={"maxiter":700}) 

    return BFGS

# w = opt(func=FDM_Error_Function, xr=5,method='BFGS')
# print(w)

# ===========================================================
# FDM Error Residuals - Optimization via Levenberg Marquardt
# ===========================================================

def FDM_Error_Residuals(beta_1, K,S0, r):
    
    '''Heston error function vector of residuals'''
    
    Residuals = []
    
    for row, option in data_options.options_df.iterrows():
        
        T = (option["MATURITY"].date() - option["DATE"].date()).days/365
        K=option["STRIKE"]
        residual = (finite_difference.Finite_Difference_Function_2D(beta_1, K,S0, r) - option["PRICE"])
        Residuals.append(residual)
        
    Residuals_array = np.array(Residuals)
    
    return Residuals_array


def opt_lm(func, method='lm'):

    xr=5
    args = (K,S0, r)
    
    LM = optimize.least_squares(func, xr, args=(args),method='lm', ftol=0.01)

    return LM

LM = opt_lm(func=FDM_Error_Residuals, method='lm')
print(LM)
