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

# ===========================
# Volatility functional form
# ===========================

def f(beta_1, stock, time):
    
    '''Volatility function -- function may vary'''
    
    #beta_1 = po
    
    #return exp(-1 * (1 * sqrt(stock**2 + time**2))**2) # Gaussian Radial Basis Function

    return beta_1/stock

# ============================
# Volatility surface function
# ============================

def Volatility_Function(beta_1):
    
    '''Volatility surface calculation'''
    
    #beta_1 = po

    xx, yy = np.meshgrid(t_array, s_array)

    xx_array = xx.ravel()
    yy_array = yy.ravel()

    z = zip(xx_array, yy_array)
    z_array = list(z)

    # Calculating the volatility at grid points

    l = []

    for element in z_array:

        l.append(f(beta_1, stock=element[1], time=element[0]))

    l_array = np.array(l)
    l_array_matrix = np.reshape(l_array, (len(s_array), len(t_array)))

    # Volatility matrix
    
    volatility = np.full_like(np.zeros_like(l_array_matrix), l_array_matrix)

    return volatility
