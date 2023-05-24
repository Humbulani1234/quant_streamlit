# =========================
# Volatility Meshgrid plot
# =========================
 
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

#=======================Plots==================================================================================================


def Meshgrid_Plot():
    
    '''Volatility meshgrid plot'''

    xx, yy = np.meshgrid(t_array,s_array)
    plt.figure(figsize=(8,5), dpi = 100 )
    plt.scatter(xx,yy,color="k", marker=".")
    plt.yticks(np.arange(0, S_max + 5, ds), fontsize=5)
    plt.xticks(np.arange(0, T+0.08333,dt), fontsize=5)
    
    return plt.show()

Meshgrid_Plot()

# ===========================
# Volatility 3D surface plot
# ===========================

def Volatility_Plot_Surface(beta_1):
    
    '''3D Volatility plot'''
    
    volatility = volatility_function.Volatility_Function(beta_1)
    xx, yy = np.meshgrid(t_array, s_array)
    
    fig = plt.figure(figsize = (12, 8))
    ax = fig.add_subplot(projection = "3d")

    surf = ax.plot_surface(xx, yy, volatility, rstride=2, cstride=2, cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=True)
    
    ax.set_xlabel = ("Time_to_maturity")
    ax.set_ylabel = ("Stock price")
    ax.set_zlabel = ("volatility")

    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    return plt.show()

Volatility_Plot_Surface(10)
