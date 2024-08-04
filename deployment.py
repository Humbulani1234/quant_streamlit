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
import streamlit as st
from PIL import Image

import finite_difference
import volatility_function
import volatility_plot
import data_options
import gradient_based

# =========
# Settings
# =========

pd.set_option("display.width",100000)
pd.set_option("display.max_columns",50000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.float_format", lambda x: "%.3f" %x)
warnings.filterwarnings("ignore")
#st.set_option('deprecation.showPyplotGlobalUse', False)

#decimal.getcontext().prec = 4

# ===================================================================================================================================

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


#=========Widgets and Titles=============================================================================

st.title("Option Price Calculation")
image = Image.open('better.jpg')
st.image(image,use_column_width=True)
st.subheader("This model will calculate the price of an option")
classifier_name = st.sidebar.selectbox('Select model', ('Dupire', 'Heston'))

#===================================Datasets============================================================================

def get_dataset(name):
    
    if name=='Dupire':
        data = data_options.options_df
        st.dataframe(data)
        st.write('Shape of dataframe:', data.shape)

    else:
        data = data_options.options_df
        st.dataframe(data)
        st.write('Shape of dataframe:', data.shape)

#===================================================Data Visualization===================================

def get_data(name):

    if name=='Volatility Mesh Grid':

        volatility_plot.Meshgrid_Plot()
        st.pyplot()

    else:
        volatility_plot.Volatility_Plot_Surface(beta_1=10)
        st.pyplot()


# def get_data2(name):

#     if name=='Volatility Surface':

#         Decision_tree.Cross_Validate_Alphas(Decision_tree.DT_Classification_fit, train_test1.X_train, train_test1.Y_train, randomstate=42\
#             , ccpalpha=0)
#         st.pyplot()

#     # elif name=='Monte Carlo Simulation':

#     #     Decision_tree.Ideal_Alpha(Decision_tree.DT_Classification_fit, train_test1.X_train, train_test1.Y_train, threshold_1=0.0019\
#     #         , threshold_2=0.0021, randomstate=42, ccpalpha=0)
#     #     st.pyplot()

#     else:

#         Decision_tree.Plot_DT(Decision_tree.DT_Classification_fit, train_test1.X_train, train_test1.Y_train, randomstate=42\
#             , ccpalpha=Decision_tree.ideal_ccp_alpha)
#         st.pyplot()

# ====================================Logistic Prediction=============================================================

def dupire_fea():

    Interest_Rate = st.text_input("Customer")
    Dividend = st.slider("AGE", 0,100)
    Strike_Price = st.slider("CHILDREN", 0, 10)
    Stock_Price = st.slider("PERS_H", 0, 10)

    button_clicked = st.button('Submit', key=30)    

    def update_variables():    
 
        if button_clicked:    
            
            inputs = [Interest_Rate, Dividend, Strike_Price, Stock_Price]    
            
            #prediction = finite_difference.Finite_Difference_Function_2D(10, t_array, s_array, 50, T, S_max,4000, ds, dt, 0.1, N, M)  

            st.subheader('Customer {} probability of default is: {}'.format(NAME , prediction))
            st.success('Successfully executed the model')
        

    update_variables()

def heston_fea():

    Interest_Rate = st.text_input("Customer")
    Dividend = st.slider("AGE", 0,100,key=10)
    Strike_Price = st.slider("CHILDREN", 0, 10,key=11)
    Stock_Price = st.slider("PERS_H", 0, 10,key=12)
    AGE = st.slider("AGE", 0,100,key=10)
    CHILDREN = st.sider("CHILDREN", 0, 10,key=11)
    PERS_H = st.slider("PERS_H", 0, 10,key=12)
    TMADD = st.sideb.slider("TMADD", 0, 1000,key=13)
    TMJOB1 = st.sideb.slider("TMJOB1", 0, 1000,key=14)

    button_clicked = st.button('Submit', key=28)    

    def update_variables2():    
    
        if button_clicked:  

            inputs = [Interest_Rate, Dividend, Strike_Price, Stock_Price]    
            
            #prediction = finite_difference.Finite_Difference_Function_2D(10, t_array, s_array, 50, T, S_max,4000, ds, dt, 0.1, N, M) 
          
            st.subheader('Customer will default if value is 1 and not if 0: {}'.format( prediction))
            st.success('Successfully executed the model')
        

    update_variables2()

# ========================================================Estimation of Parameters==================================================
 
def opt(func, xr, method=None):

    #bnds = (0, float("inf"))
    args = (K,S0, r)

    params = optimize.minimize(func, xr, args=args, method=None, options={"maxiter":700}) 

    return params

def estimation(opti, func, xr, method=None):

    xr = st.sidebar.slider("xr", 0,100)

    button_clicked = st.sidebar.button('Submit', key=32)    

    def update_variables():    
 
        if button_clicked:        
            
            #args=(K,S0,r)
            estimation_algorithm = opti(func, xr, method=None)  

            st.sidebar.subheader('Parameter: {}'.format(estimation_algorithm.x))
            st.sidebar.success('Successfully executed the model')

    update_variables()

def dupire_estimation(name):

    if name=='BFGS':

        estimation(opt, gradient_based.FDM_Error_Function,xr=5,method='BFGS')

    if name=='SLQSP':
        estimation(opt, gradient_based.FDM_Error_Function,xr=5,method='SLQSP')

    else:
        pass


# def heston_estimation():

#     Interest_Rate = st.sidebar.text_input("Customer", key=90)
#     Dividend = st.sidebar.slider("AGE", 0,100,key=10)
#     Strike_Price = st.sidebar.slider("CHILDREN", 0, 10,key=11)
#     Stock_Price = st.sidebar.slider("PERS_H", 0, 10,key=12)
#     AME = st.sidebar.text_input("Customer", key=90)
#     AGE = st.sidebar.slider("AGE", 0,100,key=10)
#     CHILDREN = st.sidebar.slider("CHILDREN", 0, 10,key=11)
#     PERS_H = st.sidebar.slider("PERS_H", 0, 10,key=12)
#     TMADD = st.sidebar.slider("TMADD", 0, 1000,key=13)
#     TMJOB1 = st.sidebar.slider("TMJOB1", 0, 1000,key=14)

#     button_clicked = st.button('Submit', key=28)    

#     def update_variables2():    
    
#         if button_clicked:  

#             inputs = [Interest_Rate, Dividend, Strike_Price, Stock_Price]    
            
#             #prediction = finite_difference.Finite_Difference_Function_2D(10, t_array, s_array, 50, T, S_max,4000, ds, dt, 0.1, N, M) 
          
#             st.subheader('Customer will default if value is 1 and not if 0: {}'.format( prediction))
#             st.success('Successfully executed the model')
        

#     update_variables2()

# =======================================================Final=====================================================================

def get_classifier(name):
    
    if name=='Dupire':

        dataset_name = st.sidebar.selectbox('Select dataset', ('Dupire_data', 'Other'))
        visualization_name=st.sidebar.selectbox('Select Visuals', ('Volatility Mesh Grid','Volatility Surface'))
        estimation_algorithm=st.sidebar.selectbox('Select Algorithm', ('BFGS','SLQSP','lm'))
        get_dataset(dataset_name)
        dupire_estimation(estimation_algorithm)
        #get_dataset2(diagnostics_name)
        get_data(visualization_name)
        dupire_fea()

    else:

        dataset_name = st.sidebar.selectbox('Select dataset', ('Heston_data', 'Other'))
        decision_name = st.sidebar.selectbox('Select Visuals',('Volatilty Surface','Monte Carlo Simulation'))
        get_dataset(dataset_name)
        #get_data2(decision_name)
        heston_fea()

get_classifier(classifier_name)
