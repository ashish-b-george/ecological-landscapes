#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday Oct 3 14:21:40 2019
@author: ashish

reads data from experiment,  converts it into our format, saves data required for figures
"""

import numpy as np
import pickle # in python3, pickle uses cPickle automatically, there is no cPickle
import os
import glob
import pandas as pd
import sys
if os.getcwd().startswith("/Users/ashish"):
    import matplotlib as mpl
else:  
    sys.path.append('/usr3/graduate/ashishge/ecology/Ecology-Stochasticity/')
    import matplotlib as mpl
    mpl.use('Agg')### can not use pyplot on cluster without agg backend

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


#nice_fonts = { #"text.usetex": True, # Use LaTex to write all text
#"font.family": "serif",
## Use 10pt font in plots, to match 10pt font in document
#"axes.labelsize": 10, "font.size": 10,
## Make the legend/label fonts a little smaller
#"legend.fontsize": 8, "xtick.labelsize": 8,"ytick.labelsize": 8 }
#mpl.rcParams.update(nice_fonts)


linReg_object = LinearRegression()
linReg_noIntercept_object = LinearRegression(fit_intercept=False)
#### data_dir="/Users/ashish/Documents/GitHub/endpoints/data/" # for [Maynard, Miller, Allesina] data
destfold="//Users/ashish/Dropbox/research/ecology/Exp Data/"
if not os.path.exists(destfold): os.makedirs(destfold)
data_file="/Users/ashish/Dropbox/research/ecology/Exp Data/Langenheder_2010_ExpData_forPython.csv"



df=pd.read_csv(data_file)
exp_data_dict={} 

###### in Species presence absence columns, replace 'x' with 1 and NaN with 0
for k in df.keys():
    if 'SL' in k:
        df.replace({k: {float('nan'): 0, 'x': 1}}, inplace=True)



####### check if metabolic rate is roughly the same by fitting time series as linear function of time,  and looking at the R^2
#######should we allow an offset?
        
t_array=np.array([0,8,16,24,32,40,48]).reshape(-1, 1) ## we only have a single feature, with one sample at each time point



substrates_list= list(set(df['subtrate ID']))         
expts_in_each_substrate={}  
for substrate in substrates_list:
    expts_in_each_substrate.update({substrate:{'substrate name':substrate,'sigma vectors':[],'metabolic rate': [], 'intensity at end':[]  }})
  
    
R2_list=[]
R2_list_noIntercept=[] 
rate_list=[]
intercept_list=[]
rate_list_noIntercept=[]
   
t_array=np.array([0,8,16,24,32,40,48]).reshape(-1, 1) ## we only have a single feature, with one sample at each time point
for idx, row in df.iterrows():
    dye_signal=np.zeros(7)
    sigma_vectors=np.zeros(6).astype(int)
    
    dye_signal[1:]=row.iloc[10:16]
    
    ########### regression without intercept --this data is NOT analyzed further  #########
    linReg_noIntercept_object.fit(t_array,dye_signal)
    dye_predict=linReg_noIntercept_object.predict(t_array)
    R2_list_noIntercept.append( r2_score(dye_signal, dye_predict) )
    rate_list_noIntercept.append(linReg_noIntercept_object.coef_[0])  
    
    
    ########### regression with intercept  --this data is analyzed  ##########    
    linReg_object.fit(t_array,dye_signal)
    dye_predict=linReg_object.predict(t_array)
    R2_list.append( r2_score(dye_signal, dye_predict) )
    rate_list.append(linReg_object.coef_[0])
    intercept_list.append(linReg_object.intercept_)

    #### to plot some examples.
    #if idx<4:Linear_regression_landscape.plot_performance(dye_signal, dye_predict, "dye observed", "dye predicted",  title="dye darkening in time",filename=None, text='$R^2=$'+'{:.2}'.format( r2_score(dye_signal, dye_predict) ),  ms_val=6, alpha_val=1.  )
    
    ########## saving sigma vectors, measured metabolic activity , and final intensity of dye.
    sigma_vector=row.iloc[3:9].values.astype(int)    
    substrate=row['subtrate ID']
    expts_in_each_substrate[substrate]['metabolic rate'].append(linReg_object.coef_[0])
    expts_in_each_substrate[substrate]['sigma vectors'].append(sigma_vector)
    expts_in_each_substrate[substrate]['intensity at end'].append(dye_signal[-1])



######## appending the data point with no species present to each substrate dictionary ##########   
for substrate in substrates_list: 
    expts_in_each_substrate[substrate]['metabolic rate'].append( 0.0 )
    expts_in_each_substrate[substrate]['sigma vectors'].append( np.zeros(6).astype(int) )
    expts_in_each_substrate[substrate]['intensity at end'].append(0.0 )
    
    
    
fig = plt.figure(figsize=(3.5,3.5)) 
ax = fig.add_subplot(111)
plt.hist(R2_list,edgecolor='black', linewidth=2)
ax.set_ylabel(r'frequency')
ax.set_xlabel(r'$R^2$ metabolic activity fit in time')
ax.text(0.05, 0.95, r'$<R^2>=$'+'{:.2}'.format(np.mean(R2_list)), verticalalignment='top', transform=ax.transAxes)                   
plt.tight_layout()    
plt.savefig(destfold+"R2_rate_of_metabolic_activity.png",dpi=200) 

fig = plt.figure(figsize=(3.5,3.5)) 
ax = fig.add_subplot(111)
plt.hist(R2_list_noIntercept,edgecolor='black', linewidth=2)
ax.set_ylabel(r'frequency')
ax.set_xlabel(r'$R^2$ metabolic activity fit in time')  
ax.text(0.05, 0.95, r'$<R^2>=$'+'{:.2}'.format(np.mean(R2_list_noIntercept)), verticalalignment='top', transform=ax.transAxes)                 
plt.tight_layout()    
plt.savefig(destfold+"R2_noIntercept_rate_of_metabolic_activity.png",dpi=200) 
    
   
exp_data_dict.update({'activity_fit_in_time: R2_list':R2_list,'activity_fit_in_time: R2_list_noIntercept':R2_list_noIntercept,
                      'activity_fit_in_time: rate_list':rate_list,'activity_fit_in_time: rate_list_noIntercept':rate_list_noIntercept,
                      'activity_fit_in_time: rate_list':rate_list,'activity_fit_in_time: rate_list_noIntercept':rate_list_noIntercept,
                      'activity_fit_in_time: intercept_list':intercept_list})
        
pickle.dump(exp_data_dict, open(destfold+'Exp_data_Langenheder_processed.dat', 'wb') ) 



for idx,substrate in enumerate(substrates_list):
    
    dest_SubFolder=destfold+'Substrate'+str(idx)+'/'
    if not os.path.exists(dest_SubFolder): os.makedirs(dest_SubFolder)    
    pickle.dump(expts_in_each_substrate[substrate], open(dest_SubFolder+'Substrate'+str(idx)+'.dat', 'wb') ) 

    

plt.close("all")    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    