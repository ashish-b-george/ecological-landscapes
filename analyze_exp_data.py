#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:53:20 2019
@author: ashish

Code to run analysis and regression scripts on the Langenheder exp data
"""

import numpy as np
import pickle # in python3, pickle uses cPickle automatically, there is no cPickle
import os
import glob
import pandas as pd
import sys
from copy import deepcopy
if os.getcwd().startswith("/Users/ashish"):
    import matplotlib as mpl
else:  
    sys.path.append('/usr3/graduate/ashishge/ecology/Ecology-Stochasticity/')
    import matplotlib as mpl
    mpl.use('Agg')### can not use pyplot on cluster without agg backend

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import Linear_regression_landscape
import analyze_landscape 
import RandomForestRegression_onCRdata as RF_vs_linReg
import time

base_dir="/Users/ashish/Dropbox/research/ecology/Exp Data/"
folder_name_list=[base_dir+'Substrate0/',
                  base_dir+'Substrate1/',
                  base_dir+'Substrate2/',
                  base_dir+'Substrate3/',
                  base_dir+'Substrate4/',
                  base_dir+'Substrate5/',
                  base_dir+'Substrate6/']
 
analysis_overwrite=True 


substrate_list=[]
# =============================================================================
# r2RF_fitting_list=[]
# r2LinReg_fitting_list=[]
# r2RF_3foldCV_list=[]
# r2LinReg_3foldCV_list=[]
# r2RF_LOOCV_list=[]
# r2LinReg_LOOCV_list=[]
# =============================================================================


for fold in folder_name_list:
    print (fold)
    analysis_fold=fold
    file_suffix=None## finds the first file the matches patter with glob.glob   
    if file_suffix==None:## finds the first file the matches patter with glob.glob
        ctr=0
        for file_suffix in glob.glob(fold+'S*.dat'):
            file_suffix=file_suffix.replace(fold,'')
            file_suffix=file_suffix.replace('.dat','')
            print ("file_suffix is now ", file_suffix)
            ctr+=1
        assert ctr>0,"no file found"
        assert ctr<2,"we can only have one file that matches the pattern in the folder!"
       
    ####################        Reading data           #################### 
    data = pickle.load(open(fold+file_suffix+'.dat', 'rb')) 
    if os.path.isfile(fold+'analysis_'+file_suffix+'.dat') and analysis_overwrite==False:
        analysis_dict=pickle.load(open(fold+'analysis_'+file_suffix+'.dat', 'rb')) 
    else:
        analysis_dict={}
        analysis_overwrite=True
     
        
    substrate_name=data['substrate name']
    analysis_dict.update({'substrate name':substrate_name})
    sigma_vectors= np.asarray(data['sigma vectors']).astype(int)
    metabolic_rate=np.asarray(data['metabolic rate'])
    intensity_at_end=np.asarray(data['intensity at end'])
    
    sigma_vectors_final= deepcopy(sigma_vectors) ### we assume that all the species survived, but we don't actually know this.
    print ('shape of sigma is ', np.shape(sigma_vectors))
    
    S=6
    n_exp=64
    
    data_fractions_to_estimate_from=[0.2, 0.25, 0.4, 0.5]
  
    analysis_options=['FourierPS','r/s','n_max_min','basic','greedy_walk','RandomWalk','basic','SWO','FDC','estimation_from_limited_data','spatial_correlation','variance_explained'] 
    
    start = time.time()
    adjacency_matrix=analyze_landscape.generate_adjacency_matrix(sigma_vectors, S, n_exp )
    end = time.time()
    print("time to make adjacency matrix",end-start)

    
    
    start = time.time()
    metabolic_dict=analyze_landscape.analyze_for_chosen_metric(metabolic_rate, sigma_vectors, sigma_vectors_final, adjacency_matrix, S, n_exp, 'metabolic rate', 
                                                 analysis_fold, file_suffix, analysis_options, fraction_limited_data=data_fractions_to_estimate_from)
    analysis_dict.update(metabolic_dict) 
    plt.close("all")
    end = time.time()
    print("time to analyze",end-start)
    pickle.dump(analysis_dict, open(analysis_fold+'analysis_'+file_suffix+'.dat', 'wb') )  
    
# =============================================================================
#     r2_RF, r2_LinReg, _, _ =RF_vs_linReg.Rforest_vs_LinReg_when_fitting(sigma_vectors, metabolic_rate, fold, n_estimatorsRF = 10, rand_state=0, fitness_name="metabolic productivity",return_stuff=True)   
#     r2RF_fitting_list.append(r2_RF)    
#     r2LinReg_fitting_list.append(r2_LinReg)    
#         
#     r2_RF, r2_LinReg, _, _ =RF_vs_linReg.Rforest_vs_LinReg_kFoldCV(sigma_vectors, metabolic_rate, fold, n_estimatorsRF = 10, rand_state=0, fitness_name="metabolic productivity", return_stuff=True)        
#     r2RF_3foldCV_list.append(r2_RF)    
#     r2LinReg_3foldCV_list.append(r2_LinReg)
#     
#     r2_RF, r2_LinReg, _, _ =RF_vs_linReg.Rforest_vs_LinReg_LOOCV(sigma_vectors, metabolic_rate, fold, n_estimatorsRF = 10, rand_state=0, fitness_name="metabolic productivity", return_stuff=True)   
#     r2RF_LOOCV_list.append(r2_RF)    
#     r2LinReg_LOOCV_list.append(r2_LinReg)    
# =============================================================================

    
    substrate_list.append(substrate_name)    
    
        
        


    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        