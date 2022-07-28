#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 18:05:52 2019

@author: ashish

Performs ensemble learning : by running random forest on the linear regression. 
i.e, the train data is bagged and given to many different trees , each finds their best parameters
We find the best set of parameters to not get biased by some outlier.

To find the best number of leaves/ depth of the trees by performing a 
grid search on the parameters before testing on the test data, we can follow this artilce:
https://medium.com/datadriveninvestor/random-forest-regression-9871bc9a25eb

Simple example:    
https://www.geeksforgeeks.org/random-forest-regression-in-python/
"""

import numpy as np
import pickle # in python3, pickle uses cPickle automatically, there is no cPickle
import os
import glob
import sys
#import pandas
#import argparse
#from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict, GridSearchCV
if os.getcwd().startswith("/Users/ashish"):
    from my_CR_model_functions import *
else:    
    sys.path.append('/usr3/graduate/ashishge/ecology/ecological-landscapes/')
    from my_CR_model_functions import *

if os.getcwd().startswith("/Users/ashish"):
    from my_CR_model_functions import rms
    import matplotlib as mpl
else:  
    sys.path.append('/usr3/graduate/ashishge/ecology/ecological-landscapes/')
    from my_CR_model_functions import rms
    import matplotlib as mpl
    mpl.use('Agg')### can not use pyplot on cluster without agg backend

import matplotlib.pyplot as plt
#from analyze_landscape import total_abundance, diversity_index
import Linear_regression_landscape 


def Rforest_vs_LinReg_when_fitting(sigma_vectors, biomass, destfold, n_estimatorsRF = 10, rand_state=None, fitness_name="biomass", return_stuff=False, suffix=''):
    RandomForestReg_object = RandomForestRegressor(n_estimators = n_estimatorsRF, random_state = rand_state)
    linReg_object = LinearRegression()

    ############### fitting all the data  ################   
    RandomForestReg_object.fit(sigma_vectors,biomass)
    y_fitRF=RandomForestReg_object.predict(sigma_vectors)    
    Rsq_RF=r2_score(biomass,y_fitRF)   
    Linear_regression_landscape.plot_performance(biomass,y_fitRF, fitness_name+" observed", fitness_name+" fit",  title="Random forest fitting",
                                                 filename=destfold+suffix+'Random_forest_fit.png', text='$R^2=$'+'{:.2}'.format(Rsq_RF), text_xpos=0.05, text_ypos=0.99  )

    linReg_object.fit(sigma_vectors,biomass)
    y_fitLinReg=linReg_object.predict(sigma_vectors)
    Rsq_LinReg=r2_score(biomass,y_fitLinReg)   
    Linear_regression_landscape.plot_performance(biomass,y_fitLinReg, fitness_name+" observed", fitness_name+" fit",  title="Linear Regression fitting",
                                                 filename=destfold+suffix+'linReg_fit.png', text='$R^2=$'+'{:.2}'.format(Rsq_LinReg), text_xpos=0.05, text_ypos=0.99  )
    if return_stuff==True:
        return Rsq_RF, Rsq_LinReg, y_fitRF, y_fitLinReg
    else:
        return None
    

def Rforest_vs_LinReg_Allbutone(sigma_vectors, biomass, N_obs, destfold, n_estimatorsRF = 10, rand_state=None, fitness_name="biomass", return_stuff=False, suffix=''):
    RandomForestReg_object = RandomForestRegressor(n_estimators = n_estimatorsRF, random_state = rand_state)
    linReg_object = LinearRegression()
    ############### Predicting from 1, all, all but one  ################        
    sigma_vectors_train, N_obs_train, sigma_vectors_test, N_obs_test, idx_train, idx_test= Linear_regression_landscape.splitData_single_All_AllButOne_as_trainingdata(sigma_vectors, N_obs, return_indices=True )
    
    RandomForestReg_object.fit(sigma_vectors_train,biomass[idx_train])   
    y_preRF=RandomForestReg_object.predict(sigma_vectors_test)    
    Rsq_RF=r2_score(biomass[idx_test],y_preRF)   
    Linear_regression_landscape.plot_performance(biomass[idx_test],y_preRF, fitness_name+" observed", fitness_name+" predicted",  title="Random forest prediction\nfrom 1, All, All but 1",
                                                 filename=destfold+suffix+'Random_forest_Allbutone.png', text='$R^2=$'+'{:.2}'.format(Rsq_RF), text_xpos=0.05, text_ypos=0.99  )
    
    linReg_object.fit(sigma_vectors_train,biomass[idx_train])   
    y_preLinReg=linReg_object.predict(sigma_vectors_test)    
    Rsq_LinReg=r2_score(biomass[idx_test],y_preLinReg) 
    Linear_regression_landscape.plot_performance(biomass[idx_test],y_preLinReg, fitness_name+" observed", fitness_name+" fit",  title="Linear Regression prediction\nfrom 1, All, All but 1",
                                                 filename=destfold+suffix+'linReg_Allbutone.png', text='$R^2=$'+'{:.2}'.format(Rsq_LinReg), text_xpos=0.05, text_ypos=0.99  )
    if return_stuff==True:
        return Rsq_RF, Rsq_LinReg, y_preRF, y_preLinReg
    else:
        return None



def Rforest_vs_LinReg_kFoldCV(sigma_vectors, biomass, destfold, n_estimatorsRF = 10, rand_state=None, K=3, fitness_name="biomass", return_stuff=False, suffix=''):
    RandomForestReg_object = RandomForestRegressor(n_estimators = n_estimatorsRF, random_state = rand_state)
    linReg_object = LinearRegression()
    
    ############### 3fold Cross validation (LOOCV) ################
    y_RandForest_3foldCV = cross_val_predict(RandomForestReg_object, sigma_vectors, biomass, cv=K)
    y_linReg_3foldCV = cross_val_predict(linReg_object, sigma_vectors, biomass, cv=K)
    
    
    
    Linear_regression_landscape.plot_performance(biomass, y_RandForest_3foldCV , fitness_name+" observed", fitness_name+" predicted",  title="Random forest 3foldCV",
                                                 filename=destfold+suffix+'Random_forest_3CV.png', text='$R^2=$'+'{:.2}'.format( r2_score(biomass,y_RandForest_3foldCV) ), 
                                                 text_xpos=0.05, text_ypos=0.99  )
    
    Linear_regression_landscape.plot_performance(biomass, y_linReg_3foldCV , fitness_name+" observed", fitness_name+" predicted",  title="Linear Regression "+str(K)+"foldCV",
                                                 filename=destfold+suffix+'linReg_'+str(K)+'CV.png', text='$R^2=$'+'{:.2}'.format( r2_score(biomass, y_linReg_3foldCV) ), 
                                                 text_xpos=0.05, text_ypos=0.99  )
    if return_stuff==True:
        return r2_score(biomass,y_RandForest_3foldCV), r2_score(biomass, y_linReg_3foldCV), y_RandForest_3foldCV , y_linReg_3foldCV
    else:
        return None

    
    

    
def Rforest_vs_LinReg_LOOCV(sigma_vectors, biomass, destfold, n_estimatorsRF = 10, rand_state=None, fitness_name="biomass", return_stuff=False, suffix=''):
    RandomForestReg_object = RandomForestRegressor(n_estimators = n_estimatorsRF, random_state = rand_state)
    linReg_object = LinearRegression()
    
    ############### Leave-One-Out Cross validation (LOOCV) ################
    '''
    Leave-One-Out Cross validation (LOOCV, used by Maynard,Alessina 2019) is K fold cross validation with K=N (size of data set)
    '''
    y_RandForest_LOOCV = cross_val_predict(RandomForestReg_object, sigma_vectors, biomass, cv=len(biomass))
    y_linReg_LOOCV = cross_val_predict(linReg_object, sigma_vectors, biomass, cv=len(biomass))
    
    Linear_regression_landscape.plot_performance(biomass, y_RandForest_LOOCV, fitness_name+" observed", fitness_name+" predicted",  title="Random forest LOOCV",
                                                 filename=destfold+suffix+'Random_forest_LOOCV.png', text='$R^2=$'+'{:.2}'.format( r2_score(biomass,y_RandForest_LOOCV) ), 
                                                 text_xpos=0.05, text_ypos=0.99  )
    
    Linear_regression_landscape.plot_performance(biomass, y_linReg_LOOCV, fitness_name+" observed", fitness_name+" predicted",  title="Linear Regression LOOCV",
                                                 filename=destfold+suffix+'linReg_LOOCV.png', text='$R^2=$'+'{:.2}'.format( r2_score(biomass, y_linReg_LOOCV) ), 
                                                 text_xpos=0.05, text_ypos=0.99  )
    
    if return_stuff==True:
        return r2_score(biomass,y_RandForest_LOOCV), r2_score(biomass, y_linReg_LOOCV), y_RandForest_LOOCV , y_linReg_LOOCV
    else:
        return None

    

def main():
    base_dir="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/"
#    folder_name_list=[base_dir+'Crossfeeding_fig_M4_compiled/']
    #folder_name_list=[base_dir+'Crossfeeding_fig_M3_compiled-a very degenerate case/',
    #                      base_dir+'Crossfeeding_fig_M4_compiled/',
    #                      base_dir+'Crossfeeding_fig_M6_compiled/',
    #                      base_dir+'Crossfeeding_fig_M9_compiled/',
    #                      base_dir+'Crossfeeding_fig_M12_compiled/',
    #                      base_dir+'Crossfeeding_fig_M15_compiled/',
    #                      base_dir+'Crossfeeding_fig_M18_compiled/',
    #                      base_dir+'Crossfeeding_fig_M24_compiled/',
    #                      base_dir+'CRmodel_figA_niche_compiled/',
    #                      base_dir+'CRmodel_figB_2interactions_compiled/',
    #                      base_dir+'CRmodel_figB_3interactions_compiled/',
    #                      base_dir+'CRmodel_figC_6interactions_compiled/',
    #                      base_dir+'Crossfeeding_fig_2consumptions_compiled/',
    #                      base_dir+'Crossfeeding_fig_6consumptions_compiled/',
    #                      ]
    folder_name_list= [base_dir+'/Crossfeeding_alleatR0_meanC2_compiled/', base_dir+'/Crossfeeding_alleatR0_meanC3_compiled/', 
                       base_dir+'/Crossfeeding_alleatR0_meanC4_compiled/', base_dir+'/Crossfeeding_alleatR0_meanC5_compiled/',
                       base_dir+'/Crossfeeding_alleatR0_meanC6_compiled/', base_dir+'/Crossfeeding_alleatR0_meanC8_compiled/',
                       base_dir+'/Crossfeeding_alleatR0_meanC10_compiled/',base_dir+'/Crossfeeding_alleatR0_meanC12_compiled/']
    
    
    
    for fold in folder_name_list:   
        print (fold) 
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
    
        destfold=fold+"/RandomForest/"

        if not os.path.exists(destfold): os.makedirs(destfold)
    
    
    
        data = pickle.load(open(fold+file_suffix+'.dat', 'rb')) 
        sigma_vectors= (data['initial_abundance'].T>0).astype(int)
        N_obs=data['steady_state'].T    
        biomass=np.sum(N_obs, axis=1)
        
        
        
        print (np.shape(biomass), np.shape(sigma_vectors))
        
        Rforest_vs_LinReg_when_fitting(sigma_vectors, biomass, destfold, n_estimatorsRF = 10, rand_state=0)
        Rforest_vs_LinReg_Allbutone(sigma_vectors, biomass, N_obs, destfold, n_estimatorsRF = 10, rand_state=0)
        Rforest_vs_LinReg_kFoldCV(sigma_vectors, biomass, destfold, n_estimatorsRF = 10, rand_state=0, K=3)
        Rforest_vs_LinReg_LOOCV(sigma_vectors, biomass, destfold, n_estimatorsRF = 10, rand_state=0)
         
    
    
        '''
        should we try
        stratified k fold CV to keep number of occurrence of a species the same in train and test?
        sklearn.model_selection.StratifiedKFold(n_splits=’warn’, shuffle=False, random_state=None)
        '''
        plt.show()
       
    
if __name__ == '__main__':
    main()   
        





