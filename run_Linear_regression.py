#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:12:44 2019

@author: ashish
Runs the linear regression functions on the data
"""
import numpy as np
import pickle # in python3, pickle uses cPickle automatically, there is no cPickle
import os
import argparse
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from Linear_regression_landscape import *
import sys
import glob
if os.getcwd().startswith('/Users/ashish'):
    import matplotlib as mpl
else: ### can not use pyplot on cluster without agg backend
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
nice_fonts = { #"text.usetex": True, # Use LaTex to write all text
"font.family": "serif",
# Use 10pt font in plots, to match 10pt font in document
"axes.labelsize": 10, "font.size": 10,
# Make the legend/label fonts a little smaller
"legend.fontsize": 8, "xtick.labelsize": 8,"ytick.labelsize": 8 }
mpl.rcParams.update(nice_fonts)



popn_cutoff=1e-6 
pseudoCount=1e-10


def main():
    ##################  file location and such things  ########################
    base_dir="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/"
    folder_name_list=[base_dir+'Crossfeeding_fig_M6_compiled/',
                      ]
    #file_suffix='S10M10_P1_c4_l0.5_typeII_sigma20.0'
    file_suffix=None
#    analysis_rewrite= False ## if true, it deletes current dictionary and replaces with new dictionary
#    analysis_overwrite=True ## overwrites existing values of  fitness functions analyzed only      
    #fold="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/Crossfeeding_fig_M6_compiled/"
    for fold in folder_name_list:
        #destfold=fold[:-1]+"_plots/"
        destfold=fold
        if not os.path.exists(destfold): os.makedirs(destfold)
        if file_suffix==None:## finds the first file the matches patter with glob.glob
            ctr=0
            for file_suffix in glob.glob(fold+'S*.dat'):
                file_suffix=file_suffix.replace(fold,'').replace('.dat','')
                print ("file_suffix is now ", file_suffix)
                ctr+=1
            assert ctr==1,"we can only have one file that matches the pattern in the folder!"
        print (fold, destfold, file_suffix)
    #run_linear_regressions(fold, destfold, file_suffix)
def run_linear_regressions(fold, destfold, file_suffix,  analysis_overwrite=True):
        assert analysis_overwrite==True and analysis_rewrite==True,'other method should not be used untile code is ready.'
        ##################  reading and initializing data  ########################
        data = pickle.load(open(fold+file_suffix+'.dat', 'rb')) 
        if os.path.isfile(fold+'linReg_'+file_suffix+'.dat') and analysis_overwrite==False:
            linReg_dict=pickle.load(open(fold+'linReg_'+file_suffix+'.dat', 'rb')) 
        else:
            linReg_dict={}
        sigma_vectors= (data['initial_abundance'].T>0).astype(int)
        N_obs=data['steady_state'].T
        
        S=len(sigma_vectors[0])
        n_exp=len(sigma_vectors)
        
        
        
        ######################      for LPA:  ######################      
        Y_vectors_Lin=N_to_Ncutoff(N_obs, sigma_vectors,popn_cutoff) ## Yvectors is the vector of observations to fit.
        Y_vectors_pred_Lin,interaction_matrix_Lin, R2_Lin, R2_Lin_VW, reg_dict=perform_Linear_Regression_of_abundance_vector(Y_vectors_Lin, sigma_vectors, S, expansionOrder=1)
        
        R2_pred_clipped=r2_score(Y_vectors_Lin, np.clip(Y_vectors_pred_Lin, 0, None), multioutput='variance_weighted')## R2 if a negative prediction was set  to 0
        reg_dict.update({'R2_pred_clipped':R2_pred_clipped})
        plot_performance(Y_vectors_Lin, Y_vectors_pred_Lin, r'$N_{observed}$', r'$N_{predicted}$',title= r'Linear Presence Absence', 
                         text='$R^2_{vw}=$'+ str( round(R2_Lin_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_Lin, 2) )+'\n$R^2_{clip}=$'+ str( round(R2_pred_clipped, 2) ), 
                         filename=destfold+file_suffix+"_LPA.png")
        linReg_dict.update({'Linear':reg_dict})
        
        
        ######predicting from splitting into train& test alon pwise, AllButOne , etc###########
        sigma_train, Ytrain, sigma_test, Ytest=splitData_upto_pairwise_as_trainingdata(sigma_vectors, Y_vectors_Lin )
        Y_pred_pwise, _, R2_pwise, R2_pwise_VW, reg_dict=perform_Linear_Regression_of_abundance_vector(Ytrain, sigma_train,  S, expansionOrder=1, Y_vectors_test= Ytest, sigma_vectors_test=sigma_test)
        plot_performance(Ytest,Y_pred_pwise, r'$N_{observed}$', r'$N_{predicted}$',title= r'from upto pairwise data', 
                         text='$R^2_{vw}=$'+ str( round(R2_pwise_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_pwise, 2) ), 
                         filename=destfold+file_suffix+"_LPA_pwise.png")
        linReg_dict.update({'Linear pwise':reg_dict})
        
        sigma_train, Ytrain, sigma_test, Ytest=splitData_single_All_AllButOne_as_trainingdata(sigma_vectors, Y_vectors_Lin )
        Y_pred_Sm1, _, R2_Sm1, R2_Sm1_VW, reg_dict=perform_Linear_Regression_of_abundance_vector(Ytrain, sigma_train,  S, expansionOrder=1, Y_vectors_test= Ytest, sigma_vectors_test=sigma_test)
        plot_performance(Ytest,Y_pred_Sm1, r'$N_{observed}$', r'$N_{predicted}$',title= r'single, all, all but one data', 
                         text='$R^2_{vw}=$'+ str( round(R2_Sm1_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_Sm1, 2) ), 
                         filename=destfold+file_suffix+"_LPA_Sm1.png")
        
        linReg_dict.update({'Linear Sm1':reg_dict})
        
        
        
        ######################       for EPA: (fits logN!)  ######################      
        Y_vectors_Log=transform_to_log_abundance(N_obs, sigma_vectors, popn_cutoff=1e-6, pseudoCount=1e-10)
        Y_vectors_pred_Log,interaction_matrix_Log, R2_Log, R2_Log_VW, reg_dict=perform_Linear_Regression_of_abundance_vector(Y_vectors_Log, sigma_vectors, S, expansionOrder=1)
        
        R2_Log_linspace=r2_score(np.power(10,Y_vectors_Log),np.power(10,Y_vectors_pred_Log))
        R2_Log_linspace_VW=r2_score(np.power(10,Y_vectors_Log),np.power(10,Y_vectors_pred_Log),multioutput='variance_weighted')
        reg_dict.update({'R2_Log_linspace':R2_Log_linspace,'R2_Log_linspace_VW':R2_Log_linspace_VW})
        
        plot_performance(Y_vectors_Log, Y_vectors_pred_Log, r'log $N_{observed}$', r'log $N_{predicted}$',title= r'Exponential Presence Absence',
                         text='$R^2_{vw}=$'+ str( round(R2_Log_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_Log, 2) ) ,
                         filename=destfold+file_suffix+"_EPA.png")
        plot_performance(np.power(10,Y_vectors_Log), np.power(10,Y_vectors_pred_Log), r'$N_{observed}$', r'$N_{predicted}$', 
                         text='$R^2_{vw}=$'+ str( round(R2_Log_linspace_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_Log_linspace, 2) ) ,
                         filename=destfold+file_suffix+"_EPA_linspace.png")
        linReg_dict.update({'Log':reg_dict})
        
        ######################      for SqrtN:  ######################      
        Y_vectors_Sqrt=np.sqrt(N_to_Ncutoff(N_obs, sigma_vectors, popn_cutoff))
        
        Y_vectors_pred_Sqrt,interaction_matrix_Sqrt, R2_Sqrt, R2_Sqrt_VW, reg_dict=perform_Linear_Regression_of_abundance_vector(Y_vectors_Sqrt, sigma_vectors, S, expansionOrder=1)
        
        R2_pred_clipped=r2_score(Y_vectors_Sqrt, np.clip(Y_vectors_pred_Sqrt, 0, None), multioutput='variance_weighted')## R2 if a negative prediction was set  to 0
        reg_dict.update({'R2_pred_clipped':R2_pred_clipped})
        
        plot_performance(Y_vectors_Sqrt, Y_vectors_pred_Sqrt, r'$\sqrt{ N_{observed} }$', r'$\sqrt{ N_{predicted} }$',title= r'$\sqrt{N}$ model', 
                         text='$R^2_{vw}=$'+ str( round(R2_Sqrt_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_Sqrt, 2) )+'\n$R^2_{clip}=$'+ str( round(R2_pred_clipped, 2) ), 
                         filename=destfold+file_suffix+"_Sqrt.png")
        
        R2_Sqrt_linspace=r2_score(Y_vectors_Sqrt**2, Y_vectors_pred_Sqrt**2)
        R2_Sqrt_linspace_VW=r2_score(Y_vectors_Sqrt**2, Y_vectors_pred_Sqrt**2,multioutput='variance_weighted')
        reg_dict.update({'R2_Log_linspace':R2_Log_linspace,'R2_Log_linspace_VW':R2_Log_linspace_VW})
        
        plot_performance(Y_vectors_Sqrt**2, Y_vectors_pred_Sqrt**2, r'$N_{observed} $', r'$N_{predicted} $',title= r'$\sqrt{N}$ model', 
                         text='$R^2_{vw}=$'+ str( round(R2_Sqrt_linspace_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_Sqrt_linspace, 2) )+'\n$R^2_{clip}=$'+ str( round(R2_pred_clipped, 2) ), 
                         filename=destfold+file_suffix+"_Sqrt_linspace.png")
        linReg_dict.update({'Square root':reg_dict})
        
        ######predicting from splitting into train& test alon pwise, AllButOne , etc###########
        sigma_train, Ytrain, sigma_test, Ytest=splitData_upto_pairwise_as_trainingdata(sigma_vectors, Y_vectors_Sqrt)
        Y_pred_pwise, _, R2_pwise, R2_pwise_VW, reg_dict=perform_Linear_Regression_of_abundance_vector(Ytrain, sigma_train,  S, expansionOrder=1, Y_vectors_test= Ytest, sigma_vectors_test=sigma_test)
        R2_Sqrt_linspace=r2_score(Ytest**2, Y_pred_pwise**2)
        R2_Sqrt_linspace_VW=r2_score(Ytest**2, Y_pred_pwise**2,multioutput='variance_weighted')
        plot_performance(Ytest**2, Y_pred_pwise**2, r'$N_{observed}$', r'$N_{predicted}$',title= r'from upto pairwise data', 
                         text='$R^2_{vw}=$'+ str( round(R2_Sqrt_linspace_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_Sqrt_linspace, 2) ), 
                         filename=destfold+file_suffix+"_Sqrt_pwise.png")
        linReg_dict.update({'Square root pwise':reg_dict})
        
        sigma_train, Ytrain, sigma_test, Ytest=splitData_single_All_AllButOne_as_trainingdata(sigma_vectors, Y_vectors_Sqrt)
        Y_pred_Sm1, _, R2_Sm1, R2_Sm1_VW, reg_dict=perform_Linear_Regression_of_abundance_vector(Ytrain, sigma_train,  S, expansionOrder=1, Y_vectors_test= Ytest, sigma_vectors_test=sigma_test)
        R2_Sqrt_linspace=r2_score(Ytest**2, Y_pred_Sm1**2)
        R2_Sqrt_linspace_VW=r2_score(Ytest**2, Y_pred_Sm1**2,multioutput='variance_weighted')
        plot_performance(Ytest,Y_pred_Sm1, r'$N_{observed}$', r'$N_{predicted}$',title= r'single, all, all but one data', 
                         text='$R^2_{vw}=$'+ str( round(R2_Sqrt_linspace_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_Sqrt_linspace, 2) ), 
                         filename=destfold+file_suffix+"_Sqrt_Sm1.png")
        linReg_dict.update({'Square root Sm1':reg_dict})
        
        
        
        
        ######################       for N^2:   ######################      
        Y_vectors_Sqr=np.square(N_to_Ncutoff(N_obs,sigma_vectors, popn_cutoff))
        
        Y_vectors_pred_Sqr,interaction_matrix_Sqr, R2_Sqr, R2_Sqr_VW, reg_dict=perform_Linear_Regression_of_abundance_vector(Y_vectors_Sqr, sigma_vectors, S, expansionOrder=1)
        Y_pred_Sqr_clipped=np.clip(Y_vectors_pred_Sqr, 0, None)
        R2_pred_clipped=r2_score(Y_vectors_Sqr, Y_pred_Sqr_clipped, multioutput='variance_weighted')## R2 if a negative prediction was set  to 0
        reg_dict.update({'R2_pred_clipped':R2_pred_clipped})
        
        plot_performance(Y_vectors_Sqr, Y_vectors_pred_Sqr, r'$N^2_{observed} $', r'$N^2_{predicted} $',title= r'$N^2$ model', 
                         text='$R^2_{vw}=$'+ str( round(R2_Sqr_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_Sqr, 2) )+'\n$R^2_{clip}=$'+ str( round(R2_pred_clipped, 2) ), 
                         filename=destfold+file_suffix+"_Sqr.png")
        
        R2_Sqr_linspace=r2_score(  np.sqrt(Y_vectors_Sqr), np.sqrt( Y_pred_Sqr_clipped) )
        R2_Sqr_linspace_VW=r2_score(np.sqrt(Y_vectors_Sqr), np.sqrt( Y_pred_Sqr_clipped),multioutput='variance_weighted')
        reg_dict.update({'R2_Log_linspace':R2_Log_linspace,'R2_Log_linspace_VW':R2_Log_linspace_VW})
        
        plot_performance(np.sqrt(Y_vectors_Sqr), np.sqrt( Y_pred_Sqr_clipped), r'$N_{observed} $', r'$N_{predicted} $',title= r'$N^2$ model', 
                         text='$R^2_{vw}=$'+ str( round(R2_Sqr_linspace_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_Sqr_linspace, 2) )+'\n$R^2_{clip}=$'+ str( round(R2_pred_clipped, 2) ), 
                         filename=destfold+file_suffix+"_Sqr_linspace.png")
        
        linReg_dict.update({'Square':reg_dict})
        
        ######predicting from splitting into train& test alon pwise, AllButOne , etc###########
        sigma_train, Ytrain, sigma_test, Ytest=splitData_upto_pairwise_as_trainingdata(sigma_vectors, Y_vectors_Sqr)
        Y_pred_pwise, _, R2_pwise, R2_pwise_VW, reg_dict=perform_Linear_Regression_of_abundance_vector(Ytrain, sigma_train,  S, expansionOrder=1, Y_vectors_test= Ytest, sigma_vectors_test=sigma_test)
        Y_pred_pwise=np.clip(Y_pred_pwise, 0, None)
        R2_Sqr_linspace=r2_score(np.sqrt(Ytest), np.sqrt( Y_pred_pwise))
        R2_Sqr_linspace_VW=r2_score(np.sqrt(Ytest), np.sqrt( Y_pred_pwise),multioutput='variance_weighted')
        plot_performance(np.sqrt(Ytest), np.sqrt( Y_pred_pwise), r'$N_{observed}$', r'$N_{predicted}$',title= r'from upto pairwise data', 
                         text='$R^2_{vw}=$'+ str( round(R2_Sqr_linspace_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_Sqr_linspace, 2) ), 
                         filename=destfold+file_suffix+"_Sqr_pwise.png")
        linReg_dict.update({'Square pwise':reg_dict})
        
        sigma_train, Ytrain, sigma_test, Ytest=splitData_single_All_AllButOne_as_trainingdata(sigma_vectors, Y_vectors_Sqr)
        Y_pred_Sm1, _, R2_Sm1, R2_Sm1_VW, reg_dict=perform_Linear_Regression_of_abundance_vector(Ytrain, sigma_train,  S, expansionOrder=1, Y_vectors_test= Ytest, sigma_vectors_test=sigma_test)
        Y_pred_Sm1=np.clip(Y_pred_Sm1, 0, None)
        R2_Sqr_linspace=r2_score(np.sqrt(Ytest), np.sqrt( Y_pred_Sm1))
        R2_Sqr_linspace_VW=r2_score(np.sqrt(Ytest), np.sqrt( Y_pred_Sm1),multioutput='variance_weighted')
        plot_performance(np.sqrt(Ytest), np.sqrt( Y_pred_Sm1), r'$N_{observed}$', r'$N_{predicted}$',title= r'single, all, all but one data', 
                         text='$R^2_{vw}=$'+ str( round(R2_Sqr_linspace_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_Sqr_linspace, 2) ), 
                         filename=destfold+file_suffix+"_Sqr_Sm1.png")
        linReg_dict.update({'Square Sm1':reg_dict})
        
        ############## saving data ##########
        pickle.dump(linReg_dict, open(fold+'linReg_'+file_suffix+'.dat', 'wb') )        

if __name__ == '__main__':
    main()





        
#X_train, X_test, Y_train, Y_test = train_test_split(X_all_i[i], Y_all_i[i], test_size=Fraction_test)
############ plotting results of 2 models side by side############## 
#fig = plt.figure(figsize=(7,3.5)) 
#ax1 = fig.add_subplot(1,2,1) 
#ax2 = fig.add_subplot(1, 2 ,2)
#ax1.plot(Y_vectors_Lin,Y_vectors_pred_Lin,'bo')
#ax2.plot(np.power(10,Y_vectors_Log),np.power(10,Y_vectors_pred_Log),'bo')
#ax1.text(0.05, 0.85,'$R^2=$'+ str( round(R2_Lin, 2) ) , horizontalalignment='left', transform=ax1.transAxes)  
#ax2.text(0.05, 0.85,'$R^2=$'+ str( round(R2_Log_linspace, 2) ) , horizontalalignment='left', transform=ax2.transAxes) 
#ax1.text(0.05, 0.91,'Variance Weighted $R^2=$'+ str( round(R2_Lin_VW, 2) ) , horizontalalignment='left', transform=ax1.transAxes)  
#ax2.text(0.05, 0.91,'Variance Weighted $R^2=$'+ str( round(R2_Log_linspace_VW, 2) ) , horizontalalignment='left', transform=ax2.transAxes) 
#ax1.plot(ax1.get_xlim(),ax1.get_xlim(),'k--')   
#ax2.plot(ax2.get_xlim(),ax2.get_xlim(),'k--')  
#ax1.set_ylabel(r'$N_{predicted}$')
#ax1.set_xlabel(r'$N_{observed}$')
#ax2.set_xlabel(r'$N_{observed}$') 
#ax2.set_ylabel(r'$N_{predicted}$') 
#ax1.set_title(r'Linear Presence Absence',weight="bold")
#ax2.set_title(r'Exponential Presence Absence',weight="bold")    
#if fold.startswith('/Users/ashish'): plt.tight_layout()
#else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)   
#

