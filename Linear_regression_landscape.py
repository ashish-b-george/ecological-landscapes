#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 16:27:45 2019
This code performs linear regression without subsetting the data, the subsetting made the other code difficult to adapt.
Performs linear regression with/without regularization on abundance data from different initial conditions,
computes R^2 and out of sample error
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
import sys
import glob
if os.getcwd().startswith('/Users/ashish'):
    import matplotlib as mpl
    
else: ### can not use pyplot on cluster without agg backend   
    import matplotlib as mpl
    mpl.use('Agg')
    sys.path.append('/usr3/graduate/ashishge/ecology/Ecology-Stochasticity/')
    
import matplotlib.pyplot as plt
from my_CR_model_functions import rms,create_replicate_lists
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="data path", default=None) ## only a single folder handled for now!
#    parser.add_argument("-p", help="param name", default=None)
#    parser.add_argument("-v", help="param value", default=None)
    args = parser.parse_args()
    if args.d is not None:
        folder_name_list=list(args.d)## only a single folder handled for now!!
    else:
    
        ##################  file location and such things  ########################
    #    base_dir="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/"
    #    folder_name_list=[base_dir+'Crossfeeding_fig_M4_compiled/']
    #    folder_name_list=[base_dir+'Crossfeeding_fig_M3_compiled-a very degenerate case/',
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
    #                      base_dir+'Crossfeeding_fig_4consumptions_compiled/',
    #                      base_dir+'Crossfeeding_fig_6consumptions_compiled/',
    #                      base_dir+'Crossfeeding_fig_8consumptions_compiled/',
    #                      ]
        
    
    #    base_dir="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/Crossfeeding_AllEatR0/"   
    #    folder_name_list= [base_dir+'/Crossfeeding_alleatR0_meanC2_compiled/', base_dir+'/Crossfeeding_alleatR0_meanC3_compiled/', 
    #                       base_dir+'/Crossfeeding_alleatR0_meanC4_compiled/', base_dir+'/Crossfeeding_alleatR0_meanC5_compiled/',
    #                       base_dir+'/Crossfeeding_alleatR0_meanC6_compiled/', base_dir+'/Crossfeeding_alleatR0_meanC8_compiled/',                       
    #                       base_dir+'/Crossfeeding_alleatR0_meanC10_compiled/',base_dir+'/Crossfeeding_alleatR0_meanC12_compiled/']
    # 
    
    #    base_dir="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/Crossfeeding_uniformC/"
    #    folder_name_list=[base_dir+'/Dbinary_SM10_meanC2_compiled/', base_dir+'/Dbinary_SM10_meanC3_compiled/', 
    #                       base_dir+'/Dbinary_SM10_meanC4_compiled/', base_dir+'/Dbinary_SM10_meanC5_compiled/',
    #                       base_dir+'/Dbinary_SM10_meanC6_compiled/', base_dir+'/Dbinary_SM10_meanC8_compiled/',
    #                       base_dir+'/Dbinary_SM10_meanC10_compiled/']
    #   
       
    #    base_dir="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/Crossfeeding_binaryD/"
    #    folder_name_list=[base_dir+'/Dbinary_SM10_meanC2_compiled/', base_dir+'/Dbinary_SM10_meanC3_compiled/', 
    #                       base_dir+'/Dbinary_SM10_meanC4_compiled/', base_dir+'/Dbinary_SM10_meanC5_compiled/',
    #                       base_dir+'/Dbinary_SM10_meanC6_compiled/', base_dir+'/Dbinary_SM10_meanC8_compiled/',
    #                       base_dir+'/Dbinary_SM10_meanC10_compiled/'] 
        
    #    folder_name_list=[base_dir+'/Dbinary_IncreasedVarianceINmgrSM10_meanC2_compiled/', base_dir+'/Dbinary_IncreasedVarianceINmgrSM10_meanC3_compiled/', 
    #                       base_dir+'/Dbinary_IncreasedVarianceINmgrSM10_meanC4_compiled/', base_dir+'/Dbinary_IncreasedVarianceINmgrSM10_meanC5_compiled/',
    #                       base_dir+'/Dbinary_IncreasedVarianceINmgrSM10_meanC6_compiled/', base_dir+'/Dbinary_IncreasedVarianceINmgrSM10_meanC8_compiled/',
    #                       base_dir+'/Dbinary_IncreasedVarianceINmgrSM10_meanC10_compiled/'] 
        
        n_replicates=2
        base_fold="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/CR_binary_MWgamma_meanC_compiled/"
        destfold_list=[base_fold+'/SM10_meanC2/', base_fold+'/SM10_meanC3/', 
                       base_fold+'/SM10_meanC4/', base_fold+'/SM10_meanC6/', 
                       base_fold+'/SM10_meanC8/', base_fold+'/SM10_meanC10/']
        parameter_list=[1,2,3,4,6,7,8]
        destfold_list,parameter_list=create_replicate_lists(parameter_list,destfold_list, n_replicates)
        folder_name_list=destfold_list
    #    folder_name_list=[fold[:-1]+'_compiled/' for fold in destfold_list]

    
#    analysis_rewrite= False ## if true, it deletes current dictionary and replaces with new dictionary
#    analysis_overwrite=True ## overwrites existing values of  fitness functions analyzed only      
    #fold="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/Crossfeeding_fig_M6_compiled/"
    for fold in folder_name_list:
        #destfold=fold[:-1]+"_plots/"
        destfold=fold
        print (fold)
        file_suffix=None
        if not os.path.exists(destfold): os.makedirs(destfold)
        if file_suffix==None:## finds the first file the matches patter with glob.glob
            ctr=0
            for file_suffix in glob.glob(fold+'S*.dat'):
                file_suffix=file_suffix.replace(fold,'').replace('.dat','')
                print ("file_suffix is now ", file_suffix)
                ctr+=1

            assert ctr>0,"no file found"
            assert ctr<2,"we can only have one file that matches the pattern in the folder!"

        run_linear_regressions(fold, destfold, file_suffix)


    

def splitData_upto_pairwise_as_trainingdata(sigma_vectors, N_obs, return_indices=False  ):        
        ##data with upto 2 species present is training data
        n_exp=len(sigma_vectors)
        sigma_vectors_train=[]
        N_obs_train=[]
        sigma_vectors_test=[]
        N_obs_test=[]  
        if return_indices==True:
            idx_test=[]
            idx_train=[]
     
        for i in range(n_exp):
            if np.sum(sigma_vectors[i])<=2:
                sigma_vectors_train.append(sigma_vectors[i])
                N_obs_train.append(N_obs[i])
                if return_indices==True:idx_train.append(i)
            else:
                sigma_vectors_test.append(sigma_vectors[i])
                N_obs_test.append(N_obs[i]) 
                if return_indices==True:idx_test.append(i)
                
        if return_indices==True:
            return np.asarray(sigma_vectors_train), np.asarray(N_obs_train), np.asarray(sigma_vectors_test), np.asarray(N_obs_test), np.asarray(idx_train), np.asarray(idx_test) 
        else:
            return np.asarray(sigma_vectors_train), np.asarray(N_obs_train), np.asarray(sigma_vectors_test), np.asarray(N_obs_test) 

    
def splitData_single_All_AllButOne_as_trainingdata(sigma_vectors, N_obs, return_indices=False ):        
        ##data with upto 1 species present, and >=S-1 species present is training data
        n_exp=len(sigma_vectors)
        S=len(sigma_vectors[0])
        sigma_vectors_train=[]
        N_obs_train=[]
        sigma_vectors_test=[]
        N_obs_test=[]  
        if return_indices==True:
            idx_test=[]
            idx_train=[]
     
        for i in range(n_exp):
            if np.sum(sigma_vectors[i])<=1:
                sigma_vectors_train.append(sigma_vectors[i])
                N_obs_train.append(N_obs[i])
                if return_indices==True:idx_train.append(i)              
            elif np.sum(sigma_vectors[i])>=S-1:
                sigma_vectors_train.append(sigma_vectors[i])
                N_obs_train.append(N_obs[i])
                if return_indices==True:idx_train.append(i)               
            else:
                sigma_vectors_test.append(sigma_vectors[i])
                N_obs_test.append(N_obs[i])  
                if return_indices==True:idx_test.append(i)
        
        if return_indices==True:
            return np.asarray(sigma_vectors_train), np.asarray(N_obs_train), np.asarray(sigma_vectors_test), np.asarray(N_obs_test), np.asarray(idx_train), np.asarray(idx_test) 
        else:
            return np.asarray(sigma_vectors_train), np.asarray(N_obs_train), np.asarray(sigma_vectors_test), np.asarray(N_obs_test)   


def perform_Linear_Regression_of_fitness_scalar(Y_fitness, sigma_vectors, S, expansionOrder=1) :
    linReg_object = LinearRegression(fit_intercept=False) # polynomial features already generates a constant term
    poly = PolynomialFeatures(expansionOrder,interaction_only=True)
    X_sigma=poly.fit_transform(sigma_vectors)
    linReg_object.fit(X_sigma,Y_fitness)
    Y_fitness_pred=linReg_object.predict(X_sigma)
    R2_val=r2_score(Y_fitness,Y_fitness_pred)
    Exp_var=explained_variance_score(Y_fitness,Y_fitness_pred)
#    print ('R^2 is ',R2_val)
    reg_dict={'Y_fitness_pred':Y_fitness_pred,'coefficients':linReg_object.coef_, 'R2': R2_val, 'Explained variance':Exp_var }
    return reg_dict,linReg_object

   
def perform_Linear_Regression_of_abundance_vector(Y_vectors_train, sigma_vectors_train, S, expansionOrder=1, Y_vectors_test=None, sigma_vectors_test=None) :
    ##expansionOrder is order at which to perform the fit.
    linReg_object = LinearRegression(fit_intercept=False) # we remove consntatn term here and in poly features
    interaction_matrix=[]
    sum_of_squares=np.zeros(S)
    residual_square=np.zeros(S)   
    if Y_vectors_test is None:   ## TEST &TRAIN are the same, we caclulate R^2 and on the data we fit.
        Y_vectors_test=Y_vectors_train
        sigma_vectors_test=sigma_vectors_train
   
    Y_vectors_pred=np.empty_like(Y_vectors_test)
    for i in range(S):
        Y_i=Y_vectors_train[:,i]
        sigma_i=sigma_vectors_train[:,i]       
        poly = PolynomialFeatures(expansionOrder,interaction_only=True)
        poly_sigma=poly.fit_transform(sigma_vectors_train)[:,1:] #we do not want the constant, since we constrain constant=0, and a_ii performs role of a_i   
        #X_i=(sigma_vectors.T*sigma_i).T   
        #OR
        #X_i=np.swapaxes( np.swapaxes(sigma_vectors,0,1)*sigma_i, 0,1)      
        X_i=(poly_sigma.T*sigma_i).T  
        linReg_object.fit(X_i,Y_i)
        interaction_matrix.append(linReg_object.coef_)
        '''
        To mutliply a vector across a matrix elementwise:
        You can automatically broadcast the vector against the outermost axis of an array. 
        So, you can transpose the array to swap the axis you want to the outside, multiply, then transpose it back:
        https://stackoverflow.com/questions/19388152/numpy-element-wise-multiplication-of-an-array-and-a-vector   
        when you want to multply across axis 2:
        ares = (a.transpose(0,1,3,2) * v).transpose(0,1,3,2)
        '''    
        
        Ytest_i=Y_vectors_test[:,i]
        sigmatest_i=sigma_vectors_test[:,i]       
        poly_sigmatest=poly.fit_transform(sigma_vectors_test)[:,1:] #we do not want the constant, since we constrain constant to 0!        
        Xtest_i=(poly_sigmatest.T*sigmatest_i).T  
        Y_pred=linReg_object.predict(Xtest_i)
        Y_vectors_pred[:,i]=Y_pred
            
        sum_of_squares[i]=np.sum( np.square( Ytest_i.mean()-Ytest_i ))
        residual_square[i]=np.sum( np.square( Ytest_i-Xtest_i@linReg_object.coef_  )  ) 
#        print ("intercept is",linReg_object.intercept_)
#        assert np.all(Y_pred-X_i@linReg_object.coef_ ==0)
        
    interaction_matrix  =np.asarray(interaction_matrix)         
    '''
    if we are treating species as identical, then caclulating the R^2 as in the manual calculation,
    equivalent to 'variance weighted' R^2 makes sense (and not  R^2). This is because R2VW is equivalent to summing up all the squared residuals
    of all the fits and all the sum of squares (variances) of all fits and then taking the ratio. Instead of summing up the ratio of each fit.
    '''
    R2_avg=r2_score(Y_vectors_test,Y_vectors_pred)##  avg of the R2 of prediction for each species
    R2_VW=r2_score(Y_vectors_test,Y_vectors_pred, multioutput='variance_weighted')  ##avg Weighted  by the variance in each species
    R2_manual=1- np.sum(residual_square)/np.sum(sum_of_squares)## this is R2_VW!
    print ("R2avg, R2_VW, R2_manual, ",R2_avg, R2_VW ,R2_manual)    
    EV_avg=explained_variance_score(Y_vectors_test,Y_vectors_pred)
    EV_VW=explained_variance_score(Y_vectors_test,Y_vectors_pred, multioutput='variance_weighted')     
    print ("explained variance score  EVavg, EV_VW", EV_avg, EV_VW)   ## equals R2 if mean of ypred and ydata are the same.
    #cannot calculate nonzero like this : R2_nonzero=r2_score(Y_vectors[sigma_vectors!=0],Y_vectors_pred[sigma_vectors!=0])    
    #R2_nonzero calculated incorrectly because subsettng flattens the array
    
    reg_dict={'Y_vectors_pred':Y_vectors_pred,'interaction_matrix':interaction_matrix, 'R2_avg': R2_avg, 'R2_VW': R2_VW}
    
    return Y_vectors_pred,interaction_matrix, R2_avg, R2_VW, reg_dict



def N_to_Ncutoff(N_obs, sigma_vectors,popn_cutoff=1e-6):
    N_cutoff=N_obs
    N_cutoff[N_obs<popn_cutoff]=0.0
    assert np.all(N_cutoff[sigma_vectors==0]==0),'Nf=0 if Ni=0'
    return N_cutoff

def transform_to_log_abundance(N_obs, sigma_vectors, popn_cutoff=1e-6, pseudoCount=1e-10):
#    N_pseudo=N_obs
#    N_pseudo[N_obs<popn_cutoff]=pseudoCount # ## if a species was present initally, it goes to the pseudocount, not below 
#    Y_vectors=np.log10(N_pseudo)
#    Y_vectors[sigma_vectors==0]=0  ## if a species wasnt present intially,its final value is zero
    ## we need it to be zero, since we are setting the intercept of the fit to 0.  

    N_pseudo=N_obs+pseudoCount
    Y_vectors=np.log10(N_pseudo)
    Y_vectors[sigma_vectors==0]=0 
    return Y_vectors

def rand_jitter(arr):
    stdev = .01*(np.max(arr)-np.min(arr))
    return arr + np.random.randn(arr.size).reshape(np.shape(arr)) * stdev

def plot_performance(xdata, ydata, xlabel, ylabel, title=None, filename=None, text=None, text_xpos=0.05, text_ypos=0.99 ,ms_val=4, alpha_val=0.05 ) :    
    fig = plt.figure(figsize=(3.5,3.5)) 
    ax1 = fig.add_subplot(1,1,1) 
    ax1.plot(rand_jitter(xdata), rand_jitter(ydata),'bo', mec='None', markersize=ms_val, alpha=alpha_val )   
    if text!=None:
        ax1.text(text_xpos, text_ypos, text , horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes)     
    ax1.plot(ax1.get_xlim(),ax1.get_xlim(),'k--')    
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(xlabel)
    if title!=None:ax1.set_title(title,weight="bold") 
    ax1.ticklabel_format(axis='both',useOffset=False)
    #ax1.ticklabel_format(axis='both', style='sci', scilimits=(20,20))
    #ax1.ticklabel_format(useOffset=False, useLocale=False)
    if os.getcwd().startswith('/Users/ashish'): plt.tight_layout()
    else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1) 
    if filename!=None:plt.savefig(filename,dpi=300)
        






def run_linear_regressions(fold, destfold, file_suffix,  analysis_overwrite=True):
#        assert analysis_overwrite==True,'other method should not be used untile code is ready.'
        ##################  reading and initializing data  ########################
        data = pickle.load(open(fold+file_suffix+'.dat', 'rb')) 
        if np.all(data['reached_SteadyState']==True):
            print ("all runs had reached steady state")            
        else:
            print ("NOT SS!")
            destfold=destfold+'/notSS/'
            if not os.path.exists(destfold): os.makedirs(destfold)
        
        
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
        reg_dict_Prefixed ={'Linear'+': '+k: v for k, v in reg_dict.items()} ##add a prefix to all measured values to identify the linear Reg used 
        linReg_dict.update(reg_dict_Prefixed)        
#        linReg_dict.update({'Linear':reg_dict})
        
        
        ######predicting from splitting into train& test alon pwise, AllButOne , etc###########
        sigma_train, Ytrain, sigma_test, Ytest=splitData_upto_pairwise_as_trainingdata(sigma_vectors, Y_vectors_Lin )
        Y_pred_pwise, _, R2_pwise, R2_pwise_VW, reg_dict=perform_Linear_Regression_of_abundance_vector(Ytrain, sigma_train,  S, expansionOrder=1, Y_vectors_test= Ytest, sigma_vectors_test=sigma_test)
        plot_performance(Ytest,Y_pred_pwise, r'$N_{observed}$', r'$N_{predicted}$',title= r'from upto pairwise data', 
                         text='$R^2_{vw}=$'+ str( round(R2_pwise_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_pwise, 2) ), 
                         filename=destfold+file_suffix+"_LPA_pwise.png")
        reg_dict_Prefixed ={'Linear pwise'+': '+k: v for k, v in reg_dict.items()} ##add a prefix to all measured values to identify the linear Reg used 
        linReg_dict.update(reg_dict_Prefixed) 
        
        sigma_train, Ytrain, sigma_test, Ytest=splitData_single_All_AllButOne_as_trainingdata(sigma_vectors, Y_vectors_Lin )
        Y_pred_Sm1, _, R2_Sm1, R2_Sm1_VW, reg_dict=perform_Linear_Regression_of_abundance_vector(Ytrain, sigma_train,  S, expansionOrder=1, Y_vectors_test= Ytest, sigma_vectors_test=sigma_test)
        plot_performance(Ytest,Y_pred_Sm1, r'$N_{observed}$', r'$N_{predicted}$',title= r'single, all, all but one data', 
                         text='$R^2_{vw}=$'+ str( round(R2_Sm1_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_Sm1, 2) ), 
                         filename=destfold+file_suffix+"_LPA_Sm1.png")
        reg_dict_Prefixed ={'Linear Sm1'+': '+k: v for k, v in reg_dict.items()} ##add a prefix to all measured values to identify the linear Reg used 
        linReg_dict.update(reg_dict_Prefixed) 

        
        
        
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
        reg_dict_Prefixed ={'Log'+': '+k: v for k, v in reg_dict.items()} ##add a prefix to all measured values to identify the linear Reg used 
        linReg_dict.update(reg_dict_Prefixed) 

        
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
        reg_dict_Prefixed ={'Square root'+': '+k: v for k, v in reg_dict.items()} ##add a prefix to all measured values to identify the linear Reg used 
        linReg_dict.update(reg_dict_Prefixed) 

        
        ######predicting from splitting into train& test alon pwise, AllButOne , etc###########
        sigma_train, Ytrain, sigma_test, Ytest=splitData_upto_pairwise_as_trainingdata(sigma_vectors, Y_vectors_Sqrt)
        Y_pred_pwise, _, R2_pwise, R2_pwise_VW, reg_dict=perform_Linear_Regression_of_abundance_vector(Ytrain, sigma_train,  S, expansionOrder=1, Y_vectors_test= Ytest, sigma_vectors_test=sigma_test)
        R2_Sqrt_linspace=r2_score(Ytest**2, Y_pred_pwise**2)
        R2_Sqrt_linspace_VW=r2_score(Ytest**2, Y_pred_pwise**2,multioutput='variance_weighted')
        plot_performance(Ytest**2, Y_pred_pwise**2, r'$N_{observed}$', r'$N_{predicted}$',title= r'from upto pairwise data', 
                         text='$R^2_{vw}=$'+ str( round(R2_Sqrt_linspace_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_Sqrt_linspace, 2) ), 
                         filename=destfold+file_suffix+"_Sqrt_pwise.png")
        reg_dict_Prefixed ={'Square root pwise'+': '+k: v for k, v in reg_dict.items()} ##add a prefix to all measured values to identify the linear Reg used 
        linReg_dict.update(reg_dict_Prefixed) 

        
        sigma_train, Ytrain, sigma_test, Ytest=splitData_single_All_AllButOne_as_trainingdata(sigma_vectors, Y_vectors_Sqrt)
        Y_pred_Sm1, _, R2_Sm1, R2_Sm1_VW, reg_dict=perform_Linear_Regression_of_abundance_vector(Ytrain, sigma_train,  S, expansionOrder=1, Y_vectors_test= Ytest, sigma_vectors_test=sigma_test)
        R2_Sqrt_linspace=r2_score(Ytest**2, Y_pred_Sm1**2)
        R2_Sqrt_linspace_VW=r2_score(Ytest**2, Y_pred_Sm1**2,multioutput='variance_weighted')
        plot_performance(Ytest,Y_pred_Sm1, r'$N_{observed}$', r'$N_{predicted}$',title= r'single, all, all but one data', 
                         text='$R^2_{vw}=$'+ str( round(R2_Sqrt_linspace_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_Sqrt_linspace, 2) ), 
                         filename=destfold+file_suffix+"_Sqrt_Sm1.png")
        reg_dict_Prefixed ={'Square root Sm1'+': '+k: v for k, v in reg_dict.items()} ##add a prefix to all measured values to identify the linear Reg used 
        linReg_dict.update(reg_dict_Prefixed) 


        
        
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
        reg_dict_Prefixed ={'Square'+': '+k: v for k, v in reg_dict.items()} ##add a prefix to all measured values to identify the linear Reg used 
        linReg_dict.update(reg_dict_Prefixed)

        
        ######predicting from splitting into train& test alon pwise, AllButOne , etc###########
        sigma_train, Ytrain, sigma_test, Ytest=splitData_upto_pairwise_as_trainingdata(sigma_vectors, Y_vectors_Sqr)
        Y_pred_pwise, _, R2_pwise, R2_pwise_VW, reg_dict=perform_Linear_Regression_of_abundance_vector(Ytrain, sigma_train,  S, expansionOrder=1, Y_vectors_test= Ytest, sigma_vectors_test=sigma_test)
        Y_pred_pwise=np.clip(Y_pred_pwise, 0, None)
        R2_Sqr_linspace=r2_score(np.sqrt(Ytest), np.sqrt( Y_pred_pwise))
        R2_Sqr_linspace_VW=r2_score(np.sqrt(Ytest), np.sqrt( Y_pred_pwise),multioutput='variance_weighted')
        plot_performance(np.sqrt(Ytest), np.sqrt( Y_pred_pwise), r'$N_{observed}$', r'$N_{predicted}$',title= r'from upto pairwise data', 
                         text='$R^2_{vw}=$'+ str( round(R2_Sqr_linspace_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_Sqr_linspace, 2) ), 
                         filename=destfold+file_suffix+"_Sqr_pwise.png")
        reg_dict_Prefixed ={'Square pwise'+': '+k: v for k, v in reg_dict.items()} ##add a prefix to all measured values to identify the linear Reg used 
        linReg_dict.update(reg_dict_Prefixed)

        
        sigma_train, Ytrain, sigma_test, Ytest=splitData_single_All_AllButOne_as_trainingdata(sigma_vectors, Y_vectors_Sqr)
        Y_pred_Sm1, _, R2_Sm1, R2_Sm1_VW, reg_dict=perform_Linear_Regression_of_abundance_vector(Ytrain, sigma_train,  S, expansionOrder=1, Y_vectors_test= Ytest, sigma_vectors_test=sigma_test)
        Y_pred_Sm1=np.clip(Y_pred_Sm1, 0, None)
        R2_Sqr_linspace=r2_score(np.sqrt(Ytest), np.sqrt( Y_pred_Sm1))
        R2_Sqr_linspace_VW=r2_score(np.sqrt(Ytest), np.sqrt( Y_pred_Sm1),multioutput='variance_weighted')
        plot_performance(np.sqrt(Ytest), np.sqrt( Y_pred_Sm1), r'$N_{observed}$', r'$N_{predicted}$',title= r'single, all, all but one data', 
                         text='$R^2_{vw}=$'+ str( round(R2_Sqr_linspace_VW, 2) )+'\n$R^2_{avg}=$'+ str( round(R2_Sqr_linspace, 2) ), 
                         filename=destfold+file_suffix+"_Sqr_Sm1.png")
        reg_dict_Prefixed ={'Square Sm1'+': '+k: v for k, v in reg_dict.items()} ##add a prefix to all measured values to identify the linear Reg used 
        linReg_dict.update(reg_dict_Prefixed)

        
        ############## saving data ##########

        pickle.dump(linReg_dict, open(destfold+'linReg_'+file_suffix+'.dat', 'wb') ) 
        plt.close("all")

if __name__ == '__main__':
    main()

