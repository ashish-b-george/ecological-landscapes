#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 17:05:24 2019

@author: ashish

makes summary plots across parameters values of data created by landscape analysis & linear regression
"""
import numpy as np
import pickle # in python3, pickle uses cPickle automatically, there is no cPickle
import os
import glob
import sys
import pandas as pd
from scipy.stats.stats import pearsonr 
from copy import deepcopy 

if os.getcwd().startswith("/Users/ashish"):
    import matplotlib as mpl
else:  
    sys.path.append('/usr3/graduate/ashishge/ecology/ecological-landscapes/')
    import matplotlib as mpl
    mpl.use('Agg')### can not use pyplot on cluster without agg backend
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from my_CR_model_functions import rms,create_replicate_lists
nice_fonts = { #"text.usetex": True, # Use LaTex to write all text
"font.family": "serif",
# Use 10pt font in plots, to match 10pt font in document
"axes.labelsize": 10, "font.size": 10,
# Make the legend/label fonts a little smaller
"legend.fontsize": 8, "xtick.labelsize": 8,"ytick.labelsize": 8 }
mpl.rcParams.update(nice_fonts)

#from scipy,interpolate import  griddata
import scipy


base_dir="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/"



###### varying C Consumer ResourceModel (no crossfeeding)
#destfolder='/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/summary_plots2/CRmodel_varying_meanC/'
#folder_name_list=[base_dir+'CRmodel_figA_niche_compiled/',
#                  base_dir+'CRmodel_figB_2interactions_compiled/',
#                  base_dir+'CRmodel_figB_3interactions_compiled/',
#                  base_dir+'CRmodel_figC_6interactions_compiled/'
#                  ]
#Folders_Label="R types consumed per species"
#Folders_values=[1,2,3,6]


###### varying C with crossfeeding 
#destfolder=base_dir+'/summary_plots2/Crossfeeding_varying_meanC/'
#folder_name_list=[base_dir+'Crossfeeding_fig_2consumptions_compiled/',
#                  base_dir+'Crossfeeding_fig_M12_compiled/',
#                  base_dir+'Crossfeeding_fig_4consumptions_compiled/',
#                  base_dir+'Crossfeeding_fig_6consumptions_compiled/',
#                  base_dir+'Crossfeeding_fig_8consumptions_compiled/'
#                  ]
#Folders_Label="R types consumed per species"
#Folders_values=[2,3,4,6,8]


#### varying meanC with crossfeeding where all species eat R0
#destfolder='/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/summary_plots2/Crossfeeding_AllEatR0/'
#base_dir="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/Crossfeeding_AllEatR0/"
#destfolder='/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/summary_plots2/Crossfeeding_AllEatR0_increasedVar/'
#base_dir="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/Crossfeeding_AllEatR0_increasedVar/"
#folder_name_list= [base_dir+'/Crossfeeding_alleatR0_meanC2_compiled/', base_dir+'/Crossfeeding_alleatR0_meanC3_compiled/', 
#                       base_dir+'/Crossfeeding_alleatR0_meanC4_compiled/', base_dir+'/Crossfeeding_alleatR0_meanC5_compiled/',
#                       base_dir+'/Crossfeeding_alleatR0_meanC6_compiled/', base_dir+'/Crossfeeding_alleatR0_meanC8_compiled/',
#                       base_dir+'/Crossfeeding_alleatR0_meanC10_compiled/',base_dir+'/Crossfeeding_alleatR0_meanC12_compiled/']
#Folders_Label="R types consumed per species"
#Folders_values=[2,3,4,5,6,8,10,12]

#### varying meanC with crossfeeding where all species eat R0 and consumption matrix is from a uniform distribution
#destfolder='/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/summary_plots2/Crossfeeding_uniformC/'
#base_dir="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/Crossfeeding_uniformC/"
#destfolder='/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/summary_plots2/Crossfeeding_uniformC_increasedVar/'
#base_dir="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/Crossfeeding_uniformC_increasedVar/"
#folder_name_list=[base_dir+'/Dbinary_SM10_meanC2_compiled/', base_dir+'/Dbinary_SM10_meanC3_compiled/', 
#                   base_dir+'/Dbinary_SM10_meanC4_compiled/', base_dir+'/Dbinary_SM10_meanC5_compiled/',
#                   base_dir+'/Dbinary_SM10_meanC6_compiled/', base_dir+'/Dbinary_SM10_meanC8_compiled/',
#                   base_dir+'/Dbinary_SM10_meanC10_compiled/']
#Folders_Label="R types consumed per species"
#Folders_values=[2,3,4,5,6,8,10]

#### varying meanC with crossfeeding with binary leakage matrix (only some species eat R0)
#destfolder='/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/summary_plots2/Crossfeeding_binaryD/'  
#base_dir="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/Crossfeeding_binaryD/"
#folder_name_list=[base_dir+'/Dbinary_SM10_meanC2_compiled/', base_dir+'/Dbinary_SM10_meanC3_compiled/', 
#                   base_dir+'/Dbinary_SM10_meanC4_compiled/', base_dir+'/Dbinary_SM10_meanC5_compiled/',
#                   base_dir+'/Dbinary_SM10_meanC6_compiled/', base_dir+'/Dbinary_SM10_meanC8_compiled/',
#                   base_dir+'/Dbinary_SM10_meanC10_compiled/'] 
#Folders_Label="R types consumed per species"
#Folders_values=[2,3,4,5,6,8,10]

#### varying meanC with crossfeeding with binary leakage matrix (only some species eat R0), with increase variance in m,g, w etc
#destfolder='/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/summary_plots2/Crossfeeding_binaryD_increasedVar/'  
#base_dir="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/Crossfeeding_binaryD/"
#folder_name_list=[base_dir+'/Dbinary_IncreasedVarianceINmgrSM10_meanC2_compiled/', base_dir+'/Dbinary_IncreasedVarianceINmgrSM10_meanC3_compiled/', 
#                   base_dir+'/Dbinary_IncreasedVarianceINmgrSM10_meanC4_compiled/', base_dir+'/Dbinary_IncreasedVarianceINmgrSM10_meanC5_compiled/',
#                   base_dir+'/Dbinary_IncreasedVarianceINmgrSM10_meanC6_compiled/', base_dir+'/Dbinary_IncreasedVarianceINmgrSM10_meanC8_compiled/',
#                   base_dir+'/Dbinary_IncreasedVarianceINmgrSM10_meanC10_compiled/'] 
#
#Folders_Label="R types consumed per species"
#Folders_values=[2,3,4,5,6,8,10]


#### varying M with crossfeeding
#destfolder=base_dir+'/summary_plots2/Crossfeeding_varyingM/'
#folder_name_list=[base_dir+'Crossfeeding_fig_M4_compiled/',
#                  base_dir+'Crossfeeding_fig_M6_compiled/',
#                  base_dir+'Crossfeeding_fig_M9_compiled/',
#                  base_dir+'Crossfeeding_fig_M12_compiled/',
#                  base_dir+'Crossfeeding_fig_M15_compiled/',
#                  base_dir+'Crossfeeding_fig_M18_compiled/',
#                  base_dir+'Crossfeeding_fig_M24_compiled/'
#                  ]
#Folders_Label="number of resources"
#Folders_values=[4,6,9,12,15,18,24]


# =============================================================================
# ############# for runs with replicates:
# plot_with_error_bars=True
# 
# #base_fold="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/CR_mGamma_wGamma_CbinarydiagGamma_SM10_meanC_compiled/"
# #folder_name_list=[base_fold+'/SM10_meanC1/', base_fold+'/SM10_meanC2/', 
# #   base_fold+'/SM10_meanC3/', base_fold+'/SM10_meanC4/', 
# #   base_fold+'/SM10_meanC5/', base_fold+'/SM10_meanC6/',
# #   base_fold+'/SM10_meanC7/', base_fold+'/SM10_meanC8/']
# #Folders_Label="R types consumed per species"
# #Folders_values=[1,2,3,4,5,6,7,8]
# #file_suffix='S10M10_mw_gamma'
# #destfolder='/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/summary_plots2/CR_mGamma_wGamma_CbinarydiagGamma_SM10_meanC_compiled/'
# 
# #base_fold="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/CF_wGamma_AllRSupplied_meanC_compiled/"
# #destfolder='/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/summary_plots2/CF_wGamma_AllRSupplied_meanC_compiled/'
# #folder_name_list=[base_fold+'/SM10_meanC1/', base_fold+'/SM10_meanC2/', 
# #           base_fold+'/SM10_meanC3/', base_fold+'/SM10_meanC4/', 
# #           base_fold+'/SM10_meanC5/', base_fold+'/SM10_meanC6/',
# #           base_fold+'/SM10_meanC7/', base_fold+'/SM10_meanC8/',
# #           base_fold+'/SM10_meanC9/', base_fold+'/SM10_meanC10/'] 
# #Folders_Label="R types consumed per species"
# #Folders_values=[1,2,3,4,5,6,7,8,9,10]
# #file_suffix='S10M10_w_gamma'
# 
# #base_fold="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/CF_wGamma_meanC_compiled/"
# #folder_name_list=[base_fold+'/SM10_meanC2/', 
# #           base_fold+'/SM10_meanC3/', base_fold+'/SM10_meanC4/', 
# #           base_fold+'/SM10_meanC5/', base_fold+'/SM10_meanC6/',
# #           base_fold+'/SM10_meanC7/', base_fold+'/SM10_meanC8/',
# #           base_fold+'/SM10_meanC9/', base_fold+'/SM10_meanC10/'] 
# #Folders_Label="R types consumed per species"
# #Folders_values=[2,3,4,5,6,7,8,9,10]
# #file_suffix='S10M10_w_gamma'
# 
# #base_fold="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/CF_wGamma_l_compiled/"
# #folder_name_list=[
# #        base_fold+'/SM10_l1/', base_fold+'/SM10_l2/', 
# #           base_fold+'/SM10_l3/', base_fold+'/SM10_l4/', 
# #           base_fold+'/SM10_l5/', base_fold+'/SM10_l6/',
# #           base_fold+'/SM10_l7/', base_fold+'/SM10_l8/',
# #           base_fold+'/SM10_l9/']
# #Folders_Label="leakage fraction, l"
# #Folders_values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# #file_suffix='S10M10_w_gamma'
# 
# 
# #base_fold="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/CR_wGamma_CbinarydiagGamma_SoverR_varyS_compiled/"
# #destfolder='/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/summary_plots2/CR_wGamma_CbinarydiagGamma_SoverR_varyS_compiled/'
# #folder_name_list=[base_fold+'/S6_SoverR2/', base_fold+'/S8_SoverR2/', 
# #                  base_fold+'/SM10_meanC5/', base_fold+'/S12_SoverR2/'
# #                  , base_fold+'/S14_SoverR2/']
# #Folders_Label="number of species"
# #Folders_values=[6,8,10,12,14]
# #file_suffix='S_SoverR_S_w_gamma' 
# =============================================================================

# =============================================================================
# =============================================================================
# ##################### For  phase plot folders ############## 
# plot_with_error_bars=True
# 
# base_fold="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/CR_cvxpt_2D_SM_CfixednumGamma_meanC_compiled_AVG/"
# parameter_list=OrderedDict([ ('SM',[ 4, 8, 12, 16] ), ('mean_c', '1 to param1')])
# 
# #base_fold="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/CR_cvxpt_2D_SM2_CfixednumGamma_meanC_compiled_AVG/"
# #parameter_list=OrderedDict([ ('SM2-',[4, 8, 12, 16 ] ), ("mean_c", "1 to param1{'step_size' : 2, 'UB_X' : 2}")])
# destfold_list=[base_fold] 
# parameter_name='2D'
# parameter_names=list(parameter_list.keys())
# folder_name_list,parameter_list=create_replicate_lists(parameter_list,destfold_list, 0) ### creates a longer list enumerating all the folders.
# Folders_Label=parameter_names
# Folders_values=parameter_list
# file_suffix='S_2D_w_gamma'
# =============================================================================
plot_with_error_bars=True
base_fold="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/CF_CfixedxGamma01Special_l_compiled_AVG/"
destfold_list=[base_fold]
parameter_name='l'
parameter_list={parameter_name:[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
folder_name_list,parameter_list, df_keyname_prefix_list=create_replicate_lists(parameter_list,destfold_list,0, return_dfkeynames=True) ### creates a longer list enumerating all the replicates.
Folders_Label=parameter_name
Folders_values=parameter_list
file_suffix='S_l_w_gamma'

# =============================================================================
# plot_with_error_bars=True
# base_fold="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/CF_CfixedxGamma0Special_meanC_compiled_AVG/"
# destfold_list=[base_fold]
# parameter_name='mean_c'
# parameter_list={parameter_name:[1,2,3,4,5,6,7,8,9,10,11,12]}
# folder_name_list,parameter_list, df_keyname_prefix_list=create_replicate_lists(parameter_list,destfold_list,0, return_dfkeynames=True)
# file_suffix='S_mean_c_w_gamma'
# Folders_Label=parameter_name
# Folders_values=parameter_list
# =============================================================================

# =============================================================================
# =============================================================================
destfolder=base_fold.replace("/binaryIC_all/", "/binaryIC_all/summary_plots2/")
if not os.path.exists(destfolder): os.makedirs(destfolder)
structure_plots=True

fitness_measure_list=['Shannon Diversity', 'Simpsons Diversity', 'Species 0']
analysis_metric_list=['r/s','FDC','R2_reg','R2_reg_w0','variance_explained_reg',
                      'p_Greediest_walk_found_global_max',
                      'mean_relative_optimum_Greedy','mean_relative_optimum_Greediest',
                      'coefficient of variation','mean','N_maxima_without_extinction','N_saddle','N_maxima'] #,'variance_explained_reg'
structure_metric_list=['Abundance vector fit Subsetted R2 Flattened','Abundance vector fit Subsetted R2avg','Abundance vector fit Subsetted R2','uniqueness','effective number of states','avg fraction species survived','S_star']

# =============================================================================
#     ###########  for testing, smaller number of params ############
# fitness_measure_list=['total biomass']
# analysis_metric_list=['r/s'] 
# structure_metric_list=['Abundance vector fit Subsetted R2 Flattened']
# 
# =============================================================================

extrema_lists=[]#=['maxima_without_extinction','maxima','minima_without_extinction','minima']



linReg_is_used=False
linReg_measure_list=['Linear', 'Linear pwise']
linReg_analysis_metric_list=['R2_avg', 'R2_VW']



#fitness_correlation_list=[ [ 'total biomass','Simpsons Diversity' ]]
#correlation_list=[ 
#         'total biomass: r/s','total biomass: maxima_without_extinction' , 'Linear: R2_VW'  ,'Linear: R2_avg' ,
#        ]

############### make a list of the quantiities we want to plot the correlation of:
correlation_list=['Cmatrix sparsity',
                  'landscape structure: Abundance vector fit Subsetted R2avg',                 
                  'Shannon Diversity: r/s','Shannon Diversity: mean_relative_optimum_Greediest' ,
                  'Shannon Diversity: R2_reg_w0','tShannon Diversity: PS_ratio',
                  'Simpsons Diversity: r/s', 'Simpsons Diversity: mean_relative_optimum_Greediest', 
                  'Simpsons Diversity: R2_reg_w0','Simpsons Diversity: PS_ratio',
                  'Species 0: r/s', 'Species 0: mean_relative_optimum_Greediest', 
                  'Species 0: R2_reg_w0','Species 0: PS_ratio',]
    
#for j, fitness_measure in enumerate(fitness_measure_list):
#    for k, metric_name in enumerate(analysis_metric_list):
#            key_name=fitness_measure+': '+metric_name
#            correlation_list.append(key_name)
#for j, fitness_measure in enumerate(linReg_measure_list):
#    for k, metric_name in enumerate(linReg_analysis_metric_list):
#            key_name=fitness_measure+': '+metric_name
#            correlation_list.append(key_name)
#for k, metric_name in enumerate(structure_metric_list):
#            key_name='landscape structure: '+metric_name
#            correlation_list.append(key_name)            
#            
            

            
I,J,K=len(folder_name_list), len(fitness_measure_list), len(analysis_metric_list)
metric_array=np.empty(( I,J,K ))
if plot_with_error_bars:
    metric_array_errorbars=np.empty(( I,J,K ))

if linReg_is_used:
    Jl,Kl=len(linReg_measure_list), len(linReg_analysis_metric_list)
    linReg_metric_array=np.empty(( I,Jl,Kl ))
if structure_plots==True:   
    Ks=len(structure_metric_list)
    structure_metric_array=np.empty((I,Ks))
    if plot_with_error_bars:
        structure_metric_array_errorbars=np.empty((I,Ks))

successfully_run_folders=np.zeros(I, dtype=bool)

    

for i,fold in enumerate(folder_name_list):
    
    if file_suffix==None:## finds the first file the matches patter with glob.glob,  only for non-averaged
        ctr=0
        for file_suffix in glob.glob(fold+'S*.dat'):
            file_suffix=file_suffix.replace(fold,'').replace('.dat','')
#            print ("file_suffix is now ", file_suffix)
            ctr+=1
        assert ctr==1,"we can only have one file that matches the pattern in the folder!"
    
    analysis_filename=fold+'analysis_'+file_suffix+'.dat'    
    if plot_with_error_bars:
        analysis_filename=fold+'AvgAnalysis_'+file_suffix+'.dat'    
    
    analysis_data=pickle.load(open(analysis_filename, 'rb')) 
     
    if 'Number of replicates' in analysis_data:
        if analysis_data['Number of replicates']==0:
            metric_array[i,:,:]=0
            structure_metric_array[i,:]=0
        else:
            successfully_run_folders[i]=True
            for j, fitness_measure in enumerate(fitness_measure_list):
                for k, metric_name in enumerate(analysis_metric_list):
                    key_name=fitness_measure+': '+metric_name
                    if 'variance_explained_reg' in metric_name or 'R2_reg'in metric_name :
                        metric_array[i,j,k]=analysis_data[key_name][1] ## changed to first order on 9/24, was [2] earlier 
                        if plot_with_error_bars:    
                            metric_array_errorbars[i,j,k]=np.sqrt(analysis_data[key_name+'-Variance'][1]/analysis_data['Number of replicates'])
                    
                    elif metric_name in extrema_lists:
                        metric_array[i,j,k]=analysis_data[key_name].size
                            
                            
                    else:
                        metric_array[i,j,k]=analysis_data[key_name]  
                        if plot_with_error_bars:    
                            metric_array_errorbars[i,j,k]=np.sqrt(analysis_data[key_name+'-Variance']/analysis_data['Number of replicates'])
                                              
            if structure_plots==True:
                for k, metric_name in enumerate(structure_metric_list):
                    key_name='landscape structure'+': '+metric_name
                    structure_metric_array[i,k]=analysis_data[key_name]
                    if plot_with_error_bars:    
                            structure_metric_array_errorbars[i,k]=np.sqrt(analysis_data[key_name+'-Variance']/analysis_data['Number of replicates'])
                                
            if linReg_is_used:
                linReg_filename=fold+'linReg_'+file_suffix+'.dat'
                linReg_data=pickle.load(open(linReg_filename, 'rb'))    
                for j, linReg_measure in enumerate(linReg_measure_list):
                    for k, linReg_metric_name in enumerate(linReg_analysis_metric_list):
                        key_name=linReg_measure+': '+linReg_metric_name
                        linReg_metric_array[i,j,k]=linReg_data[key_name]  
                                     


            ######### to make correlation plot ######
            sub_dictionary={}
            if isinstance(Folders_Label,list):                
                #Folder_label_for_dict='-'.join(Folders_Label)
                for idx in range(len(Folders_Label)):
                    sub_dictionary.update({Folders_Label[idx]:Folders_values[i][idx]})
            else:
                sub_dictionary.update({Folders_Label:Folders_values[i]})
                
            for key,v in analysis_data.items():          
                if key in correlation_list: 
                    value=v         
                    if 'variance_explained_reg' in key or 'R2_reg'in key:
                        value=analysis_data[key][1] ## changed to first order on 9/24, was [2] earlier 
                    elif key.split(':')[1][1:] in extrema_lists:
                        value=analysis_data[key].size
                    sub_dictionary.update({key:value})
                    
            if 'Cmatrix sparsity'in correlation_list and 'Cmatrix sparsity' not in analysis_data.items():
                if 'SM' in sub_dictionary and 'mean_c' in sub_dictionary :
                    sparsity=sub_dictionary['mean_c']*1./sub_dictionary['SM']
                    sub_dictionary.update({'Cmatrix sparsity':sparsity})
                    
                
            if linReg_is_used:        
                for key,v in linReg_data.items():
                        if key in correlation_list:
                            sub_dictionary.update({key:v}) 
            if i==0:
                corr_df=pd.DataFrame(deepcopy(sub_dictionary), index=[0]) 
            else:
                corr_df=corr_df.append(deepcopy(sub_dictionary), ignore_index=True)



if not isinstance(Folders_values[0],list): ## runs vary a single variable          
    for j, fitness_measure in enumerate(fitness_measure_list):
        for k, metric_name in enumerate(analysis_metric_list):
            fig = plt.figure(figsize=(3.5,3.5)) 
            ax = fig.add_subplot(111)
            if plot_with_error_bars:   
                ax.errorbar(Folders_values,metric_array[:,j,k],yerr=metric_array_errorbars[:,j,k]  )
            else:
                ax.plot(Folders_values,metric_array[:,j,k] ,'bo',mec='None')            
            ax.set_ylabel(metric_name)
            ax.set_xlabel(Folders_Label) 
            ax.set_title(fitness_measure) 
     
            if fold.startswith('/Users/ashish'): plt.tight_layout()
            else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
            plt.savefig(destfolder+fitness_measure+metric_name.replace('/','_')+".png",dpi=200)
    
    if linReg_is_used:
        for j, linReg_measure in enumerate(linReg_measure_list):
            for k, linReg_metric_name in enumerate(linReg_analysis_metric_list):
                fig = plt.figure(figsize=(3.5,3.5)) 
                ax = fig.add_subplot(111)
                ax.plot(Folders_values,linReg_metric_array[:,j,k] ,'bo',mec='None')       
                ax.set_ylabel(linReg_metric_name)
                ax.set_xlabel(Folders_Label) 
                ax.set_title(fitness_measure) 
         
                if fold.startswith('/Users/ashish'): plt.tight_layout()
                else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
                plt.savefig(destfolder+linReg_measure+linReg_metric_name.replace('/','_')+".png",dpi=200)        
            
    
    if structure_plots==True:
        for k, metric_name in enumerate(structure_metric_list):
            fig = plt.figure(figsize=(3.5,3.5)) 
            ax = fig.add_subplot(111)
            if plot_with_error_bars:   
                ax.errorbar(Folders_values,structure_metric_array[:,k],yerr=structure_metric_array_errorbars[:,k] )
            else:
                ax.plot(Folders_values,structure_metric_array[:,k] ,'bo',mec='None')
            ax.set_ylabel(metric_name)
            ax.set_xlabel(Folders_Label)  
            ax.set_title("landscape structure")  
            if fold.startswith('/Users/ashish'): plt.tight_layout()
            else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
            plt.savefig(destfolder+"structure_"+metric_name.replace('/','-')+".png",dpi=200)



else: ### make 2d heatmaps
    Xval=np.asarray(parameter_list)[successfully_run_folders,0]
    Yval=np.asarray(parameter_list)[successfully_run_folders,1]
    
    
    Yval=Yval/Xval
    xlabel_2D=r'Number of Species & Resources, $S,M$'
    ylabel_2D=r'sparsity, $\phi$'
    destfolder=destfolder+''
    
  
# =============================================================================
#     xlabel_2D=r'Number of Species & Resources, $S,M$'
#     ylabel_2D=r'Number of Rs consumed $M_c$'
#     destfolder=destfolder+'RConsumed-'
# =============================================================================
    
    
    
    grid_x, grid_y = np.mgrid[Xval.min():Xval.max():100j, Yval.min():Yval.max():100j]
    
 
    

    for j, fitness_measure in enumerate(fitness_measure_list):
        for k, metric_name in enumerate(analysis_metric_list):
            Zval=metric_array[successfully_run_folders,j,k]
       
            grid_nearest=scipy.interpolate.griddata(np.vstack([Xval,Yval]).T, Zval, (grid_x, grid_y), method='nearest',rescale=True) 
            grid_linear=scipy.interpolate.griddata(np.vstack([Xval,Yval]).T, Zval, (grid_x, grid_y), method='linear',rescale=True) 
            grid_cubic=scipy.interpolate.griddata(np.vstack([Xval,Yval]).T, Zval, (grid_x, grid_y), method='cubic',rescale=True) 
            
            fig = plt.figure(figsize=(6,6))            
            plot_handle=plt.imshow(grid_nearest.T, extent=(Xval.min(),Xval.max(),Yval.min(),Yval.max()), origin='lower', aspect='auto')# choose 20 contour levels, just to show how good its interpolation is
            fig.colorbar(plot_handle)
            plt.xlabel(xlabel_2D, fontweight='bold')
            plt.ylabel(ylabel_2D, fontweight='bold')
            plt.title(fitness_measure+' '+metric_name, fontweight='bold')
            if fold.startswith('/Users/ashish'): plt.tight_layout()
            else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
            plt.savefig(destfolder+fitness_measure+metric_name.replace('/','_')+"-nearest.png",dpi=200)
            
            fig = plt.figure(figsize=(6,6))            
            plot_handle=plt.imshow(grid_linear.T, extent=(Xval.min(),Xval.max(),Yval.min(),Yval.max()), origin='lower', aspect='auto')# choose 20 contour levels, just to show how good its interpolation is
            fig.colorbar(plot_handle)
            plt.xlabel(xlabel_2D, fontweight='bold')
            plt.ylabel(ylabel_2D, fontweight='bold')
            plt.title(fitness_measure+' '+metric_name, fontweight='bold')
            if fold.startswith('/Users/ashish'): plt.tight_layout()
            else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
            plt.savefig(destfolder+fitness_measure+metric_name.replace('/','_')+"-linear.png",dpi=200)
    
            fig = plt.figure(figsize=(6,6))            
            plot_handle=plt.imshow(grid_cubic.T, extent=(Xval.min(),Xval.max(),Yval.min(),Yval.max()), origin='lower', aspect='auto')# choose 20 contour levels, just to show how good its interpolation is
            fig.colorbar(plot_handle)
            plt.xlabel(xlabel_2D, fontweight='bold')
            plt.ylabel(ylabel_2D, fontweight='bold')
            if fold.startswith('/Users/ashish'): plt.tight_layout()
            else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
            plt.savefig(destfolder+fitness_measure+metric_name.replace('/','_')+"-cubic.png",dpi=200)
           
            ####### smoothed spline extrapolates outside the physical domain unfortunately; works okay if we dont use the S=2 and 4 data points ####
            spline_object_smoothed = scipy.interpolate.bisplrep(Xval,Yval, Zval)
            z_spline_smoothed= scipy.interpolate.bisplev(grid_x[:,0], grid_y[0,:], spline_object_smoothed)  
                        
            fig = plt.figure(figsize=(6,6))            
            plt.pcolor(grid_x, grid_y, z_spline_smoothed)
            plt.colorbar()
            plt.xlabel(xlabel_2D, fontweight='bold')
            plt.ylabel(ylabel_2D, fontweight='bold')
            plt.title(fitness_measure+' '+metric_name, fontweight='bold')
            if fold.startswith('/Users/ashish'): plt.tight_layout()
            else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
            plt.savefig(destfolder+fitness_measure+metric_name.replace('/','_')+"-spline.png",dpi=200)


# =============================================================================
#             ####### spline without smmoothing was not very good.####
#             spline_object_without_smoothing = scipy.interpolate.bisplrep(Xval,Yval, Zval, s=0)
#             z_spline_without_smoothing= scipy.interpolate.bisplev(grid_x[:,0], grid_y[0,:], spline_object_without_smoothing)  
#                       
#             fig = plt.figure(figsize=(6,6))            
#             plt.pcolor(grid_x, grid_y, z_spline_without_smoothing)
#             plt.colorbar()
#             plt.xlabel(xlabel_2D, fontweight='bold')
#             plt.ylabel(ylabel_2D, fontweight='bold')
#             plt.title(fitness_measure+' '+metric_name, fontweight='bold')
#             if fold.startswith('/Users/ashish'): plt.tight_layout()
#             else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
#             plt.savefig(destfolder+fitness_measure+metric_name.replace('/','_')+"-spl-no-smoothing-pcolor.png",dpi=200)
# =============================================================================
# =============================================================================
#            ######## interp 2D did not work well####
#             f_2dinterp=scipy.interpolate.interp2d(Xval,Yval, Zval, kind='linear', copy=True, bounds_error=False, fill_value=1.0)#, fill_value=nan)
#             z_2dinterp= f_2dinterp(grid_x[:,0], grid_y[0,:])
#             fig = plt.figure(figsize=(6,6))            
#             plt.pcolor(grid_x, grid_y, z_2dinterp)
#             plt.colorbar()
#             plt.xlabel(xlabel_2D, fontweight='bold')
#             plt.ylabel(ylabel_2D, fontweight='bold')
#             plt.title(fitness_measure+' '+metric_name, fontweight='bold')
#             if fold.startswith('/Users/ashish'): plt.tight_layout()
#             else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
#             plt.savefig(destfolder+fitness_measure+metric_name.replace('/','_')+"-interp2D.png",dpi=200)
# =============================================================================
# =============================================================================
#            ########  other ways to interpolate or draw heatmaps that I tried      ###############
#             fig = plt.figure(figsize=(6,6)) 
#             plot_handle=plt.tricontourf(Xval,Yval, Zval, 20) # choose 20 contour levels, just to show how good its interpolation is
#             plt.plot(Xval,Yval, 'ko')
#             fig.colorbar(plot_handle)
#             if fold.startswith('/Users/ashish'): plt.tight_layout()
#             else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
#             plt.savefig(destfolder+fitness_measure+metric_name.replace('/','_')+"-tricontour.png",dpi=200,)
#                       
#             fig = plt.figure(figsize=(6,6))            
#             plot_handle=plt.tripcolor(Xval,Yval, Zval, 5) # choose 20 contour levels, just to show how good its interpolation is
#             plt.plot(Xval,Yval, 'ko')
#             fig.colorbar(plot_handle)
#             if fold.startswith('/Users/ashish'): plt.tight_layout()
#             else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
#             plt.savefig(destfolder+fitness_measure+metric_name.replace('/','_')+"-tripcolor.png",dpi=200)         
# =============================================================================
       
            
    if structure_plots==True:
        for k, metric_name in enumerate(structure_metric_list):
            
            Zval=structure_metric_array[successfully_run_folders,k]
            grid_nearest=scipy.interpolate.griddata(np.vstack([Xval,Yval]).T, Zval, (grid_x, grid_y), method='nearest',rescale=True) 
            grid_linear=scipy.interpolate.griddata(np.vstack([Xval,Yval]).T, Zval, (grid_x, grid_y), method='linear',rescale=True) 
            grid_cubic=scipy.interpolate.griddata(np.vstack([Xval,Yval]).T, Zval, (grid_x, grid_y), method='cubic',rescale=True) 
            
            fig = plt.figure(figsize=(6,6))            
            plot_handle=plt.imshow(grid_nearest.T, extent=(Xval.min(),Xval.max(),Yval.min(),Yval.max()), origin='lower', aspect='auto')# choose 20 contour levels, just to show how good its interpolation is
            fig.colorbar(plot_handle)
            plt.xlabel(xlabel_2D, fontweight='bold')
            plt.ylabel(ylabel_2D, fontweight='bold')
            plt.title(metric_name, fontweight='bold')
            if fold.startswith('/Users/ashish'): plt.tight_layout()
            else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
            plt.savefig(destfolder+"structure_"+metric_name.replace('/','-')+"-nearest.png",dpi=200)
            
            fig = plt.figure(figsize=(6,6))            
            plot_handle=plt.imshow(grid_linear.T, extent=(Xval.min(),Xval.max(),Yval.min(),Yval.max()), origin='lower', aspect='auto')# choose 20 contour levels, just to show how good its interpolation is
            fig.colorbar(plot_handle)
            plt.xlabel(xlabel_2D, fontweight='bold')
            plt.ylabel(ylabel_2D, fontweight='bold')
            plt.title(metric_name, fontweight='bold')
            if fold.startswith('/Users/ashish'): plt.tight_layout()
            else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
            plt.savefig(destfolder+"structure_"+metric_name.replace('/','-')+"-linear.png",dpi=200)
            
            fig = plt.figure(figsize=(6,6))            
            plot_handle=plt.imshow(grid_cubic.T, extent=(Xval.min(),Xval.max(),Yval.min(),Yval.max()), origin='lower', aspect='auto')# choose 20 contour levels, just to show how good its interpolation is
            fig.colorbar(plot_handle)
            if fold.startswith('/Users/ashish'): plt.tight_layout()
            else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
            plt.savefig(destfolder+fitness_measure+metric_name.replace('/','_')+"-cubic.png",dpi=200)
           
            ####### smoothed spline object works okay if we dont use the S=2 and 4 data points, but it extrapolates outside the physical domain.####
            spline_object_smoothed = scipy.interpolate.bisplrep(Xval,Yval, Zval)
            z_spline_smoothed= scipy.interpolate.bisplev(grid_x[:,0], grid_y[0,:], spline_object_smoothed)  
                        
            fig = plt.figure(figsize=(6,6))            
            plt.pcolor(grid_x, grid_y, z_spline_smoothed)
            plt.colorbar()
            plt.xlabel(xlabel_2D, fontweight='bold')
            plt.ylabel(ylabel_2D, fontweight='bold')
            plt.title(fitness_measure+' '+metric_name, fontweight='bold')
            if fold.startswith('/Users/ashish'): plt.tight_layout()
            else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
            plt.savefig(destfolder+fitness_measure+metric_name.replace('/','_')+"-spline.png",dpi=200)

    fig = plt.figure(figsize=(6,6))            
    plt.plot(Xval,Yval, 'ko')
    plt.xlabel(xlabel_2D, fontweight='bold')
    plt.ylabel(ylabel_2D, fontweight='bold')
    plt.title(metric_name, fontweight='bold')
    if fold.startswith('/Users/ashish'): plt.tight_layout()
    else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
    plt.savefig(destfolder+'data_points.png',dpi=200)
    
    
    
    
    
corr_labels=corr_df.columns.values.tolist()
def shorten_labels(corr_labels):
    corr_labels= [label.replace('total biomass', 'Biomass') for label in corr_labels]
    corr_labels=[label.replace('R types consumed per species', '#R/species') if len(label)>12 else label for label in corr_labels]
    corr_labels=[label.replace('mean_c', '#R/species') if len(label)>12 else label for label in corr_labels]
    corr_labels= [label.replace('p_Greedy_walk_found_global_max', 'P(succesful greedy search)') for label in corr_labels]  
    corr_labels= [label.replace('p_Greediest_walk_found_global_max', 'P(succesful greediest search)') for label in corr_labels]  
    corr_labels= [label.replace('mean_relative_optimum_Greedy', 'relative optimum found') for label in corr_labels] 
    corr_labels= [label.replace('landscape structure: Abundance vector fit Subsetted R2', 'vector fit R2') for label in corr_labels]                             
    corr_labels= [label.replace('without_extinction', 'noE') for label in corr_labels]
    corr_labels= [label.replace('variance_explained_reg', 'variance exp-O(1)') for label in corr_labels]
    corr_labels= [label.replace('Resource', 'R') for label in corr_labels]
    corr_labels= [label.replace('Simpsons Diversity', 'Diversity') for label in corr_labels]
    corr_labels= [label.replace('maxima', 'Max') for label in corr_labels]
    corr_labels= [label.replace('minima', 'Min') for label in corr_labels]
    corr_labels= [label.replace('Linear', 'Lin') for label in corr_labels]
    corr_labels= [label.replace('number of', '#') for label in corr_labels]  
    corr_labels= [label.replace('coefficient of variation', 'CV') for label in corr_labels]  
    corr_labels= [label.replace('landscape structure: ', 'landcscape: ') for label in corr_labels]  
    corr_labels= [label.replace('effective ', 'eff ') for label in corr_labels]  
    corr_labels=[label.replace(': ', ':\n') if len(label)>12 else label for label in corr_labels]
    
    
    return corr_labels
corr_labels=shorten_labels(corr_labels)
corr_filenames= [c.replace('\n','') for c in corr_labels] 
corr_filenames= [c.replace(':','_') for c in corr_filenames] 
corr_filenames= [c.replace('/','_') for c in corr_filenames]                   
# calculate the correlation matrix
corr = corr_df.corr(method='spearman')

# plot the heatmap
fig = plt.figure(figsize=(10,10)) 
sns.heatmap(corr,vmin=-1., vmax=1.0, cmap='RdBu',
        xticklabels=corr_labels,
        yticklabels=corr_labels)
if fold.startswith('/Users/ashish'): plt.tight_layout()
else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
plt.savefig(destfolder+"correlation_matrix.png",dpi=200)

fig = plt.figure(figsize=(10,10)) 
sns.clustermap(corr,vmin=-1., vmax=1.0, cmap='RdBu',
        xticklabels=corr_labels,
        yticklabels=corr_labels)
if fold.startswith('/Users/ashish'): plt.tight_layout()
else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
plt.savefig(destfolder+"correlation_matrix_Clustered.png",dpi=200)

plt.close('all')
xlist,ylist=np.where(np.abs(corr.values)>0.4)

######### plotting highly correlated pairs #############
for i,j in zip(xlist,ylist) :   
    
    if i >j: ## we want to plot only one pair, and dont want to plot the diagonal either
            print (i,j)
            print(corr_labels[i],corr_labels[j])
            print(corr_df.values[i],corr_df.values[j])
            fig= plt.figure(figsize=(6,6)) 
            ax1 = fig.add_subplot(1,1,1) 
#            plt.scatter(corr_df[ corr_df.keys()[i]  ],corr_df[ corr_df.keys()[j]  ])
            plt.scatter(corr_df.values[:,i ],corr_df.values[:,j])
            plt.xlabel(corr_labels[i], fontweight='bold')
            plt.ylabel(corr_labels[j], fontweight='bold')
            plt.text(0.05, 0.9, r'$\rho =$'+'{:.2f}'.format(corr.values[i,j]),horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes)         
            plt.title('correlation', fontweight='bold')
            if fold.startswith('/Users/ashish'): plt.tight_layout()
            else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
            plt.savefig(destfolder+'Corr-'+corr_filenames[i]+'-'+corr_filenames[j]+'.png',dpi=200)


#for l, fitness_and_measure in enumerate(correlation_list):
#    fitness=fitness_and_measure[0]
#    metric=fitness_and_measure[1]
#
#    
#    if 'Linear' in fitness1:
#        j=linReg_measure_list.index(fitness1) 
#        k=linReg_analysis_metric_list.index(metric1) 
#        Xval=linReg_metric_array[:,j,k]
#    elif 'structure' in fitness1:
#        j=0
#        k=structure_metric_list.index(metric1) 
#        Xval=structure_metric_array[:,j,k]
#    else:
#        j=fitness_measure_list.index(fitness1) 
#        k=analysis_metric_list.index(metric1) 
#        Xval=metric_array[:,j,k]
#        
#    if 'Linear' in fitness2:
#        j=linReg_measure_list.index(fitness2) 
#        k=linReg_analysis_metric_list.index(metric2) 
#        Xval=linReg_metric_array[:,j,k]
#    elif 'structure' in fitness2:
#        j=0
#        k=structure_metric_list.index(metric2) 
#        Xval=structure_metric_array[:,j,k]
#    else:
#        j=fitness_measure_list.index(fitness2) 
#        k=analysis_metric_list.index(metric2) 
#        Yval=metric_array[:,j,k]     
#
#    
#    pearson_corr=pearsonr(Xval,Yval)[0]
#    metric1=metric1.replace('/','-')
#    metric2=metric2.replace('/','-')
#    fig = plt.figure(figsize=(3.5,3.5)) 
#    ax = fig.add_subplot(111)
#    ax.plot(Xval, Yval ,'bo',mec='None')
#    plt.text(0.05, 0.95,r'r=' + '{:.1e}'.format( pearson_corr ) , horizontalalignment='left', transform=ax.transAxes) 
#    ax.set_ylabel(fitness2+' '+metric2)
#    ax.set_xlabel(fitness1+' '+metric1)  
#    if fold.startswith('/Users/ashish'): plt.tight_layout()
#    else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
#    plt.savefig(destfolder+"corr_"+fitness1+'_'+metric1+'&'+fitness2+'_'+metric2+".png",dpi=200)




       
plt.close('all')
