#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:11:21 2020

@author: ashish

this aggregates analysis data into a table format for correlation analysis before averaging
"""

import numpy as np
import pickle # in python3, pickle uses cPickle automatically, there is no cPickle
import os
import glob
import sys
import pandas as pd
import time
from scipy.stats.stats import pearsonr 
from copy import deepcopy 
from collections import OrderedDict
if os.getcwd().startswith("/Users/ashish"):
    from my_CR_model_functions import rms,create_replicate_lists
    import matplotlib as mpl
else:  
    sys.path.append('/usr3/graduate/ashishge/ecology/Ecology-Stochasticity/')
    from my_CR_model_functions import rms,create_replicate_lists
    import matplotlib as mpl
    mpl.use('Agg')### can not use pyplot on cluster without agg backend


'''
The names of the folders to average over needs to be provided to the script.
Below are different sets of folders corresponding to different sets of simulations.
(see analyze_landscape.py)
'''


# =============================================================================
############# for simulation folders ##################
# =============================================================================

aggregate_data_fold=None
file_suffix=None
n_replicates=10



# =============================================================================
# base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CF_CfixedxGamma_Rsupply_constTotR_compiled/"
# parameter_name='Rsupply_constTotR'
# R0_supplied= 240.
# 
# # =============================================================================
# # base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CF_CfixedxGamma_Rsupply_constR0_compiled/"
# # parameter_name='Rsupply_constR0'
# # R0_supplied= 40.
# # =============================================================================
# 
# destfold_list=[base_fold]
# parameter_list={parameter_name:[1,2,3,4,5,6,7,8,9,10,11,12]}
# destfold_list,parameter_list, df_keyname_prefix_list=create_replicate_lists(parameter_list,destfold_list, 0, return_dfkeynames=True) ### creates a longer list enumerating all the replicates.
# 
# =============================================================================

# =============================================================================
# base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CF_CfixedxGamma0Special_meanC_compiled/"
# destfold_list=[base_fold]
# parameter_name='mean_c'
# parameter_list={parameter_name:[1,2,3,4,5,6,7,8,9,10,11,12]}
# destfold_list,parameter_list, df_keyname_prefix_list=create_replicate_lists(parameter_list,destfold_list,0, return_dfkeynames=True)### enumerates all the folders and associated dictionary keys
# =============================================================================


base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CF_CfixedxGamma01Special_l_compiled/"
destfold_list=[base_fold]
parameter_name='l'
#parameter_list={parameter_name:[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
parameter_list={parameter_name:[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
destfold_list,parameter_list, df_keyname_prefix_list=create_replicate_lists(parameter_list,destfold_list,0, return_dfkeynames=True) ### creates a longer list enumerating all the replicates.


# =============================================================================
# base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CR_cvxpt_2D_SM_CfixednumGamma_meanC_compiled/"
# parameter_list=OrderedDict([ ('SM',[2, 4, 8, 12, 16] ), ('mean_c', '1 to param1')])
# 
# #base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CR_cvxpt_2D_SM2_CfixednumGamma_meanC_compiled/"
# #parameter_list=OrderedDict([ ('SM2-',[4, 8, 12, 16 ] ), ("mean_c", "1 to param1{'step_size' : 2, 'UB_X' : 2}")])
# #
# #base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CR_cvxpt_2D_SMp5_CfixednumGamma_meanC_compiled/"
# #parameter_list=OrderedDict([ ('SMp5',[8, 12, 16 ] ), ("mean_c", "1 to param1{'step_size' : 1, 'UB_X' : 0.5}")])
# 
# destfold_list=[base_fold] 
# parameter_name='2D'
# parameter_names=list(parameter_list.keys())
# destfold_list,parameter_list, df_keyname_prefix_list=create_replicate_lists(parameter_list,destfold_list,0, return_dfkeynames=True) ### enumerates all the folders and associated dictionary keys
# 
# =============================================================================









aggregate_data_fold=base_fold[:-1]+"_AGG/"
folder_name_list=destfold_list
if aggregate_data_fold==None: aggregate_data_fold = base_fold

'''
we want the aggregate data to have column_names (keys) of the form 'param1Nameparam1Val: param2Nameparam2Val: fitnessMeasureName: MetricName'
and rows 0-N_replicates where rownumber corresponds to replicate simulation number and value corresponds to the particular value in that simulation
'''

#fitness_measures_to_aggregate=['landscape structure','total biomass','Shannon Diversity','Simpsons Diversity',
#                               'Species 0','Species 1','Species 2','Species 3','Species 4',
#                               'Species 0 coex_product','Species 2 coex_product','Species 3 coex_product',
#                               'Species 0,3 coex_product','Species 2,3 coex_product' ]

fitness_measures_partNames_to_aggregate=['landscape structure','Shannon Diversity','Simpsons Diversity',
                               'Species 0','Species 1','Species 2','Species 3','Clark_M3' ]


ruggedness_metrics_partialNames_to_aggregate=['est-R2','est-EV','est-FDC','est-rs','est-ranked_FDC',
                                              'est-EffNumStates','est-FNeutral','est-nnCorr','est-CorrL']
ruggedness_metrics_to_aggregate=['degenerate_landscape_flag',
                                'mean', 'variance','coefficient of variation',
                                'r/s','r/s_w0','PS_ratio','Power_spectrum',
                                'N_maxima','N_maxima_without_extinction','N_saddle','N_minima',
                                'total_links','f_neutral_links','f_up_links','f_down_links',
                                'SWO', 'FDC','ranked_FDC',
                                'nn_correlation','corr_length_nn',
                                'R2_reg','variance_explained_reg','R2_linear','variance_explained_linear',                                
                                'Greedy_walk_Failed_Flag',
                                'p_Greediest_walk_found_global_max','mean_relative_optimum_Greediest','var_relative_optimum_Greediest',
                                'p_Greedy_walk_found_global_max','mean_relative_optimum_Greedy', 'var_relative_optimum_Greedy',   
                                'relative_optimum_Greediest_zero_sp','Greediest_walk_steps_zero_sp',
                                'relative_optimum_Greediest_one_sp','Greediest_walk_steps_one_sp',
                                'relative_optimum_Greediest_all_sp','Greediest_walk_steps_all_sp',
                                'relative_optimum_Greediest_typical_sp','Greediest_walk_steps_typical_sp',
                                'Abundance vector fit Subsetted R2avg','Abundance vector fit Subsetted R2avg_unclipped',
                                'uniqueness', 'effective number of states','frequency of states','avg fraction species survived','S_star',                               
                                'Random_Walk_AutoCorrelation']# this is too big an array, typically length 1000 so not worth it..,'Random_Walk_Nsteps','Random_Walk_Navg':n_averages

combined_search_protocols_keys=['compared_search_protocols','n_wells','n_iterations','n_averages','biomass_conversion','biomass_conversion_factor',
                                'dilution_factor_list', 
                                'SpeciesAddition_relopt_avg','Greediest_relopt_avg']#'SpeciesAddition_relopt_list','Greediest_relopt_list',
combined_search_protocols_keys_suffixed= [k+'-fixed_conversion'for k in combined_search_protocols_keys]
combined_search_protocols_partialKeys=['Bottleneck_relopt_avg','Combined_relopt_avg']

ruggedness_metrics_to_aggregate=ruggedness_metrics_to_aggregate+combined_search_protocols_keys+combined_search_protocols_keys_suffixed
ruggedness_metrics_partialNames_to_aggregate=ruggedness_metrics_partialNames_to_aggregate+combined_search_protocols_partialKeys

##'Random_Walk_AutoCorrelation' is truncated to save space as typically it has 1000 time points


# =============================================================================
############# for experimental folders ##################
# =============================================================================
# =============================================================================
# 
# 
# base_dir="/Users/ashish/Dropbox/research/ecology/Exp Data/"
# folder_name_list=[base_dir+'Substrate0/',
#                   base_dir+'Substrate1/',
#                   base_dir+'Substrate2/',
#                   base_dir+'Substrate3/',
#                   base_dir+'Substrate4/',
#                   base_dir+'Substrate5/',
#                   base_dir+'Substrate6/']
# =============================================================================






agg_analysis_dict={}

for fold_idx, lvl1fold in enumerate(folder_name_list):
    print (lvl1fold)
    if len(glob.glob(lvl1fold+'S*.dat') )>0:
        print ("data file in lvl1, there were no replicates")
        print ( glob.glob(lvl1fold+'S*.dat') )        
        sys.exit(1)               
    for replicate_i in range(n_replicates):
        fold=lvl1fold+str(replicate_i)+'/'
        
        if file_suffix==None:## finds the first file the matches patter with glob.glob
            ctr=0                
            for file_suffix in glob.glob(fold+'S*.dat'):
                file_suffix=file_suffix.replace(fold,'').replace('.dat','')
                ctr+=1
            if ctr==0:
                print ("this folder didnt complete", fold)
                continue
            assert ctr==1,print("we can only have one file that matches the pattern in the folder!",fold)
        analysis_filename=fold+'analysis_'+file_suffix+'.dat'  
        simulation_filename=fold+file_suffix+'.dat' 
        if not os.path.isfile(analysis_filename):
            print ('this replicate did not run',replicate_i)
            continue
        analysis_data=pickle.load(open(analysis_filename, 'rb'))
        simulation_data=pickle.load(open(simulation_filename, 'rb'))
        if 'reached_SteadyState' in analysis_data:
            if analysis_data['reached_SteadyState']==False:
                continue ### this particular analysis file is not used.
            if 'popn_cutoff_OK' not in analysis_data:
                print ('popn_cutoff_OK was not in the file in ')
                print (fold)
            elif analysis_data['popn_cutoff_OK']==False:
                print  ('popn_cutoff was NOT OK in:')
                print (fold)

        if os.path.isfile(analysis_filename ):
            for key in analysis_data.keys():   
                if ': ' not in key : continue ## skip this key, since its not in the format
                fitness_measure_name,metric_name=key.split(': ')
                
#                if fitness_measure_name in fitness_measures_to_aggregate and (metric_name in ruggedness_metrics_to_aggregate or any(partial_name in metric_name for partial_name in ruggedness_metrics_partialNames_to_aggregate) ):                     
                if  any(part_name in fitness_measure_name for part_name in fitness_measures_partNames_to_aggregate) and (metric_name in ruggedness_metrics_to_aggregate or any(partial_name in metric_name for partial_name in ruggedness_metrics_partialNames_to_aggregate) ):                     
                    df_keyname=  df_keyname_prefix_list[fold_idx]+': '+key                  
                    if df_keyname not in agg_analysis_dict.keys():
                        agg_analysis_dict.update({df_keyname:[None] * n_replicates})
                    if 'Random_Walk_AutoCorrelation' not in df_keyname:
                        agg_analysis_dict[df_keyname][replicate_i]=analysis_data[key]
                    else:
                        agg_analysis_dict[df_keyname][replicate_i]=analysis_data[key][:min(20,len(analysis_data[key]))]


        if os.path.isfile(simulation_filename ): ### saves some stuff from the simulation data as well.
            for key in ['passed_params', 'exp_case', 'passed_assumptions'] :      
                df_keyname=  df_keyname_prefix_list[fold_idx]+': '+key
                if df_keyname not in agg_analysis_dict.keys():
                        agg_analysis_dict.update({df_keyname:[None] * n_replicates})
                #print (simulation_filename, simulation_data[key] )
                agg_analysis_dict[df_keyname][replicate_i]=simulation_data[key] 
            


if not os.path.exists(aggregate_data_fold): os.makedirs(aggregate_data_fold)      
pickle.dump(agg_analysis_dict, open(aggregate_data_fold+'AggAnalysis_'+file_suffix+'.dat', 'wb') )  
                 
#df_keyname_prefix

























