#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 18:11:07 2019

@author: ashish

This averages the output of analysis data over replicas, and provided error bars for measurements as well.
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
import argparse
averaged_data_fold=None



'''
The names of the folders to average over needs to be provided to the script.
Below are different sets of folders corresponding to different sets of simulations.
(see analyze_landscape.py)
'''



#base_fold="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/CR_mGamma_wGamma_CbinarydiagGamma_SM10_meanC_compiled/"
#folder_name_list=[base_fold+'/SM10_meanC1/', base_fold+'/SM10_meanC2/', 
#   base_fold+'/SM10_meanC3/', base_fold+'/SM10_meanC4/', 
#   base_fold+'/SM10_meanC5/', base_fold+'/SM10_meanC6/',
#   base_fold+'/SM10_meanC7/', base_fold+'/SM10_meanC8/']


#base_fold="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/CF_wGamma_meanC_compiled/"
#folder_name_list=[base_fold+'/SM10_meanC1/', base_fold+'/SM10_meanC2/', 
#           base_fold+'/SM10_meanC3/', base_fold+'/SM10_meanC4/', 
#           base_fold+'/SM10_meanC5/', base_fold+'/SM10_meanC6/',
#           base_fold+'/SM10_meanC7/', base_fold+'/SM10_meanC8/',
#           base_fold+'/SM10_meanC9/', base_fold+'/SM10_meanC10/'] 

#base_fold="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/CF_wGamma_l_compiled/"
#folder_name_list=[base_fold+'/SM10_l0/',
#        base_fold+'/SM10_l1/', base_fold+'/SM10_l2/', 
#           base_fold+'/SM10_l3/', base_fold+'/SM10_l4/', 
#           base_fold+'/SM10_l5/', base_fold+'/SM10_l6/',
#           base_fold+'/SM10_l7/', base_fold+'/SM10_l8/',
#           base_fold+'/SM10_l9/']



# =============================================================================
# base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CF_CfixedxGamma0Special_meanC_compiled/"
# destfold_list=[base_fold]
# parameter_name='mean_c'
# parameter_list={parameter_name:[1,2,3,4,5,6,7,8,9,10,11,12]}
# destfold_list,parameter_list, df_keyname_prefix_list=create_replicate_lists(parameter_list,destfold_list,0, return_dfkeynames=True)### enumerates all the folders and associated dictionary keys
# =============================================================================


# =============================================================================
# n_replicates=10
# base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CF_CfixedxGamma01Special_l_compiled/"
# destfold_list=[base_fold]
# parameter_name='l'
# parameter_list={parameter_name:[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
# destfold_list,parameter_list=create_replicate_lists(parameter_list,destfold_list, 0) ### creates a longer list enumerating all the replicates.
# =============================================================================





base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CR_cvxpt_2D_SM_CfixednumGamma_meanC_compiled/"
parameter_list=OrderedDict([ ('SM',[2, 4, 8, 12, 16] ), ('mean_c', '1 to param1')])

#base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CR_cvxpt_2D_SM2_CfixednumGamma_meanC_compiled/"
#parameter_list=OrderedDict([ ('SM2-',[4, 8, 12, 16 ] ), ("mean_c", "1 to param1{'step_size' : 2, 'UB_X' : 2}")])

#base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CR_cvxpt_2D_SMp5_CfixednumGamma_meanC_compiled/"
#parameter_list=OrderedDict([ ('SMp5',[8, 12, 16 ] ), ("mean_c", "1 to param1{'step_size' : 1, 'UB_X' : 0.5}")])

destfold_list=[base_fold] 
parameter_name='2D'
parameter_names=list(parameter_list.keys())
destfold_list,parameter_list=create_replicate_lists(parameter_list,destfold_list, 0) ### creates a longer list enumerating all the folders.



averaged_data_fold=base_fold[:-1]+"_AVG/"
folder_name_list=destfold_list
if averaged_data_fold==None: averaged_data_fold = base_fold

names_of_lists_to_average=['est-','variance_explained_reg','R2_reg','variance_explained_reg_w0','R2_reg_w0','Power_spectrum']
names_to_propagate_unchanged=['fitness_measure_name','fitness_measure_suffix']
names_of_success_flags=['degenerate_landscape_flag','Greedy_walk_Failed_Flag']
file_suffix=None 
for lvl1fold in folder_name_list:
    print (lvl1fold)
    if len(glob.glob(lvl1fold+'S*.dat') )>0:
        print ("data file in lvl1, there were no replicates")
        print ( glob.glob(lvl1fold+'S*.dat') )        
        sys.exit(1)
                
    total_number_of_replicates=0
    avg_analysis_data={}
    avg_analysis_data.clear()
    for fold in glob.glob(lvl1fold+'*/'):     
        if file_suffix==None:## finds the first file the matches patter with glob.glob
            ctr=0                
            for file_suffix in glob.glob(fold+'S*.dat'):
                file_suffix=file_suffix.replace(fold,'').replace('.dat','')
    #            print ("file_suffix is now ", file_suffix)
                ctr+=1
            if ctr==0:
#                print ("this folder didnt complete", fold)
                continue
            assert ctr==1,print("we can only have one file that matches the pattern in the folder!",fold)
        analysis_filename=fold+'analysis_'+file_suffix+'.dat'
        linReg_filename=fold+'linReg_'+file_suffix+'.dat'
        
        start = time.time()       
        if os.path.isfile(analysis_filename ): ### if it had reached steady state, there would be a file here
            total_number_of_replicates+=1## count updated only if file had reached steady state.
            print('total_number_of_replicates after ', fold.replace(base_fold,''),"   = ",total_number_of_replicates)
            analysis_data=pickle.load(open(analysis_filename, 'rb'))                         
            
################ checking landscape of a particular was analysed or not due to it being degenerate #################       
            fitness_measure_list=[]
            fitness_measures_that_were_degenerate=[]
            for key in analysis_data.keys():           
                fitness_measure_name=key.split(': ')[0]  
                fitness_measure_list.append(fitness_measure_name)

            
            for fitness_measure_name in list(set(fitness_measure_list)):
                fitness_replica_count_name=fitness_measure_name+': Number of replicates'
                if fitness_measure_name+': degenerate_landscape_flag' in analysis_data:
                    if fitness_replica_count_name not in avg_analysis_data: 
                           avg_analysis_data.update({fitness_replica_count_name:0})                           
                    if analysis_data[fitness_measure_name+': degenerate_landscape_flag']==False:  
                           avg_analysis_data[fitness_replica_count_name]=avg_analysis_data[fitness_replica_count_name]+1
                    else:
                        fitness_measures_that_were_degenerate.append(fitness_measure_name)
                else:
                    if fitness_replica_count_name not in avg_analysis_data: 
                           avg_analysis_data.update({fitness_replica_count_name:1}) 
                    else:
                           avg_analysis_data[fitness_replica_count_name]=avg_analysis_data[fitness_replica_count_name]+1


            print ("\n degenerate fitness measures in fold=", fold.replace(base_fold,''),"  were:\n",fitness_measures_that_were_degenerate)

            for key, value in analysis_data.items():
                
                fitness_measure_name=key.split(': ')[0]
                if fitness_measure_name in fitness_measures_that_were_degenerate:
                    continue ##### this fitness measure was a degenerate landscape and so skip this one!              
                
                #### some values are just saved, some lists are averaged  and any floats or integers are averaged as well.                                
                if any(partial_name in key for partial_name in names_to_propagate_unchanged): 
                    if key not in avg_analysis_data:   
                        avg_analysis_data.update({key:value})
                elif any(partial_name in key for partial_name in names_of_success_flags): ## make a list of True False values for these flags       
                    if key in avg_analysis_data:                        
                        avg_analysis_data[key].append(value)                                               
                    else:
                        avg_analysis_data.update({key:[value]})                       
                    if 'Greedy_walk_Failed_Flag' in key:                        
                        if value ==True:
                            print (fitness_measure_name)
                            print (analysis_data[fitness_measure_name+': mean'])
                            print (analysis_data[fitness_measure_name+': variance'])
                            print (analysis_data[fitness_measure_name+': N_maxima'])
                            print ('Greedy_walk_Failed_Flag was true, greedy walk failed even when landscape was non-degenerate!')
#                            sys.exit(1)
                            
                elif isinstance(value,(int,float,np.integer,np.float)):                                                              
                    if key in avg_analysis_data:     
                        
                        avg_analysis_data[key]=avg_analysis_data[key]+value
                        avg_analysis_data[key+'-Variance']=avg_analysis_data[key+'-Variance']+value**2
                    else:
                        avg_analysis_data.update({key:value})
                        avg_analysis_data.update({key+'-Variance':value**2})


                ### there are actually some of these partial names as floats and ints as well, (eg. maxima and n_maxima), but they are picked up in the above elif
                elif any(partial_name in key for partial_name in names_of_lists_to_average): 
                    if key in avg_analysis_data: 
                        try:
                            avg_analysis_data[key]=np.asarray(avg_analysis_data[key])+np.array(value)
                            avg_analysis_data[key+'-Variance']=np.asarray(avg_analysis_data[key+'-Variance'])+np.array(value)**2
                        except:
                            print ('unexpected lists are not averaged--like in estimated ruggedness')
                            print ('key was:', key)
                            print ('shapes were:',np.shape(np.asarray(avg_analysis_data[key])),np.shape(np.asarray(value)) )
                        
                    else:
                        avg_analysis_data.update({key:np.array(value)})
                        avg_analysis_data.update({key+'-Variance':np.array(value)**2})
                        

                    
        end = time.time()
        print("time to read and add data was: ",end-start)                              
    ### now we we need to divide appropriately to get mean and variance
    avg_analysis_data.update({'Number of replicates':total_number_of_replicates})     
    
    list_of_keys_to_avg=list(avg_analysis_data.keys() ) ## might be more than the current replicate since it might have failed for some fitness measures
    list_of_keys_to_avg[:]=[x for x in list_of_keys_to_avg if "-Variance" not in x] ## dont want to average this key either
    list_of_keys_to_avg[:]=[x for x in list_of_keys_to_avg if "Number of replicates" not in x] ## we don't want to average the number of replicate key!

    
    for key in list_of_keys_to_avg: ## value of boolean list is evaluated for some reason?
        fitness_measure_name=key.split(': ')[0]
        fitness_replica_count_name=fitness_measure_name+': Number of replicates'       
        replica_count=avg_analysis_data[fitness_replica_count_name]
        value=avg_analysis_data[key]

        if replica_count>0: ## if all runs didnt fail/ end,
            if isinstance(value,(int,float,np.integer,np.float)) and not isinstance(value,bool): ## bool is a subclass of init in python :/

                avg_analysis_data[key]=avg_analysis_data[key]*1./replica_count

                if key+'-Variance' in avg_analysis_data:

                    avg_analysis_data[key+'-Variance']=avg_analysis_data[key+'-Variance']*1./replica_count - avg_analysis_data[key]**2
                    avg_analysis_data[key+'-Variance']=round(avg_analysis_data[key+'-Variance'],8) #  to remove precision errors giving negative values.   
                    if avg_analysis_data[key+'-Variance']<0:                        
                        if np.any(np.isnan(avg_analysis_data[key])) or np.any(avg_analysis_data[key]>1e8) or np.any(avg_analysis_data[key]<-1e8): 
                            print ("\n negative variance due to NaNs: \n")
                            print (key)
                            print (avg_analysis_data[key], avg_analysis_data[key+'-Variance'])
                            ## if the array was interger and assigned NaN values, np.isnan will not catch it. 
                            ##Also np.abs will not work so comparison needs to be done separately.
                        else:
                            print ("negative variance")
                            print (key)
                            print (avg_analysis_data[key], avg_analysis_data[key+'-Variance'])
                            print (replica_count)    
                            sys.exit(1)
                
            elif any(partial_name in key for partial_name in names_of_lists_to_average): 
                avg_analysis_data[key]=avg_analysis_data[key]*1./replica_count
                if key+'-Variance' in avg_analysis_data:
                    avg_analysis_data[key+'-Variance']=avg_analysis_data[key+'-Variance']*1./replica_count - avg_analysis_data[key]**2
                    avg_analysis_data[key+'-Variance']=np.around(avg_analysis_data[key+'-Variance'], decimals=8)#  to remove precision errors giving negative values.
                    if np.any(avg_analysis_data[key+'-Variance']<0):
                        if np.any(np.isnan(avg_analysis_data[key])) or np.any(avg_analysis_data[key]>1e8) or np.any(avg_analysis_data[key]<-1e8): 
                            print ("\n negative variance due to NaNs: \n")
                            print (key)
                            print (avg_analysis_data[key], avg_analysis_data[key+'-Variance'])
                            ## if the array was interger and assigned NaN values, np.isnan will not catch it. 
                            ##Also np.abs will not work so comparison needs to be done separately.
                        else:
                            print ("negative variance for list")
                            print (key)
                            print (avg_analysis_data[key], avg_analysis_data[key+'-Variance'])
                            print (np.any(np.isnan(avg_analysis_data[key])), np.any(avg_analysis_data[key]>1e8), np.any(avg_analysis_data[key]<-1e8))
                            print (replica_count)                   
                            sys.exit(1)    
    
    fold_to_put_data_in= lvl1fold.replace(base_fold, averaged_data_fold)
    if not os.path.exists(fold_to_put_data_in): os.makedirs(fold_to_put_data_in)      
    pickle.dump(avg_analysis_data, open(fold_to_put_data_in+'AvgAnalysis_'+file_suffix+'.dat', 'wb') )  
    print ("Note: this averaging script does not check if steady state was reached")
    print (avg_analysis_data.keys())    
    









         
            
            
            