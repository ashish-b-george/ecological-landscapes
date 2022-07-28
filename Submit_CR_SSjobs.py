#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 18:24:28 2019

@author: ashish
submits find steady state jobs on cluster, makes a input params file for each separate job 
"""
'''
needs to be run on cluster every time. ( just python3 did not work for a while dirichlet distribution was broken, & module load python3/3.6.5 was used)
module load python3/3.6.9
module load cvxpy/1.0.25
export PACKAGEINSTALLDIR=/projectnb/qedksk/ashishge/mypython
export PYTHONPATH=$PACKAGEINSTALLDIR/lib/python3.6/site-packages:$PYTHONPATH
'''

import stat
import subprocess
import pickle # in python3, pickle uses cPickle automatically, there is no cPickle
import os
import numpy as np
from functools import partial
import sys
if os.getcwd().startswith("/Users/ashish"):
    from my_CR_model_functions import *

else:   
    sys.path.append('/usr3/graduate/ashishge/ecology/ecological-landscapes/')
    from my_CR_model_functions import *
from community_simulator import *
from community_simulator.usertools import *
from community_simulator.visualization import *
from community_simulator.analysis import *
from collections import OrderedDict 

home="/usr3/graduate/ashishge/ecology/ecological-landscapes/"
code_name="generate_CR_steady_state_data.py"
this_file="Submit_CR_SSjobs.py"

print ("current wokring directory is", os.getcwd())




'''
Crossfeeding simulations with resource supply varied in one of two ways: 
sum of all Resource supplied is kept constant or supply of each resource is kept constant as number of resources supplied is varied.
'''

# =============================================================================
# n_replicates=10
# 
# base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CF_CfixedxGamma_Rsupply_constTotR/"
# parameter_name='Rsupply_constTotR'
# R0_supplied= 240.
# 
# #base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CF_CfixedxGamma_Rsupply_constR0/"
# #parameter_name='Rsupply_constR0'
# #R0_supplied= 40.
# 
# destfold_list=[base_fold]
# parameter_list={parameter_name:[1,2,3,4,5,6,7,8,9,10,11,12]}
# destfold_list,parameter_list=create_replicate_lists(parameter_list,destfold_list, n_replicates) ### creates a longer list enumerating all the replicates.
# 
# =============================================================================


'''
crossfeeding simulations where the number of resources consumed, or niche overlap, is varied.
'''

# =============================================================================
# n_replicates=10
# base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CF_CfixedxGamma0Special_meanC/"
# destfold_list=[base_fold]
# parameter_name='mean_c'
# parameter_list={parameter_name:[1,2,3,4,5,6,7,8,9,10,11,12]}
# destfold_list,parameter_list=create_replicate_lists(parameter_list,destfold_list, n_replicates) ### creates a longer list enumerating all the replicates.
# =============================================================================

'''
crossfeeding simulations where the leakage fraction is varied
'''


n_replicates=10
base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CF_CfixedxGamma01Special_l/"
destfold_list=[base_fold]
parameter_name='l'
parameter_list={parameter_name:[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
destfold_list,parameter_list=create_replicate_lists(parameter_list,destfold_list, n_replicates) ### creates a longer list enumerating all the replicates.


'''
Consumer-resource model simulations where number of resources consumed is varied as well as number of species S and number of resources M
Like fig.2
Three scenarios are simulated S=M, S=M/2 and S=2M.
'''
# =============================================================================
# # ##################### For creating 2D phase plots. ############## 
# n_replicates=10
# 
# base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CR_cvxpt_2D_SM_CfixednumGamma_meanC/"
# parameter_list=OrderedDict([ ('SM',[4, 8, 12, 16] ), ('mean_c', '1 to param1')])
# 
# #base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CR_cvxpt_2D_SM2_CfixednumGamma_meanC/"
# #parameter_list=OrderedDict([ ('SM2-',[4, 8, 12, 16 ] ), ("mean_c", "1 to param1{'step_size' : 2, 'UB_X' : 2}")])
# 
# #base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CR_cvxpt_2D_SMp5_CfixednumGamma_meanC/"
# #parameter_list=OrderedDict([ ('SMp5',[8, 12, 16 ] ), ("mean_c", "1 to param1{'step_size' : 1, 'UB_X' : 0.5}")])
# 
# destfold_list=[base_fold] 
# parameter_name='2D'
# parameter_names=list(parameter_list.keys())
# destfold_list,parameter_list=create_replicate_lists(parameter_list,destfold_list, n_replicates) ### creates a longer list enumerating all the replicates. 
# 
# =============================================================================



##################### for testing ##############
#n_replicates=1
#base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/testing_replicates/"
#destfold_list=[base_fold]
#parameter_name='mean_c'
#parameter_list={parameter_name:[2,5,10]}
#destfold_list,parameter_list=create_replicate_lists(parameter_list,destfold_list, n_replicates) ### creates a longer list enumerating all the replicates.





##################### the actual job creation and submission loop ##############
job_number=int(np.random.rand()*10000)
print (job_number)
for parameter_idx, destfold in enumerate(destfold_list):
#    random_suffix=str(int(np.random.rand()*1000))
#    jobname = "SS" +random_suffix        
    jobname = "J" +str(job_number)
    
    
    if not os.path.exists(destfold): os.makedirs(destfold)
    
    
    #################     copying these files to the runfolder ####################
    copy_command="cp -rf "+this_file+" "+destfold+this_file
    exit_status = subprocess.call(copy_command, shell=True)
    if exit_status == 1:  # Check to make sure the job submitted
        print ("\n Job {0} failed to submit".format(copy_command) )
    copy_command="cp -rf "+code_name+" "+destfold+code_name
    exit_status = subprocess.call(copy_command, shell=True)
    if exit_status == 1:  # Check to make sure the job submitted
        print ("\n Job {0} failed to submit".format(copy_command))
        
    #################     generating parameters ####################    
    '''
    parameters have to be generated according the scenario required. So the appropriate "exp_case" needs to be chosen
    to match the simulation scenario chosen  above
    
    '''
    
    
    
    
    ########## for Cross feeding model with a single supplied resource########
    exp_case={
              'params':'w_gamma', ## 11/29    ## changed on 10/30
              'c':'binary_fixedNum_x_gamma_0_Special',  #'binary_fixedNum_x_gamma_0_1_Special', #'binary_AllEatR0_gamma', ##11/29 ## changed on 10/30## changed on 3-17
              'mean_c':6, #11/29  # for plos is 10, but it aso had S,M=200,100
              'c0':0.0, ## for plos parameters=0.01                          
              'w_params':[1.,10.,1.], 
              'resource_supply':'external',
              'D':'dirichlet',  ## changed back to dirichlet 11/13  # changed from 'dirichlet' to 'binary' because of cluster issues on 10/30/
              'D_param': .5,   ## is nleak=3 for binary, and 0.5 for dirichlet to make the matrix intermediately sparse (about a quarter of the entries are large)
              'response':'type II',
              'sampling_N0':'all',
              'l':0.5, 
              'sigma_max':20.,
              'R0_case':'single',# 'single' means R0 is a constant # for 'uniform' , max is passed so double the value if you want to fix mean
              'R0_params':240.  ## 240=20*12= meanR0*S for CR figure. to keep max energy flux similar
              } 
        
# =============================================================================
#     ########## for Cross feeding model varying numbers  supplied resource########
#     exp_case={
#               'params':'w_gamma', ## 11/29    ## changed on 10/30
#               'c':'binary_fixedNum_x_gamma',  #'binary_fixedNum_x_gamma_0_1_Special', #'binary_AllEatR0_gamma', ##11/29 ## changed on 10/30## changed on 3-17
#               'mean_c':6, #11/29  # for plos is 10, but it aso had S,M=200,100
#               'c0':0.0, ## for plos parameters=0.01                          
#               'w_params':[1.,10.,1.], 
#               'resource_supply':'external',
#               'D':'dirichlet',  ## changed back to dirichlet 11/13  # changed from 'dirichlet' to 'binary' because of cluster issues on 10/30/
#               'D_param': .5,   ## is nleak=3 for binary, and 0.5 for dirichlet to make the matrix intermediately sparse (about a quarter of the entries are large)
#               'response':'type II',
#               'sampling_N0':'all',
#               'l':0.5, 
#               'sigma_max':20.,
#               } 
# =============================================================================
      
# =============================================================================
#     ########## for Cross feeding model with  all resources supplied ########
#     exp_case={
#               'params':'w_gamma',    ## changed on 10/30
#               'c':'binary_x_gamma', ##11/29 ## changed on 10/30
#               'mean_c':2,  # for plos is 10, but it aso had S,M=200,100
#               'c0':0.0, ## for plos parameters=0.01                          
#               'w_params':[1.,10.,1.], 
#               'resource_supply':'external',
#               'D':'dirichlet',  ## changed back to dirichlet 11/13  # changed from 'dirichlet' to 'binary' because of cluster issues on 10/30/
#               'D_param': .5,   ## is nleak=3 for binary, and 0.5 for dirichlet to make the matrix intermediately sparse (about a quarter of the entries are large)
#               'response':'type II',
#               'sampling_N0':'all',
#               'l':0.5, 
#               'sigma_max':20.,
#               'R0_case':'gamma',# for a uniform distribution, max is passed so double the value if you want to fix mean
#               'R0_params':[10.,2.]  ## 240=20*12= meanR0*S for CR figure. to keep max energy flux similar
#               } 
# =============================================================================

# =============================================================================
#     ####### for CR models, every resource is supplied, and type II growth (so no cvxpy). ########
#     exp_case={
#               'params':'w_gamma',    
#               'c':'binary_diag_x_gamma', #'binary_x_gamma''binary_diag_x_gamma'
#              #'m_params':[1.,4.,0.05],## used  wolfram alpha to make sure m never is tooo small
#               'w_params':[1.,10.,1.], ## changed frm mw_params to w_params on 11/25## changed from 1,5,2, to 1,10,1 on 11/21.
#               #2nd and 3rd element gives k and theta for gamma. 
#               #1st element is the desired mean randomnumber,btained  by rescaling. This is because gamma distribution is weird for K<1
#               'mean_c':2,  # 
#               'c0':0.0, ##   small background consumption to ensure all resources are consumed.                    
#               'resource_supply':'external',
#               'response':'type II',
#               'sampling_N0':'all',
#               'l':0.,
#               'sigma_max':20.,
#               'R0_case':'gamma',# for a uniform distribution, max is passed so double the value if you want to fix mean
#               'R0_params':[10.,2.]  ##changed to 10,2 on 11/21. because R0 for gamma[4,5]varied too much, some species did not survive on their own. 
#               } 
# =============================================================================
    
# =============================================================================
#         ####### for CR models, with type I growth for cvxpy  ########
#     exp_case={
#               'params':'w_gamma',    
#               'c':'binary_fixedNum_x_gamma', #'binary_x_gamma''binary_diag_x_gamma' 'binary_fixedNum_x_gamma'
#              #'m_params':[1.,4.,0.05],## used  wolfram alpha to make sure m never is tooo small
#               'w_params':[1.,10.,1.], ## changed frm mw_params to w_params on 11/25## changed from 1,5,2, to 1,10,1 on 11/21.
#               #2nd and 3rd element gives k and theta for gamma. 
#               #1st element is the desired mean randomnumber,btained  by rescaling. This is because gamma distribution is weird for K<1
#               'mean_c':2,  # 
#               'c0':0.0, ##   small background consumption to ensure all resources are consumed.                    
#               'resource_supply':'external',
#               'response':'type I',
#               'sampling_N0':'all',
#               'l':0.,
#               'sigma_max':20.,
#               'R0_case':'gamma',# for a uniform distribution, max is passed so double the value if you want to fix mean
#               'R0_params':[10.,2.],  ##changed to 10,2 on 11/21. because R0 for gamma[4,5]varied too much, some species did not survive on their own. 
#               'tau':1.
#                 } 
# =============================================================================
    

    
    S=12### default values which can be changed below.
    M=12 
    ############      #############
    if parameter_name=='mean_c':
        exp_case[parameter_name]=parameter_list[parameter_idx]
        print (parameter_name , " was changed to ", exp_case[parameter_name])
    elif parameter_name=='SoverR_S':
        S=parameter_list[parameter_idx]
        M=S
        exp_case['mean_c']=int(S/SoverR)
        print ("S,M = ",S,M,", and mean_c is ",exp_case['mean_c'])  
    elif parameter_name=='l':
        exp_case[parameter_name]=parameter_list[parameter_idx]
        print (parameter_name , " was changed to ", exp_case[parameter_name])
        
    elif parameter_name=='Rsupply_constR0' or parameter_name=='Rsupply_constTotR' :
        exp_case.update({'R0_case': parameter_name})
        exp_case.update({'R0_params':[parameter_list[parameter_idx],R0_supplied]})
        print ('R0_params is now',exp_case['R0_params'] )
        
    elif parameter_name=='2D':
        if parameter_names[0]!='SM' and parameter_names[0]!='SM2-' and parameter_names[0]!='SMp5':
            exp_case[parameter_names[0]]=parameter_list[parameter_idx][0]
            print (parameter_names[0] , " was changed to ", exp_case[parameter_names[0]])
        elif parameter_names[0]=='SM':
            exp_case['S']=parameter_list[parameter_idx][0]
            exp_case['M']=parameter_list[parameter_idx][0]
            print (parameter_names[0] , " was changed to ", exp_case['S'],exp_case['M'])
        elif parameter_names[0]=='SM2-':
            exp_case['S']=parameter_list[parameter_idx][0]
            exp_case['M']=parameter_list[parameter_idx][0]*2
            print (parameter_names[0] , " was changed to ", exp_case['S'],exp_case['M'])            
        elif parameter_names[0]=='SMp5':
            exp_case['S']=parameter_list[parameter_idx][0]
            exp_case['M']=int(round(parameter_list[parameter_idx][0]*0.5))
            print (parameter_names[0] , " was changed to ", exp_case['S'],exp_case['M'])
            
        if parameter_names[1]!='SM' and parameter_names[1]!='SM2-' and parameter_names[1]!='SMp5':
            exp_case[parameter_names[1]]=parameter_list[parameter_idx][1]
            print (parameter_names[1] , " was changed to ", exp_case[parameter_names[1]])
        elif parameter_names[1]=='SM':
            exp_case['S']=parameter_list[parameter_idx][1]
            exp_case['M']=parameter_list[parameter_idx][1]
            print (parameter_names[1], " was changed to ", exp_case['S'],exp_case['M'])   
        elif parameter_names[1]=='SM2-':
            exp_case['S']=parameter_list[parameter_idx][1]
            exp_case['M']=parameter_list[parameter_idx][1]*2
            print (parameter_names[1], " was changed to ", exp_case['S'],exp_case['M'])   

        elif parameter_names[0]=='SMp5':
            exp_case['S']=parameter_list[parameter_idx][1]
            exp_case['M']=int(round(parameter_list[parameter_idx][1]*0.5))
            print (parameter_names[0] , " was changed to ", exp_case['S'],exp_case['M'])
                               
    else:
        print ("error, unknown parameter name",parameter_name)
    
    
    if 'S' in exp_case.keys():
        S=exp_case['S']
    if 'M' in exp_case.keys():
        M=exp_case['M']
    print ("S,M=",S,M)

 
    
    if exp_case['sampling_N0']=='all':### we run from 1,0 initial conditions. 
        # R0 is not used, it is made later.
        N0_all,_=make_all_combinations_IC(S, M, all_absent=True) 
        nexp=2**S  ##include case where all species are absent 
        njobs=8
        if parameter_name=='2D':
            njobs=2
            print ("2D is using cvxpt, so each job should be very quick")
        if njobs>nexp:
            njobs=1
    else:
        nexp=1000 # total number of experiments
        njobs=20
    
    assert nexp%njobs==0,'we want all L to be the same1'
    L=int(nexp/njobs)
    assert L>1,'we want more than a single well to prevent errors.'
    
    
    file_suffix='S_'+parameter_name+'_'+exp_case['params']
    

    
    
    
    c=make_consumption_matrix(S, M, exp_case) 
    D=make_leakage_matrix(S, M, exp_case)   
    R0_alpha=make_R0_alpha(M, exp_case)      
    params, assumptions=make_params_and_assumptions(S, M, c, D, R0_alpha, exp_case) 
    Nmax= 10*np.sum(R0_alpha) # some arbitrary number for the range over which abundance values should be generated
    
    
    #################     moving to rundir #################### 
    os.chdir(destfold) 
    st = os.stat(code_name)#### changes the file permissions just in case
    os.chmod(code_name, st.st_mode | stat.S_IEXEC) 
    
    
    #################     making initial abundances and params files for each subexperiment and submitting jobs #################### 
    for i in range(njobs):
        ##Make initial abundances of species in the wells 
        if exp_case['sampling_N0']=='uniform':
            N0 = np.random.rand(S*L).reshape(S,L)*Nmax
        elif exp_case['sampling_N0']=='all':   
            N0 = N0_all[:,i*L:(i+1)*L]                
        R0 =np.swapaxes( np.tile(R0_alpha,(L,1)),0,1 )
    
        file_suffix_i=file_suffix+'-'+str(i) # separate file suffix for each job is reqd

        input_file_name=destfold+'input'+'-'+str(i)+'.dat'
        output_file="out"+'-'+str(i)+".txt"
        
        
        input_dict={'destfold':destfold,'S':S,'M':M,'L':L,'exp_case':exp_case,'R0_alpha':R0_alpha,
                'N0':N0,'R0':R0,'file_suffix':file_suffix_i,'c':c,'D':D,'params':params,'assumptions':assumptions}
       
        pickle.dump( input_dict, open(input_file_name, 'wb') ) # in destfold.
    
        subscript_str = f"""#!/bin/bash -l
module load python3/3.6.9
module load cvxpy/1.0.25
export PACKAGEINSTALLDIR=/projectnb/qedksk/ashishge/mypython
export PYTHONPATH=$PACKAGEINSTALLDIR/lib/python3.6/site-packages:$PYTHONPATH
#$ -j y
#$ -l h_rt=12:00:00
#$ -N {jobname}
#$ -o {output_file}
python {code_name} -i {input_file_name}"""
    
        with open('sub'+'-'+str(i)+'.sh', "w") as script: ## adding suffix to sub just in case since sub is overwritten usually
            print(subscript_str, file = script)
            script.close()        
    
        ###### FOR qsubbing the job:
        qsub_command = "qsub sub"+'-'+str(i)+".sh"
      
        '''
        qsub_command = "qsub -pe omp 4 sub"+'-'+str(i)+".sh"# -pe omp 4 to make sure node has atleast 4 processors                                                
        ###### if you want to just run it on the cluster computer not as a submitted job
        qsub_command = "./sub"+'-'+str(i)+".sh"
        os.chmod('sub'+'-'+str(i)+'.sh', st.st_mode | stat.S_IEXEC)
        '''
        exit_status = subprocess.call(qsub_command, shell=True)
        if exit_status == 1:  # Check to make sure the job submitted        
            print("\n Job {0} failed to submit".format(qsub_command))   
        
   
    os.chdir( home )                                                                                                                                                                                                                                                             
    ########### submitting a dependent script that compiles the data ###########
    compile_job_name="compile"+str(job_number)
    assert base_fold+'/' in destfold, "the subfolders need to start with /"
    assert base_fold[-1]=='/', 'basefold assumed to end like this'
    compiled_dest_folder=destfold.replace(base_fold, base_fold[:-1]+'_compiled')
    analysis_plots_dest_folder=destfold.replace(base_fold, base_fold[:-1]+'_compiled_with_plots')    
    output_file=compile_job_name+".txt"
    subscript_str = f"""#!/bin/bash -l
module load python3/3.6.9
export PACKAGEINSTALLDIR=/projectnb/qedksk/ashishge/mypython
export PYTHONPATH=$PACKAGEINSTALLDIR/lib/python3.6/site-packages:$PYTHONPATH
#$ -j y
#$ -l h_rt=10:00:00
#$ -N {compile_job_name}
#$ -o {output_file}
python compile_data.py -p {destfold} -d {compiled_dest_folder} -s {file_suffix}"""
    print ("compile from", destfold, "to", compiled_dest_folder)
    with open('sub_dependent.sh', "w") as script:
            print(subscript_str, file = script)
            script.close()
    qsub_command = "qsub -hold_jid "+jobname+" sub_dependent.sh"    
    print(qsub_command)
    exit_status = subprocess.call(qsub_command, shell=True)
    if exit_status == 1:  # Check to make sure the job submitted        
        print("\n Job {0} failed to submit".format(qsub_command))
        
        
    ######### submitting a dependent script that analyzes the landscape after compilation ########
    analysis_job_name="analysis"+str(job_number)   
    output_file=analysis_job_name+".txt"
    subscript_str = f"""#!/bin/bash -l
module load python3/3.6.9
export PACKAGEINSTALLDIR=/projectnb/qedksk/ashishge/mypython
export PYTHONPATH=$PACKAGEINSTALLDIR/lib/python3.6/site-packages:$PYTHONPATH
#$ -j y
#$ -l h_rt=48:00:00
#$ -N {analysis_job_name}
#$ -o {output_file}
#$ -m n
python analyze_landscape.py -d {compiled_dest_folder} -p {analysis_plots_dest_folder}"""
 ## options '#$ -m ea'  for email on jon ending or aborting, '#$ -m n' for never email
#    subscript_name='sub_analysis3.sh'
    subscript_name='sub_analysis'+str(job_number)+'.sh'
    with open(subscript_name, "w") as script:
            print(subscript_str, file = script)
            script.close()
    qsub_command = "qsub -hold_jid "+compile_job_name+" "+subscript_name
   
    print(qsub_command)
    exit_status = subprocess.call(qsub_command, shell=True)
    if exit_status == 1:  # Check to make sure the job submitted        
        print("\n Job {0} failed to submit".format(qsub_command))
        
    job_number+=1 

print ('if submitting more than one job at a time,change the name of sub_analysis.sh and sub_dependent.sh files to ensure that submissions dont interfere!')
       
print("Done submitting jobs!")




