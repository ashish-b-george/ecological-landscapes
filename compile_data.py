#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:48:28 2019

@author: ashish

reads all the data files from various sub-jobs and makes it one data file
module load python3/3.6.9
export PACKAGEINSTALLDIR=/projectnb/qedksk/ashishge/mypython
export PYTHONPATH=$PACKAGEINSTALLDIR/lib/python3.6/site-packages:$PYTHONPATH
"""

import stat
import subprocess
import pickle # in python3, pickle uses cPickle automatically, there is no cPickle
import os
from os.path import isfile, join
import argparse
import numpy as np
from community_simulator import *
from community_simulator.usertools import *
from community_simulator.visualization import *
from community_simulator.analysis import *
if os.getcwd().startswith("/Users/ashish"):
    from my_CR_model_functions import *
else:
    import sys
    sys.path.append('/usr3/graduate/ashishge/ecology/Ecology-Stochasticity/')
    from my_CR_model_functions import *
    

parser = argparse.ArgumentParser()
parser.add_argument("-p", help="data path", default=None)
parser.add_argument("-d", help="destfolder", default=None)
parser.add_argument("-s", help="output file suffix", default=None)
args = parser.parse_args()
 
    
data_path="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CR_mGamma_wGamma_CbinarydiagGamma_SM10_meanC/SM10_meanC1/0/" if args.p ==None else args.p
output_suffix='S10M10_mw_gamma' if args.s ==None else args.s
destfold=data_path[:-1]+"_compiled/" if args.d ==None else args.d
#destfold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CR_mGamma_wGamma_CbinarydiagGamma_SM10_meanC_compiled/SM10_meanC1/0/"


if not os.path.exists(destfold): os.makedirs(destfold)

print ("input folder is", data_path)
print ("output folder is", destfold)
onlyfiles = [f for f in os.listdir(data_path) if isfile(join(data_path, f))]

input_files = [f for f in onlyfiles if "input" in f]


for idx,ipt in enumerate(input_files): 
    input_data = pickle.load(open(data_path+ipt, 'rb')) 
    opt=ipt.replace('input',output_suffix)  
    if os.path.isfile(data_path+opt):
        output_data = pickle.load(open(data_path+opt, 'rb')) 
    else:
        #exit, this data is incomplete, so discard it
        print ("the file ",data_path+opt,"did not exist, so not compiling")
        sys.exit(1)
    
    if idx==0:
        input_dict=input_data
        N0=input_data['N0']
        R0=input_data['R0']
        L=len(N0[0])
      
        output_dict=output_data
        N0_out=output_data['initial_abundance']
        R0_out=output_data['initial_resources']
        NSS_out=output_data['steady_state']
        RSS_out=output_data['steady_state_resources']
        reached_SteadyState=output_data['reached_SteadyState']
        run_suffix=ipt.replace('input-','')
        print (np.shape(N0),np.shape(N0_out),np.shape(NSS_out))
        
        params=output_data['passed_params']
        assumptions=output_data['passed_assumptions']
        exp_case=output_data['exp_case']
        
    else:
        
        N0=np.append(N0,input_data['N0'],axis=1)
        R0=np.append(R0,input_data['R0'],axis=1)
        
        N0_out=np.append(N0_out,output_data['initial_abundance'],axis=1)
        R0_out=np.append(R0_out,output_data['initial_resources'],axis=1)       
        NSS_out=np.append(NSS_out,output_data['steady_state'],axis=1)
        RSS_out=np.append(RSS_out,output_data['steady_state_resources'],axis=1)       
        reached_SteadyState=np.append(reached_SteadyState,output_data['reached_SteadyState'])
       
        run_suffix=np.append(run_suffix, ipt.replace('input-','') )
        
print (np.shape(N0),np.shape(N0_out),np.shape(NSS_out),np.shape(reached_SteadyState))


if np.any(N0!=N0_out):
    print ("not equal ICs?", (N0-N0_out)[np.where(N0!=N0_out)] )

        
        
        
        
        
to_write={'initial_abundance':N0_out, 'steady_state':NSS_out, 'L':L, 'n_jobs':len(input_files),'passed_params':params, 'passed_assumptions':assumptions, 'exp_case':exp_case,
          'initial_resources':R0_out, 'steady_state_resources':RSS_out,'reached_SteadyState':reached_SteadyState}    
pickle.dump( to_write, open(destfold+output_suffix+'.dat', 'wb') )        
        
        
        
        
        
    