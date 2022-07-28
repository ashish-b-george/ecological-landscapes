#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:01:16 2019
@author: ashish

Runs community simulator package to find steady states for various initial species compositions
in scenarios where cvxopt is used, so can be run only locally
Currently runs for all configurations, needs to be extended to run for a well-chosen subset of possible configurations
"""
import pickle # in python3, pickle uses cPickle automatically, there is no cPickle
import os
import numpy as np
from IPython.display import Image
from community_simulator import *
from community_simulator.usertools import *
from community_simulator.visualization import *
import matplotlib.pyplot as plt
import seaborn as sns
colors = sns.color_palette()

from my_CR_model_functions import *

destfold="/Users/ashish/Downloads/ecology_simulation_data/test/"
Sval=5
Mval=5
exp_case={'c':'halfresource', 'params':'equal', 'resource_supply':'external'}
file_suffix='S'+str(Sval)+'M'+str(Mval)+'_'+exp_case['params']
if '_Cnoise' in exp_case['c']:
    file_suffix=file_suffix+'_Cnoise'
       
def generate_all_steady_state_data(S, M, destfold, file_suffix, exp_case):    
    if not os.path.exists(destfold): os.makedirs(destfold)
    
    c= make_consumption_matrix(S, M, exp_case) 
    # for testing:  c=np.array([[1., 0.],[.1, 1.]] ) - self-renewing and external resource supply give very different answers!  
    print(exp_case)         
    print(c)

    D=np.eye(M) ## no cross-feeding-right? l is always 0 currently.
    R0_alpha=make_R0_alpha(M, exp_case)
    params, assumptions=make_params_and_assumptions(S, M, c, D, R0_alpha, exp_case)    
    print (assumptions)
    
    #Construct dynamics functions
    def dNdt(N,R,params):
        return MakeConsumerDynamics(assumptions)(N,R,params)
    def dRdt(N,R,params):
        return MakeResourceDynamics(assumptions)(N,R,params)
    dynamics_range = [dNdt,dRdt]
       
    ##Make initial abundances of species in the wells
    init_state=make_all_combinations_IC(S, M)
    
    #Initialize plate
    CR_object = Community(init_state,dynamics_range,params,parallel=True)
    initial_abundance=CR_object.N.values
    #Calculate steady state
    CR_object.SteadyState(supply=exp_case['resource_supply'])
    print ("steady state is:-")
    print (CR_object.N)
    
    to_write={'initial_abundance':initial_abundance, 'steady_state':CR_object.N.values, 'passed_params':params, 'passed_assumptions':assumptions, 'exp_case':exp_case}    
    pickle.dump( to_write, open(destfold+'data'+file_suffix+'.dat', 'wb') )      


generate_all_steady_state_data(Sval, Mval, destfold, file_suffix, exp_case)













