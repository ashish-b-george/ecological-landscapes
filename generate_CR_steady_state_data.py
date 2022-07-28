#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:41:40 2019

@author: ashish

Runs community simulator package to find steady states for various initial nonzero abundances of species
via ODE integration
"""
import pickle # in python3, pickle uses cPickle automatically, there is no cPickle
import os
import numpy as np
import pandas as pd
import sys
import time
import warnings
from community_simulator import *
from community_simulator.usertools import *
from community_simulator.visualization import *
from community_simulator.analysis import *
if os.getcwd().startswith("/Users/ashish"):    
    from my_CR_model_functions import *
else:    
    sys.path.append('/usr3/graduate/ashishge/ecology/Ecology-Stochasticity/')
    from my_CR_model_functions import *    
import argparse
from copy import deepcopy
import scipy


   
parser = argparse.ArgumentParser()
parser.add_argument("-i", help="input params file", default=None)
args = parser.parse_args()
input_file=args.i
use_cutoff=True   
abundance_cutoff=1e-3  ## changed from 1e-6 to 1e-3 on 12/11
propagation_time_step=10 ## changed from 100 on 9/27/2019
save_time_series=False

if input_file==None: # use the parameters below
#    input_data = pickle.load(open('/Users/ashish/Downloads/S14_meanc1_0/input-1.dat', 'rb')) 
#    destfold='/Users/ashish/Downloads/S14_meanc1_0/'
#    input_data = pickle.load(open('/Users/ashish/Downloads/S14_meanc9_1/input-meanc9.dat', 'rb')) 
#    destfold='/Users/ashish/Downloads/S14_meanc9_1/'
    input_data = pickle.load(open('/Users/ashish/Downloads/CF_CfixedxGamma01Special_l/l0/1/input-0.dat', 'rb')) 
    destfold='/Users/ashish/Downloads/CF_CfixedxGamma01Special_l/l0/testing/'
    
        
#    print ("error, please pass parameters")
#    sys.exit(1)
    
    
# =============================================================================
#         destfold="/Users/ashish/Downloads/ecology_simulation_data/Steady_state_counting/figure1_CR_crossfeeding_time_series/"
# #        S=8
# #        M=8
# #        L=10
# #        exp_case={'c':'binary',
# #                  'mean_c':2,
# #                  'params':'test1',          
# #                  'resource_supply':'external',
# #                  'D': 'dirichlet',
# #                  'l':0.8,
# #                  'response':'type II'
# #                  }               
#         ##Make initial abundances of species in the wells       
#         #N0 = np.random.rand(S*L).reshape(S,L)*50 
#         ####### set resources
# #        R0_alpha=make_R0_alpha(M, exp_case)                     
# #        R0 =np.swapaxes( np.tile(R0_alpha,(L,1)),0,1 )*0.0 ## start off resources at zero concentration for the figure1       
#         #### set parameters for a particular figure
#         S, M, L, N0,R0_alpha, R0, exp_case=set_figure_params('figure1_CR_crossfeeding_time_series')   # figure1_CR_crossfeeding_time_series , figure1_CR_time_series ,etc.    
#         file_suffix='S'+str(S)+'M'+str(M)+'_'+exp_case['params']
#         if '_Cnoise' in exp_case['c']:
#             file_suffix=file_suffix+'_Cnoise'
#         if exp_case['l']>0:
#             file_suffix=file_suffix+'_l'+str(exp_case['l'])
#         if exp_case['response']!='type I':
#             file_suffix=file_suffix+'_'+str(exp_case['response']).replace(' ','')        
#         c=make_consumption_matrix(S, M, exp_case) 
#         D=make_leakage_matrix(S, M, exp_case)        
#         params, assumptions=make_params_and_assumptions(S, M, c, D, R0_alpha, exp_case)   
# =============================================================================
        
else: # input_file was provided
    input_data = pickle.load(open(input_file, 'rb')) 
    destfold=input_data['destfold']
S=input_data['S']
M=input_data['M']
L=input_data['L']
exp_case=input_data['exp_case']
R0_alpha=input_data['R0_alpha']    
N0 =input_data['N0']
R0 =input_data['R0']
file_suffix=input_data['file_suffix']
c=input_data['c']
D =input_data['D']
params=input_data['params']
assumptions =input_data['assumptions']
if 'use_cutoff' in exp_case:use_cutoff=exp_case['use_cutoff']
##True by default
if 'abundance_cutoff' in exp_case:abundance_cutoff=exp_case['abundance_cutoff']
#default value is abundance_cutoff= 1e-6

if not os.path.exists(destfold): os.makedirs(destfold)

# =============================================================================
# print(exp_case) 
# print (assumptions)        
# =============================================================================
cross_feeding=True if params['l']>0 else False
nonlinearResponse=True if assumptions['response']!='type I'else False 



removed_a_well_from_init_state=False
def remove_no_species_IC(N0, R0, L): ## if an IC had no species, we don't run dynamics or cvxpt on it since we know the answer  
    if np.any(np.all(N0 == 0, axis=0)):
        assert len(np.where(np.all(N0 == 0, axis=0)==True)[0]==1), 'we dont expect more than one zero IC'
        idx_0IC=int(np.where(np.all(N0 == 0, axis=0)==True)[0])
        print ("there was a zero IC we removed at ", idx_0IC)
        R0=np.delete(R0,idx_0IC, axis=1 )
        N0=np.delete(N0,idx_0IC, axis=1 )
        L-=1         
        return N0, R0, L, True
    else:
        return N0, R0, L, False
        

if not(cross_feeding or nonlinearResponse or save_time_series):
    N0, R0, L, removed_a_well_from_init_state=  remove_no_species_IC(N0, R0, L)        
    print ('since we are using cvxpt, we should remove 0 IC if it exists')
 
    
    
    
    
init_state = [N0,R0]    

#Construct dynamics functions
def dNdt(N,R,params):
    return MakeConsumerDynamics(assumptions)(N,R,params)
def dRdt(N,R,params):
    return MakeResourceDynamics(assumptions)(N,R,params)
dynamics_range = [dNdt,dRdt]



Simulation_start_time=time.time()

#MyPlate = Community(init_state,dynamics_range,params,parallel=True)
#since cluster sometimes doesnt have as many processors available as pool() thinks.
MyPlate = Community(init_state,dynamics_range,params,parallel=False)
initial_abundance=MyPlate.N.values
initial_resources=MyPlate.R.values


    

reached_SteadyState=False
ODEint_error=False
currentTime=1e-5 ## to have logspaced time intervals, I add a pseudo count.
if cross_feeding or nonlinearResponse or save_time_series:    
    #need to pass a list of parameters to use the map function
    if type(MyPlate.params) is not list:
        params_list = [MyPlate.params for k in range(len(MyPlate.N.T))]
    else:
        params_list = MyPlate.params
    ctr=0    
    while (reached_SteadyState==False and ODEint_error==False) :   
        if save_time_series:
#            print ("ctr is ", ctr)
            if ctr==0:
                t_traj_arr, Ntraj_arr, Rtraj_arr=Propagate_and_return_trajectories(MyPlate,T0=currentTime,Duration=propagation_time_step, nsteps=1000,
                                                                               log_time=True,show_plots=True)
            else:
                t_traj_arr, Ntraj_arr, Rtraj_arr=Propagate_and_return_trajectories(MyPlate,T0=currentTime,Duration=propagation_time_step, nsteps=8,
                                                                               log_time=False,show_plots=False)
            if ctr==0:
                N_time_arr=deepcopy(Ntraj_arr) #### Ntime arr is of the shape [nwells, timepoints, species ]
                R_time_arr=deepcopy(Rtraj_arr)
                time_arr=deepcopy(t_traj_arr)
            else:
                N_time_arr=np.append(N_time_arr, Ntraj_arr, axis=1)
                R_time_arr=np.append(R_time_arr, Rtraj_arr, axis=1)
                time_arr=np.append(time_arr, t_traj_arr, axis=1)           
        else:
            MyPlate.Propagate(propagation_time_step) 
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    MyPlate.Propagate(propagation_time_step) 
                except  scipy.integrate.odepack.ODEintWarning:
                    print ("ODEint error:", sys.exc_info()[0])
                    print ('So we need to say we did not reach steady state and get out')
                    ODEint_error=True
            
#                e = sys.exc_info()[0]
#                print( "<p>Error: %s</p>" % e )
            
# =============================================================================
#            ### old way of calculating dlogNdt
#         dNdt_f = np.asarray(list(map(MyPlate.dNdt,MyPlate.N.T.values,MyPlate.R.T.values,params_list)))
#         idx_nonzero=np.nonzero(MyPlate.N.T.values)
#         dlogNdt_f=dNdt_f[idx_nonzero]/MyPlate.N.T.values[idx_nonzero]
# =============================================================================
        
        ### new way of calculating dlogNdt has less roundoff error. 12/11/19
        
        N_survived=deepcopy(MyPlate.N.T.values)
        
        
        if use_cutoff:
            N_survived[N_survived<=abundance_cutoff]=0.0
            N_survived[N_survived>abundance_cutoff]=1.
            
        else:
            N_survived[N_survived>0]=1
            N_survived[N_survived<=0.]=0.0
        dlogNdt_f=np.asarray(list(map(MyPlate.dNdt,N_survived,MyPlate.R.T.values,params_list)))
        
        
        if use_cutoff: ### removes small values to remove errors from machine precision, and avoid problems with stiffness
            N_cutoff=MyPlate.N.values 
            N_cutoff[N_cutoff< abundance_cutoff]=0.0
            R_values=MyPlate.R.values 
            if exp_case['resource_supply']=='self-renewing': # in a chemostat, no cutoff is imposed.
                R_values[R_values< abundance_cutoff]=0.0            
            MyPlate.N = pd.DataFrame(N_cutoff, index = MyPlate.N.index, columns = MyPlate.N.keys())    
            MyPlate.R = pd.DataFrame(R_values, index = MyPlate.R.index, columns = MyPlate.R.keys())            
        ctr+=1
        currentTime+=propagation_time_step
#       if np.all( np.asarray( list(map(rms,dNdt_f)) )<1e-4 ): ## before July 1st 2019
        if np.all( np.asarray( list(map(rms,dlogNdt_f)) )<1e-3 ):## changed cutoff to 1e-3 on 12/11 ## change to log and change in cutoff to 1e-5 made on July1st2019
            print('steady state has been reached')
            reached_SteadyState=True 
        N_final=MyPlate.N.values
        R_final=MyPlate.R.values
        
        if ctr>500:#ctr>50000: ## changed on 11/20 and 11/21
            print ('too long.')
            break
   
else:    #Calculate steady state directly using cvxopt     
#    assert destfold.startswith("/Users/ashish/"), 'cvxopt not installed on cluster!'
    MyPlate.SteadyState(supply=exp_case['resource_supply'],verbose=True)
    Verification_start_time=time.time()
# =============================================================================
#     ######################### old way of verifying the steady state ############
#     
#     if type(MyPlate.params) is not list:
#         params_list = [MyPlate.params for k in range(len(MyPlate.N.T))]
#     else:
#         params_list = MyPlate.params
#                
#     dNdt_f = np.asarray(list(map(MyPlate.dNdt,MyPlate.N.T.values,MyPlate.R.T.values,params_list)))
#     idx_nonzero=np.nonzero(MyPlate.N.T.values)
#     dlogNdt_f=dNdt_f[idx_nonzero]/MyPlate.N.T.values[idx_nonzero]
#     if np.all( np.asarray( list(map(rms,dlogNdt_f)) )<1e-5 ): ## change to log and change in cutoff made on July1st2019
#         print('steady state has been reached')
#         reached_SteadyState=True   
#     else:
#         reached_SteadyState=False      
#     print ( "max rate of change at SS was:", np.max( list(map(rms,dlogNdt_f)) ) )
#     
#     N_cutoff=MyPlate.N.T.values
#     N_cutoff[MyPlate.N.T.values< 1e-3]=0.0
#     idx_nonzero=np.nonzero(N_cutoff)
#     dlogNdt_f=dNdt_f[idx_nonzero]/MyPlate.N.T.values[idx_nonzero]
#     if np.all( np.asarray( list(map(rms,dlogNdt_f)) )<1e-5 ): ## change to log and change in cutoff made on July1st2019
#         print('steady state has been reached')
#         reached_SteadyState=True   
#     else:
#         reached_SteadyState=False      
#     print ( "max rate of change at SS with cutoff ", np.max( list(map(rms,dlogNdt_f)) ) )
# =============================================================================
    ###### new way of checkign steady state started on 12/11/19 ########
    check_SS=validate_simulation(MyPlate,N0)
    print (check_SS)
    reached_SteadyState=True
    if check_SS['Failures']!=0 or check_SS['Invasions']!=0:
        reached_SteadyState=False    
    if check_SS['Mean Accuracy'] < 1e-3:
        print ('mean accuracy was very low' )
        reached_SteadyState=False
    Verification_end_time=time.time()
    print ("verification ran for ",(Verification_end_time-Verification_start_time)/60. ," mins"   )
  
    N_final=MyPlate.N.values
    R_final=MyPlate.R.values
    
    ######################### appending values if an empty well was removed ############
    if removed_a_well_from_init_state:
        N_final=np.append(N_final,np.zeros(S)[:,np.newaxis],axis=1)
        R_final=np.append(R_final, R0_alpha[:,np.newaxis],axis=1)
        initial_resources=np.append(initial_resources, R0_alpha[:,np.newaxis],axis=1)
        initial_abundance=np.append(initial_abundance, np.zeros(S)[:,np.newaxis],axis=1)
        print ('we added back the empty well')

print ('reached_SteadyState',reached_SteadyState, 'ODEint_error',ODEint_error)
to_write={'initial_abundance':initial_abundance, 'steady_state': N_final, 'passed_params':params, 'passed_assumptions':assumptions, 'exp_case':exp_case,
          'initial_resources':initial_resources, 'steady_state_resources':R_final,'reached_SteadyState':reached_SteadyState, 'ODEint_error':ODEint_error}
           
if save_time_series: ##Ntime arr is of the shape [nwells, timepoints, species ]
    to_write.update({'abundance_time_series':N_time_arr,'resource_time_series':R_time_arr,'timepoints_time_series':time_arr})
    print(np.shape(N_time_arr))
pickle.dump( to_write, open(destfold+file_suffix+'.dat', 'wb') )   






######### comparing results to simply propagating
#MyPlate2 = Community(init_state,dynamics_range,params,parallel=False)
#MyPlate2.Propagate(4*propagation_time_step) 

print (validate_simulation(MyPlate,N0))

Simulation_end_time=time.time()
print ("simulation ran for ",(Simulation_end_time-Simulation_start_time)/60. ," mins"   )







