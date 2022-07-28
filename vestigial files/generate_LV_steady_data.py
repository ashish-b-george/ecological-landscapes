#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:43:49 2019

@author: ashish
Find steady states for various initial nonzero abundances of species of Lotkka Volterra model
via ODE integration
"""
import pickle # in python3, pickle uses cPickle automatically, there is no cPickle
import os
import numpy as np
import pandas as pd
from scipy import integrate
from copy import deepcopy
import matplotlib.pyplot as plt
#from community_simulator import *
#from community_simulator.usertools import *
#from community_simulator.visualization import *
#from community_simulator.analysis import *
#if os.getcwd().startswith("/Users/ashish"):
#    from my_CR_model_functions import *
#else:
#    import sys
#    sys.path.append('/usr3/graduate/ashishge/ecology/ecological-landscapes/')
#    from my_CR_model_functions import *
   

use_cutoff=True   
abundance_cutoff=1e-6 
propagation_time_step=10


def rms(y):
    return np.sqrt(np.mean(y**2))

def set_LV_params(S,exp_case):
   LV_params={}
   if exp_case['alpha']=='gaussian' and  exp_case['gamma']==0:
       alpha=np.random.normal(exp_case['mean_alpha'], exp_case['std_alpha'], S*S).reshape(S,S)       
       ### alpha is the LV interaction matrix
       ### alpha[i,j] describes the impact of species j on the abundance of species i
       ### +ve alpha implies competition, -ve alpha is mutualism       
       mu_bunin=S*np.mean(alpha)
       sigma_bunin=np.sqrt(S* np.var(alpha))
       print ('sigma_bunin=',sigma_bunin)
       assert sigma_bunin <1.,'too large a sigma?' ##Guy's paper says sqrt(2) is when you get multiple stable attractors for gamma=0.
       assert mu_bunin >-1.,'too much mutualism means blowup'
       LV_params.update({'alpha':alpha})
   if exp_case['r']=='one':
       LV_params.update({'r':1.})
   if exp_case['K']=='one':
       LV_params.update({'K':1.})
   return LV_params
#Construct dynamics functions
def dNdt_LV(N, dummy_t, params): # t is not used.
    return (params['r']*N*(params['K']-N)/params['K'] -N * np.dot(params['alpha'],N))


def integrate_ODE_LV(N,time_steps,params):
#    print (np.shape(N), np.shape(params))
    out = integrate.odeint(dNdt_LV,N.T,time_steps,args=(params,),mxstep=10000,atol=1e-4)
    #out = integrate.odeint(dNdt_LV, N, t, args=(params['r'],params['K'],params['alpha']), mxstep=10000,atol=1e-4)[-1]
    return out      

def Propagate_and_return_trajectories_LV(N, n_wells, params_list, T0=0,Duration=1, nsteps=10, log_time=True, show_plots=False ):
        """
        Propagate the state variables forward in time according to dNdt        
        T = time interval for propagation        
        """
        T=Duration
        if log_time==True:
            assert T0!=0, 'we cant have T0=0 for logspaced time, add a pseudo count atleast.'
            t_arr=10**(np.linspace(np.log10(T0),np.log10(T0+T),nsteps))
        else:
            t_arr=np.linspace(T0,T0+T,nsteps)
        
        time_step_list=[t_arr for i in range(n_wells)]
   
        integ_output=list(map(integrate_ODE_LV, N.T, time_step_list, params_list ))
                     
        assert len(integ_output)==L and len(integ_output)>1, 'integration for a subset of wells not supported.'
        Ntraj_arr=np.array(integ_output)

        Nf=Ntraj_arr[:,-1,:].squeeze().T
        
        if show_plots:
            for w in range(n_wells):
                fig, ax = plt.subplots(1)           
                if log_time:
                    ax.semilogx(time_step_list[w],Ntraj_arr[w])                    
                else:
                    ax.plot(time_step_list[w],Ntraj_arr[w])    
                    ax.set_ylabel('Species Abundance')
                    ax.set_xlabel('Time')
                plt.show()
        
        return Nf, np.asarray(time_step_list), Ntraj_arr


def set_figure_params_LV(fig_name):    
    if fig_name=='figure1_competitive_LV_time_series':
        S=20
        L=3
        N0 = np.random.rand(S*L).reshape(S,L)*0.2
        #### notation for lotka volterra model the same as in [F Roy, G Biroli, G Bunin, and C Cammarota 2019]
        exp_case={'alpha':'gaussian',
                  'mean_alpha':0.5,
                  'std_alpha':0.15, ## standard deviation
                  'K':'one',
                  'r':'one',
                  'gamma':0.,###
                  'params':'figure1_competitive_LV_time_series'
                  }

    elif fig_name=='figure1_mutualistic_LV_time_series':
        S=4
        L =10
        N0 = np.random.rand(S*L).reshape(S,L)*4.
        #### notation for lotka volterra model the same as in [F Roy, G Biroli, G Bunin, and C Cammarota 2019]
        exp_case={'alpha':'gaussian',
                  'mean_alpha':-0.15,
                  'std_alpha':0.1, ## standard deviation
                  'K':'one',
                  'r':'one',
                  'gamma':0.,###
                  'params':'figure1_mutualistic_LV_time_series'
                  }
        
    else:
        print ('error in fig_name chosen')

    return S,L,N0, exp_case


destfold="/Users/ashish/Downloads/ecology_simulation_data/Steady_state_counting/figure1_mutualistic_LV_time_series/"
if not os.path.exists(destfold): os.makedirs(destfold)
#### notation for lotka volterra model params similar to [F Roy, G Biroli, G Bunin, and C Cammarota 2019]
#S=3
#L =2
#exp_case={'alpha':'gaussian',
#          'mean_alpha':0.,
#          'std_alpha':0.1, ## standard deviation
#          'K':'one',
#          'r':'one',
#          'gamma':0.,###
#          'params':'try1'
#          }
#N=inp.random.rand(S*L).reshape(S,L)*50 


S,L,initial_abundance, exp_case=set_figure_params_LV('figure1_mutualistic_LV_time_series')
N=initial_abundance
  
                
file_suffix='S'+str(S)+'_'+exp_case['params']
LV_params=set_LV_params(S,exp_case)   
#print(exp_case) 
#need to pass a list of parameters to use the map function
if type(LV_params) is not list:
    params_list = [LV_params for k in range(L)]
else:
    params_list = LV_params


reached_SteadyState=False  
ctr=0  
currentTime=1e-5## to have logspaced time intervals, I add a pseudo count.
while (reached_SteadyState==False) :  
    print ("ctr is ", ctr)
    time_step_list=[propagation_time_step for k in range(L)]
    #N = np.asarray(list(map(integrate_ODE, N.T,time_step_list, params_list ))).squeeze().T
    if ctr==0:
        N, t_traj_arr, Ntraj_arr=Propagate_and_return_trajectories_LV(N, L, params_list, T0=currentTime, Duration= propagation_time_step,nsteps=1000, 
                                                                  log_time=True, show_plots=True )
    else:
        N, t_traj_arr, Ntraj_arr=Propagate_and_return_trajectories_LV(N, L, params_list, T0=currentTime, Duration= propagation_time_step,nsteps=8, 
                                                                  log_time=False, show_plots=True )   

    if ctr==0:
        N_time_arr= deepcopy(Ntraj_arr)
        time_arr=deepcopy(t_traj_arr)
    else:
        N_time_arr=np.append(N_time_arr, Ntraj_arr, axis=1)
        time_arr=np.append(time_arr, t_traj_arr, axis=1) 
    if ctr==0:
        N_time_arr=deepcopy(Ntraj_arr) #### Ntime arr is of the shape [nwells, timepoints, species ]
        time_arr=deepcopy(t_traj_arr)
    else:
        N_time_arr=np.append(N_time_arr, Ntraj_arr, axis=1)
        time_arr=np.append(time_arr, t_traj_arr, axis=1)     

    
    dNdt_f = np.asarray(list(map(dNdt_LV,N.T,time_step_list,params_list)))
#    break
    idx_nonzero=np.nonzero(N.T)
    dlogNdt_f=dNdt_f[idx_nonzero]/N.T[idx_nonzero]
    if use_cutoff: ### removes small values to remove errors from machine precision, and avoid problems with stiffness
        N_cutoff=N
        N_cutoff[N_cutoff< abundance_cutoff]=0.0         
        N=N_cutoff         
    ctr+=1
    currentTime+=propagation_time_step
    if np.all( np.asarray( list(map(rms,dlogNdt_f)) )<1e-5 ): ## change to log and change in cutoff made on July1st2019
        print('steady state has been reached')
        reached_SteadyState=True      
    if ctr>2000:
        print ('too long.')
        break

to_write={'initial_abundance':initial_abundance, 'steady_state':N, 'passed_params':LV_params,  'exp_case':exp_case,
          'abundance_time_series':N_time_arr,'timepoints_time_series':time_arr, 'reached_SteadyState':reached_SteadyState}    
pickle.dump( to_write, open(destfold+file_suffix+'.dat', 'wb') )   



'''
to plot trajectories one can use this:
'''
#t_arr, Ntraj_arr, Rtraj_arr=Propagate_and_return_trajectories(MyPlate,1.6,show_plots=True)
#fig,ax=plt.subplots()
#StackPlot(MyPlate.N,ax=ax,unique_color=False,drop_zero=False,labels=True)
#plt.show()
#fig,ax=plt.subplots()
#StackPlot(MyPlate.R,ax=ax,unique_color=False,drop_zero=False,labels=True)
#plt.show()










