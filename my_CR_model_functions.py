#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 16:09:27 2019

Useful functions when using CR_Model
"""
import pickle # in python3, pickle uses cPickle automatically, there is no cPickle
import os
import sys
import numpy as np
#from IPython.display import Image
if os.getcwd().startswith("/Users/ashish"):
    sys.path.append('/Users/ashish/Documents/GitHub/community-simulator/')
    import matplotlib as mpl
    import matplotlib.pyplot as plt
else:  
    sys.path.append('/usr3/graduate/ashishge/ecology/Ecology-Stochasticity/')
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

from community_simulator import *
from community_simulator.usertools import *
from community_simulator.visualization import *
from community_simulator.essentialtools import IntegrateWell

import ast

from collections import OrderedDict 
#import seaborn as sns
#colors = sns.color_palette()

def rms(y):
    return np.sqrt(np.mean(y**2))


def create_replicate_lists(parameter_list,destfold_list, n_replicates, return_dfkeynames=False):
    newfold_list=[]
    newparam_list=[]
    df_keyname_list=[]
    
    if isinstance(parameter_list,list) : ## single parameter being varied
        assert len(destfold_list)== len(parameter_list),'both need to be supplied'
        for param, fold in zip(parameter_list,destfold_list):    
            for i in range(n_replicates):
                newfold_list.append(fold+str(i)+'/')
                newparam_list.append(param)
            
    
    elif isinstance(parameter_list,dict):## two  parameteres are varied, folder names need to be assigned given
        if len(parameter_list.keys())==1:
            key1,=parameter_list.keys()
            print (parameter_list.keys(),parameter_list)
            param_list=parameter_list[key1]
            if len(destfold_list)==1: ## we name the folders here                
                for param in param_list:
                    suffix=key1+str(param)
                    df_keyname=key1+str(param)
                    if n_replicates>0:
                        for i in range(n_replicates):
                            newfold_list.append(destfold_list[0]+'/'+suffix+'/'+str(i)+'/')
                            newparam_list.append(param)                       
                            df_keyname_list.append(df_keyname)
                    else:
                        newfold_list.append(destfold_list[0]+'/'+suffix+'/')
                        newparam_list.append(param)                       
                        df_keyname_list.append(df_keyname)
                        
            else: ## named folders are provided
                for param, fold in zip(param_list,destfold_list):    
                    for i in range(n_replicates):
                        newfold_list.append(fold+str(i)+'/')
                        newparam_list.append(param)   
            
            
        elif len(parameter_list.keys())==2:
            key1,key2=parameter_list.keys()
            assert len(destfold_list)==1, "we name the folders in here!"
            fold=destfold_list[0]
            
            if '1 to param' in parameter_list[key2]:
                step_size=1
                UB_X=1
                LB=1
                UB_temp=None
                if '1 to param1' in parameter_list[key2]:
                    temp_str=parameter_list[key2].replace('1 to param1','')
                
                
                if len(temp_str)>0:
                    temp_dict= ast.literal_eval(temp_str)
                    if  'step_size' in  temp_dict.keys():
                        step_size=int(temp_dict['step_size'])
                    if  'LB' in  temp_dict.keys():
                        LB=int(temp_dict['LB'])
                    if  'UB_X' in  temp_dict.keys():
                        UB_X=float(temp_dict['UB_X'])
                    if 'UB' in  temp_dict.keys():
                        UB_temp=int(temp_dict['UB'])
                    if 'UB' in  temp_dict.keys() and 'UB_X' in  temp_dict.keys():
                        print ('both UB and UB_X shouldnt be specified together')
                        sys.exit(1)
                    
                                        
                assert isinstance(parameter_list[key1],list)    
                assert isinstance(parameter_list[key1][0],int)  

                for param1 in parameter_list[key1]:
                    if UB_temp==None:
                        UB=int(round(UB_X*param1+1))
                    else:
                        UB=UB_temp
                    print ('UB=',UB)
                    for  param2 in range(LB, UB, step_size): ## 1,2,.. till =param1
                        suffix=key1+str(param1)+key2+str(param2)
                        df_keyname=key1+str(param1)+': '+key2+str(param2)
                        if n_replicates>0:
                            for i in range(n_replicates):
                                newparam_list.append([param1,param2])
                                newfold_list.append( fold+'/'+suffix+'/'+str(i)+'/' )  
                                df_keyname_list.append(df_keyname)
                        else:
                            newparam_list.append([param1,param2])
                            newfold_list.append( fold+'/'+suffix+'/' )  
                            df_keyname_list.append(df_keyname)
            else:
                assert isinstance(parameter_list[key1],list)    
                assert isinstance(parameter_list[key2],list)    
                for param1 in parameter_list[key1]:
                    for  param2 in parameter_list[key2]:
                        suffix=key1+str(param1)+key2+str(param2)  
                        df_keyname=key1+str(param1)+': '+key2+str(param2)
                         
                        if n_replicates>0:
                            for i in range(n_replicates):
                                newparam_list.append([param1,param2])                                             
                                newfold_list.append( fold+'/'+suffix+'/'+str(i)+'/' )
                                df_keyname_list.append(df_keyname)
                        else:
                            newparam_list.append([param1,param2])                                             
                            newfold_list.append( fold+'/'+suffix+'/' )
                            df_keyname_list.append(df_keyname)
                    
    else:
        print ("parameter list format not supported")
        sys.exit(1)
    
    if return_dfkeynames:
        assert len(df_keyname_list)>0,'this isnt implemented for some cases in a single parameter being varied'
        return newfold_list,newparam_list,df_keyname_list
    else:
        return newfold_list,newparam_list
    

def set_figure_params(fig_name):
    
    if fig_name=='figure1_CR_time_series':
        S=4
        M=6
        L=10
        N0=np.random.rand(S*L).reshape(S,L)*50 
        exp_case={'c':'binary_fixedNum',
                  'mean_c':2,
                  'params':'CR_fig1',          
                  'resource_supply':'self-renewing',
                  'D': 'dirichlet',
                  'l':0.0,
                  'sigma_max':20., ## resource uptake rate saturation
                  'response':'type II'
                  }
        
        R0_alpha=make_R0_alpha(M, exp_case)                     
        R0 =np.swapaxes( np.tile(R0_alpha,(L,1)),0,1 )*0.0
        
    elif fig_name=='figure1_CR_crossfeeding_time_series':
        S=40
        M=40
        L=3
        N0=np.random.rand(S*L).reshape(S,L)*50 
        exp_case={'c':'binary_fixedNum',
                  'mean_c':10,
                  'params':'CR_fig1',          
                  'resource_supply':'external',
                  'D': 'dirichlet',
                  'D_param': 0.5, ## to make the matrix intermediately sparse (about a quarter of the entries are large)
                  'l':0.5,
                  'response':'type II',
                  'sigma_max':20., ## resource uptake rate saturation
                  'R0_case':'single',# for a uniform distribution, max is passed so double the value if you want to fix mean
                  'R0_params':100.  ## 800=20*40= meanR0*S would to keep max energy flux similar to Bobby's PLoS paper
                  }
        R0_alpha=make_R0_alpha(M, exp_case)                     
        R0 =np.swapaxes( np.tile(R0_alpha,(L,1)),0,1 )*0.0
        
       
    else:
        print ('error in fig_name chosen')

    return S,M,L,N0, R0_alpha, R0, exp_case


def binary_diagonal_fixed_numInteractions_matrix(S, M, n_ones):
    ## makes a matrix with 1s on diagonal where each row has 'n_ones'  nonzero elements
    assert S==M,"the diagonal form makes sense only when S==M"
    assert n_ones>=1 and n_ones<=M
    c=np.diag(np.ones(M)) 
    if n_ones==1: 
        print("diagonal matrix, non-interacting case")
    else:   
        for i in range (S):
            indices=np.arange(M)
            indices=np.delete(indices,i)
            c[i,np.random.choice(indices, n_ones-1,replace=False)]=1.0      
    return c
 
def binary_fixed_numInteractions_matrix(S, M, n_ones):
    ## makes a matrix where each row has 'n_ones'nonzero elements, no diagonal constraint.
    ##removed the  'ensures that all resources are used' constraint on 1/10
    # remoed on 1/10:   assert S*n_ones*0.8>=M,"we cannot have all resources consumed, and be reasonably random otherwise"
    assert n_ones>=1 and n_ones<=M    
    #all_resources_consumed=False
    #while all_resources_consumed==False:
    c=np.zeros((S,M)) 
    for i in range (S):
        c[i,np.random.choice(M, n_ones,replace=False)]=1.0   
    return c


def binary_fixed_numInteractions_SpecialSpecies_matrix(S, M, n_ones, always_consumes_list, never_consumes_list):
    ## makes a matrix where each row has 'n_ones'nonzero elements, no diagonal constraint.
    ## 1st set of special species always consume R0
    ## 2nd set of  species species nevers consume R0
    assert n_ones>=1 and n_ones<=M    

    c=np.zeros((S,M)) 
    for i in range (S):
        if i in always_consumes_list:
            c[i,0]=1.0
            if n_ones>1:
                c[i,1+np.random.choice(M-1, n_ones-1,replace=False)]=1.0 
        if i in never_consumes_list:
            if n_ones<M:
                c[i,1+np.random.choice(M-1, n_ones,replace=False)]=1.0
            else:
                c[i,:]=1.0
        else:
            c[i,np.random.choice(M, n_ones,replace=False)]=1.0   
    return c


def binary_AllEatR0_fixed_numInteractions_matrix(S, M, n_ones):
    ## makes a matrix where each row has 'n_ones'nonzero elements, 
    ## all species eat Resource 0
    ## does NOT ensures that all resources are used.    
    assert n_ones>=1 and n_ones<=M        
    c=np.zeros((S,M))
    c[:,0]=1.
    if n_ones>1:
        for i in range (S):
            c[i, 1+np.random.choice(M-1, n_ones-1,replace=False) ]=1.0    ## should choose resource between 1 & M, not 0&M.
    return c
           

def truncated_normal(mu,sigma,size):
    truncated_normal_variable=np.random.normal(mu,sigma,size)
    ctr=0
    while np.any(truncated_normal_variable<0):
        idx_negative=np.where(truncated_normal_variable<0)[0]
        truncated_normal_variable[idx_negative]=np.random.normal(mu,sigma,idx_negative.size)
        ctr+=1
        assert ctr<10,"did this too many times, you should try different mu and sigma!"
    return   truncated_normal_variable 


def make_consumption_matrix(S, M, exp_case):
    #Construct consumption  matrix
    c=np.zeros((S,M))
    if exp_case['c']=='non_interacting':    # each species consumes only its resource 
        assert S==M, "non interacting only possible when S<=M, and S=M is efficient"
        c=np.diag(np.ones(M))
    if exp_case['c'].startswith('2resource'):   # each species consumes 2 resources, one diagonal (its own) and one other           
        if S==M:
            for i in range (S):
                    c[i,min(i,M-1)]=1
                    temp=np.random.randint(S)
                    while temp==min(i,M-1): temp=np.random.randint(S) ## want each species to have 2 distinct resources                                   
                    c[i,temp]=1
        else:
            print ("not implemented yet")
            return None
        if exp_case['c']=='2resource_Cnoise': c=c*(1 + 0.1*np.random.rand(S*M).reshape(S,M) )
                            
    if exp_case['c'].startswith('halfresource'): # each species consumes M/2 resources (rounded down).
        for i in range (S):
                c[i,np.random.choice(M,int(M/2),replace=False)]=1
        if np.any( np.sum(c,axis=0)==0 ): print("some resources are not consumed by anyone :\n", np.sum(c,axis=0))         
        if exp_case['c']=='halfresource_Cnoise': c=c*(1 + 0.1*np.random.rand(S*M).reshape(S,M) )
   
    if exp_case['c']=='binary':
        c=BinaryRandomMatrix(S,M,exp_case['mean_c']*1.0/M)
    if exp_case['c']=='binary_x_gamma':
        c=BinaryRandomMatrix(S,M,exp_case['mean_c']*1.0/M)
        c=c * (np.random.gamma(10,1,S*M)/10.).reshape(S,M) ## to keep each element ~1 on average.
        
    '''
     from wikipedia
     If X is a random variable~ Gamma (k,theta)
     then for any c>0
     cX is a random variable ~ Gamma (k,ctheta)
     similar to scaling of Poisson distributions
    '''
        
    if exp_case['c']=='binary_diag':
        c=binary_diagonal_fixed_numInteractions_matrix(S,M,int(exp_case['mean_c']))
    if exp_case['c']=='binary_diag_x_gamma':
        c=binary_diagonal_fixed_numInteractions_matrix(S,M,int(exp_case['mean_c']))
        c=c * (np.random.gamma(10,1,S*M)/10.).reshape(S,M)  ## to keep each element ~1 on average.
    
    if exp_case['c']=='binary_fixedNum':
        c=binary_fixed_numInteractions_matrix(S,M,int(exp_case['mean_c']))   
    if exp_case['c']=='binary_fixedNum_x_gamma':
        c=binary_fixed_numInteractions_matrix(S,M,int(exp_case['mean_c']))   
        c=c * (np.random.gamma(10,1,S*M)/10.).reshape(S,M) ## to keep each element ~1 on average.
        
    if exp_case['c']=='binary_fixedNum_x_gamma_0_1_Special':
        ## species 0 always eats R0 and species 1 never eats R0 unless meanc=M!
        c=binary_fixed_numInteractions_SpecialSpecies_matrix(S,M,int(exp_case['mean_c']),[0],[1] ) 
        c=c * (np.random.gamma(10,1,S*M)/10.).reshape(S,M)
    if exp_case['c']=='binary_fixedNum_x_gamma_0_Special':
        ## species 0 always eats R0
        c=binary_fixed_numInteractions_SpecialSpecies_matrix(S,M,int(exp_case['mean_c']),[0],[] ) 
        c=c * (np.random.gamma(10,1,S*M)/10.).reshape(S,M)
    
    if exp_case['c']=='binary_AllEatR0':
        c=binary_AllEatR0_fixed_numInteractions_matrix(S,M,int(exp_case['mean_c']))    
    if exp_case['c']=='binary_AllEatR0_gamma':
        c=binary_AllEatR0_fixed_numInteractions_matrix(S,M,int(exp_case['mean_c']))    
        c=c * (np.random.gamma(10,1,S*M)/10.).reshape(S,M)
        
    if exp_case['c']=='binary_AllEatR0_uniform': ## each nonzero element is chosen from a uniform distribution in [0,1]
        c=binary_AllEatR0_fixed_numInteractions_matrix(S,M,int(exp_case['mean_c']))    
        c=c*np.random.rand(S,M)
        
    if 'c0' in exp_case:
        c=c+exp_case['c0']
        
    else:
        print ("error, exp_case c was incorrectly given",exp_case['c'])
        sys.exit(1)

        
    assert np.any(c>0.01),"Cant be all zero!"    
    return c

def make_leakage_matrix(S, M, exp_case):   
    if 'D' in exp_case:   
        if exp_case['D']=='diagonal':    
            D=np.diag(np.ones(M))
        elif exp_case['D']=='uniform':    
            D=np.ones((M,M))*1./M
        elif exp_case['D']=='dirichlet':        
            dirichlet_param=exp_case['D_param'] if 'D_param' in exp_case else 1. ##1 means uniform over the simplex 
            D=np.random.dirichlet(dirichlet_param*np.ones(M),M).reshape(M,M)        # if  dirichlet_param is small~0.0 leaks to a single type.            
            print ('dirichlet_param was'+str(dirichlet_param))
        elif exp_case['D']=='binary': 
            n_leak=exp_case['D_param'] if 'D_param' in exp_case else exp_case['mean_c']  ## leaks to n_leak resource types
            d0=exp_case['d0'] if 'd0' in exp_case else 0.0 ## background value
            norm_factor= 1.-(M-n_leak)*d0 
            assert norm_factor>0.5, "background leakage is the majority, doesnt make that much sense"
            D=np.zeros((M,M))+d0
            for i in range(M):
                idx_chosen=np.random.choice(np.arange(M), n_leak, replace=False)
                D[i,idx_chosen]=norm_factor/(n_leak)   
            print ('n_leak was'+str(n_leak))
    else:
        assert exp_case['l']==0.0, "we are looking at the no leakage case"
        D=np.ones((M,M))*1./M
        
    return D

def make_R0_alpha(M, exp_case):
    if 'R0_case' in exp_case:
        if exp_case['R0_case']  =='uniform':
            if 'R0_params' in exp_case: Rmax=exp_case['R0_params']
            else:Rmax=20
            R0=np.random.rand(M)*Rmax
        elif exp_case['R0_case']  =='single':
            Rval=exp_case['R0_params'] if 'R0_params' in exp_case else 20.
            R0=np.zeros(M)
            R0[0]=Rval   
        elif exp_case['R0_case']  =='gamma':
            if 'R0_params' in exp_case:k,theta=exp_case['R0_params']           
            else: k,theta=20,1
            R0=np.random.gamma(k,theta,M)  
        elif exp_case['R0_case']  =='Rsupply_constR0': ## specify number of supplied resources at fixed abundance of each
            num, Rval=exp_case['R0_params'] 
            num=int(num) 
            R0=np.zeros(M)
            Rval=1.0*Rval 
            R0[:num]=Rval 
            
        elif exp_case['R0_case'] =='Rsupply_constTotR': ## specify number of supplied resources at fixed total resource abundance
            num, Rval=exp_case['R0_params'] 
            num=int(num) 
            R0=np.zeros(M)
            Rval=Rval *1./num
            R0[:num]=Rval 
                           
    else:
        if 'R0_params' in exp_case:k,theta=exp_case['R0_params']           
        else: k,theta=20,1           
        R0=np.random.gamma(k,theta,M)         
    
    if exp_case['params'].startswith('PLoS'):
        R0=np.zeros(M)
        if exp_case['params']=='PLoS2':R0[0]=28.
        if exp_case['params']=='PLoS3':R0[0]=1000.
    
        
    return R0
    

def make_params_and_assumptions(S, M, c, D, R0_alpha, exp_case):
    #Make parameter list
    if exp_case['params']=='equal':
        params = {'c':c,
                  'D':D,
                  'm':0.1*np.ones(S),
                  'w':0.1*np.ones(M),
                  'g':1.1*np.ones(S),
                  'l':0.0,
                  'r':2.1*np.ones(M),
                  'R0':R0_alpha*np.ones(M),
                  'tau':.1*np.ones(M),
                  'sigma_max':1.
                  }
    elif exp_case['params']=='noisy':
        params = {'c':c,
                  'D':D,
                  'm':0.1*( np.ones(S)+0.1*np.random.rand(S) ),
                  'w':0.1*( np.ones(M)+0.1*np.random.rand(M) ),
                  'g':1.1*( np.ones(S)+0.1*np.random.rand(S) ),
                  'l':0.0,
                  'r':0.1*( np.ones(M)+0.1*np.random.rand(M) ),
                  'R0':R0_alpha*( np.ones(M)+0.1*np.random.rand(M) ),
                  'tau':.1*( np.ones(M)+0.1*np.random.rand(M) ),
                  'sigma_max':1.
                  }
        
    elif exp_case['params']=='test1':
        params = {'c':c,
                  'D':D,
                  'm':truncated_normal(0.1, 0.01, S),
                  'w':np.random.rand(M),
                  'g':truncated_normal(1., 0.01, S),
                  'l':0.0,
                  'r':truncated_normal(.1,0.01, M),
                  'R0':R0_alpha,
                  'tau':0.1/R0_alpha,
                  'sigma_max':1.
                  }

    elif exp_case['params']=='P1':
        params = {'c':c,
                  'D':D,
                  'm':truncated_normal(1., 0.1, S),
                  'w':truncated_normal(1., 0.1, M),
                  'g':truncated_normal(1., 0.1, S),
                  'l':0.0,
                  'r':truncated_normal(1., 0.1, M),
                  'R0':R0_alpha,
                  'tau':0.1/R0_alpha, ## this division is not good. constant tau is chemostat dilution rate!
                  'sigma_max':20.
                  }
    
    elif exp_case['params'].startswith('PLoS'): 
    ## refer to page 5 in SI of Bobbly's PLoSpaper. D matrix and R0 are not the same.    
        params = {'c':c,
                  'D':D,
                  'm':truncated_normal(1., 0.1, S),
                  'w':1.,
                  'g':1.,
                  'l':0.0,
                  'r':truncated_normal(1., 0.1, M),
                  'R0':R0_alpha,
                  'tau':1., ## not sure what this should be
                  'sigma_max':20.
                  } 
        if exp_case['params']=='PLoS2':params['l']=0.6 
        if exp_case['params']=='PLoS3':params['l']=0.9 
             
    elif exp_case['params']=='CR_fig':
        params = {'c':c,
                  'D':D,
                  'm':truncated_normal(1., 0.1, S),
                  'w':truncated_normal(1., 0.1, M),
                  'g':truncated_normal(1., 0.1, S),
                  'l':0.0,
                  'r':truncated_normal(1., 0.1, M),
                  'R0':R0_alpha,
                  'tau':0.1,
                  'sigma_max':20.
                  }
                
    elif exp_case['params']=='CR_fig2':
        params = {'c':c,
                  'D':D,
                  'm':truncated_normal(1., 0.2, S),
                  'w':truncated_normal(1., 0.2, M),
                  'g':truncated_normal(1., 0.2, S),
                  'l':0.0,
                  'r':truncated_normal(1., 0.2, M),
                  'R0':R0_alpha,
                  'tau':0.1,
                  'sigma_max':20.
                  }
        
        
    elif exp_case['params']=='CR_gamma':
        assert 'w_params' in exp_case," need params for gamma distribution"
        mean, k, theta=exp_case['w_params']## used to be called mw params
        params = {'c':c,
                  'D':D,
                  'm':np.random.gamma(k,theta,S)*mean/(k*theta),
                  'w':np.random.gamma(k,theta,M)*mean/(k*theta),
                  'g':np.random.gamma(k,theta,S)*mean/(k*theta),
                  'l':0.0,
                  'r':truncated_normal(1., 0.2, M), ## only used for self-renewing resources
                  'R0':R0_alpha,
                  'tau':0.1,
                  'sigma_max':20.
                  }
        
    elif exp_case['params']=='w_gamma':
        assert 'w_params' in exp_case," need params for gamma distribution"
        mean, k, theta=exp_case['w_params']
        params = {'c':c,
                  'D':D,
                  'm':truncated_normal(1., 0.1, S),
                  'w':np.random.gamma(k,theta,M)*mean/(k*theta),
                  'g':truncated_normal(1., 0.1, S),
                  'l':0.0,
                  'r':truncated_normal(1., 0.1, M), ## only used for self-renewing resources
                  'R0':R0_alpha,
                  'tau':0.1,
                  'sigma_max':20.
                  }
        
    elif exp_case['params']=='mw_gamma':
        assert 'w_params' in exp_case," need params for gamma distribution"
        assert 'm_params' in exp_case," need params for gamma distribution"
        meanM, kM, thetaM=exp_case['m_params']
        meanW, kW, thetaW=exp_case['w_params']

        params = {'c':c,
                  'D':D,
                  'm':np.random.gamma(kM,thetaM,S)*meanM/(kM*thetaM),
                  'w':np.random.gamma(kW,thetaW,M)*meanW/(kW*thetaW),
                  'g':truncated_normal(1., 0.1, S),
                  'l':0.0,
                  'r':truncated_normal(1., 0.1, M), ## only used for self-renewing resources
                  'R0':R0_alpha,
                  'tau':0.1,
                  'sigma_max':20.
                  }   
                

   
    for item in ['m','w','g','l','tau','r','sigma_max']: # modifying any parameter if supplied
        if item in exp_case.keys():
            params[item] = exp_case[item]

    '''
    Bobby's assumption dictionary is much bigger, refer  to a_default in user_tools.py
    We use only a subset of the assumptions since we generate and supply c and D matrices separately,
    and we do not incorporate any family structure or resource types.
    '''  
    assumptions = {
                  'regulation':'independent',
                  'supply':'external',   # external or self-renewing
                  'response':'type I',
                  'R0_food': params['R0'], #unperturbed fixed point for supplied food
                 'exp_case': exp_case # just adding this here , not used in Community-Simulator
                 }

    for item in ['regulation','supply','response']: # modifying any parameter if supplied
         if item in exp_case.keys():
            assumptions[item] = exp_case[item]
    assert exp_case['resource_supply']!='logistic',"its called self-renewing!"
   
    return params, assumptions


def make_all_combinations_IC(S, M, R0_alpha=20,all_absent=False):
    '''
    all combinations of initial abudances are created in the various wells
    '''
    L = 2**S-1 if all_absent==False else 2**S # need L to be atleast 2^(S-1)if we want all species to appear atleast once.
    N0 = np.zeros((S,L))
    for i in range(L): 
        if all_absent:
            binary_rep=bin(i)[2:]
        else:
            binary_rep=bin(i+1)[2:]
        for j in range(len(binary_rep)):  
            N0[S-j-1,i]= int( binary_rep[len(binary_rep)-j-1] )
    
    if isinstance(R0_alpha,int):   
        R0 = R0_alpha* np.ones((M,L))
    else:
        R0 =np.swapaxes( np.tile(R0_alpha,(L,1)),0,1 )
    
    init_state = [N0,R0]    
    return init_state





def sample_IC_space(S, M):
    print("need to write a function to efficiently sample the state space.")
    
  
    
def Propagate_and_return_trajectories(CommunityInstance,T0=0,Duration=1,nsteps=10,compress_resources=False,compress_species=True,log_time=True,show_plots=False ):
        """
        Function's like Bobb'y propagate except that the trajectories are returned instead of just final state
        
        Propagate the state variables forward in time according to dNdt, dRdt.
        
        T = time interval for propagation
        
        compress_resources specifies whether zero-abundance resources should be
            ignored during the propagation. Tshis makes sense when the resources
            are non-renewable.
            
        compress_species specifies whether zero-abundance species should be
            ignored during the propagation. This always makes sense for the
            models we consider. But for user-defined models with new parameter
            names, it must be turned off, since the package does not know how
            to compress the parameter matrices properly.
        """
        #CONSTRUCT FULL SYSTEM STATE
        y_in = CommunityInstance.N.append(CommunityInstance.R).values
        T=Duration
        if log_time==True:
            assert T0!=0, 'we cant have T0=0 for logspaced time, add a pseudo count atleast.'
            
        
        
        #PACKAGE SYSTEM STATE AND PARAMETERS IN LIST OF DICTIONARIES
        if isinstance(CommunityInstance.params,list):
            well_info = [{'y0':y_in[:,k],'params':CommunityInstance.params[k]} for k in range(CommunityInstance.n_wells)]
        else:
            well_info = [{'y0':y_in[:,k],'params':CommunityInstance.params} for k in range(CommunityInstance.n_wells)]
        
        #PREPARE INTEGRATOR FOR PARALLEL PROCESSING
        IntegrateTheseWells = partial(IntegrateWell,CommunityInstance,T0=T0,T=T,compress_resources=compress_resources,
                                      compress_species=compress_species,log_time=log_time,ns=nsteps,return_all=True)
        
        #INITIALIZE PARALLEL POOL AND SEND EACH WELL TO ITS OWN WORKER
        if CommunityInstance.parallel:
            pool = Pool()
#            print (pool.map(IntegrateTheseWells,well_info))
            integ_output=pool.map(IntegrateTheseWells,well_info)
#            t, traj = pool.map(IntegrateTheseWells,well_info)
#            t, traj = np.asarray(pool.map(IntegrateTheseWells,well_info)).squeeze().T
            pool.close()
            print ('parallel has not been tested!')
        else:   
            integ_output=list(map(IntegrateTheseWells,well_info))
                              
        assert len(integ_output)==CommunityInstance.n_wells and len(integ_output)>1, 'integration for a subset of wells not supported. For a single well, use TestWell'

        t_arr=[]
        Ntraj_arr = []
        Rtraj_arr = []

        for w in range(CommunityInstance.n_wells):
            t_arr.append(integ_output[w][0])
            
            Ntraj_arr.append(integ_output[w][1][:,:CommunityInstance.S] )
            Rtraj_arr.append(integ_output[w][1][:, CommunityInstance.S:] ) 

        t_arr=np.array(t_arr)
        Ntraj_arr=np.array(Ntraj_arr)
        Rtraj_arr=np.array(Rtraj_arr)
       
        #print(np.shape(t_arr),np.shape(Ntraj_arr),np.shape(Rtraj_arr))
        
        Nf=Ntraj_arr[:,-1,:].squeeze().T
        Rf=Rtraj_arr[:,-1,:].squeeze().T
 
        if show_plots:
            for w in range(CommunityInstance.n_wells):
                fig, axs = plt.subplots(2,sharex=True)           
                if log_time:
                    axs[0].semilogx(t_arr[w],Ntraj_arr[w])
                    axs[1].semilogx(t_arr[w],Rtraj_arr[w])
                else:
                    axs[0].plot(t_arr[w],Ntraj_arr[w])
                    axs[1].plot(t_arr[w],Rtraj_arr[w])
                    axs[0].set_ylabel('Consumer Abundance')
                    axs[1].set_ylabel('Resource Abundance')
                    axs[1].set_xlabel('Time')
                plt.show()
                              
        #UPDATE STATE VARIABLES WITH RESULTS OF INTEGRATION
        CommunityInstance.N = pd.DataFrame(Nf,
                              index = CommunityInstance.N.index, columns = CommunityInstance.N.keys())
        CommunityInstance.R = pd.DataFrame(Rf,
                              index = CommunityInstance.R.index, columns = CommunityInstance.R.keys()) 
        
        return t_arr, Ntraj_arr, Rtraj_arr
        
        
        
        
        #UPDATE STATE VARIABLES WITH RESULTS OF INTEGRATION
#        CommunityInstance.N = pd.DataFrame(y_out[:CommunityInstance.S,:],
#                              index = CommunityInstance.N.index, columns = CommunityInstance.N.keys())
#        CommunityInstance.R = pd.DataFrame(y_out[CommunityInstance.S:,:],
#                              index = CommunityInstance.R.index, columns = CommunityInstance.R.keys())    
    