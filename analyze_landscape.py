#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 19:28:30 2019

@author: ashish

Computes quanitities of interest in the landscape of Consumer-Resource models
using the community simulator package 
"""


import numpy as np
import pickle # in python3, pickle uses cPickle automatically, there is no cPickle
import os
import glob
#import pandas
import argparse
#from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import minimize
import sys
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import comb
import itertools
if os.getcwd().startswith("/Users/ashish"):
    sys.path.append('/Users/ashish/Documents/GitHub/Ecology-Stochasticity/')
    from my_CR_model_functions import rms,create_replicate_lists
    import matplotlib as mpl
else:  
    sys.path.append('/usr3/graduate/ashishge/ecology/Ecology-Stochasticity/')
    from my_CR_model_functions import rms,create_replicate_lists
    import matplotlib as mpl
    mpl.use('Agg')### can not use pyplot on cluster without agg backend
#import networkx as nx
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from Linear_regression_landscape import perform_Linear_Regression_of_fitness_scalar, perform_Linear_Regression_of_abundance_vector,N_to_Ncutoff
########### and  Random Forest regression  uses this file.! :/
import RandomForestRegression_onCRdata as RF_vs_linReg
from collections import OrderedDict

nice_fonts = { #"text.usetex": True, # Use LaTex to write all text
"font.family": "serif",
# Use 10pt font in plots, to match 10pt font in document
"axes.labelsize": 10, "font.size": 10,
# Make the legend/label fonts a little smaller
"legend.fontsize": 8, "xtick.labelsize": 8,"ytick.labelsize": 8 }
mpl.rcParams.update(nice_fonts)


popn_cutoff=1e-3 #update on 9/25 in accordance with CF simulations. 

machine_precision=1e-6




####################        Function definitions           #################### 
def total_abundance(N):
    ## computes the total biomass in each plate
    return np.sum(N, axis=1)

def number_of_surviving_species(N, cutoff=1):
    ## computes number of species with abundance> cutoff
    return np.sum( N>cutoff, axis=1)

def focal_species_abundance(N, Sidx):
    ## computes the abundance of a focal species at index=idx
    return N[:,Sidx]

def two_species_product(N, Sidx1, Sidx2,sigma_vectors_final):
    ## computes the product of the abundance of two focal species    
    ### to prevent any big number*0 kind of errors..
    indices1=np.where(sigma_vectors_final[:,Sidx1]==1)[0]
    indices2=np.where(sigma_vectors_final[:,Sidx2]==1)[0]
    coex_indices=np.intersect1d(indices1,indices2)
    print ("\n number of coexistence points", coex_indices.size, np.shape(coex_indices), indices1.size, indices2.size)
    
    if coex_indices.size>0:         
        N1=N[:,Sidx1]
        N2=N[:,Sidx2]
        print ('mean N1, N2:', np.mean(N1),  np.mean(N2), np.mean(N1[coex_indices]), np.mean(N2[coex_indices]))
        N1_x_N2=np.zeros(len(N))
        N1_x_N2[coex_indices]=N1[coex_indices] *N2[coex_indices]
        return N1_x_N2
    else:
        return np.zeros(len(N))
    
    
# =============================================================================
# def Clark_type_function(N):
#     ## computes a funciton of the form w_i N_i + A_ij Ni Nj
#     assert len(N)==2**16,'function defined for 16 species only'
#     w_i=np.array([0.90011087, 0.58424366, 0.01088101, 0.51320863, 0.39194486,0.15840141, 0.28115222, 0.35206516, 0.91801849, 
#                   0.71234526,0.31231176, 0.29621652, 0.75654075, 0.84559851, 0.76025346,0.90809062]) ## uniform between 0 and 1.
# 
#     A_ij=0.
#     return np.sum(N, axis=1)
#     
# =============================================================================
        
    
    
def butyrate_M3_Clark(Nf, sigma_final):
    sp_list=['RI', 'AC','ER', 'CC','CG', 'CA','EL', 'DF','BF', 'DP','BV']
    idx_list=[ 7, 10, 8,  3,  2,  6,  4,  5,  1,  9, 11] ## randomly assigned species to index
    sp_to_idx_dict={}
    idx_to_sp_dict={}
    for sp,i in zip(sp_list,idx_list):
        sp_to_idx_dict.update({sp:i})
        idx_to_sp_dict.update({i:sp})   
    f_mono=43.1*Nf[:,sp_to_idx_dict['RI']] + 32.2*sigma_final[:,sp_to_idx_dict['AC']] 
    + 28.8*Nf[:,sp_to_idx_dict['ER']] + 16.2*Nf[:,sp_to_idx_dict['CC']] + 5.6*sigma_final[:,sp_to_idx_dict['CC']]
    
    f_inter=10.2*Nf[:,sp_to_idx_dict['CG']] *sigma_final[:,sp_to_idx_dict['AC']]
    +9.3*Nf[:,sp_to_idx_dict['CA']] *sigma_final[:,sp_to_idx_dict['RI']]
    +8.0*sigma_final[:,sp_to_idx_dict['EL']]*sigma_final[:,sp_to_idx_dict['AC']]
    +6.3*Nf[:,sp_to_idx_dict['DF']] *sigma_final[:,sp_to_idx_dict['RI']]
    +6.0*Nf[:,sp_to_idx_dict['ER']] *sigma_final[:,sp_to_idx_dict['DP']]
    +5.2*Nf[:,sp_to_idx_dict['DF']] *sigma_final[:,sp_to_idx_dict['AC']]
    +4.1*sigma_final[:,sp_to_idx_dict['CA']] *sigma_final[:,sp_to_idx_dict['AC']]
    -6.1*Nf[:,sp_to_idx_dict['RI']] *sigma_final[:,sp_to_idx_dict['ER']]
    -6.2*Nf[:,sp_to_idx_dict['BF']] *sigma_final[:,sp_to_idx_dict['AC']]
    -6.4*Nf[:,sp_to_idx_dict['CC']] *sigma_final[:,sp_to_idx_dict['DP']]
    -6.7*Nf[:,sp_to_idx_dict['ER']] *sigma_final[:,sp_to_idx_dict['AC']]
    -10.3*sigma_final[:,sp_to_idx_dict['DP']] *sigma_final[:,sp_to_idx_dict['AC']]
    -13.3*Nf[:,sp_to_idx_dict['ER']] *sigma_final[:,sp_to_idx_dict['BV']]

    return f_mono+f_inter
    
    
    
    

    
    
    
    
  
    
    
    

def focal_species_coexistence_product(N, Sidx,sigma_vectors_final):
    ## computes the product of the abundance of a focal species  with the abundance of all OTHER species left in the community
    indices=np.where(sigma_vectors_final[:,Sidx]==1)[0]   
    N_foc_x_otherN=np.zeros(len(N))
    if indices.size>0:
        N_focal=N[:,Sidx] 
        N_foc_x_otherN[indices]= ( N_focal * (total_abundance(N)-N_focal)  )[indices]
        return N_focal* (total_abundance(N)-N_focal)
    else:
        return np.zeros(len(N))

#strength_list=[0.001,0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1., 2.] 
def generate_noisy_measurements(fitness, kind='multiplicative', strength=0.1, randseed=None)  :
    if randseed is not None:
        np.random.seed(randseed)## fix the seed uniquely for each value of strength
    if kind== 'multiplicative':
        noisy_fitness=fitness+strength*fitness*np.random.normal(size=len(fitness))
    elif kind=='additive':
        noisy_fitness=fitness+strength*np.mean(fitness)*np.random.normal(size=len(fitness))
    np.random.seed()
    return noisy_fitness
 

def remove_zero_data_point(sigma_vectors, fitness=None, sigma_vectors_final=None, n_exp=None):
    idx0=int( np.where( np.sum(sigma_vectors, axis=1)==0)[0])
    if fitness is not None:
        return(np.delete(fitness, idx0))
    elif sigma_vectors_final is not None:
        return(np.delete(sigma_vectors_final, idx0,axis=0))
    elif n_exp is not None:
        return n_exp-1
    else:
        return(np.delete(sigma_vectors, idx0,axis=0))
    


def subsetting_for_focal_species(N, sigma_vectors, sigma_vectors_final, adjacency_matrix, fitness, n_exp, S, Sidx):
    '''
    returns subsetted matrices of N, sigma and adj corresponding to S-1 species system 
    with the focal species always present
    '''   
    indices=np.where(sigma_vectors[:,Sidx]==1)[0]
    ## first remove experiments without focal species
    Nsub=N[indices]
    Sigma_sub=sigma_vectors[indices] 
    fit_sub=fitness[indices]
    Sigmaf_sub=sigma_vectors_final[indices] 
    ## now remove the data corresponding to focal species
    Nsub=np.delete(Nsub,Sidx,1)
    Sigma_sub=np.delete(Sigma_sub,Sidx,1)  
    Sigmaf_sub=np.delete(Sigmaf_sub,Sidx,1)  
    ## for the adjacency matrix, data without focal species should be removed
    ## AND address values in the array should be changed  corresponding to the new subsetted matrices!    
    adj_sub=[]#adj_sub=np.zeros((int(n_exp/2),S-1))is fine, but we want to catch any errors in case.
    ctr=0
    for i in range (n_exp):       
        if i in indices: ## if species Sidx was present, the corresponding adjacency is recorded
            temp=[]
            for j in adjacency_matrix[i]:
                if j in indices:
                    temp.append(int(np.where(indices==j)[0][0]) )                  
            adj_sub.append(temp) 
            ctr+=1
#    print (ctr)
    return Nsub, Sigma_sub, Sigmaf_sub, np.asarray(adj_sub), fit_sub


def diversity_index(N, sigma_vectors, S, n_exp): 
    ## calculates the simpson's diversity/ dominance index 
    '''
    Simpson's diversity index is defined as 1-(\sigma_i p_i ^2) , it [varies from (0, 1-1/S)] -lower is more diverse
    Simpson's Dominance/inverse index is defined as  1/(\sigma_i p_i ^2) , it [varies from (1,S)] 
    equitability or evenness index is the 1/(\sigma_i p_i ^2) *1/S i.e, it is the diversity index normalized to lie between 1/S &1 
    '''
    biomass=total_abundance(N)
    S_in_IC=total_abundance(sigma_vectors) ## number of species in each initial codnitns
    p_vector=np.zeros(N.shape)
    SimpsonsDiv=np.zeros(n_exp)
    SimpsonsDom=np.zeros(n_exp)
    SimpsonsE=np.zeros(n_exp)
    p_vector[biomass>0]=np.apply_along_axis(np.divide, 0, N[biomass>0], biomass[biomass>0])
    SimpsonsDiv[biomass>0]=1.-np.sum( np.square(p_vector[biomass>0]),axis=1 )
    SimpsonsDom[biomass>0]=1./np.sum( np.square(p_vector[biomass>0]),axis=1 )
    SimpsonsE[biomass>0]=SimpsonsDom[biomass>0]/S_in_IC[biomass>0]
    return SimpsonsDiv, SimpsonsDom, SimpsonsE

def Shannon_diversity(N, sigma_vectors, S, n_exp): 
    ## calculates the Shannon  diversity
#    N[N<popn_cutoff]=0.0
    biomass=total_abundance(N)
    p_vector=np.zeros(N.shape)
    ShannonDiv=np.zeros(n_exp)
    p_vector[biomass>0]=np.apply_along_axis(np.divide, 0, N[biomass>0], biomass[biomass>0])
    for i in range(n_exp):
        p_i=p_vector[i][np.nonzero(p_vector[i])]
        if len( p_i )>1:
            ShannonDiv[i]=-np.sum( p_i *np.log(p_i) )
    
    normalized_ShannonDiv=ShannonDiv/np.log(S)
    return ShannonDiv, normalized_ShannonDiv

def Fourier_roughness (fitness, sigma_vectors, S, n_exp ):
    sigma_tilde=2*sigma_vectors-1
    expansionOrder=S
    Power_spectrum=np.zeros(expansionOrder+1)
    Fourier_coefficients=[]    
    
    if S<=12: ### we can do the numpy functin method of calculation Pspec and Fourier coefficients
        '''
        poly_sigma is an array of shape (2^S X 2^S), hence for S>=16, system might not have enough memory to save the array
        So a different method of calculating fourier roughness is required for S>15; this needs to be written manually if required.
        '''     
        poly = PolynomialFeatures(expansionOrder,interaction_only=True)
        poly_sigma=poly.fit_transform(sigma_tilde)
        '''
        poly = PolynomialFeatures(3) on array [a, b, c]
        produces [1, a, b, c, a^2, b^2, c^2, ab, bc, ca, a^3, b^3, c^3, a^2b, ....] which has a lot of redundant terms
        poly = PolynomialFeatures(interaction_only=True) produces only unique combinations and not permutations, i.e., on array [a, b, c]
        produces [1, a, b, c, ab, bc, ca, abc]  ie., 2^3= 8 elements which is what we want!
        method  in the function "def transform(self, X):" is at:   
        https://github.com/scikit-learn/scikit-learn/blob/b194674c4/sklearn/preprocessing/_data.py#L1516
        '''            
        for i in range(len(poly_sigma[0])):
            Fourier_coefficients.append( np.dot(fitness,poly_sigma[:,i])/n_exp  )
        Fourier_coefficients=np.array(Fourier_coefficients)
        beg=0
        end=0
        for i in range(S+1):
            beg=int(end)
            end=int(end+comb(S,i))
    #        print(i,beg,end)
            Power_spectrum[i]=np.sum(np.square(Fourier_coefficients[beg:end]))
            if end>len(Fourier_coefficients):
                print ('all coefficients werent estimated',i, end,len(Fourier_coefficients))
                break
    
    elif S>12: ### we do the manual way which uses less memory. We do not save Fourier coefficients, only Pspec.
        a= np.arange(S)
        Fourier_coefficients='array is too large!'
        for k in range(0,expansionOrder+1):
            if k==0:
                Fi=np.mean(fitness)              
                Power_spectrum[k]=float(Fi)**2 
#                Fourier_coefficients.append(float(Fi))               
            else:
                for combination in itertools.combinations(a,k):                       
                    if k==1:
                        XP=sigma_tilde[:,combination]   ## Selectes a particular column, indexed by combination             
                    else:
                       XP=np.prod(sigma_tilde[:,combination],axis=1 )   ## does a product over all the columns indexed by combination                       
                    
                    Fi=np.dot(fitness,XP)*1./n_exp                
                    Power_spectrum[k]=Power_spectrum[k] + float(Fi)**2                    

                  
    PS_ratio = np.sum(Power_spectrum[2:])/np.sum(Power_spectrum[1:])
    beta_coeffs=Power_spectrum[1:]/np.sum(Power_spectrum[1:])
   
    return  Fourier_coefficients, Power_spectrum,  PS_ratio , beta_coeffs



def r_over_s(fitness, sigma_vectors, remove0=False):
    if remove0:
        fitness=remove_zero_data_point(sigma_vectors, fitness=fitness)
        sigma_vectors=remove_zero_data_point(sigma_vectors)
    
    
    linReg_object = LinearRegression()
    linReg_object.fit(sigma_vectors,fitness)
#    print ("r/s coeffs & intercept ",linReg_object.coef_, linReg_object.intercept_)
    coefs=linReg_object.coef_
    coefs=np.append(coefs,linReg_object.intercept_) ## coefs[-1] is the mean.
    s_slope=np.mean( np.abs(coefs[:-1])  ) ##s should not include the mean!
    r_roughness=np.std(fitness-linReg_object.predict(sigma_vectors))
    
    return r_roughness,s_slope

def generate_adjacency_matrix(sigma_vectors, S, n_exp ):
    adjacency_matrix=np.zeros((n_exp,S)).astype(int)
    for i in range(n_exp):
        ref_vec=sigma_vectors[i]      
#       np.apply along axis is very slow, we do not need it here though cos of array broadcasting!                  
#        difference_between_ICs=np.apply_along_axis(np.subtract, 1, sigma_vectors, ref_vec)
#        diff_magnitude=np.sum(np.abs(difference_between_ICs),axis=1)        
        diff_magnitude=np.sum(np.abs(sigma_vectors- ref_vec),axis=1)
        
        adjacency_matrix[i]=np.where(diff_magnitude==1)[0]
    return adjacency_matrix
def generate_adjacency_list_from_Subsampling(sigma_vectors ):
    adjacency_list=[]
    for i in range(len(sigma_vectors)):
        ref_vec=sigma_vectors[i]         
        diff_magnitude=np.sum(np.abs(sigma_vectors- ref_vec),axis=1)
        
        adjacency_list.append(np.where(diff_magnitude==1)[0])
    return adjacency_list


def check_if_a_nearest_neighbour_is_identical_PA(sigmaF_i, sigmaF_nn, dummy):
          
    diff_bw_sigma=sigmaF_nn-sigmaF_i    
    if np.any(np.sum(np.abs(diff_bw_sigma),axis=1)==0): # two of the final states had the same species present
        return True
    else:
        return False
    
    
def check_if_IC_and_SS_have_identical_PA(sigma_i, sigmaF_i):
                
    if np.array_equal(sigma_i,sigmaF_i ): 
        return True
    else:
        return False 
    
    
def count_invadable_directions(sigmaF_i, sigmaF_nn):
    diff_bw_sigma=sigmaF_nn-sigmaF_i     
    if np.any(diff_bw_sigma==1): ## that means in a nearest neighbour, a new species was present in the final state.
        ctr=0
        for j in range(len(sigmaF_nn)): 
            if np.any(diff_bw_sigma[j]==1):
                ctr+=1
        return ctr
    
    else:
        return 0  
 
    
    

    
    
    
def spatial_correlation(fitness, sigma_vectors, adjacency_matrix ):   
    
    '''
    will accept weirdly shaped adjacency matrix to account for scenario where we have some sampled landscape with each point
    having different numbers of neighbors
    '''
    avg_fitness=np.mean(fitness)
    var_fitness=np.var(fitness)
    if var_fitness< machine_precision: ## all the points are basically the same. perfect correlation.
        nn_correlation=1.
        corr_length_nn=np.NaN
        return nn_correlation, corr_length_nn
        
    link_ctr=0 
    cum_sum=0.
    for i in range(len(fitness)):
        cum_sum+=(fitness[i]-avg_fitness) * np.sum(fitness[adjacency_matrix[i]]-avg_fitness)
        link_ctr+=len(adjacency_matrix[i])
    nn_correlation= (cum_sum*1./link_ctr)/var_fitness
    corr_length_nn=-1./np.log(nn_correlation) ## estimate of corr. length from nearest neighbor data, more sophisticated ways requires correlation at different distances.
    return nn_correlation, corr_length_nn


def spatial_correlation_biased_sample(fitness, sigma_vectors, adjacency_matrix ):       
    '''
    If the sample is biased such that a few point have many nearest neighbors while
    other points have very few, it is better to calculate Znn by computing moments of 
    F(sigma) and F(sigma_nn) separately since they can differ by a lot.
    
    However, this would still give you a biased estimate answer 
    as data points with more nearest neighbors contribute more to Znn    
    '''
    avg_fitness=np.mean(fitness)
    var_fitness=np.var(fitness)
    fitness_nn= []  
    for i in range(len(fitness)):
        fitness_nn.extend(fitness[adjacency_matrix[i]])        ##fitness of nearest neighbors
    fitness_nn=np.ravel(fitness_nn)
    avg_fitness_nn=np.mean(fitness_nn)
    var_fitness_nn=np.var(fitness_nn)
    if var_fitness< machine_precision: ## all the points are basically the same. perfect correlation.
        nn_correlation=1.
        corr_length_nn=np.NaN
        return nn_correlation, corr_length_nn      
    link_ctr=0 
    cum_sum=0.
    for i in range(len(fitness)):
        cum_sum+=(fitness[i]-avg_fitness) * np.sum(fitness[adjacency_matrix[i]]-avg_fitness_nn)
        link_ctr+=len(adjacency_matrix[i])
    nn_correlation= (cum_sum*1./link_ctr)/ (np.sqrt(var_fitness)*np.sqrt(var_fitness_nn))
    corr_length_nn=-1./np.log(nn_correlation) ## estimate of corr. length from nearest neighbor data, more sophisticated ways requires correlation at different distances.
    return nn_correlation, corr_length_nn





def F_Neutral_Links(fitness, sigma_vectors, adjacency_matrix, rel_threshold=1e-4):
    '''
    When fitnesses are almost equal but numerical precision makes them different, we need to identify these as flat instead of greater or lesser.    
    we can compute number of flat edges in the graph along with maxima and minima at vertices..
    '''     
    total_links=0
    neutral_links=0
    up_links=0
    down_links=0### number of uplinks= number of down links, and total links =sum of all three.

    '''
    neutral links depends only on fitness difference. It does not check for sigma vectors in the final state
    Therefore it is a function defined purely on the landscape without any additional information.
    '''

    for i in range(len(fitness)):     
        
        ####### compute fitness difference between point and nearest neighbours  ####### 
        if np.abs(fitness[i])>=machine_precision : ## if fitness!=0.0
            relative_diff= (fitness[i]- fitness[adjacency_matrix[i]] )/ np.abs(fitness[i]) ## abs to find maxima correctly in case fitness is actually negative,
        else: ## if fitness ==0.0, rel_diff is simply +-1. if identical then 0
            relative_diff=2.*(fitness[i]> fitness[adjacency_matrix[i]] )-1.  
            relative_diff[fitness[i]== fitness[adjacency_matrix[i]]]=0.0
        
        total_links+=len(relative_diff)
        down_links+=np.sum(relative_diff>=rel_threshold)
        up_links+=np.sum(relative_diff<=-rel_threshold)
        neutral_links+=np.sum(np.abs(relative_diff)<=rel_threshold)

    links_dict={'total_links':total_links/2,'f_neutral_links':neutral_links*1./total_links,
                'f_up_links':up_links*1./total_links,'f_down_links':down_links*1./total_links}
    return links_dict

def find_extrema(fitness, sigma_vectors, sigma_vectors_final, adjacency_matrix, S, n_exp, rel_threshold=1e-4 ):
    '''
    When fitnesses are almost equal but numerical precision makes them different, we need to identify these as flat instead of greater or lesser.    
    we can compute number of flat edges in the graph along with maxima and minima at vertices..
    '''    
    maxima_list=[]
    minima_list=[]
    pure_saddle_list=[]

    maxima_without_extinction=[] ## the species in the end is the same as the species in the beginning
    minima_without_extinction=[] ## and hence there is only one of this in a set of degenerate maxima 
    nsp_in_maxima_without_extinction=[]
    nsp_in_minima_without_extinction=[]
    N_invadable_directions_max_no_ext=[]
    N_invadable_directions_min_no_ext=[]   
    
    
    total_links=0
    neutral_links=0
    up_links=0
    down_links=0### number of uplinks= number of down links, and total links =sum of all three.

    '''
    neutral links depends only on fitness difference. It does not check for sigma vectors in the final state
    Therefore it is a function defined purely on the landscape without any additional information.
    '''
    for i in range(n_exp):        
        ####### compute fitness difference between point and nearest neighbours  ####### 
        if np.abs(fitness[i])>=machine_precision : ## if fitness!=0.0
            relative_diff= (fitness[i]- fitness[adjacency_matrix[i]] )/ np.abs(fitness[i]) ## abs to find maxima correctly in case fitness is actually negative,
        else: ## if fitness ==0.0, rel_diff is simply +-1. if identical then 0
            relative_diff=2.*(fitness[i]> fitness[adjacency_matrix[i]] )-1.  
            relative_diff[fitness[i]== fitness[adjacency_matrix[i]]]=0.0
        
        total_links+=len(relative_diff)
        down_links+=np.sum(relative_diff>=rel_threshold)
        up_links+=np.sum(relative_diff<=-rel_threshold)
        neutral_links+=np.sum(np.abs(relative_diff)<=rel_threshold)
#        assert total_links==neutral_links+up_links+down_links, print ('link numbers dont add up?',total_links, neutral_links,up_links,down_links,
#                                                                    '\n rel_diff was: ',relative_diff,'\n and counts were:  ',  np.sum(relative_diff>=rel_threshold) ,
#                                                                    np.sum(relative_diff<=-rel_threshold),np.sum(np.abs(relative_diff)<=rel_threshold)  )


        #######     see if the point is a maxima    ####### 
        if np.all( relative_diff>= -rel_threshold   ) : ## counted as flat if very slightly negative
            if np.any( relative_diff>= rel_threshold ):
                maxima_list.append(i)
                if check_if_IC_and_SS_have_identical_PA(sigma_vectors[i], sigma_vectors_final[i]):
                    maxima_without_extinction.append(i)
                    nsp_in_maxima_without_extinction.append(np.sum(sigma_vectors[i]))
                    N_invadable_directions_max_no_ext.append( count_invadable_directions(sigma_vectors_final[i], sigma_vectors_final[adjacency_matrix[i]])  )               
#                if check_if_a_nearest_neighbour_is_identical_PA(sigma_vectors_final[i], sigma_vectors_final[adjacency_matrix[i]], popn_cutoff ): 
#                    maxima_identical_to_nn.append(i)
#                if np.all( relative_diff>= rel_threshold ):
#                    absolute_maxima.append(i) 
            else:pure_saddle_list.append(i)    ### only flat directions exist                                               
        
        #######     see if the point is a minima    #######                                  
        elif np.all( relative_diff<= rel_threshold   ) :
            if np.any( relative_diff<= -rel_threshold ):
                minima_list.append(i) 
                if check_if_IC_and_SS_have_identical_PA(sigma_vectors[i], sigma_vectors_final[i]):
                    minima_without_extinction.append(i)
                    nsp_in_minima_without_extinction.append(np.sum(sigma_vectors[i]))
                    N_invadable_directions_min_no_ext.append( count_invadable_directions(sigma_vectors_final[i], sigma_vectors_final[adjacency_matrix[i]])  )                   
            else:pure_saddle_list.append(i)                                
    
    
    
    
    
    links_dict={'total_links':total_links/2,'f_neutral_links':neutral_links*1./total_links,
                'f_up_links':up_links*1./total_links,'f_down_links':down_links*1./total_links}
    
    
#    if len(maxima_without_extinction)==0:
#        print ("no maxima without extinction !? \n")
#            if os.getcwd().startswith("/Users/ashish"):
#                input("Press Enter to continue...")
                                
    return np.asarray(maxima_list), np.asarray(minima_list), np.asarray(maxima_without_extinction), np.asarray(minima_without_extinction), np.asarray(nsp_in_maxima_without_extinction), np.asarray(nsp_in_minima_without_extinction), np.asarray(N_invadable_directions_max_no_ext), np.asarray(N_invadable_directions_min_no_ext),  np.asarray(pure_saddle_list), links_dict


def fitness_distance_correlation(fitness, sigma_vectors, S, n_exp, maxima_without_extinction=None, global_maxima_loc=None):
    '''
    as defined in Eq.10 of  [A Comparison of Predictive Measures of Problem Difficulty in Evolutionary Algorithms Naudt, Kallel]
    '''        
    if global_maxima_loc==None: # if location of global maximum is not provided, we have to find it.           
        if len(maxima_without_extinction)==0:
            print ('FDC doesnt make sense.')
            FDC=None
            return FDC
        global_maxima_loc=maxima_without_extinction[np.argmax(fitness[maxima_without_extinction])]    
    
    def distance_from_max(sigma):
        return( np.abs(sigma-sigma_vectors[global_maxima_loc]).sum()  )
    distance_vectors=np.apply_along_axis(distance_from_max, 1, sigma_vectors)    
    FDC_num= np.sum(   (fitness-np.mean(fitness)) * ( distance_vectors-np.mean(distance_vectors) )  ) /n_exp
    FDC_denom=np.std(fitness)*np.std(distance_vectors) 
    FDC=FDC_num/FDC_denom   
    return FDC


def ranked_fitness_distance_correlation(fitness, sigma_vectors, S, n_exp, rel_threshold=1e-4):
    sorted_idx=np.argsort(fitness)  ## we want highest rank for highest fitness for this to resemble FDC
    sorted_fitness=fitness[sorted_idx]
    sorted_sigma_vectors=sigma_vectors[sorted_idx]
    ranked_fitness=1.*np.ones(len(sorted_fitness))
    fitness_diff=np.diff(sorted_fitness)

    rel_fitness_diff=np.empty_like(fitness_diff)
    
    if np.any(np.abs(sorted_fitness[1:])<machine_precision):
        idx_nonzero=np.where(np.abs(sorted_fitness[1:])>machine_precision)
        idx_zero=np.where(np.abs(sorted_fitness[1:])<machine_precision)
        rel_fitness_diff[idx_nonzero]=fitness_diff[idx_nonzero]/(sorted_fitness[1:])[idx_nonzero]
        rel_fitness_diff[idx_zero]=np.sign(fitness_diff[idx_zero])

    else:
       
        rel_fitness_diff=fitness_diff/sorted_fitness[1:]

    
    '''
    we assign fitnesses that are almost equal the average rank using the procedure below
    '''
    current_rank=1
    last_rank=0
    for i in range(len(rel_fitness_diff)+1):## rel_diff is shorter than sorted_diff
        if rel_fitness_diff[min(i,len(rel_fitness_diff)-1)]>rel_threshold: ## we need to use the last rel. difference twice, to evaluate penultimate and last ranks
            drank=current_rank-last_rank
            if drank==1:           
                ranked_fitness[i]=current_rank
                last_rank=current_rank
                
            else:  
                ranked_fitness[i-drank+1:i+1]= (last_rank+1.+ current_rank)*1./2.                 
                last_rank=current_rank
                                
        elif i==len(rel_fitness_diff): ## then we need to assign ranks anyway
            drank=current_rank-last_rank
#            print ('last i', last_rank, current_rank, drank)
            if drank==1:   
                ranked_fitness[i]=current_rank
            else:
                ranked_fitness[i-drank+1:i+1]= (last_rank+1.+ current_rank)*1./2. 
                   
        current_rank+=1  
          
#    print (last_rank, current_rank)
#    if rel_fitness_diff[-1]>rel_threshold: 
#        drank=current_rank-last_rank
#    
#    if rel_fitness_diff[-1]>rel_threshold:
#        ranked_fitness[-1]=len(ranked_fitness)
#    else:
#        ranked_fitness[-1]=ranked_fitness[-2] ## a very small error here, shouldnt really make a difference.

    rankedFDC= fitness_distance_correlation(ranked_fitness, sorted_sigma_vectors, S, n_exp,  global_maxima_loc=len(ranked_fitness)-1)
    

    return rankedFDC 
    

def average_pairwise_distance(sigma_vec):
    distance=0
    for sigma in sigma_vec:
        distance+= np.abs(sigma_vec-sigma).sum()
    distance =distance*1./len(sigma_vec)**2

    return distance

def sitewise_optimizability(fitness, sigma_vectors, S, n_exp, adjacency_matrix):
    '''
    as defined in Eq.14 of  [A Comparison of Predictive Measures of Problem Difficulty in Evolutionary Algorithms Naudt, Kallel]
    '''
    n_exp=len(sigma_vectors)
    sigma_final_swo=np.zeros_like(sigma_vectors)
    for i in range(n_exp):
        sigma_f=deepcopy(sigma_vectors[i])
        NN=adjacency_matrix[i]
        delta_fitness= fitness[NN]-fitness[i]       
        
        if np.any(delta_fitness>0):
            idx_to_flip=np.abs(sigma_vectors[NN[np.where(delta_fitness>0)[0]]]-sigma_vectors[i]).sum(axis=0).astype(bool)
#            print(idx_to_flip)
            sigma_f[idx_to_flip]=1-sigma_f[idx_to_flip]
# =============================================================================
#         if np.any(delta_fitness==0):
#             addressing delta_fitness=0 by randomly choosing 0 and 1 is not really necessary for a binary alphabet that is fully sampled.           
# =============================================================================            
        sigma_final_swo[i]=deepcopy(sigma_f)


    denom_dist=average_pairwise_distance(sigma_vectors)
    num_dist=average_pairwise_distance(sigma_final_swo)
    print ("SWO: distance init", denom_dist, "distance final", num_dist )
    SW0_ratio=num_dist/denom_dist
    return SW0_ratio, sigma_final_swo         
    



def do_random_walk(Nsteps, fitness, adjacency_matrix, S, n_exp, Initial_node_list):
    ### performs a random walk and returns fitness trajectory   
    fitness_trajectory=np.zeros(( len(Initial_node_list), Nsteps))
    node_trajectory=np.zeros(( len(Initial_node_list), Nsteps))
    
    for idx,Init_node in enumerate(Initial_node_list):
        node=Init_node
        for t in range(Nsteps):           
            fitness_trajectory[idx,t]=fitness[node]
            node_trajectory[idx,t]=node
            node=adjacency_matrix[node][np.random.choice(S)]        
    return fitness_trajectory,node_trajectory

def random_walk_performance(fitness, adjacency_matrix, S, n_exp, Nsteps=100, n_init_nodes=1 ):

    Initial_node_list=np.random.choice(np.arange(n_exp), size=n_init_nodes, replace=False )   
    fitness_trajectory,_=do_random_walk(Nsteps, fitness, adjacency_matrix, S, n_exp, Initial_node_list)
        
    def normalized_auto_correlate(x):  #### calculate autocorrelation,divide by number of data points that contributed to  each step
       return np.correlate(x, x, mode = 'full')[-len(x):] / ( np.arange(1+len(x))[::-1][:-1] ) 

    
    def mean_plus_s_with_overcounting(x): ## ftplus s is biased to the later values, so this mean counts in the appropriate fashion
        freq_bias=np.arange(len(x))+1.
        return freq_bias.dot(x)/np.sum(freq_bias)
        
    #### calculate autocorrelation, take mean across ensembles, 
    ft_ft_plus_s= np.mean( list(map(normalized_auto_correlate, fitness_trajectory)), axis=0   )  #/ ( np.arange(Nsteps)[::-1][:-1] )
    mean_ft=np.mean(fitness_trajectory )    
    #mean_ft_plus_s=np.mean( list(map( mean_t_plus_s, fitness_trajectory, np.arange(Nsteps) )), axis=0   ) [:-1]   
    var_ft=np.mean( list(map(np.var, fitness_trajectory))  )    
    Weinbergers_AutoCorrelation= (ft_ft_plus_s-mean_ft*mean_ft)/  var_ft     
#    print ('autocorr',Weinbergers_AutoCorrelation[:10])
    
######## method 2 averaged weinberger's correlation across ensembles instead of averaging each term, was more error prone    
#    ft_ft_plus_s= np.asarray( list(map(normalized_auto_correlate, fitness_trajectory))) 
#    mean_alongT_ft=np.mean(fitness_trajectory,axis=1 )    
#    var_alongT_ft=np.asarray( list(map(np.var, fitness_trajectory))  )    
#    def a_minus_b_dividedby_c(a,b,c):
#        return (a-b)/c    
#    Weinbergers_AutoCorrelation= np.mean(  list(map(a_minus_b_dividedby_c , ft_ft_plus_s, np.square(mean_alongT_ft), var_alongT_ft)),   axis=0)
#     print ('method 2',Weinbergers_AutoCorrelation[:10])

    return Weinbergers_AutoCorrelation
    
    



   

def do_greediest_walk(fitness, adjacency_matrix,S, n_exp, Initial_node_list, max_fitness, global_maxima_noE, rel_threshold=1e-4):
    ### this performs the greediest walk where all neighbors are evaluated and the best one is selected.
    Greediest_walk_found_global_max_list=[]
    Greediest_walk_found_global_max_noE_list=[]
    Greediest_walk_steps_list=[]
    Greediest_walk_evaluations_list=[]
    Greediest_walk_relative_optimum_achieved=[]

    for Init_node in Initial_node_list:
        reached_a_maxima=False
        num_steps=0
        node=Init_node
#        print ("starting for node", node)
        while (reached_a_maxima==False):
            current_fitness=fitness[node]
            neighbors_fitness=fitness[adjacency_matrix[node]]                
            relative_diff= (neighbors_fitness-current_fitness  )/ np.abs(current_fitness+machine_precision)    
            
            if np.all( relative_diff<= rel_threshold   ) : ## we reached a local maxima
                reached_a_maxima=True
                relative_diff= (max_fitness-current_fitness)/np.abs(current_fitness+machine_precision)
                relative_optimum_achieved=current_fitness/(max_fitness+machine_precision) ## if fitness is all negative, this measure is weird...
                Greediest_walk_relative_optimum_achieved.append(relative_optimum_achieved)
                if  relative_diff <=rel_threshold : # we reached the global maxima                      
                    Greediest_walk_found_global_max_list.append(1)
                    if node==global_maxima_noE: 
                        Greediest_walk_found_global_max_noE_list.append(1)
                    else:
                        Greediest_walk_found_global_max_noE_list.append(0)
                else:
                    Greediest_walk_found_global_max_list.append(0)
                Greediest_walk_steps_list.append(num_steps)
                Greediest_walk_evaluations_list.append( (S-1)*num_steps+1  ) # each step except first, we dont need to evaluate the back step             
            else:
                num_steps+=1
                node=adjacency_matrix[node][np.argmax(neighbors_fitness)]
                
    return np.asarray(Greediest_walk_found_global_max_list), np.asarray(Greediest_walk_found_global_max_noE_list), np.asarray(Greediest_walk_steps_list), np.asarray(Greediest_walk_evaluations_list), np.asarray(Greediest_walk_relative_optimum_achieved)

def do_greedy_walk(fitness, adjacency_matrix,S, n_exp, Initial_node_list, max_fitness, global_maxima_noE, rel_threshold=1e-4):  
    ### this performs the greedy walk walk where neighbors are evaluated one by one, and the first improvement is selected.               
    Greedy_walk_found_global_max_list=[]
    Greedy_walk_found_global_max_noE_list=[]
    Greedy_walk_steps_list=[]
    Greedy_walk_evaluations_list=[]
    Greedy_walk_relative_optimum_achieved=[]
    
    for Init_node in Initial_node_list:
        reached_a_maxima=False
        num_steps=0
        num_evaluations=0
        node=Init_node
        prev_node=node
#        print ("starting for node", node)
        while (reached_a_maxima==False):
            current_fitness=fitness[node]
            #print (num_steps,max_fitness-current_fitness, (max_fitness-current_fitness)/(current_fitness+machine_precision) )
#            print(current_fitness)
            neighbors_fitness=fitness[adjacency_matrix[node]]  
            relative_diff= (neighbors_fitness -current_fitness)/ np.abs(current_fitness+machine_precision)   

            if np.all( relative_diff<= rel_threshold   ) : ## we reached a local maxima
                reached_a_maxima=True
                relative_diff= (max_fitness-current_fitness)/np.abs(current_fitness+machine_precision) 
                relative_optimum_achieved=current_fitness/(max_fitness+machine_precision)
                Greedy_walk_relative_optimum_achieved.append(relative_optimum_achieved)
                if  relative_diff <=rel_threshold : # we reached the global maxima                      
                    Greedy_walk_found_global_max_list.append(1)
 
                    if node==global_maxima_noE: 
                        Greedy_walk_found_global_max_noE_list.append(1)
                    else:
                        Greedy_walk_found_global_max_noE_list.append(0)
                else:
                    Greedy_walk_found_global_max_list.append(0)
                    
                Greedy_walk_steps_list.append(num_steps)
                if num_evaluations>0:
                    Greedy_walk_evaluations_list.append( num_evaluations+S-1 )
                else:
                    Greedy_walk_evaluations_list.append( num_evaluations+S )
            else:
                num_steps+=1
                order_array=np.arange(S)
                np.random.shuffle( order_array )  
                
                for i in order_array:
                    new_node=adjacency_matrix[node][i]                    
                    if new_node!=prev_node: ## we evaluate it, else we don't evaluate it.       
                        new_fitness=fitness[new_node]
                        num_evaluations+=1                       
                        relative_diff= (new_fitness-current_fitness)/np.abs(current_fitness+machine_precision)                 
                        if  relative_diff >rel_threshold : # we travel to this node
                            prev_node=node
                            node=new_node
                            break
    return np.asarray(Greedy_walk_found_global_max_list), np.asarray(Greedy_walk_found_global_max_noE_list), np.asarray(Greedy_walk_steps_list), np.asarray(Greedy_walk_evaluations_list), np.asarray(Greedy_walk_relative_optimum_achieved)
       
    #return Greedy_walk_found_global_max_list, Greedy_walk_found_global_max_noE_list, Greedy_walk_steps_list, Greedy_walk_evaluations_list

def performance_of_greedy_walk(fitness, sigma_vectors, adjacency_matrix, S, n_exp, maxima_without_extinction, rel_threshold=1e-4 ):
#    if np.any(fitness<0):
#        print ('greedy walk cannot handle negative fitnesses at the moment.')
    
    '''
    with noisy measurements maxima need not be maxima without extinction, hence commenting out below
    and using a new max and defn.
    '''
    max_fitness=np.max(fitness)
    global_maxima_noE=np.argmax(fitness)     
    if len(maxima_without_extinction)>=1:
        if np.max(fitness[maxima_without_extinction])>=max_fitness*(1-rel_threshold):
            global_maxima_noE=maxima_without_extinction[np.argmax(fitness[maxima_without_extinction])]
   
             
# =============================================================================
#     if len(maxima_without_extinction)>=1:
#         max_fitness=np.max(fitness[maxima_without_extinction])
#         if max_fitness<=np.max(fitness)*(1-rel_threshold):
#           print ( 'the max fitness is not in the without extinction list...?',S)
#           print (max_fitness,np.max(fitness),np.max(fitness)*(1-rel_threshold) )         
#           results_dict={ 'Greedy_walk_Failed_Flag':True}
#           return results_dict 
#         global_maxima_noE=maxima_without_extinction[np.argmax(fitness[maxima_without_extinction])]
#         if np.count_nonzero(fitness[maxima_without_extinction]-max_fitness)!=len(maxima_without_extinction)-1:
#             print("non-unique global maxima")
#             #print (np.count_nonzero(fitness[maxima_without_extinction]-max_fitness), fitness[maxima_without_extinction])
#             results_dict={'Greedy_walk_Failed_Flag':True}
#             return results_dict
#     else:   
#         results_dict={'Greedy_walk_Failed_Flag':True}
#         return results_dict
# =============================================================================
       
    #print ('fitness global maxima is', max_fitness,'no: max no E',len(maxima_without_extinction) )

    
    
    
    ##### starting from zero species,1 species, all species and half species (typical) point######
    num_initial_species=np.sum(sigma_vectors, axis=1)
    idx_zero_sp=int(np.where(num_initial_species==0)[0])
    idx_one_sp=int(np.random.choice( np.where(num_initial_species==1)[0]   ))
    idx_all_sp=int(np.where(num_initial_species==S)[0])
    idx_typical_sp=int(np.random.choice( np.where(num_initial_species==int(S/2))[0]   ))
    
    _, _, Greediest_walk_steps_zero_sp, _, Greediest_walk_relative_optimum_achieved_zero_sp=do_greediest_walk(fitness, adjacency_matrix, S, n_exp, [idx_zero_sp], max_fitness, global_maxima_noE, rel_threshold=1e-4)
    _, _, Greediest_walk_steps_one_sp, _, Greediest_walk_relative_optimum_achieved_one_sp=do_greediest_walk(fitness, adjacency_matrix, S, n_exp, [idx_one_sp], max_fitness, global_maxima_noE, rel_threshold=1e-4)
    _, _, Greediest_walk_steps_all_sp, _, Greediest_walk_relative_optimum_achieved_all_sp=do_greediest_walk(fitness, adjacency_matrix, S, n_exp, [idx_all_sp], max_fitness, global_maxima_noE, rel_threshold=1e-4)
    _, _, Greediest_walk_steps_typical_sp, _, Greediest_walk_relative_optimum_achieved_typical_sp=do_greediest_walk(fitness, adjacency_matrix, S, n_exp, [idx_typical_sp], max_fitness, global_maxima_noE, rel_threshold=1e-4)
    
    
    ##### starting from all points.######
    Initial_node_arr=np.arange(n_exp) 
    Greediest_walk_found_global_max_arr, Greediest_walk_found_global_max_noE_arr, Greediest_walk_steps_arr, Greediest_walk_evaluations_arr, Greediest_walk_relative_optimum_achieved_arr=do_greediest_walk(fitness, adjacency_matrix, S, n_exp, Initial_node_arr, max_fitness, global_maxima_noE, rel_threshold=1e-4)
    Greedy_walk_found_global_max_arr, Greedy_walk_found_global_max_noE_arr, Greedy_walk_steps_arr, Greedy_walk_evaluations_arr, Greedy_walk_relative_optimum_achieved_arr=do_greedy_walk(fitness, adjacency_matrix, S, n_exp, Initial_node_arr, max_fitness, global_maxima_noE, rel_threshold=1e-4)
    
    
    p_Greediest_walk_found_global_max=np.sum(Greediest_walk_found_global_max_arr)*1./len(Greediest_walk_found_global_max_arr)
    p_Greedy_walk_found_global_max=np.sum(Greedy_walk_found_global_max_arr)*1./len(Greedy_walk_found_global_max_arr)
    mean_steps_Greediest=np.mean(Greediest_walk_steps_arr)
    mean_evals_Greediest=np.mean(Greediest_walk_steps_arr)
    if np.any(Greediest_walk_found_global_max_arr==1):
        mean_steps_Greediest_global=np.mean(Greediest_walk_steps_arr[np.argwhere(Greediest_walk_found_global_max_arr==1)[0]])
        mean_evals_Greediest_global=np.mean(Greediest_walk_steps_arr[np.argwhere(Greediest_walk_found_global_max_arr==1)[0]])   

    else:
        mean_steps_Greediest_global=-1
        mean_evals_Greediest_global=-1

    
    mean_steps_Greedy=np.mean(Greedy_walk_steps_arr)
    mean_evals_Greedy=np.mean(Greedy_walk_steps_arr)
    
    if np.any(Greedy_walk_found_global_max_arr==1):
        mean_steps_Greedy_global=np.mean(Greedy_walk_steps_arr[np.argwhere(Greedy_walk_found_global_max_arr==1)[0]])    
        mean_evals_Greedy_global=np.mean(Greedy_walk_steps_arr[np.argwhere(Greedy_walk_found_global_max_arr==1)[0]])
    else:
        mean_steps_Greedy_global=-1  
        mean_evals_Greedy_global=-1
    mean_relative_optimum_Greedy=round(np.mean(Greedy_walk_relative_optimum_achieved_arr), 4)
    mean_relative_optimum_Greediest=round(np.mean(Greediest_walk_relative_optimum_achieved_arr), 4)
    
    var_relative_optimum_Greedy=round(np.var(Greedy_walk_relative_optimum_achieved_arr), 4)
    var_relative_optimum_Greediest=round(np.var(Greediest_walk_relative_optimum_achieved_arr), 4)
    
#    print ("p's are ",p_Greediest_walk_found_global_max,p_Greediest_walk_found_global_max)
#    print ("mean_steps are ",mean_steps_Greediest_global,mean_steps_Greediest)
#    print ("mean_steps are ",mean_steps_Greedy_global,mean_steps_Greedy)
    print ("mean rel optimums is",mean_relative_optimum_Greediest,mean_relative_optimum_Greedy)
    print ("rel optimums from 0,1, all and tpyical are: ",Greediest_walk_relative_optimum_achieved_zero_sp,Greediest_walk_relative_optimum_achieved_one_sp, 
           Greediest_walk_relative_optimum_achieved_all_sp, Greediest_walk_relative_optimum_achieved_typical_sp)
    
    print ("walk steps from 0,1, all and tpyical are: ",Greediest_walk_steps_zero_sp,Greediest_walk_steps_one_sp, Greediest_walk_steps_all_sp,Greediest_walk_steps_typical_sp)
    
    mean_relative_function=np.mean(fitness)*1./max_fitness
    results_dict={
                  'Greedy_walk_Failed_Flag':False,
                  'mean_relative_function':mean_relative_function,
                  
                  'p_Greediest_walk_found_global_max':p_Greediest_walk_found_global_max,
                  'mean_steps_Greediest':mean_steps_Greediest,'mean_steps_Greediest_global':mean_steps_Greediest_global,
                  'mean_evals_Greediest':mean_evals_Greediest,'mean_evals_Greediest_global':mean_evals_Greediest_global,
                  'Greediest_walk_found_global_max_arr':Greediest_walk_found_global_max_arr, 'Greediest_walk_found_global_max_noE_arr': Greediest_walk_found_global_max_noE_arr,
                  'Greediest_walk_steps_arr':Greediest_walk_steps_arr, 'Greediest_walk_evaluations_arr':Greediest_walk_evaluations_arr,
                  'mean_relative_optimum_Greediest':mean_relative_optimum_Greediest,'var_relative_optimum_Greediest':var_relative_optimum_Greediest,

                  
                  'p_Greedy_walk_found_global_max':p_Greedy_walk_found_global_max,
                  'mean_steps_Greedy':mean_steps_Greedy,'mean_steps_Greedy_global':mean_steps_Greedy_global,
                  'mean_evals_Greedy':mean_evals_Greedy,'mean_evals_Greedy_global':mean_evals_Greedy_global,
                  'Greedy_walk_found_global_max_arr':Greedy_walk_found_global_max_arr, 'Greedy_walk_found_global_max_noE_arr': Greedy_walk_found_global_max_noE_arr,
                  'Greedy_walk_steps_arr':Greedy_walk_steps_arr, 'Greedy_walk_evaluations_arr':Greedy_walk_evaluations_arr,
                  'mean_relative_optimum_Greedy':mean_relative_optimum_Greedy,'var_relative_optimum_Greedy':var_relative_optimum_Greedy,
                  
                  'relative_optimum_Greediest_zero_sp':round(float(Greediest_walk_relative_optimum_achieved_zero_sp),4),
                  'Greediest_walk_steps_zero_sp':round(float(Greediest_walk_steps_zero_sp),4),
                  'relative_optimum_Greediest_one_sp':round(float(Greediest_walk_relative_optimum_achieved_one_sp),4),
                  'Greediest_walk_steps_one_sp':round(float(Greediest_walk_steps_one_sp),4),
                  'relative_optimum_Greediest_all_sp':round(float(Greediest_walk_relative_optimum_achieved_all_sp),4),
                  'Greediest_walk_steps_all_sp':round(float(Greediest_walk_steps_all_sp),4),
                  'relative_optimum_Greediest_typical_sp':round(float(Greediest_walk_relative_optimum_achieved_typical_sp),4),
                  'Greediest_walk_steps_typical_sp':round(float(Greediest_walk_steps_typical_sp),4)                 
                  }

    return results_dict



def perform_bottleneck(starting_community_indices, Nf,sigma_vectors,sigma_vectors_final,  dilution_factor, biomass_conversion_factor)  :   
    ending_community_indices=[]
    ending_community_indices.append(starting_community_indices[0]) # one community is not bottlenecked, like in all of Alvaro's protocols 
    for idx in starting_community_indices[1:]:        
        surviving_species_before_bottleneck=np.nonzero(sigma_vectors_final[idx])[0]
#        print ('idx',idx,'surviving_species_before_bottleneck', surviving_species_before_bottleneck)
        Ncells_community=Nf[idx]/biomass_conversion_factor 
        tot_cells=np.sum(Ncells_community)   
        tot_cells_after_dilution= np.random.poisson( tot_cells* dilution_factor )
        
        if tot_cells_after_dilution==0: ## all cells killed by dilution
            sigma_vector_ending_community=np.zeros(len(sigma_vectors_final[idx])).astype(int)
            
        else:
            ### we multinomial sample only the species with nonzero abundances to prevent rounding off errors creating species
            N_nonzero_cells=Ncells_community[surviving_species_before_bottleneck]
            p_nonzero_cells=N_nonzero_cells/np.sum(N_nonzero_cells) 
            try:          
                N_nonzero_cells_after_dilution=np.random.multinomial(tot_cells_after_dilution, p_nonzero_cells).astype(int) ## actually, we don't really need to multinomial sample as we only care if bottlenecking removes a species entirely, but this is easier to follow
            except:
                print ('pvals were bad:',p_nonzero_cells, '\n', N_nonzero_cells,'\n',tot_cells_after_dilution)
                print('tot_cells_before',tot_cells)
                print('Ncells_community',Ncells_community)
                print (len(surviving_species_before_bottleneck))
                exit(1)
                print (shit)
#            print ('N_nonzero_cells diluted and N_nonzero_cells afer sampling:\n',N_nonzero_cells* dilution_factor, N_nonzero_cells_after_dilution)
            
            if np.any(N_nonzero_cells_after_dilution==0):#bottlenecking killed some species
#                print('bottlenecking killed some species')
                nonzero_idx=np.nonzero(N_nonzero_cells_after_dilution)[0]           
                surviving_species_after_bottleneck=surviving_species_before_bottleneck[nonzero_idx]    
                sigma_vector_ending_community=np.zeros(len(sigma_vectors_final[idx])).astype(int)               
                for j in surviving_species_after_bottleneck:
                    sigma_vector_ending_community[j]=1
            else:   
                sigma_vector_ending_community=np.zeros(len(sigma_vectors_final[idx])).astype(int)            
                sigma_vector_ending_community[:]=sigma_vectors_final[idx]
            
       
       
        idx_ending_community=None
        for i in range(len(sigma_vectors)):
            if np.array_equal(sigma_vectors[i],sigma_vector_ending_community ):
                idx_ending_community=i
        if idx_ending_community is None:
            print ('NOT found!',idx_ending_community )
            print (sigma_vectors[idx_ending_community])
            print (sigma_vector_ending_community)
            print (shit)
#        diff_sigma=sigma_vectors-sigma_vector_ending_community
#        idx_ending_community=int( np.where( np.sum( np.abs(diff_sigma),axis=1 )==0)[0])
        
        ending_community_indices.append(idx_ending_community)
   
    return np.array(ending_community_indices).astype(int)    
    

def add_single_species(starting_communities, sigma_vectors, sigma_vectors_final, S, n_wells) :
    ending_community_indices=[]     
    
    assert (n_wells <=S+1),'there is only so many species you can knock in at once'
    ### we choose n_wells random species. we add one of them to each well.
    species_to_add=np.random.choice(np.arange(S),size=n_wells-1,replace=False)   
#    sigma_vector_ending_community=np.empty_like(sigma_vectors_final)
    for i in range(n_wells):
        sigma_vector_ending_community=np.empty_like(sigma_vectors_final[0])
        np.copyto(sigma_vector_ending_community,sigma_vectors_final[ starting_communities[i] ])
#        sigma_vector_ending_community= sigma_vectors_final[ starting_communities[i] ][:]
        
        if i> 1:# no species is added to the first well like Alvaro's protocol, and in bottlenecking
            sigma_vector_ending_community[ species_to_add[i-1] ]=1 ## add a species
            
        diff_sigma=sigma_vectors-sigma_vector_ending_community
        idx_ending_community=int( np.where( np.sum( np.abs(diff_sigma),axis=1 )==0)[0])
        ending_community_indices.append(idx_ending_community)
      
    return np.array(ending_community_indices).astype(int)         

def compare_search_protocols(Nfull, fit_full, sigma_full, sigmaFinal_full, adjacency_matrix, S, n_exp, dilution_factor_list=[1e-6], biomass_conversion='adaptive', rel_threshold=1e-4, n_iterations=4, n_averages=10  ):    
    '''
    these protocols are run on the full landscapes and not sublandscapes (that we use for focal species analysis)
    since sublandscapes are not closed under bottlenecking. (bottlenecking and combined protocoles can visit points outside the sublandscape)   
    Therefore, greedy search is also simulated here on the full landscapes and saved for easy comparison
    Additionally, comparing ruggedness measures of sublandscapes with search on full landscapes would also be incorrect
    simulations of these protocols required Nfull and sigma_final since they involve bottlenecking 
    '''   
    
    assert n_exp==len(fit_full), 'these protocoles are only done on the complete landscapes, not sublandscapes'
    if biomass_conversion=='fixed': ## then abundance of 10-3 is one cell
        biomass_conversion_factor=popn_cutoff

    elif biomass_conversion=='adaptive': 
        ## we fix the total abundance of the average community on the landscape to be 10^9 cells- 
        ##number of cells tend to be around 10^9 to 10^12 at steady state in chemostat [Chang, Villa,..Sanchez;  Wides,Milo]
        biomass_conversion_factor=np.mean(np.sum( Nfull,axis=1))/1e9
    
    max_fit_full=np.max(fit_full)
    dummy_global_max=np.argmax(fit_full)
    
    
    n_wells=S+1   ## number of parallel tests
    
    results_dict={'n_wells':n_wells,
                  'n_iterations': n_iterations,
                  'n_averages':n_averages,
                  'biomass_conversion':biomass_conversion,
                  'biomass_conversion_factor': biomass_conversion_factor,
                  'dilution_factor_list':dilution_factor_list, 
                  'SpeciesAddition_relopt_list':[],
                  'Greediest_relopt_list':[]}
    # if biomass conversion protocol is 'fixed', then a suffix is appended
    for dil_idx, dilution_factor in enumerate(dilution_factor_list):
        results_dict.update({'Bottleneck_relopt_list-dilution'+str(dil_idx): [],
                             'Combined_relopt_list-dilution'+str(dil_idx): []})

   
    for k in range(n_averages):   
        initial_communities = np.random.choice(np.arange(0,n_exp),size=n_wells,replace=False)  #### choose S initial communities to start with   
        best_initial_community=initial_communities[np.argmax(fit_full[initial_communities])] ## index of best of the S starting communities

        ##### protocol1- bottlnecking ####
        for dil_idx, dilution_factor in enumerate(dilution_factor_list):            
            test_communities=np.ones(n_wells,dtype=int)*best_initial_community ## index of the S replicates if the starting community
#            print ('dilution factor=',dilution_factor)
            
            for i in range(n_iterations):     
          
                ending_community_indices=perform_bottleneck(test_communities,Nfull,sigma_full,sigmaFinal_full,  dilution_factor, biomass_conversion_factor)   
                ### select the best community and propagate to next level
                best_community=ending_community_indices[np.argmax(fit_full[ending_community_indices])]
                test_communities=np.ones(n_wells,dtype=int)*best_community
                           
            results_dict['Bottleneck_relopt_list-dilution'+str(dil_idx)].append(fit_full[best_community]/max_fit_full)
#            print ('the highest function bottlenecking found was:', fit_full[best_community])
        
        
        ##### protocol2- addition of a single species greedily ####
        test_communities=np.ones(n_wells,dtype=int)*best_initial_community ## index of the S replicates if the starting community
        for i in range(n_iterations): 

            ending_community_indices=add_single_species(test_communities, sigma_full, sigmaFinal_full, S, n_wells) 
            
            
            ### select the best community and propagate to next level
            best_community=ending_community_indices[np.argmax(fit_full[ending_community_indices])]
            test_communities=np.ones(n_wells,dtype=int)*best_community

        results_dict['SpeciesAddition_relopt_list'].append(fit_full[best_community]/max_fit_full)
#        print ('the highest function greedy species addition found was:', fit_full[best_community])
                   
        ##### protocol3- Combination of bottlenecking and single species addition ####
        for dil_idx, dilution_factor in enumerate(dilution_factor_list):
            test_communities=np.ones(n_wells,dtype=int)*best_initial_community ## index of the S replicates if the starting community
#            print ('dilution factor=',dilution_factor)
            for i in range(n_iterations):     
          
                bottlenecked_community_indices=perform_bottleneck(test_communities,Nfull,sigma_full,sigmaFinal_full,  dilution_factor, biomass_conversion_factor)                
                ending_community_indices=add_single_species(bottlenecked_community_indices, sigma_full, sigmaFinal_full, S, n_wells) 
               
                ### select the best community and propagate to next level
                best_community=ending_community_indices[np.argmax(fit_full[ending_community_indices])]
                test_communities=np.ones(n_wells,dtype=int)*best_community
                
            results_dict['Combined_relopt_list-dilution'+str(dil_idx)].append(fit_full[best_community]/max_fit_full)
#            print ('the highest function combined found was:', fit_full[best_community])
            
        ##### protocol4- the greedy walk- ####           
        _, _, Greediest_walk_steps, _, Greediest_walk_relative_optimum_achieved=do_greediest_walk(fit_full, adjacency_matrix, S, n_exp, [best_initial_community], max_fit_full, dummy_global_max, rel_threshold=1e-4)    
        results_dict['Greediest_relopt_list'].append(float(Greediest_walk_relative_optimum_achieved))
#        print ('the highest function greedy found was:', Greediest_walk_relative_optimum_achieved*max_fit_full)
    
    results_dict.update({'SpeciesAddition_relopt_avg':np.mean(results_dict['SpeciesAddition_relopt_list']),
                        'Greediest_relopt_avg':np.mean(results_dict['Greediest_relopt_list'])} )
    for dil_idx, dilution_factor in enumerate(dilution_factor_list):
        results_dict.update({'Bottleneck_relopt_avg-dilution'+str(dil_idx):np.mean(results_dict['Bottleneck_relopt_list-dilution'+str(dil_idx)]),
                        'Combined_relopt_avg-dilution'+str(dil_idx):np.mean(results_dict['Combined_relopt_list-dilution'+str(dil_idx)]) })

    if biomass_conversion=='fixed':
        results_dict_suffixed= {k+'-fixed_conversion': v for k, v in results_dict.items()}
        return results_dict_suffixed
    else:
        return results_dict
    
def variance_explained(fitness, sigma_vectors, S, n_exp, remove0=False, minExpOrder=3):
    if remove0:
        fitness=remove_zero_data_point(sigma_vectors, fitness=fitness)
        n_exp=remove_zero_data_point(sigma_vectors, n_exp=n_exp)
        sigma_vectors=remove_zero_data_point(sigma_vectors)
    
    var_exp_dict={}    
    R2_list=[]
    EV_list=[]
#    for i in range (int(S/2 +1)): 
    for i in range ( int(min(minExpOrder,S-1)) ):

        reg_dict,_=perform_Linear_Regression_of_fitness_scalar(fitness, sigma_vectors, S, expansionOrder=i) 
        var_exp_dict.update({f'order {i}':reg_dict})

#        print ("R^2, Variance explained=",reg_dict['R2'],reg_dict['Explained variance'])
        R2_list.append(reg_dict['R2'])
        EV_list.append(reg_dict['Explained variance'])
#        print(f"time for lin reg {i}",end-start)
    return np.asarray(R2_list), np.asarray(EV_list), var_exp_dict


def calc_error_in_estimate(M_true, M_est_list):
    if np.abs(M_true)>1e-5:
        error_in_estimate=(np.asarray(M_est_list)-M_true)/np.abs(M_true)
    else:
        error_in_estimate=np.zeros_like(M_est_list)*1. ### need to make float as nan is a float
        error_in_estimate[:]=np.NaN
    return error_in_estimate

def estimate_variance_explained_from_sparse_data(fitness, sigma_vectors, S, n_exp, frac_data_list=[0.1, 0.2, 0.3, 0.5], n_replicates=1):
    R2_list=[]
    EV_list=[]
    frac_data_list_used=[]
    
    def variance_explained_by_linear_model(fitness, sigma_vectors, S, n_exp):
        reg_dict,_=perform_Linear_Regression_of_fitness_scalar(fitness, sigma_vectors, S, expansionOrder=1)
        return  reg_dict ['R2'], reg_dict['Explained variance']
    
    R2_true, EV_true=variance_explained_by_linear_model(fitness, sigma_vectors, S, n_exp)
         
    for idx, frac in enumerate(frac_data_list):
        if frac*n_exp>3* S+1: ##then we fit, otherwise its too small.
            frac_data_list_used.append(frac)
            R2_list.append([])
            EV_list.append([])
            
            for x in range(n_replicates): 
                nexp_sub=int(frac*n_exp)
                all_species_1and0_present=False
                while (not all_species_1and0_present):
                    expts_chosen=np.random.choice(np.arange(n_exp), size=nexp_sub, replace=False)
                    fit_sub=fitness[expts_chosen]
                    sig_sub=sigma_vectors[expts_chosen]
                    for i in range(S):
                        if np.any(sig_sub[:,i]==1) and np.any(sig_sub[:,i]==0):
                            all_species_1and0_present=True
                        else:
                            all_species_1and0_present=False
                            break ## try picking again so that all speacing have a present and an absent.
                                
                R2_est, EV_est=variance_explained_by_linear_model(fit_sub, sig_sub, S, nexp_sub)
                curr_idx=len(frac_data_list_used)-1
                R2_list[curr_idx].append(R2_est)
                EV_list[curr_idx].append(EV_est)
    
    if len(frac_data_list_used)>0:
        error_in_estimate_R2= calc_error_in_estimate(R2_true, R2_list)     
        error_in_estimate_EV= calc_error_in_estimate(EV_true, EV_list)            
        results_dict={'est-R2_frac_data_list':frac_data_list_used,
                          'est-R2_n_points_sampled':np.array(n_exp*np.array(frac_data_list_used)).astype(int),
                          'est-R2_estimate_list':R2_list,
                          'est-R2_true':R2_true,
                          'est-R2_error_in_estimate':error_in_estimate_R2,
                          'est-EV_frac_data_list':frac_data_list_used,
                          'est-EV_n_points_sampled':np.array(n_exp*np.array(frac_data_list_used)).astype(int),
                          'est-EV_estimate_list':EV_list,
                          'est-EV_true':EV_true,
                          'est-EV_error_in_estimate':error_in_estimate_EV                     
                          }   
    else:
        results_dict={'est-R2_frac_data_list':[0],'est-EV_error_in_estimate':[0]}
        
    return results_dict  

def estimate_FDC_from_sparse_data(fitness, sigma_vectors, maxima_without_extinction, S, n_exp, frac_data_list=[0.1, 0.2, 0.3, 0.5], n_replicates=1):
    FDC_list=[] 
    ranked_FDC_list=[] 
    frac_data_list_used=[]
    FDC_true=fitness_distance_correlation(fitness, sigma_vectors, S, n_exp, maxima_without_extinction=maxima_without_extinction)    
    ranked_FDC_true=ranked_fitness_distance_correlation(fitness, sigma_vectors, S, n_exp)
        
    
    if FDC_true!=None:     
        for idx, frac in enumerate(frac_data_list):
            if frac*n_exp>3* S+1: ##otherwise its a very small landscape.                
                frac_data_list_used.append(frac)
                FDC_list.append([])
                ranked_FDC_list.append([])
                
                for x in range(n_replicates):                
                    nexp_sub=int(frac*n_exp)
                    expts_chosen=np.random.choice(np.arange(n_exp), size=nexp_sub, replace=False)
                    fit_sub=fitness[expts_chosen]
                    sig_sub=sigma_vectors[expts_chosen]
                    glob_max=np.argmax(fit_sub)## we might not have sampled the global max without extinction, so we just choose the answer from argmax.       
                    FDC_est=fitness_distance_correlation(fit_sub, sig_sub, S, nexp_sub, global_maxima_loc=glob_max)                    
                    ranked_FDC_est=ranked_fitness_distance_correlation(fit_sub, sig_sub, S, nexp_sub)                    
                    
                    curr_idx=len(frac_data_list_used)-1
                    FDC_list[curr_idx].append(FDC_est)
                    ranked_FDC_list[curr_idx].append(ranked_FDC_est)                    
                
                
                
    if len(frac_data_list_used)>0:
        error_in_estimate_FDC= calc_error_in_estimate(FDC_true, FDC_list) 
        error_in_estimate_rFDC= calc_error_in_estimate(FDC_true, FDC_list) 
        print ('this FDC might differ from true FDC even at frac=1! since we dont necessarily maxima without extinction, here we just use argmax to find peak fitness ')
        results_dict={'est-FDC_frac_data_list':frac_data_list_used,
                          'est-FDC_n_points_sampled':np.array(n_exp*np.array(frac_data_list_used)).astype(int),
                          'est-FDC_estimate_list':FDC_list,
                          'est-FDC_true':FDC_true,
                          'est-FDC_error_in_estimate':error_in_estimate_FDC,                        
                          'est-ranked_FDC_frac_data_list':frac_data_list_used,
                          'est-ranked_FDC_n_points_sampled':np.array(n_exp*np.array(frac_data_list_used)).astype(int),
                          'est-ranked_FDC_estimate_list':ranked_FDC_list,
                          'est-ranked_FDC_true':ranked_FDC_true,
                          'est-ranked_FDC_error_in_estimate':error_in_estimate_rFDC                          
                          }   
    else:
        results_dict={'est-FDC_frac_data_list':[0],'est-ranked_FDC_frac_data_list':[0]}
    return results_dict     

   
        

def estimate_rs_from_sparse_data(fitness, sigma_vectors,  S, n_exp, frac_data_list=[0.1, 0.2, 0.3, 0.5], n_replicates=1):
    rs_list=[] 
    frac_data_list_used=[]
    r_roughness_true,s_slope_true=r_over_s(fitness, sigma_vectors) 
    if s_slope_true>1e-5:
        rs_true=r_roughness_true/s_slope_true
    else:
        rs_true=0
    for idx, frac in enumerate(frac_data_list):         
         if frac*n_exp>3* S+1: ##then we fit, otherwise its too small.
             frac_data_list_used.append(frac)
             rs_list.append([])
             for x in range(n_replicates):
                 nexp_sub=int(frac*n_exp)
                 all_species_1and0_present=False
                 while (not all_species_1and0_present):
                    expts_chosen=np.random.choice(np.arange(n_exp), size=nexp_sub, replace=False)
                    fit_sub=fitness[expts_chosen]
                    sig_sub=sigma_vectors[expts_chosen]
                    for i in range(S):
                        if np.any(sig_sub[:,i]==1) and np.any(sig_sub[:,i]==0):
                            all_species_1and0_present=True
                        else:
                            all_species_1and0_present=False
                            break ## try picking again so that we have a 1 and 0 for each species have a present and an absent.
                
                 r_roughness,s_slope= r_over_s(fit_sub, sig_sub)
                 curr_idx=len(frac_data_list_used)-1
                 if s_slope>1e-5:
                     rs_list[curr_idx].append(r_roughness/s_slope)
                 else:
                     rs_list[curr_idx].append(0)    
        
    if len(frac_data_list_used)>0:
        error_in_estimate= calc_error_in_estimate(rs_true, rs_list) 
        results_dict={'est-rs_frac_data_list':frac_data_list_used,
                          'est-rs_n_points_sampled':np.array(n_exp*np.array(frac_data_list_used)).astype(int),
                          'est-rs_estimate_list':rs_list,
                          'est-rs_true':rs_true,
                          'est-rs_error_in_estimate':error_in_estimate
                          }  
    else:
        results_dict={'est-rs_frac_data_list':[0]}
    return results_dict




def estimate_CorrLength_FNeutral_from_sparse_data(fitness, sigma_vectors, adjacency_matrix, S, n_exp, frac_data_list=[0.1, 0.2, 0.3, 0.5], n_replicates=1):
    CorrL_list=[] 
    nnCorr_list=[] 
    FNeutral_list=[] 
    frac_data_list_used=[]
    n_points_sampled=[]
    nnCorr_true, CorrL_true=spatial_correlation(fitness, sigma_vectors, adjacency_matrix )
    links_dict=F_Neutral_Links(fitness, sigma_vectors, adjacency_matrix)
    FNeutral_true=links_dict['f_neutral_links']

    for idx, frac in enumerate(frac_data_list):         
        if frac*n_exp>3* S+1: ##then we fit, otherwise its too small.
            frac_data_list_used.append(frac)
            CorrL_list.append([])
            nnCorr_list.append([])
            FNeutral_list.append([])
            for x in range(n_replicates):
                nexp_sub=int(frac*n_exp)
                '''
                 we randomly select n_sample/2 epxeriments from the total
                 and then we pick at random one previously unselected neighbor for each to have sampled n_sample elements
                '''                 
                all_species_1and0_present=False
                while (not all_species_1and0_present):
                    all_expts=np.arange(n_exp)
                    chosen_mask=np.zeros(n_exp,dtype=bool)                   
                    idx_chosen=np.random.choice(all_expts, size=int(nexp_sub/2), replace=False)
                    chosen_mask[idx_chosen]=True
                    initally_chosen_expts=all_expts[chosen_mask]
                
                    for exp in initally_chosen_expts:
                        nn_candidates=np.intersect1d(adjacency_matrix[exp], all_expts[~chosen_mask])
                        if len(nn_candidates)>0:
                           new_addition= np.random.choice(nn_candidates)
                           chosen_mask [ new_addition]=True
                        else:## if all nearest neighbors were chosen, just choose a random other point
                            chosen_mask [ np.random.choice( all_expts[~chosen_mask])]=True
                     
                    expts_chosen=all_expts[chosen_mask]

                    fit_sub=fitness[expts_chosen]
                    sig_sub=sigma_vectors[expts_chosen]
                    for i in range(S):
                        if np.any(sig_sub[:,i]==1) and np.any(sig_sub[:,i]==0):
                            all_species_1and0_present=True
                        else:
                            all_species_1and0_present=False
                            break ## try picking again so that we have a 1 and 0 for each species have a present and an absent.
                
                adjList_sub=generate_adjacency_list_from_Subsampling(sig_sub)
#                print ('adjacency list shape is ', np.shape(adjList_sub),'\n and matrix is: \n',adjList_sub)
                links_dict=F_Neutral_Links(fit_sub, sig_sub, adjList_sub)
                nn_correlation, CorrL=spatial_correlation(fit_sub, sig_sub, adjList_sub )
                curr_idx=len(frac_data_list_used)-1
                FNeutral_list[curr_idx].append(links_dict['f_neutral_links'])
                nnCorr_list[curr_idx].append(nn_correlation)
                CorrL_list[curr_idx].append(CorrL)
            n_points_sampled.append(len(fit_sub)) ## since it can differ slightly from int(frac*n_exp)
        
    if len(frac_data_list_used)>0:
 
        results_dict={'est-CorrL_frac_data_list':frac_data_list_used,
                          'est-CorrL_n_points_sampled':np.array(n_points_sampled).astype(int),
                          'est-CorrL_estimate_list':CorrL_list,
                          'est-CorrL_true':CorrL_true,

                          'est-nnCorr_frac_data_list':frac_data_list_used,
                          'est-nnCorr_n_points_sampled':np.array(n_points_sampled).astype(int),
                          'est-nnCorr_estimate_list':nnCorr_list,
                          'est-nnCorr_true':nnCorr_true,
                          
                          'est-FNeutral_frac_data_list':frac_data_list_used,
                          'est-FNeutral_n_points_sampled':np.array(n_points_sampled).astype(int),
                          'est-FNeutral_estimate_list':FNeutral_list,
                          'est-FNeutral_true':FNeutral_true,
                          }  
    else:
        results_dict={'est-CorrL_frac_data_list':[0],'est-nnCorr_frac_data_list':[0],'est-FNeutral_frac_data_list':[0]}
    return results_dict




'''
effnumstate is actually a part of landscape structure analysis and not based on the functional landscape. so it is run there.
'''

def estimate_eff_num_states_from_sparse_data(sigma_vectors,  sigma_vectors_final, S, n_exp, frac_data_list=[0.1, 0.2, 0.3, 0.5], n_replicates=1):
    eff_num_states_list=[] 
    frac_data_list_used=[]
    
    def calc_eff_num_states(sigma_final,S):
        powers_of_2_arr=2**np.arange(S)
        community_ID_final=np.dot(sigma_final,powers_of_2_arr)
        unique, counts = np.unique(community_ID_final, return_counts=True)
        if counts.size>1:
            prob_state=counts*1.0/np.sum(counts)           
            eff_num_states=-np.sum( prob_state*np.log(prob_state))
            ## we need to renormalize the entropy estimate to correct for the smaller sample.
            eff_num_states= eff_num_states*S*1./np.log2(1.*len(sigma_final))
        else:
            ## only a single state.
            eff_num_states=0.
        return eff_num_states
    
    eff_num_states_true=calc_eff_num_states(sigma_vectors_final,S)


    for idx, frac in enumerate(frac_data_list):         
         if frac*n_exp>3* S+1: ##then we fit, otherwise its too small.
             frac_data_list_used.append(frac)
             eff_num_states_list.append([])
             for x in range(n_replicates):
                 nexp_sub=int(frac*n_exp)
                 all_species_1and0_present=False
                 while (not all_species_1and0_present):
                    expts_chosen=np.random.choice(np.arange(n_exp), size=nexp_sub, replace=False)
                    sig_sub=sigma_vectors[expts_chosen]
                    sigma_final_sub=sigma_vectors_final[expts_chosen]
                    for i in range(S):
                        if np.any(sig_sub[:,i]==1) and np.any(sig_sub[:,i]==0):
                            all_species_1and0_present=True
                        else:
                            all_species_1and0_present=False
                            break ## try picking again so that we have a 1 and 0 for each species have a present and an absent.
                
                 eff_num_states=calc_eff_num_states(sigma_final_sub,S)
                 curr_idx=len(frac_data_list_used)-1
                 eff_num_states_list[curr_idx].append(eff_num_states)
                  
        
    if len(frac_data_list_used)>0:
        results_dict={'est-EffNumStates_frac_data_list':frac_data_list_used,
                          'est-EffNumStates_n_points_sampled':np.array(n_exp*np.array(frac_data_list_used)).astype(int),
                          'est-EffNumStates_estimate_list':eff_num_states_list,
                          'est-EffNumStates_true':eff_num_states_true,
                          }  
    else:
        results_dict={'est-EffNumStates_frac_data_list':[0]}
    return results_dict

def estimate_beta_via_random_walks(fitness, sigma_vectors, adjacency_matrix, S, n_exp, fig_name=None, frac_data_list=[0.15, 0.2, 0.5, 0.8, 1. ], n_replicates=1):
    '''
    estimates Powerspectral ratio from random walk autocorrelation
    '''
    assert n_exp==2**S,"we want the full data to compare our estimates with."
    #assert int(n_exp*min(frac_data_list) )>S+1, "to have a rw autocorrelation of minimum requires length"
    r_roughness,s_slope=r_over_s(fitness, sigma_vectors) 
    Fourier_coefficients, Power_spectrum, PS_ratio_true, beta_true =Fourier_roughness (fitness, sigma_vectors, S, n_exp )

    frac_data_list_used=[]

    def set_coeff_matrix(Neq,P,S):
        coef_matrix=np.zeros((Neq,P))
        for k in range (Neq):
            for p in range (P):
               coef_matrix[k,p]= (1.-(p+1)*2./S)**k
        return coef_matrix
    
    beta_estimate_list=[]
    PS_ratio_list=[]
    for idx, frac in enumerate(frac_data_list):
        n_steps=int(n_exp*frac)
        if n_steps>S+1: #to have a rw autocorrelation of minimum requires length
            frac_data_list_used.append(frac)
            PS_ratio_list.append([])
            for ctr in range(n_replicates):
                rw_autocorr= random_walk_performance(fitness, adjacency_matrix, S, n_exp, Nsteps=n_steps, n_init_nodes=1 )
                       
                Neq=S+1 ## choosing number of equations that we will use will set to S or S+1 later, 
                P=S ## number of beta coefficients, beta[0] is actually beta1, as beta0 is not used.
                coeff_matrix=set_coeff_matrix(Neq,P,S)
    
                            
                def func_min(beta_p, coeff_matrix, r_of_s): ## function of the beta_array of length n
                    y = np.dot(coeff_matrix, beta_p) - r_of_s            
                    return np.dot(y, y)   
                
        #        bnds =[(0, None) for i in range(P)] ## we want to ensure all beta are positive.
                bnds =[(0, 1) for i in range(P)] ## we want to ensure all beta are positive.
                '''
                the first equation says that  sum of inferred beta =1, but its a soft constraint. so answer need not respect it.          
                '''
        #        print ("shapes are", np.shape(coeff_matrix), np.shape(rw_autocorr[:Neq]), P, Neq)
                
                result = minimize(func_min, np.zeros(P), args=(coeff_matrix,rw_autocorr[:Neq] ), method='L-BFGS-B', bounds=bnds, 
                                options={'disp': False}) # method='SLSQP'
                
                if ctr==0:
                    beta_estimate_list.append(result.x) ### only one beta estimate is saved.
                curr_idx=len(frac_data_list_used)-1
                PS_ratio_list[curr_idx].append(np.sum(result.x[1:])/np.sum(result.x))

        
    
    error_in_estimate= list(map(lambda x: np.sqrt(np.square( beta_true- x ).sum()), np.asarray(beta_estimate_list)) )    
    
    
    error_in_estimate_PSratio= calc_error_in_estimate(PS_ratio_true, PS_ratio_list) 
    
    if len(frac_data_list_used)>0:
        if fig_name!=None :
            fig = plt.figure(figsize=(7,3.5)) 
            ax = fig.add_subplot(121)
            ax.plot(1+np.arange(S), beta_true ,'b-o',mec='None', label='true')
            for idx, frac in enumerate(frac_data_list_used):
                ax.plot(1+np.arange(S), beta_estimate_list[idx],'*',mec='None',label='estimate, f='+str(frac))
            ax.set_ylabel(r'amplitude spectrum, $\beta_i$')
            ax.set_xlabel(r'coefficient number') 
            ax.legend(loc='best')
            ax2=fig.add_subplot(122)
            ax2.plot(frac_data_list_used, error_in_estimate ,'b-o',mec='None', label='true')
            ax2.set_ylabel(r'error in estimate')
            ax2.set_xlabel(r'fraction of data') 
            ax2.set_xscale('log')
    #        ax2.set_yscale('log')
            if fig_name.startswith('/Users/ashish'): plt.tight_layout()
            else: fig.set_tight_layout(True)  #plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)           
            plt.savefig(fig_name+'.png',dpi=120)
            
        results_dict={'est-beta_frac_data_list':frac_data_list_used,
                      'est-beta_n_points_sampled':np.array(n_exp*np.array(frac_data_list_used)).astype(int),
                      'est-beta_estimate_list':beta_estimate_list,
                      'est-beta_true':beta_true,
                      'est-beta_error_in_estimate':error_in_estimate,
                      'est-PSratio_frac_data_list':frac_data_list_used,
                      'est-PSratio_n_points_sampled':np.array(n_exp*np.array(frac_data_list_used)).astype(int),
                      'est-PSratio_estimate_list':PS_ratio_list,
                      'est-PSratio_true':PS_ratio_true,
                      'est-PSratio_error_in_estimate':error_in_estimate_PSratio                     
                      }
    else:
        results_dict={'est-beta_frac_data_list':[0],'est-PSratio_frac_data_list':[0]}
    return results_dict



def perform_Linear_Regression_of_abundance_vector_by_subsetting(Y_vectors_train, sigma_vectors_train, S, verbose=False) :
    ##no test data, only fitting.
    linReg_object = LinearRegression(fit_intercept=False) # we remove consntant term
    interaction_matrix=[]
    sum_of_squares=np.zeros(S)
    residual_square=np.zeros(S)   
    residual_square_unclipped=np.zeros(S)

    for i in range(S):        
        idx_species_present=np.where(sigma_vectors_train[:,i]==1)[0]
        Y_i=Y_vectors_train[idx_species_present,i]        
        sigma_ij=sigma_vectors_train[idx_species_present]   
#        print (np.shape(Y_i),np.shape(sigma_ij))
        linReg_object.fit(sigma_ij,Y_i)
        interaction_matrix.append(linReg_object.coef_)  
        Y_pred=linReg_object.predict(sigma_ij)
        Y_pred_unclipped=deepcopy(Y_pred)
        Y_pred[Y_pred<0]=0. ##### clipping           
        sum_of_squares[i]=np.sum( np.square( Y_i.mean()-Y_i ))
        residual_square[i]=np.sum( np.square( Y_i-Y_pred  )  ) 
        residual_square_unclipped[i]=np.sum( np.square( Y_i-Y_pred_unclipped  )  ) 
#        residual_square[i]=np.sum( np.square( Y_i-sigma_ij@linReg_object.coef_  )  ) 
#        assert np.all(Y_pred-sigma_ij@linReg_object.coef_ ==0)
        if i ==0:
            Y_vectors_pred=np.empty((len(Y_pred),S)) ## only half of the vectors are trained and predicted!
            Y_vectors_test=np.empty((len(Y_pred),S))
            Y_vectors_pred_unclipped=np.empty((len(Y_pred),S))
        
        
        Y_vectors_pred[:,i]=Y_pred
        Y_vectors_test[:,i]=Y_i
        Y_vectors_pred_unclipped[:,i]=Y_pred_unclipped
        
        
        if verbose:
            print ('species ',i,' $R^2=$ ', '{:.2e}'.format(r2_score(Y_i,Y_pred)), '{:.2f}'.format(1- residual_square[i]/sum_of_squares[i] ))
            print ('species ',i,' variance= ', '{:.2e}'.format(sum_of_squares[i]) )
            print ('species ',i,' survival fraction= ', '{:.2e}'.format(np.sum(Y_pred>1e-2 )/len(Y_pred) ) )
        
    '''
    review
    '''    
        
    print ('We clip negative abundances  before calculating R^2')    
    interaction_matrix  =np.asarray(interaction_matrix)         
    
    '''
    if we are treating species as identical, then caclulating the R^2 as in the manual calculation,
    equivalent to 'variance weighted' R^2 makes sense (and not  R^2). This is because R2VW is equivalent to summing up all the squared residuals
    of all the fits and all the sum of squares (variances) of all fits and then taking the ratio. Instead of summing up the ratio of each fit.
    '''
    R2_avg=r2_score(Y_vectors_test,Y_vectors_pred)##  avg of the R2 of prediction for each species
    R2_VW=r2_score(Y_vectors_test,Y_vectors_pred, multioutput='variance_weighted')  ##avg Weighted  by the variance in each species
    R2_manual=1- np.sum(residual_square)/np.sum(sum_of_squares)## this is R2_VW!
    R2_avg_unclipped=r2_score(Y_vectors_test,Y_vectors_pred_unclipped)
    
    
    R2_Flattened=r2_score(np.ravel(Y_vectors_test),np.ravel(Y_vectors_pred)) ### computes R^2 of each abundance predicted and observed as a separate data point
    R2_Flattened_unclipped=r2_score(np.ravel(Y_vectors_test),np.ravel(Y_vectors_pred_unclipped))
    print ("R2avg, R2_VW, R2_manual, ",R2_avg, R2_VW ,R2_manual)     
    print ("R2Flat, R2Flat unclipped", R2_Flattened, R2_Flattened_unclipped)
    
        
    if np.all(sum_of_squares<1e-3):
        '''
        all species were non-interacting,
        so the abundance of the species wasnt actually changing. there was no variation to fit (just noise from simulation endpoint)
        So we set all R^2 to 1
        '''
        print("all species were non-interacting,  R^2 is set to1")
        R2_avg, R2_VW ,R2_manual,R2_avg_unclipped=1.,1.,1.,1.
    elif np.any(sum_of_squares<1e-3):
        '''
        atleast one of the species was behaving like noninteracting.
        however if R2_VW ,R2_manual agree, then its fine; the variance weighting erased the weird R^2 
        R2_avg cannot be trusted though, so we compute it manually..
        '''        
        np.testing.assert_almost_equal(R2_VW ,R2_manual,decimal=3)
        idx_good=np.where(sum_of_squares>1e-3)[0]        
        R2_avg =1.0-np.mean(  residual_square [idx_good] / sum_of_squares [idx_good]   )
        R2_avg_unclipped =1.0-np.mean(  residual_square_unclipped [idx_good] / sum_of_squares [idx_good]   )

    
        
    reg_dict={'Y_vectors_test':Y_vectors_test,'Y_vectors_pred':Y_vectors_pred,'interaction_matrix':interaction_matrix,
              'R2_avg': R2_avg, 'R2_VW': R2_VW,'R2_Flattened':R2_Flattened, 'R2_avg_unclipped':R2_avg_unclipped}
    
    return Y_vectors_test,Y_vectors_pred,interaction_matrix, R2_avg, R2_VW, R2_Flattened, reg_dict



def LinearRegressions_on_AbundanceVector(sigma_vectors, Nf,  S, n_exp, fold, file_suffix ):

    Y_vectors=N_to_Ncutoff(Nf, sigma_vectors,popn_cutoff) ## Yvectors is the vector of observations to fit.
    '''
    fit can done in two waysL
    1. using the multiplicative model (MM) where Y_i= sigma_i * SUM[ alpha_ij*sigma_j , j]
    2. using the subsetted data where Y_i=SUM[ alpha_ij*sigma_j , j]  for data where  sigma_i =1. 
    Both should give the same fit coefficients, but R^2 should be different. ( Model 1 will have higher R^2 from all the zeros predicted correctly) 
    '''
    Y_vectors_pred_MM,interaction_matrix_MM, R2_Avg_MM, R2_VW_MM, reg_dict_MM=perform_Linear_Regression_of_abundance_vector(Y_vectors, sigma_vectors, S, expansionOrder=1)
   
    Y_vectors_Subset, Y_vectors_pred_Subset,interaction_matrix_Subset, R2_Avg_Subset, R2_VW_Subset, R2_Flattened_subset, reg_dict_Subset=perform_Linear_Regression_of_abundance_vector_by_subsetting(Y_vectors, sigma_vectors, S)
    
    
    fig = plt.figure(figsize=(3.5,3.5)) 
    ax1 = fig.add_subplot(1,1,1) 
    ax1.plot(Y_vectors_Subset, Y_vectors_pred_Subset,'bo', mec='None', markersize=4, alpha=.05 )       
    ax1.text(0.05, 0.95, r'$R^2_{\overline{0}}=$'+'{:.2f}'.format(R2_VW_Subset),color='b', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes)     
    ax1.plot(Y_vectors, Y_vectors_pred_MM,'g*', mec='None', markersize=4, alpha=.05 )       
    ax1.text(0.05, 0.88, r'$R^2 =$'+'{:.2f}'.format(R2_VW_MM),color='g', horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes)     
    
    ax1.plot(ax1.get_xlim(),ax1.get_xlim(),'k--')    
    ax1.set_ylabel('predicted abundance')
    ax1.set_xlabel('observed abundance')
    ax1.ticklabel_format(axis='both',useOffset=False)
    #ax1.ticklabel_format(axis='both', style='sci', scilimits=(20,20))
    #ax1.ticklabel_format(useOffset=False, useLocale=False)
    if os.getcwd().startswith('/Users/ashish'): plt.tight_layout()
    else: fig.set_tight_layout(True)  #plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1) 
    plt.savefig(fold+"abundance_vector_fit.png" ,dpi=120)
    
    

    '''
    added one more return variable :reg_dict_Subset['R2_avg_unclipped'] on 6/15/22
    '''
    return R2_Flattened_subset, R2_Avg_Subset, R2_VW_Subset, R2_Avg_MM, R2_VW_MM, reg_dict_Subset['R2_avg_unclipped']
    
    
def run_routine_analysis(sigma_vectors, Nf, sigma_vectors_final, adjacency_matrix, S, n_exp, fold, file_suffix, fraction_limited_data=[0.1, 0.2, 0.3, 0.5] ):
    
    structure_dict={}
    fitness_measure_name='landscape structure'
    fitness_measure_suffix='landscape_structure'
    structure_dict.update({'fitness_measure_name':fitness_measure_name,'fitness_measure_suffix':fitness_measure_suffix})
    
    
    avg_fraction_species_survived=np.sum(sigma_vectors_final)*1.0/np.sum(sigma_vectors)
    idx_all_inIC=np.where(np.sum(sigma_vectors,axis=1)==S  )[0]
    print ("IC: all", sigma_vectors[idx_all_inIC], "Final: all",sigma_vectors_final[idx_all_inIC])
    S_star=np.sum(sigma_vectors_final[idx_all_inIC]  )
    structure_dict.update({'avg fraction species survived':avg_fraction_species_survived,'S_star':S_star})
    
    
    powers_of_2_arr=2**np.arange(S)
    community_ID_final=np.dot(sigma_vectors_final,powers_of_2_arr)
    unique, counts = np.unique(community_ID_final, return_counts=True)

    
#    fig = plt.figure(figsize=(3.5,3.5)) 
#    ax = fig.add_subplot(111)
#    plt.hist(community_ID_final, bins=np.arange(n_exp), density=True)
#    ax.set_ylabel(r'frequency of a final state')
#    ax.set_xlabel(r'state') 
#    ax.set_yscale('log')                  
#    if fold.startswith('/Users/ashish'): plt.tight_layout()
#    else: fig.set_tight_layout(True)  #plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)              
#    plt.savefig(fold+file_suffix+"_structure_"+"histogram.png",dpi=120) 
    
    
    fig = plt.figure(figsize=(3.5,3.5)) 
    ax = fig.add_subplot(111)
    plt.hist(counts,bins=2**np.arange(S),edgecolor='black', linewidth=2)
    ax.set_ylabel(r'frequency of the degeneracy')
    ax.set_xlabel(r'degeneracy') 
    ax.set_xscale('log',basex=2)                  
    if fold.startswith('/Users/ashish'): plt.tight_layout()
    else: fig.set_tight_layout(True)  #plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)     
    plt.savefig(fold+file_suffix+"_structure_"+"degeneracy.png",dpi=120)
    
    uniqueness=np.sum(1.0/counts)
    prob_state=counts*1.0/np.sum(counts)
    eff_num_states=-np.sum( prob_state*np.log(prob_state))
    structure_dict.update({'frequency of states':counts,'uniqueness':uniqueness, 'effective number of states':eff_num_states})    
    print("uniqueness and entropy are:",uniqueness,eff_num_states)
    
    
    R2_Flattened_subset, R2_Avg_Subset, R2_VW_Subset, R2_Avg_MM, R2_VW_MM, R2_avg_Subset_unclipped=LinearRegressions_on_AbundanceVector(sigma_vectors, Nf,  S, n_exp, fold, file_suffix )
    structure_dict.update({'Abundance vector fit Subsetted R2 Flattened':R2_Flattened_subset, 
                           'Abundance vector fit Subsetted R2':R2_VW_Subset, 'Abundance vector fit Subsetted R2avg':R2_Avg_Subset,
                           'Abundance vector fit Subsetted R2avg_unclipped':R2_avg_Subset_unclipped,
                           'Abundance vector fit MM R2':R2_VW_MM, 'Abundance vector fit MM R2avg':R2_Avg_MM})
    
        
    results_dict=estimate_eff_num_states_from_sparse_data(sigma_vectors,  sigma_vectors_final, S, n_exp, frac_data_list=fraction_limited_data, n_replicates=10)
    structure_dict.update(results_dict)
    ####### add a prefix to all measured values identify the fitness metric used ########
    structure_dict_Prefixed ={fitness_measure_name+': '+k: v for k, v in structure_dict.items()} 
    return structure_dict_Prefixed

    
    

    
    
    
    
def analyze_for_chosen_metric(fitness, sigma_vectors, sigma_vectors_final, adjacency_matrix, S, n_exp, fitness_measure_name, fold, file_suffix, 
                              analysis_options=['FourierPS','r/s','n_max_min'],fraction_limited_data=[0.1, 0.2, 0.3, 0.5], 
                              Nfull=None, sigma_full=None, sigmaFinal_full=None, fit_full=None):
    metric_analysis_dict={}
    fitness_measure_suffix=fitness_measure_name.replace(' ','_')
    fitness_measure_suffix=fitness_measure_suffix.replace('$','')
    print (fitness_measure_suffix)
    metric_analysis_dict.update({'fitness_measure_name':fitness_measure_name,'fitness_measure_suffix':fitness_measure_suffix})
    
    print ('mean and variance was', np.mean(fitness),np.var(fitness) )
#    if 'check_SS' in analysis_options:
#        fig = plt.figure(figsize=(3.5,3.5)) 
#        ax = fig.add_subplot(111)
#        ax.plot(np.arange(n_exp),fitness ,'bo',alpha=0.1,mec='None')
#        ax.set_ylabel(fitness_measure_name)
#        ax.set_xlabel(r'experiment number')  
#        if fold.startswith('/Users/ashish'): plt.tight_layout()
#        else: fig.set_tight_layout(True)  #plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
#        plt.savefig(fold+file_suffix+"_"+fitness_measure_suffix+".png",dpi=120)

############# Ruggedness computed only if landscape is not super flat! ############################
    
    if np.abs( np.std(fitness)/np.mean(fitness) )<1e-5 or (np.abs(np.mean(fitness))<1e-5 and np.abs(np.std(fitness))<1e-5) :
        print ('this is a very flat landscape, for example landscape of focal species abundance in the non-interacting case is a constant')
        metric_analysis_dict.update({'degenerate_landscape_flag':True})
        
# =============================================================================
#         #########saving default values: ###############
#         ##turned off to have Nones for 2 species coexistence stuff
#         print ('saving default values')
#         metric_analysis_dict.update({'mean':np.mean(fitness),'variance':np.var(fitness)})
#         metric_analysis_dict.update({'r/s':0.,'roughness':0.,'slope':0.}) ##setting default r/s to 0 even though undefined actually.'''
#         metric_analysis_dict.update({'r/s_w0':0.,'roughness_w0':0.,'slope_w0':0.})
#         metric_analysis_dict.update({'nn_correlation':1.,'corr_length_nn':np.NaN})
#         metric_analysis_dict.update({'FDC':0.,'ranked_FDC':0.})            
#         metric_analysis_dict={'Greedy_walk_Failed_Flag':False,
#                   'p_Greediest_walk_found_global_max':1.,'mean_steps_Greediest':0,'mean_steps_Greediest_global':0,'mean_relative_optimum_Greediest':1.,
#                   'p_Greedy_walk_found_global_max':1.,'mean_steps_Greedy':0.,'mean_steps_Greedy_global':0, 'mean_relative_optimum_Greedy':1.}
#         metric_analysis_dict.update({'R2_linear':1., 'variance_explained_linear':1.})
#         metric_analysis_dict.update({'SWO':0.})
# =============================================================================
       
        metric_analysis_dict_Prefixed ={fitness_measure_name+': '+k: v for k, v in metric_analysis_dict.items()}  
        return metric_analysis_dict_Prefixed
    
############# ########################################## ############################
   
    else: # we actually run the analysis
        metric_analysis_dict.update({'degenerate_landscape_flag':False})
    if 'basic' in analysis_options:
        metric_analysis_dict.update({'mean':np.mean(fitness),
                                        'variance':np.var(fitness),
                                        'coefficient of variation':np.std(fitness)/np.mean(fitness) })   
    

    
        
    if 'n_max_min' in  analysis_options:
        ## does both extrema analysis and Fraction of neutral links
        #start = time.time()
#        maxima, minima, absolute_maxima, absolute_minima, maxima_without_extinction, minima_without_extinction,maxima_identical_to_nn, minima_identical_to_nn, pure_saddle=find_extrema(fitness, sigma_vectors, sigma_vectors_final, adjacency_matrix, S, n_exp, rel_threshold=1e-4  )   
        maxima, minima, maxima_without_extinction, minima_without_extinction, nsp_in_max_no_extinction, nsp_in_min_no_extinction, N_invadable_directions_max_no_extinction, N_invadable_directions_min_no_extinction, pure_saddle, links_dict=find_extrema(fitness, sigma_vectors, sigma_vectors_final, adjacency_matrix, S, n_exp, rel_threshold=1e-4  )   
        
        ## to save space, many lists are not saved in the dict. Fraction of neutral links is in links   dict
        metric_analysis_dict.update({'maxima':None,'minima':None,'N_maxima':len(maxima),'N_minima':len(minima),
                                      'maxima_without_extinction': None,'minima_without_extinction': None,
                                      'N_maxima_without_extinction': len(maxima_without_extinction),'N_minima_without_extinction': len(minima_without_extinction),
                                         'nsp_in_max_no_extinction':None,'nsp_in_min_no_extinction':None,                              
                                         'N_invadable_directions_max_no_extinction': None, 'N_invadable_directions_min_no_extinction':None,
                                         'pure_saddle':None, 'N_saddle':len(pure_saddle)})## to reduce size of analysis data, a lot of this is not saved any more
        metric_analysis_dict.update(links_dict)## for the fraction of neutral links and such  Fneut          
        #end = time.time()
        #print("time to find maxima and minima",end-start)
        
        
        
        
    if 'r/s' in analysis_options:
        r_roughness,s_slope=r_over_s(fitness, sigma_vectors) 
        if s_slope>1e-5:
            r_s_val=r_roughness/s_slope
        else:
            r_s_val=0.
                 
        metric_analysis_dict.update({'r/s':r_s_val,'roughness':r_roughness,'slope':s_slope})
        ### without 0
        r_roughness,s_slope=r_over_s(fitness, sigma_vectors,remove0=True) 
        if s_slope>1e-5:
            r_s_val=r_roughness/s_slope
        else:
            r_s_val=0.
        metric_analysis_dict.update({'r/s_w0':r_s_val,'roughness_w0':r_roughness,'slope_w0':s_slope})
        
        
    if 'FourierPS' in analysis_options:
#        print('skipping fourier')
        start = time.time()
        Fourier_coefficients, Power_spectrum, PS_ratio, beta_arr =Fourier_roughness (fitness, sigma_vectors, S, n_exp )
        metric_analysis_dict.update({'Fourier_coefficients':Fourier_coefficients,
                                        'Power_spectrum':Power_spectrum,'PS_ratio':PS_ratio})
            
        end = time.time()
        print("time to find FourierPS",end-start)    
        
    if 'estimation_from_limited_data' in analysis_options:
        
        start = time.time()
#        est_dict=estimate_beta_via_random_walks(fitness, sigma_vectors, adjacency_matrix, S, n_exp, fig_name=fold+'EstimateBeta_'+fitness_measure_suffix,frac_data_list=fraction_limited_data )
#        metric_analysis_dict.update(est_dict)
        
        est_dict=estimate_rs_from_sparse_data(fitness, sigma_vectors,  S, n_exp, frac_data_list=fraction_limited_data, n_replicates=10)
        metric_analysis_dict.update(est_dict)
        if 'n_max_min' not in  analysis_options:
            _,_, maxima_without_extinction,_,_, _,_, _,_,_,_=find_extrema(fitness, sigma_vectors, sigma_vectors_final, adjacency_matrix, S, n_exp, rel_threshold=1e-4 )   
        est_dict=estimate_FDC_from_sparse_data(fitness, sigma_vectors, maxima_without_extinction, S, n_exp, frac_data_list=fraction_limited_data, n_replicates=10 )
        metric_analysis_dict.update(est_dict)  
        
        est_dict=estimate_variance_explained_from_sparse_data(fitness, sigma_vectors, S, n_exp, frac_data_list=fraction_limited_data, n_replicates=10)
        metric_analysis_dict.update(est_dict)  
        
        est_dict=estimate_CorrLength_FNeutral_from_sparse_data(fitness, sigma_vectors, adjacency_matrix, S, n_exp, frac_data_list=fraction_limited_data, n_replicates=10)
        metric_analysis_dict.update(est_dict)  
       
        end = time.time()
        print("time to estimate from limited data",end-start)       

            
    if 'SWO' in analysis_options: ## SWO stands for sitewise optimizability
        SWO,_=sitewise_optimizability(fitness, sigma_vectors, S, n_exp, adjacency_matrix)
        metric_analysis_dict.update({'SWO':SWO})
        
    if 'spatial_correlation' in analysis_options: 
        nn_correlation, corr_length_nn=spatial_correlation(fitness, sigma_vectors, adjacency_matrix )
        metric_analysis_dict.update({'nn_correlation':nn_correlation,'corr_length_nn':corr_length_nn})
        
        
    if 'FDC' in analysis_options:
        if 'n_max_min' not in  analysis_options: ## we need "maxima_without_extinction" to find FDC
            _, _, maxima_without_extinction, _, _,_, _, _, _,_=find_extrema(fitness, sigma_vectors, sigma_vectors_final, adjacency_matrix, S, n_exp, rel_threshold=1e-4  )   
        FDC=fitness_distance_correlation(fitness, sigma_vectors, S, n_exp, maxima_without_extinction=maxima_without_extinction)
        metric_analysis_dict.update({'FDC':FDC})            
        ranked_FDC=ranked_fitness_distance_correlation(fitness, sigma_vectors, S, n_exp)
        metric_analysis_dict.update({'ranked_FDC':ranked_FDC}) 
    if 'greedy_walk' in analysis_options:

        start = time.time()                
        if 'n_max_min' not in  analysis_options: # if this was not calculated already
            _, _, maxima_without_extinction, _, _,_, _, _, _,_=find_extrema(fitness, sigma_vectors, sigma_vectors_final, adjacency_matrix, S, n_exp, rel_threshold=1e-4  )   
        results_dict=performance_of_greedy_walk(fitness, sigma_vectors, adjacency_matrix, S, n_exp, maxima_without_extinction, rel_threshold=1e-4)
        if results_dict!=None:  ## this is only performed for some non-degenerate fitness measure          
            metric_analysis_dict.update(results_dict)
        else:
            metric_analysis_dict.update({'greedy walk failed':True})
            
        end = time.time()
        print("time to find greedywalk",end-start)

    if 'compare_search_protocols' in analysis_options:
        start = time.time()     
        compared_search_protocols=True
        if Nfull is  None or sigma_full is None:
            print ('simulating these heuristics needs Nf, full landscapes')
            compared_search_protocols=False
        elif len(sigma_full)!=n_exp:
            print('bottlenecking cannot be run on sub-landscapes, like the ones used in focal species abundance')
            compared_search_protocols=False
        
        if compared_search_protocols:
            ## for adaptive  biomass conversion (avg num cells on a landscape is fixed to 1e9)
            results_dict=compare_search_protocols(Nfull, fit_full, sigma_full, sigmaFinal_full, adjacency_matrix, len(sigma_full[0]), len(sigma_full), 
                                                  dilution_factor_list=[1e-6, 1e-7, 1e-8, 1e-9, 1e-10 ], biomass_conversion='adaptive', rel_threshold=1e-4, n_iterations=int(S/2), n_averages=20  )
                                                    #old default was [1e-7, 1e-8, 5e-9, 2e-9 ]
            results_dict.update({'compared_search_protocols':compared_search_protocols})
            
            ## for fixed biomass conversion 1 cell is the popn cutoff value
            results_dict2=compare_search_protocols(Nfull, fit_full, sigma_full, sigmaFinal_full, adjacency_matrix, len(sigma_full[0]), len(sigma_full), 
                                                  dilution_factor_list=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], biomass_conversion='fixed', rel_threshold=1e-4, n_iterations=int(S/2), n_averages=20  )
            results_dict2.update({'compared_search_protocols-fixed_conversion':compared_search_protocols})
                
        metric_analysis_dict.update(results_dict)
        metric_analysis_dict.update(results_dict2)
 
            
    
    if 'variance_explained' in analysis_options:## this is variance in fitting, not in prediction!
        R2_expOrder,VarExp_expOrder,_ =variance_explained(fitness, sigma_vectors, S, n_exp,minExpOrder=6)
        metric_analysis_dict.update({'R2_reg':R2_expOrder, 'R2_linear':R2_expOrder[1], 'variance_explained_reg':VarExp_expOrder, 'variance_explained_linear':VarExp_expOrder[1]})        
        ### we don't compare with Random forest cos it's not a fair comparison, and to speed up analysis
        #        RF_vs_linReg.Rforest_vs_LinReg_when_fitting(sigma_vectors, fitness, fold, n_estimatorsRF = 10, rand_state=0, fitness_name=fitness_measure_name, suffix=fitness_measure_suffix, return_stuff=False)                   
        
# =============================================================================
#         ## without the no species present data point.
#         R2_expOrder,VarExp_expOrder,_ =variance_explained(fitness, sigma_vectors, S, n_exp,remove0=True)
#         metric_analysis_dict.update({'R2_reg_w0':R2_expOrder, 'R2_linear_w0':R2_expOrder[1],'variance_explained_reg_w0':VarExp_expOrder, 'variance_explained_linear_w0':VarExp_expOrder[1]})       
# #        fitness_w0=remove_zero_data_point(sigma_vectors, fitness=fitness)
# #        sigma_vectors_w0=remove_zero_data_point(sigma_vectors)
# #        RF_vs_linReg.Rforest_vs_LinReg_when_fitting(sigma_vectors_w0, fitness_w0, fold, n_estimatorsRF = 10, rand_state=0, fitness_name=fitness_measure_name, suffix=fitness_measure_suffix+'_w0', return_stuff=False)   
# =============================================================================
             
    if 'LOOCV' in analysis_options :
        r2_RF, r2_LinReg, _, _ =RF_vs_linReg.Rforest_vs_LinReg_LOOCV(sigma_vectors, fitness, fold, n_estimatorsRF = 10, rand_state=0, fitness_name=fitness_measure_name, suffix=fitness_measure_suffix, return_stuff=True)   
        metric_analysis_dict.update({'R2_linReg_LOOCV':r2_LinReg,'R2_RF_LOOCV':r2_RF})
    
    if 'RandomWalk' in analysis_options:
        #start = time.time()     
        n_steps=1000
        n_averages=1
        AutoCorr= random_walk_performance(fitness, adjacency_matrix, S, n_exp, Nsteps=n_steps, n_init_nodes=n_averages )
        metric_analysis_dict.update({'Random_Walk_AutoCorrelation':AutoCorr,'Random_Walk_Nsteps':n_steps,'Random_Walk_Navg':n_averages })
        fig = plt.figure(figsize=(3.5,3.5)) 
        ax = fig.add_subplot(111)
        ax.semilogy(np.arange(40), AutoCorr [:40] ,'bo',alpha=0.1,mec='None')
        ax.set_ylabel(fitness_measure_name+'AutoCorrelation')
        ax.set_xlabel(r'number of steps')  
        if fold.startswith('/Users/ashish'): plt.tight_layout()
        else: fig.set_tight_layout(True)  #plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
#        plt.show()
        plt.savefig(fold+file_suffix+"_"+fitness_measure_suffix+"_AutoCorr.png",dpi=120)
        #end = time.time()
        #print("time to find Randomwalk",end-start)
        
    if 'basic' in analysis_options:
        fig = plt.figure(figsize=(3.5,3.5)) 
        ax = fig.add_subplot(111)
        ax.plot(np.arange(n_exp),fitness ,'bo',alpha=0.1,mec='None')
        ax.set_ylabel(fitness_measure_name)
        ax.set_xlabel(r'experiment number')  
        if fold.startswith('/Users/ashish'): plt.tight_layout()
        else: fig.set_tight_layout(True)  #plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
        plt.savefig(fold+file_suffix+"_"+fitness_measure_suffix+".png",dpi=120)
        

    
    if 'FourierPS' in analysis_options:
        
        fig = plt.figure(figsize=(3.5,3.5)) 
        ax = fig.add_subplot(111)
        ax.plot(np.arange(S)+1, Power_spectrum[1:],'bo',mec='None')
        ax.set_ylabel(r'Variance decomposition')
        ax.set_xlabel(r'Order of expansion') 
        ax.set_title(r'Power spectrum of '+fitness_measure_name,weight='bold')         
        if 'r/s' in metric_analysis_dict: plt.text(0.92, 0.89,'r/s= ' + '{:.1e}'.format( metric_analysis_dict['r/s'] ) , horizontalalignment='right', transform=ax.transAxes) 
        if 'r/s_w0' in metric_analysis_dict: plt.text(0.92, 0.84,'r/s _w0= ' + '{:.1e}'.format( metric_analysis_dict['r/s_w0'] ) , horizontalalignment='right', transform=ax.transAxes) 
        plt.text(0.92, 0.95,r'PS ratio ' + '{:.1e}'.format( PS_ratio ) , horizontalalignment='right', transform=ax.transAxes) 
        
        if fold.startswith('/Users/ashish'): plt.tight_layout()
        else: fig.set_tight_layout(True)  #plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)      
        ax.set_yscale('log')
        if np.max(Power_spectrum[1:])>1e-8:
            ax.set_ylim( min(1e-8 ,np.min(Power_spectrum[1:])/5) , np.max(Power_spectrum[1:])*5)
        plt.savefig(fold+file_suffix+"_"+fitness_measure_suffix+"_PS.png",dpi=120)    
        
    if 'variance_explained' in analysis_options:
        
#        fig = plt.figure(figsize=(3.5,3.5)) 
#        ax = fig.add_subplot(111)
#        if 'variance_explained_reg' in metric_analysis_dict:
#            ax.plot(np.arange(len(VarExp_expOrder)), metric_analysis_dict['variance_explained_reg'],'bo',mec='None', label='with 0')
#        if 'variance_explained_reg_w0' in metric_analysis_dict:
#            ax.plot(np.arange(len(VarExp_expOrder)),  metric_analysis_dict['variance_explained_reg_w0'],'go',mec='None', label='without 0')
#        ax.legend(loc='best')      
#        ax.set_ylabel(r'Variance explained ')
#        ax.set_xlabel(r'Order of expansion') 
#        ax.plot(ax.get_xlim(),[1.0,1.0],'k--')                  
#        if fold.startswith('/Users/ashish'): plt.tight_layout()
#        else: fig.set_tight_layout(True)  #plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)              
#        plt.savefig(fold+file_suffix+"_"+fitness_measure_suffix+"_VarExp.png",dpi=120) 
        fig = plt.figure(figsize=(3.5,3.5)) 
        ax = fig.add_subplot(111)
        if 'variance_explained_reg' in metric_analysis_dict:
            ax.plot(np.arange(len(VarExp_expOrder)), 1.0-metric_analysis_dict['variance_explained_reg'],'bo',mec='None', label='with 0')
        if 'variance_explained_reg_w0' in metric_analysis_dict:
            ax.plot(np.arange(len(VarExp_expOrder)), 1.0- metric_analysis_dict['variance_explained_reg_w0'],'go',mec='None', label='without 0')
        ax.set_ylabel(r'Variance unexplained ')
        ax.set_xlabel(r'Order of expansion') 
        ax.set_yscale('log')   
        ax.legend(loc='best')               
        if fold.startswith('/Users/ashish'): plt.tight_layout()
        else: fig.set_tight_layout(True)  #plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)              
        plt.savefig(fold+file_suffix+"_"+fitness_measure_suffix+"_VarUnexp.png",dpi=120) 
           
    plt.close("all")
    ####### add a prefix to all measured values identify the fitness metric used ########
    metric_analysis_dict_Prefixed ={fitness_measure_name+': '+k: v for k, v in metric_analysis_dict.items()}   
    return metric_analysis_dict_Prefixed


def main():
    analysis_overwrite=False ## overwrites existing dictionary if true
    analysis_options= None
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="data path", default=None) ## only a single folder handled for now!
    parser.add_argument("-p", help="folder to put plots in", default=None)
    parser.add_argument("-a", help="analysis_options", default=None)
    
#    parser.add_argument("-v", help="param value", default=None)
    args = parser.parse_args()
    folder_name_list=[]
    plots_folders_list=[]
    analysis_options=args.a
#    analysis_options= ['only_landscape']
    print ('modified now for running only lansdscape properties')

    
    
    if args.p is not None:
        plots_folders_list.append(args.p)## only a single folder handled for now!!
        print ('plots_folders_list was provided as \n',plots_folders_list)   
    if args.d is not None:
        folder_name_list.append(args.d)## only a single folder handled for now!!
    
        
    else:              
        ## if false, reads original data and overwrite only the entries analyzed right now.
#        base_dir="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/"
#        folder_name_list=[base_dir+'/CR_wGamma_CbinarydiagGamma_meanC_R0Gamma4,5_compiled/SM10_meanC1/0/']
        
        base_dir='/Users/ashish/Downloads/'
        folder_name_list=[base_dir+'/2/']
        
                
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
    #    base_dir="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/Crossfeeding_AllEatR0_increasedVar/"    
    #    folder_name_list= [base_dir+'/Crossfeeding_alleatR0_meanC2_compiled/', base_dir+'/Crossfeeding_alleatR0_meanC3_compiled/', 
    #                       base_dir+'/Crossfeeding_alleatR0_meanC4_compiled/', base_dir+'/Crossfeeding_alleatR0_meanC5_compiled/',
    #                       base_dir+'/Crossfeeding_alleatR0_meanC6_compiled/', base_dir+'/Crossfeeding_alleatR0_meanC8_compiled/',
    #                       base_dir+'/Crossfeeding_alleatR0_meanC10_compiled/',base_dir+'/Crossfeeding_alleatR0_meanC12_compiled/']
    #    
        
        
    #    base_dir="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/Crossfeeding_uniformC/"
    #    base_dir="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/Crossfeeding_uniformC_increasedVar/"
    #    folder_name_list=[base_dir+'/Dbinary_SM10_meanC2_compiled/', base_dir+'/Dbinary_SM10_meanC3_compiled/', 
    #                       base_dir+'/Dbinary_SM10_meanC4_compiled/', base_dir+'/Dbinary_SM10_meanC5_compiled/',
    #                       base_dir+'/Dbinary_SM10_meanC6_compiled/', base_dir+'/Dbinary_SM10_meanC8_compiled/',
    #                       base_dir+'/Dbinary_SM10_meanC10_compiled/']

    
    
#        n_replicates=10
#        base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CF_wGamma_l_compiled/"
#        destfold_list=[base_fold+'/SM10_l0/',
#                base_fold+'/SM10_l1/', base_fold+'/SM10_l2/', 
#                   base_fold+'/SM10_l3/', base_fold+'/SM10_l4/', 
#                   base_fold+'/SM10_l5/', base_fold+'/SM10_l6/',
#                   base_fold+'/SM10_l7/', base_fold+'/SM10_l8/',
#                   base_fold+'/SM10_l9/']
#        parameter_name='l'
#        parameter_list=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        
#        base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CF_wGamma_meanC_compiled/"
#        base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CF_wGamma_AllRSupplied_meanC_compiled/"
#        destfold_list=[base_fold+'/SM10_meanC1/', base_fold+'/SM10_meanC2/', 
#                   base_fold+'/SM10_meanC3/', base_fold+'/SM10_meanC4/', 
#                   base_fold+'/SM10_meanC5/', base_fold+'/SM10_meanC6/',
#                   base_fold+'/SM10_meanC7/', base_fold+'/SM10_meanC8/',
#                   base_fold+'/SM10_meanC9/', base_fold+'/SM10_meanC10/'] 
#        parameter_name='mean_c'
#        parameter_list=[1,2,3,4,5,6,7,8,9,10]
        

    
# =============================================================================
# #        base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all//CR_wGamma_CbinaryGamma_meanC_compiled/"
#         base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CR_wGamma_CbinarydiagGamma_meanC_compiled/"
#         destfold_list=[base_fold+'/SM10_meanC1/', base_fold+'/SM10_meanC2/', 
#            base_fold+'/SM10_meanC3/', base_fold+'/SM10_meanC4/', 
#            base_fold+'/SM10_meanC5/', base_fold+'/SM10_meanC6/',
#            base_fold+'/SM10_meanC7/', base_fold+'/SM10_meanC8/']
#         parameter_list=[1,2,3,4,5,6,7,8]
# =============================================================================
        

# =============================================================================
#         base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CR_wGamma_CbinarydiagGamma_SoverR_varyS_compiled/"
#         destfold_list=[base_fold+'/S6_SoverR2/', base_fold+'/S8_SoverR2/', 
#            base_fold+'/S12_SoverR2/', base_fold+'/S14_SoverR2/']
#         parameter_list=[1,2,3,4,5,6,7,8]
#         parameter_name='SoverR_S'
#         SoverR=2
#         parameter_list=[6,8,12,14]
# =============================================================================


# =============================================================================
#         destfold_list,parameter_list=create_replicate_lists(parameter_list,destfold_list, n_replicates)
#         folder_name_list= destfold_list
# =============================================================================
    
    
    ###### for alleatR0, simpson diversity fails in one case because maxima without Extinction doesnt exist, probably due to bad relative thresholding

# =============================================================================
#     ####### for 2D plots
#         
#        base_fold="/projectnb/qedksk/ashishge/ecology/binaryIC_all/CR_cvxpt_2D_SM_meanC_compiled/"
#        destfold_list=[base_fold] 
#        parameter_name='2D'
#        parameter_list=OrderedDict([ ('SM',[2, 4, 6, 8 ,10, 12, 14, 16] ), ('mean_c', '1 to param1')])
#        parameter_names=list(parameter_list.keys())
#        destfold_list,parameter_list=create_replicate_lists(parameter_list,destfold_list, 10) ### creates a longer list enumerating all the folders.
#        folder_name_list=destfold_list
# =============================================================================
    
    
    
    
    
    
    
    for fold_idx, fold in enumerate(folder_name_list):
        print (fold) 
        file_suffix=None## finds the first file the matches patter with glob.glob   
        if file_suffix==None:## finds the first file the matches patter with glob.glob
            ctr=0
            for file_suffix in glob.glob(fold+'S*.dat'):
                file_suffix=file_suffix.replace(fold,'')
                file_suffix=file_suffix.replace('.dat','')
                print ("file_suffix is now ", file_suffix)
                ctr+=1
            if ctr==0:
                print ("this folder didnt complete", fold)
                continue
            assert ctr>0,"no file found"
            assert ctr<2,"we can only have one file that matches the pattern in the folder!"
           
        ####################        Reading data           #################### 
        data = pickle.load(open(fold+file_suffix+'.dat', 'rb')) 
        reached_SteadyState_Flag= np.all(data['reached_SteadyState']==True)
        
        
        
        
        ####################        choosing appropriate folder for the plots to be in         #################### 
# =============================================================================
#         if np.all(data['reached_SteadyState']==True):
#             print ("all runs had reached steady state")           
#             analysis_fold=fold            
#         else:
#             print ("NOT SS!")
#             analysis_fold=fold+'/notSS/'
#          
# =============================================================================

        analysis_fold=fold                      
        if len(plots_folders_list)!=0: ### plots should be in a separate folder
            analysis_fold= analysis_fold.replace(folder_name_list[fold_idx], plots_folders_list[fold_idx])
            
            
        if not os.path.exists(analysis_fold): os.makedirs(analysis_fold)    
        
        if os.path.isfile(fold+'analysis_'+file_suffix+'.dat') and analysis_overwrite==False:
            analysis_dict=pickle.load(open(fold+'analysis_'+file_suffix+'.dat', 'rb')) 
        else:
            analysis_dict={}
            analysis_overwrite=True
        
        #c_matrix=data['passed_params']['c']         
        sigma_vectors= (data['initial_abundance'].T>0).astype(int)
        
        analysis_dict.update({'popn_cutoff_OK':True})
        '''
        changed  sigma_vectors_final definition  on 2/13/20 (could have been a bug till now in defn of max without exctinction.)
        from: sigma_vectors_final= (data['steady_state'].T>0).astype(int)  
        to:   sigma_vectors_final= (N_obs>0).astype(int)
        to account for the popn cutoff while evaluation sigma vectors final.
        '''
        N_obs=data['steady_state'].T
        N_obs[N_obs<popn_cutoff]=0.0
        
        sigma_vectors_final= (N_obs>popn_cutoff).astype(int)#  sigma_vectors_final= (data['steady_state'].T>0).astype(int) 
        R_obs=data['steady_state_resources'].T if 'steady_state_resources' in data.keys() else None
        S=len(sigma_vectors[0])  
        M=len(R_obs[0])  
        n_exp=len(N_obs)
        print ('Starting analysis')  
        ####################        Running analysis           ####################      
        routine_analysis=True #True
        data_fractions_to_estimate_from=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        noise_strength_list=[0.001,0.003, 0.01, 0.03, 0.1, 0.3, 1., 3.] 
        diff_sigma=(sigma_vectors_final-sigma_vectors).astype(int)
        if np.any(diff_sigma>0):
            print ('round off error!, some zombie species has arisen')
            print (fold)
            print (np.where(diff_sigma==0)[0])
            print (diff_sigma[np.where(diff_sigma==0)[0][-1]])
            print (sigma_vectors[np.where(diff_sigma==0)[0][-1]])
            print (sigma_vectors_final[np.where(diff_sigma==0)[0][-1]])
            
            analysis_dict.update({'popn_cutoff_OK':False})
            if len(plots_folders_list)!=0: ### we save the data along with the plots and without the plots in the original folder as well
                analysis_data_only_fold= analysis_fold.replace(plots_folders_list[fold_idx],folder_name_list[fold_idx] )                  
                if not os.path.exists(analysis_data_only_fold): os.makedirs(analysis_data_only_fold)  
                pickle.dump(analysis_dict, open(analysis_data_only_fold+'analysis_'+file_suffix+'.dat', 'wb') ) 
            print (shit)
        
        if analysis_options==None:
            #analysis_options=['FourierPS','r/s','n_max_min','basic','greedy_walk','RandomWalk','SWO','FDC','estimation_from_limited_data','spatial_correlation','variance_explained'] #analysis_options=['LOOCV']
            #analysis_options=['greedy_walk'] #analysis_options=['LOOCV']           
            
#            analysis_options=['compare_search_protocols','r/s','n_max_min','basic','greedy_walk','FDC','estimation_from_limited_data']#,'spatial_correlation','FourierPS'] #analysis_options=['LOOCV']
#            analysis_options=['compare_search_protocols','r/s','spatial_correlation','n_max_min','basic','greedy_walk','estimation_from_limited_data']
#            analysis_options_QUICK=['r/s','spatial_correlation','n_max_min','basic']
            analysis_options=['r/s','spatial_correlation','n_max_min','basic','greedy_walk','compare_search_protocols']
            routine_analysis=False

        
        start = time.time()
        adjacency_matrix=generate_adjacency_matrix(sigma_vectors, S, n_exp )
        end = time.time()
        print("time to make adjacency matrix",end-start)

                 
        ###### running analysis on metrics.

# =============================================================================
#         ####################        Shannon Diversity          #################### 
#         start = time.time()
#         ShannonDiv,normalized_ShannonDiv=Shannon_diversity(N_obs, sigma_vectors, S, n_exp)
#         fitness=normalized_ShannonDiv
#         ShannonDiv_dict=analyze_for_chosen_metric(fitness, sigma_vectors, sigma_vectors_final, adjacency_matrix, S, n_exp, 'Shannon Diversity', 
#                                                      analysis_fold, file_suffix, analysis_options, fraction_limited_data=data_fractions_to_estimate_from, 
#                                                      Nfull=N_obs, sigma_full=sigma_vectors, sigmaFinal_full=sigma_vectors_final, fit_full=fitness)
#         analysis_dict.update(ShannonDiv_dict) 
#         plt.close("all")
#         end = time.time()
#         print("time to analyze for Shannon Div",end-start)
#         
#         start = time.time()
#         for strength in noise_strength_list:
#             fitness_suffix='Shannon Diversity-ADDnoise_'+"{:.2e}".format(strength)
#             noisy_ShannonDiv=generate_noisy_measurements(normalized_ShannonDiv, kind='additive', strength=strength)
#             noisy_ShannonDiv_dict=analyze_for_chosen_metric(noisy_ShannonDiv, sigma_vectors, sigma_vectors_final, adjacency_matrix, S, n_exp, 
#                                                      fitness_suffix, 
#                                                      analysis_fold, file_suffix, analysis_options_QUICK, fraction_limited_data=data_fractions_to_estimate_from, 
#                                                      Nfull=N_obs, sigma_full=sigma_vectors, sigmaFinal_full=sigma_vectors_final, fit_full=noisy_ShannonDiv)
#             analysis_dict.update(noisy_ShannonDiv_dict) 
#         analysis_dict.update({'Shannon Diversity-ADDnoise_strengths':noise_strength_list})
#         for strength in noise_strength_list:    
#             fitness_suffix='Shannon Diversity-MULTnoise_'+"{:.2e}".format(strength)
#             noisy_ShannonDiv=generate_noisy_measurements(normalized_ShannonDiv, kind='multiplicative', strength=strength)
#             noisy_ShannonDiv_dict=analyze_for_chosen_metric(noisy_ShannonDiv, sigma_vectors, sigma_vectors_final, adjacency_matrix, S, n_exp, 
#                                                      fitness_suffix, 
#                                                      analysis_fold, file_suffix, analysis_options_QUICK, fraction_limited_data=data_fractions_to_estimate_from, 
#                                                      Nfull=N_obs, sigma_full=sigma_vectors, sigmaFinal_full=sigma_vectors_final, fit_full=noisy_ShannonDiv)
#             analysis_dict.update(noisy_ShannonDiv_dict)   
#             
#         analysis_dict.update({'Shannon Diversity-MULTnoise_strengths':noise_strength_list})
#         end = time.time()    
#         print("time to analyze for noisy Shannon Div",end-start)    
# =============================================================================
            
# =============================================================================
#         ####################        Simpson's diversity         #################### 
#         SimpsonsDiv, SimpsonsDom, SimpsonsE=diversity_index(N_obs, sigma_vectors, S, n_exp)
#         fitness=SimpsonsDiv 
#         SimpsonsDiv_dict=analyze_for_chosen_metric(fitness, sigma_vectors, sigma_vectors_final, adjacency_matrix, S, n_exp, 'Simpsons Diversity', 
#                                                    analysis_fold, file_suffix, analysis_options, fraction_limited_data=data_fractions_to_estimate_from, 
#                                                    Nfull=N_obs, sigma_full=sigma_vectors, sigmaFinal_full=sigma_vectors_final, fit_full=fitness)
#         analysis_dict.update(SimpsonsDiv_dict)
#         plt.close("all")        
# =============================================================================
        

    
        
        ####################     Clark et al Butyrate production Function        #################### 
        start = time.time()
        
        fitness=butyrate_M3_Clark(N_obs, sigma_vectors_final)
        ClarkM3_dict=analyze_for_chosen_metric(fitness, sigma_vectors, sigma_vectors_final, adjacency_matrix, S, n_exp, 'Clark_M3', 
                                                     analysis_fold, file_suffix, analysis_options, fraction_limited_data=data_fractions_to_estimate_from,
                                                     Nfull=N_obs, sigma_full=sigma_vectors, sigmaFinal_full=sigma_vectors_final, fit_full=fitness)
        analysis_dict.update(ClarkM3_dict) 
        plt.close("all")
        end = time.time()
        print("time to analyze for resource_prod",end-start)





        
        
         
        '''
        for focal species (N), we need to worry about experiments where focal species was present at the beginning.
        This is what we call subsetted landscape.
        community function is  calculated on the full data  and then all the matrices are  subsetted so that  we consider only focal species present experiments.
        Paper reports reuslts on the subsetted landscape.
        '''   


# =============================================================================
#         ####################        Focal species abundance         #################### 
#         if S>=4: ## else we dont analyse focal species cos it doesnt make sense to.
#             start = time.time()       
# #            for i in range(S): ### to save on some time on analysis, we run for a smaller number of these.
#             for i in range(min(S,5)):#
#                 '''
#                 we don't analyze for all species to save computation time.
#                 '''                               
#                 fitness=focal_species_abundance(N_obs,i)   
#                 Nsub, sigma_sub, sigmaf_sub, adj_sub, fit_sub=subsetting_for_focal_species(N_obs, sigma_vectors, sigma_vectors_final, adjacency_matrix, fitness, n_exp, S, i)            
#                 focal_abundance_dict=analyze_for_chosen_metric(fit_sub, sigma_sub, sigmaf_sub, adj_sub, S-1, int(n_exp/2), f'Species {i}', 
#                                                                analysis_fold, file_suffix, analysis_options, fraction_limited_data=data_fractions_to_estimate_from, 
#                                                                Nfull=N_obs, sigma_full=sigma_vectors, sigmaFinal_full=sigma_vectors_final, fit_full=fitness)
#                 analysis_dict.update(focal_abundance_dict)
#                 
#                 if i <3: ## to save time on noisy runs
#                     for strength in noise_strength_list:
#                         fitness_suffix=f'Species {i}-ADDnoise_'+"{:.2e}".format(strength)
#                         fitness=focal_species_abundance(N_obs,i) 
#                         noisy_abundance_full=generate_noisy_measurements(fitness, kind='additive', strength=strength)
#                         Nsub, sigma_sub, sigmaf_sub, adj_sub, fit_sub=subsetting_for_focal_species(N_obs, sigma_vectors, sigma_vectors_final, adjacency_matrix, fitness, n_exp, S, i)            
#                         noisy_abundance_dict=analyze_for_chosen_metric(fit_sub, sigma_sub, sigmaf_sub, adj_sub, S-1, int(n_exp/2), 
#                                                                        fitness_suffix, 
#                                                                       analysis_fold, file_suffix, analysis_options, fraction_limited_data=data_fractions_to_estimate_from,
#                                                                       Nfull=N_obs, sigma_full=sigma_vectors, sigmaFinal_full=sigma_vectors_final, fit_full=noisy_abundance_full)
#                         analysis_dict.update(noisy_abundance_dict)             
#                     analysis_dict.update({f'Species {i}-ADDnoise_strengths':noise_strength_list})
#                     
#                     for strength in noise_strength_list:
#                         fitness_suffix=f'Species {i}-MULTnoise_'+"{:.2e}".format(strength)
#                         fitness=focal_species_abundance(N_obs,i) 
#                         noisy_abundance_full=generate_noisy_measurements(fitness, kind='multiplicative', strength=strength)
#                         Nsub, sigma_sub, sigmaf_sub, adj_sub, fit_sub=subsetting_for_focal_species(N_obs, sigma_vectors, sigma_vectors_final, adjacency_matrix, fitness, n_exp, S, i)            
#                         noisy_abundance_dict=analyze_for_chosen_metric(fit_sub, sigma_sub, sigmaf_sub, adj_sub, S-1, int(n_exp/2), 
#                                                                        fitness_suffix, 
#                                                                       analysis_fold, file_suffix, analysis_options, fraction_limited_data=data_fractions_to_estimate_from,
#                                                                       Nfull=N_obs, sigma_full=sigma_vectors, sigmaFinal_full=sigma_vectors_final, fit_full=noisy_abundance_full)
#                         analysis_dict.update(noisy_abundance_dict)             
#                     analysis_dict.update({f'Species {i}-MULTnoise_strengths':noise_strength_list})
#                     
#             end = time.time()
#             print('time to analyze for all Species',end-start)
# =============================================================================
        
         
# =============================================================================
#         ####################  Focal species productivity on subsetted landscape     ####################  
#         if S>=8: ## else we dont analyse focal species coexistences or 2focalspecies product 
#             
#             start = time.time()       
#             
#             for i in [0,2,3]: ### to save on some time on analysis, we run for a smaller number of these.     
#                 
#                 fitness=focal_species_coexistence_product(N_obs, i,sigma_vectors_final)             
#                 Nsub, sigma_sub, sigmaf_sub, adj_sub, fit_sub=subsetting_for_focal_species(N_obs, sigma_vectors, sigma_vectors_final, adjacency_matrix, fitness, n_exp, S, i)            
#                 focal_coex_product_dict=analyze_for_chosen_metric(fit_sub, sigma_sub, sigmaf_sub, adj_sub, S-1, int(n_exp/2), f'Species {i} coex_product', 
#                                                                analysis_fold, file_suffix, analysis_options, fraction_limited_data=data_fractions_to_estimate_from, 
#                                                                Nfull=N_obs, sigma_full=sigma_vectors, sigmaFinal_full=sigma_vectors_final, fit_full=fitness)
#                 analysis_dict.update(focal_coex_product_dict)
#                 
#                 ## noisy landscapes
#                 if i <3: ## to save time on noisy runs
#                     for strength in noise_strength_list:
#                         fitness_suffix=f'Species {i} coex_product-ADDnoise_'+"{:.2e}".format(strength)
#                         noisy_abundance_full=generate_noisy_measurements(fitness, kind='additive', strength=strength)
#                         Nsub, sigma_sub, sigmaf_sub, adj_sub, fit_sub=subsetting_for_focal_species(N_obs, sigma_vectors, sigma_vectors_final, adjacency_matrix, noisy_abundance_full, n_exp, S, i) 
#                         noisy_abundance_dict=analyze_for_chosen_metric(fit_sub, sigma_sub, sigmaf_sub, adj_sub, S-1, int(n_exp/2), 
#                                                                        fitness_suffix, 
#                                                                    analysis_fold, file_suffix, analysis_options, fraction_limited_data=data_fractions_to_estimate_from, 
#                                                                    Nfull=N_obs, sigma_full=sigma_vectors, sigmaFinal_full=sigma_vectors_final, fit_full=fitness)
#                         analysis_dict.update(noisy_abundance_dict)             
#                     analysis_dict.update({f'Species {i} coex_product-ADDnoise_strengths':noise_strength_list}) 
#                     
#                     for strength in noise_strength_list:
#                         fitness_suffix=f'Species {i} coex_product-MULTnoise_'+"{:.2e}".format(strength)
#                         noisy_abundance_full=generate_noisy_measurements(fitness, kind='multiplicative', strength=strength)
#                         Nsub, sigma_sub, sigmaf_sub, adj_sub, fit_sub=subsetting_for_focal_species(N_obs, sigma_vectors, sigma_vectors_final, adjacency_matrix, noisy_abundance_full, n_exp, S, i) 
#                         noisy_abundance_dict=analyze_for_chosen_metric(fit_sub, sigma_sub, sigmaf_sub, adj_sub, S-1, int(n_exp/2), 
#                                                                        fitness_suffix, 
#                                                                    analysis_fold, file_suffix, analysis_options, fraction_limited_data=data_fractions_to_estimate_from, 
#                                                                    Nfull=N_obs, sigma_full=sigma_vectors, sigmaFinal_full=sigma_vectors_final, fit_full=fitness)
#                         analysis_dict.update(noisy_abundance_dict)             
#                     analysis_dict.update({f'Species {i} coex_product-MULTnoise_strengths':noise_strength_list}) 
#                     
#             end = time.time()
#             print('time to analyze for focal coex product',end-start)
#             
# =============================================================================
            
# =============================================================================
#         ####################  Focal species productivity on Full Landscape     ####################      
#         if S>=8:# else we dont analyse focal species coexistences or 2focalspecies product 
#             '''to compare search protocols, one needs to simulate search and measure ruggedness on full landscape'''
#             start = time.time()       
#             ### to save on some time on analysis, we run for a smaller number of these.
#             for i in [2]:     
#                 fitness=focal_species_coexistence_product(N_obs, i,sigma_vectors_final)             
#                 
#                 focal_coex_product_dict=analyze_for_chosen_metric(fitness, sigma_vectors, sigma_vectors_final, adjacency_matrix, S, n_exp, f'Species {i} product FullLandscape', 
#                                                                analysis_fold, file_suffix, analysis_options, fraction_limited_data=data_fractions_to_estimate_from, 
#                                                                Nfull=N_obs, sigma_full=sigma_vectors, sigmaFinal_full=sigma_vectors_final, fit_full=fitness)
#                 analysis_dict.update(focal_coex_product_dict)
#             end = time.time()
#             print('time to analyze for focal coex product on full landscape',end-start)
# 
# =============================================================================
     
        
        '''
        for pariwise productivity, again we compute the community function on both full landscape and the subsetted landscape.
        The paper reports results on the full landscape.
        '''  
        
        
        
# =============================================================================
#         ####################      Pairwise productivity on subsetted landscape        ####################     
#         if S>=8:                  
#             start = time.time()       
#             ### to save on some time on analysis, we run for a smaller number of these.
#             for i,j in [[2,3],[0,3]] :     
#                 assert (j>i),'just a convention for subsetting the arrays'
#                 fitness=two_species_product(N_obs, i,j ,sigma_vectors_final)             
#                 Nsub1, sigma_sub1, sigmaf_sub1, adj_sub1, fit_sub1=subsetting_for_focal_species(N_obs, sigma_vectors, sigma_vectors_final, adjacency_matrix, fitness, n_exp, S, i)                            
#                 Nsub, sigma_sub, sigmaf_sub, adj_sub, fit_sub=subsetting_for_focal_species(Nsub1, sigma_sub1, sigmaf_sub1, adj_sub1, fit_sub1, int(n_exp/2), S-1, j-1) ###index is j-1 because the arrays have been subsetted already!!          
#                 focal_coex_product_dict=analyze_for_chosen_metric(fit_sub, sigma_sub, sigmaf_sub, adj_sub, S-2, int(n_exp/4), f'Species {i},{j} coex_product', 
#                                                                analysis_fold, file_suffix, analysis_options, fraction_limited_data=data_fractions_to_estimate_from,
#                                                                Nfull=N_obs, sigma_full=sigma_vectors, sigmaFinal_full=sigma_vectors_final,fit_full=fitness)
#                 analysis_dict.update(focal_coex_product_dict)      
#                 
#                 ## noisy landscape
#                 for strength in noise_strength_list:
#                     fitness_suffix=f'Species {i},{j} coex_product-ADDnoise_'+"{:.2e}".format(strength)
#                     noisy_fitness_full=generate_noisy_measurements(fitness, kind='additive', strength=strength)
#                     Nsub1, sigma_sub1, sigmaf_sub1, adj_sub1, fit_sub1=subsetting_for_focal_species(N_obs, sigma_vectors, sigma_vectors_final, adjacency_matrix, noisy_fitness_full, n_exp, S, i)                            
#                     Nsub, sigma_sub, sigmaf_sub, adj_sub, fit_sub=subsetting_for_focal_species(Nsub1, sigma_sub1, sigmaf_sub1, adj_sub1, fit_sub1, int(n_exp/2), S-1, j-1) ###index is j-1 because the arrays have been subsetted already!!          
#                     noisy_abundance_dict=analyze_for_chosen_metric(fit_sub, sigma_sub, sigmaf_sub, adj_sub, S-2, int(n_exp/4), 
#                                                                    fitness_suffix, 
#                                                                analysis_fold, file_suffix, analysis_options, fraction_limited_data=data_fractions_to_estimate_from, 
#                                                                Nfull=N_obs, sigma_full=sigma_vectors, sigmaFinal_full=sigma_vectors_final, fit_full=noisy_fitness_full)
#                     analysis_dict.update(noisy_abundance_dict)             
#                 analysis_dict.update({f'Species {i},{j} coex_product-ADDnoise_strengths':noise_strength_list})
#                 
#                 for strength in noise_strength_list:
#                     fitness_suffix=f'Species {i},{j} coex_product-MULTnoise_'+"{:.2e}".format(strength)
#                     noisy_fitness_full=generate_noisy_measurements(fitness, kind='multiplicative', strength=strength)
#                     Nsub1, sigma_sub1, sigmaf_sub1, adj_sub1, fit_sub1=subsetting_for_focal_species(N_obs, sigma_vectors, sigma_vectors_final, adjacency_matrix, noisy_fitness_full, n_exp, S, i)                            
#                     Nsub, sigma_sub, sigmaf_sub, adj_sub, fit_sub=subsetting_for_focal_species(Nsub1, sigma_sub1, sigmaf_sub1, adj_sub1, fit_sub1, int(n_exp/2), S-1, j-1) ###index is j-1 because the arrays have been subsetted already!!          
#                     noisy_abundance_dict=analyze_for_chosen_metric(fit_sub, sigma_sub, sigmaf_sub, adj_sub, S-2, int(n_exp/4), 
#                                                                    fitness_suffix, 
#                                                                analysis_fold, file_suffix, analysis_options, fraction_limited_data=data_fractions_to_estimate_from, 
#                                                                Nfull=N_obs, sigma_full=sigma_vectors, sigmaFinal_full=sigma_vectors_final, fit_full=noisy_fitness_full)
#                     analysis_dict.update(noisy_abundance_dict)             
#                 analysis_dict.update({f'Species {i},{j} coex_product-MULTnoise_strengths':noise_strength_list})           
#             end = time.time()
#             print('time to analyze for focal coex product',end-start)
# =============================================================================
            
# =============================================================================
#             
#         ####################      Pairwise productivity on full landscape        ####################     
#         if S>=8:# else we dont analyse focal species coexistences or 2focalspecies product 
#             '''to compare search protocols, one needs to simulate search and measure ruggedness on full landscape'''
#             start = time.time()       
#             ### to save on some time on analysis, we run for a smaller number of these.
#             for i,j in [[2,3]] :                
#                 fitness=two_species_product(N_obs, i,j ,sigma_vectors_final)             
#                 focal_coex_product_dict=analyze_for_chosen_metric(fitness, sigma_vectors, sigma_vectors_final, adjacency_matrix, S, n_exp, f'Species {i},{j} coex_product FullLandscape', 
#                                                                analysis_fold, file_suffix, analysis_options, fraction_limited_data=data_fractions_to_estimate_from, 
#                                                                Nfull=N_obs, sigma_full=sigma_vectors, sigmaFinal_full=sigma_vectors_final, fit_full=fitness)
#                 analysis_dict.update(focal_coex_product_dict)
#             end = time.time()
#             print('time to analyze for 2 species focal coex product on full landscape',end-start)
#          
#             
#             ### noisy landscapes
#             ''' 
#             using  analysis_options_QUICK, not analysis options! 
#             '''
#             for strength in noise_strength_list:
#                  fitness_suffix=f'Species {i},{j} coex_product FullLandscape-ADDnoise_'+"{:.2e}".format(strength)
#                  noisy_fitness_full=generate_noisy_measurements(fitness, kind='additive', strength=strength) 
#                  noisy_coex_dict=analyze_for_chosen_metric(noisy_fitness_full, sigma_vectors, sigma_vectors_final, adjacency_matrix, S, n_exp, fitness_suffix, 
#                                                                analysis_fold, file_suffix, analysis_options_QUICK, fraction_limited_data=data_fractions_to_estimate_from, 
#                                                             Nfull=N_obs, sigma_full=sigma_vectors, sigmaFinal_full=sigma_vectors_final, fit_full=noisy_fitness_full)
#                  analysis_dict.update(noisy_coex_dict)             
#             analysis_dict.update({f'Species {i},{j} coex_product FullLandscape-ADDnoise_strengths':noise_strength_list})
#              
#             for strength in noise_strength_list:
#                  fitness_suffix=f'Species {i},{j} coex_product FullLandscape-MULTnoise_'+"{:.2e}".format(strength)
#                  noisy_fitness_full=generate_noisy_measurements(fitness, kind='multiplicative', strength=strength)
#                  noisy_coex_dict=analyze_for_chosen_metric(noisy_fitness_full, sigma_vectors, sigma_vectors_final, adjacency_matrix, S, n_exp, fitness_suffix, 
#                                                                analysis_fold, file_suffix, analysis_options_QUICK, fraction_limited_data=data_fractions_to_estimate_from, 
#                                                             Nfull=N_obs, sigma_full=sigma_vectors, sigmaFinal_full=sigma_vectors_final, fit_full=noisy_fitness_full)
#                  analysis_dict.update(noisy_coex_dict)    
#              
#             analysis_dict.update({f'Species {i},{j} coex_product FullLandscape-MULTnoise_strengths':noise_strength_list})    
# 
# =============================================================================

    
        if routine_analysis ==True:   
            start = time.time() 
            structure_dict=run_routine_analysis(sigma_vectors, N_obs, sigma_vectors_final, adjacency_matrix, S, n_exp, analysis_fold, file_suffix, fraction_limited_data=data_fractions_to_estimate_from)
            analysis_dict.update(structure_dict)
            end = time.time()
            print('time to run routine analysis',end-start)
        
        plt.close("all")
        print ('analysis_fold was: \n',analysis_fold)
        
        analysis_dict.update({'reached_SteadyState':reached_SteadyState_Flag})
        
        pickle.dump(analysis_dict, open(analysis_fold+'analysis_'+file_suffix+'.dat', 'wb') )  

        if len(plots_folders_list)!=0: ### we save the data along with the plots and without the plots in the original folder as well
            analysis_data_only_fold= analysis_fold.replace(plots_folders_list[fold_idx],folder_name_list[fold_idx] )
            print ("separate fold for data without plots: ", analysis_data_only_fold)            
            if not os.path.exists(analysis_data_only_fold): os.makedirs(analysis_data_only_fold)  
            pickle.dump(analysis_dict, open(analysis_data_only_fold+'analysis_'+file_suffix+'.dat', 'wb') )  
        
        print ('analysis is complete')
 
if __name__ == '__main__':
        
    main()  


        












