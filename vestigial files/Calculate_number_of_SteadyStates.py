#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:07:39 2019

@author: ashish

Calculates if the steady states are unique or not by computing the distance between them
"""

from scipy.optimize import least_squares
import numpy as np
import pickle # in python3, pickle uses cPickle automatically, there is no cPickle
import os
#import pandas
import argparse
from sklearn.metrics import r2_score
import sys
from community_simulator.analysis import *
if os.getcwd().startswith("/Users/ashish"):
    from my_CR_model_functions import rms

else:  
    sys.path.append('/usr3/graduate/ashishge/ecology/ecological-landscapes/')
    from my_CR_model_functions import rms
    

fold="/Users/ashish/Downloads/ecology_simulation_data/cluster/counting_steady_states/type2/PLoS3_compiled/"
file_suffix='S80M40_PLoS3_c10_typeII_sigma20.0'
popn_cutoff=1# not implemented yet

parser = argparse.ArgumentParser()
parser.add_argument("-f", help="folder name")
parser.add_argument("-s", help="file suffix")
args = parser.parse_args()
if args.f:
    fold=args.f
if args.s:
    file_suffix=args.s   

if fold.startswith('/Users/ashish'):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
else: ### can not use pyplot on cluster without agg backend
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt



nice_fonts = { #"text.usetex": True, # Use LaTex to write all text
"font.family": "serif",
# Use 10pt font in plots, to match 10pt font in document
"axes.labelsize": 10, "font.size": 10,
# Make the legend/label fonts a little smaller
"legend.fontsize": 8, "xtick.labelsize": 8,"ytick.labelsize": 8 }
mpl.rcParams.update(nice_fonts)


#returns if distance between two abundance vectors are within a specified relative tolerance
def steady_states_are_equal(a,b,tol=1e-2):
    result= True if rms(a-b) <tol else False
    return result
def distance_between_steady_states(a,b):   
    return rms(a-b)


data = pickle.load(open(fold+file_suffix+'.dat', 'rb')) 

c_matrix=data['passed_params']['c']
S=len(c_matrix)
M=len(c_matrix[0])
print (S,M)


N0= data['initial_abundance'].T
sigma_vectors= (data['initial_abundance'].T>0).astype(int)
N_obs=data['steady_state'].T
reached_SteadyState=data['reached_SteadyState']
print (np.sum(reached_SteadyState==False))
n_exp=len(N_obs)
ref_vector=np.tile(N_obs[0],(n_exp,1)) 



### this is only checks if steady states are different from reference, need to do repeat this for every ref!
print ( np.sum( list(map(steady_states_are_equal,ref_vector,N_obs)) ) )
distance_between_SS=np.array( list(map(rms,N_obs-ref_vector)) )



fig = plt.figure(figsize=(3.5,3.5)) 
ax = fig.add_subplot(111)
for i in range(S):
    ax.plot(np.arange(n_exp),distance_between_SS ,'bo',alpha=0.1,mec='None')
plt.text(0.01, 0.89,'max distance: '+ '{:.1e}'.format( distance_between_SS.max() ) +'\npopn size rms: ' + '{:.1e}'.format( rms(N_obs[0]) ) , horizontalalignment='left', transform=ax.transAxes) 
ax.set_ylabel(r'distance bw SS')
ax.set_xlabel(r'experiment number') 
if distance_between_SS.max()/ rms(N_obs[0]) < 1e-2:
    ax.set_title(r'Only one steady state',weight="bold")  
if fold.startswith('/Users/ashish'): plt.tight_layout()
else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
plt.savefig(fold+file_suffix+"_distance_bw_SS.png",dpi=200)



idx_survivors=np.where(N_obs>1)


fig = plt.figure(figsize=(3.5,3.5)) 
ax = fig.add_subplot(111)
#for i in range(n_exp):
ax.scatter(idx_survivors[0],idx_survivors[1], s=1.,c='b')
#plt.text(0.01, 0.89,'max distance: '+ '{:.1e}'.format( distance_between_SS.max() ) +'\npopn size rms: ' + '{:.1e}'.format( rms(N_obs[0]) ) , horizontalalignment='left', transform=ax.transAxes) 
ax.set_ylabel(r'identity of surviving species')
ax.set_xlabel(r'experiment number')  
if fold.startswith('/Users/ashish'): plt.tight_layout()
else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
plt.savefig(fold+file_suffix+"_survivor_identity.png",dpi=200)

