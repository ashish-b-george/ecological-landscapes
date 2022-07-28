k#!/usr/bin/env python3
#$ -j y
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:59:11 2019
@author: ashish

Runs deterministic Poisson sampling on IPA model to find  steady state of the passaging
"""

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
nice_fonts = { #"text.usetex": True, # Use LaTex to write all text
"font.family": "serif",
# Use 10pt font in plots, to match 10pt font in document
"axes.labelsize": 10, "font.size": 10,
# Make the legend/label fonts a little smaller
"legend.fontsize": 8, "xtick.labelsize": 8,"ytick.labelsize": 8 }
mpl.rcParams.update(nice_fonts)


destfold="/Users/ashish/Downloads/ecology_simulation_data/test_passaging/"
if not os.path.exists(destfold): os.makedirs(destfold)
file_suffix='test_2'

S=2
L= 2**S-1
K0=100 # intial carrying capacity
lambda0=0.01# dilution rate
r_matrix=np.ones((S,S))
if S==4:r_matrix =np.array([[1., 0, 0., 0.],
       [0., 1., -1., .1],
       [0, 1., 1., 0],
       [0.1, .2, .2, 0.2]])
if S==2:r_matrix =np.array([[1., 0.1],
       [-0.1,1.]])
else:
    r_matrix=np.random.rand(S*S).reshape(S,S)


N_vec=np.ones(S)*10
sigma_vectors = np.zeros((L,S))
N_vectors = np.zeros((L,S))
prob_vec=np.zeros(S)

T=8# number of passages
def abundance_function(r_matrix, sigma_vec):
    return  sigma_vec * np.exp(r_matrix@sigma_vec ) 




for i in range(L): 
        binary_rep=bin(i+1)[2:]
        for j in range(len(binary_rep)):  
            sigma_vectors[i,S-j-1]= int( binary_rep[len(binary_rep)-j-1] )


fig = plt.figure(figsize=(3.5,3.5)) 
ax = fig.add_subplot(111) 

for t in range(T):
    p_absent=np.exp(-lambda0*N_vec)    
    N_vec=np.zeros(S)
    for i in range(L):               
        p_config=  (1-sigma_vectors[i])*p_absent +sigma_vectors[i]*(1-p_absent)
        N_vectors[i]=abundance_function(r_matrix, sigma_vectors[i])
        N_vec=N_vec+p_config*N_vectors[i]
    ax.plot(np.arange(S),N_vec,label="t="+str(t+1))
    
    
    
plt.legend(loc='best')   
plt.ylabel(r'abundance, $N_i$')
plt.xlabel(r'species $i$')   
if destfold.startswith('/Users/ashish'): plt.tight_layout()
else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.15, hspace=0.15)
plt.savefig(destfold+file_suffix+"abundance_evolution.png",dpi=300)  