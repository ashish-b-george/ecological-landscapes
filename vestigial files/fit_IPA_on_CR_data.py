#!/usr/bin/env python3
#$ -j y
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:59:11 2019
@author: ashish

Performs least squares fit on abundance data from different initial conditions, computes R^2 and out of sample error
"""
from scipy.optimize import least_squares
import numpy as np
import pickle # in python3, pickle uses cPickle automatically, there is no cPickle
import os
#import pandas
import argparse
from sklearn.metrics import r2_score

fold="/usr3/graduate/ashishge/ecology/steady_state_simulation_data/halfresource/"
fold="/Users/ashish/Downloads/ecology_simulation_data/2resource/"
file_suffix='S8M8_noisy'

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

destfold=fold[:-1]+"_plots/"
if not os.path.exists(destfold): os.makedirs(destfold)

nice_fonts = { #"text.usetex": True, # Use LaTex to write all text
"font.family": "serif",
# Use 10pt font in plots, to match 10pt font in document
"axes.labelsize": 10, "font.size": 10,
# Make the legend/label fonts a little smaller
"legend.fontsize": 8, "xtick.labelsize": 8,"ytick.labelsize": 8 }
mpl.rcParams.update(nice_fonts)

def abundance_function(r_matrix, sigma_vec):
    return  sigma_vec * np.exp(r_matrix@sigma_vec ) 
    
def error_prediction(r_matrix_flattened, sigma_vectors, N_obs,  S, n_exp):
    n_train_exp=len(sigma_vectors)
    r_matrix=r_matrix_flattened.reshape((S,S))
    error=0
    for i in range(n_train_exp):
        N_predicted=abundance_function( r_matrix, sigma_vectors[i] )
        error+= (N_predicted  - N_obs[i] )  @(N_predicted  - N_obs[i] )    
    return error

#def Jacobian(r_matrix_flattened, sigma_vec, N_obs,  S, n_exp):
#    r_matrix=r_matrix_flattened.reshape((S,S))
##    Jacobian_matrix=np.zeros((sigma_vec.size,r_matrix.size))
#    Jacobian_matrix=np.outer(sigma_vec,abundance_function(r_matrix, sigma_vec))
#    return  Jacobian_matrix





data = pickle.load(open(fold+'data'+file_suffix+'.dat', 'rb')) 

c_matrix=data['passed_params']['c']
S=len(c_matrix)
M=len(c_matrix[0])
print (S,M)

sigma_vectors= (data['initial_abundance'].T>0).astype(int)
N_obs=data['steady_state'].T
r_guess=np.ravel(np.eye(S))
S=len(sigma_vectors[0])
n_exp=len(sigma_vectors)


##########  plotting initial and final show that species can go extinct sometimes ############
#fig = plt.figure(figsize=(3.5,3.5)) 
#ax = fig.add_subplot(111)   
#for i in range(n_exp):
#    plt.plot(sigma_vectors[i],N_obs[i],'bo',markeredgecolor='none') 
#plt.ylabel(r'$\sigma$', weight="bold")
#plt.xlabel(r'$N_{observed}$',weight="bold")   
#if fold.startswith('/Users/ashish'): plt.tight_layout()
#plt.savefig(destfold+file_suffix+"_data.png",dpi=200)




##########  Fitting all the data ############
def fit_all_data_lsq():
    r_lsq = least_squares(error_prediction, r_guess, args=(sigma_vectors, N_obs, S, n_exp))
    
    fig = plt.figure(figsize=(3.5,3.5))  
    ax = fig.add_subplot(111) 
    ax.plot([0, np.max(N_obs)],[0, np.max(N_obs)],'k-')  
    N_predicted=[]
    for i in range(n_exp):
        N_predicted.append( abundance_function( r_lsq.x.reshape((S,S)), sigma_vectors[i] ) )
        ax.plot(N_obs[i],N_predicted[i],'bo',markeredgecolor='none') 
    plt.text(0.05, 0.8,'full $R^2=$'+ str( round( r2_score(np.ravel(N_obs), np.ravel(N_predicted) ), 2) ), horizontalalignment='left', transform=ax.transAxes)
    idx_nonzero=np.ravel(N_obs).nonzero()
    plt.text(0.05, 0.9,'nonzero $R^2=$'+ str( round( r2_score(np.ravel(N_obs)[idx_nonzero], np.ravel(N_predicted)[idx_nonzero] ), 2) ), horizontalalignment='left', transform=ax.transAxes)
    plt.ylabel(r'$N_{predicted}$', weight="bold")
    plt.xlabel(r'$N_{observed}$',weight="bold")  
    if fold.startswith('/Users/ashish'): plt.tight_layout()
    else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.15, hspace=0.15)
    plt.savefig(destfold+file_suffix+"_performace_fitting_all.png",dpi=200)


##########  Fitting only pairwise data ############)
def fit_only_pwise_data_lsq():
        #split into training and test data i.e., pwise and the rest of the data
        sigma_vectors_train=[]
        N_obs_train=[]
        sigma_vectors_test=[]
        N_obs_test=[]        
        for i in range(n_exp):
            if np.sum(sigma_vectors[i])<=2:
                sigma_vectors_train.append(sigma_vectors[i])
                N_obs_train.append(N_obs[i])
            else:
                sigma_vectors_test.append(sigma_vectors[i])
                N_obs_test.append(N_obs[i])       
        sigma_vectors_train=np.array(sigma_vectors_train)  
        N_obs_train=np.array(N_obs_train) 
        sigma_vectors_test=np.array(sigma_vectors_test)  
        N_obs_test=np.array(N_obs_test) 
        
               
        r_lsq_pwise = least_squares(error_prediction, r_guess, args=(sigma_vectors_train, N_obs_train, S, n_exp))
                
        #plot fit on all data
        fig = plt.figure(figsize=(3.5,3.5)) 
        ax = fig.add_subplot(111) 
        ax.plot([0, np.max(N_obs)],[0, np.max(N_obs)],'k-')  
        N_predicted=[] 
        for i in range(n_exp):
            N_predicted.append( abundance_function( r_lsq_pwise.x.reshape((S,S)), sigma_vectors[i] )  ) 
            plt.plot(N_obs[i],N_predicted[i],'bo',markeredgecolor='none') 
        plt.text(0.05, 0.9,'$R^2=$'+ str( round( r2_score(np.ravel(N_obs), np.ravel(N_predicted) ), 2) ), horizontalalignment='left', transform=ax.transAxes)
        plt.ylabel(r'$N_{predicted}$', weight="bold")
        plt.xlabel(r'$N_{observed}$',weight="bold")   
        if fold.startswith('/Users/ashish'): plt.tight_layout()
        else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.15, hspace=0.15)
        plt.savefig(destfold+file_suffix+"_performance_from_pwise.png",dpi=200)  
        
        #plot fit on test data
        n_test_exp=len(sigma_vectors_test)
        fig = plt.figure(figsize=(7,3.5))  
        ax1 = fig.add_subplot(1,2,1) 
        ax2 = fig.add_subplot(1, 2 ,2)
        N_predicted=[] 
        for i in range(n_test_exp):
            N_predicted.append( abundance_function( r_lsq_pwise.x.reshape((S,S)), sigma_vectors_test[i] )  ) 
            xval=np.array(N_obs_test[i])
            yval=np.array(N_predicted[i])
            idx_nonzero=sigma_vectors_test[i].nonzero()
            ax1.plot(xval,yval,'bo',markeredgecolor='none') 
            ax2.plot(xval[idx_nonzero],yval[idx_nonzero],'bo',markeredgecolor='none') 
        
        flat_N_obs_test=np.ravel(N_obs_test)
        flat_N_predicted=np.ravel(N_predicted)
        idx_nonzero=flat_N_obs_test.nonzero()
        ax1.text(0.05, 0.9,'$R^2=$'+ str( round( r2_score( flat_N_obs_test, flat_N_predicted ), 2) ), horizontalalignment='left', transform=ax1.transAxes)
        ax2.text(0.05, 0.9,'$R^2=$'+ str( round( r2_score( flat_N_obs_test[idx_nonzero], flat_N_predicted[idx_nonzero] ), 2) ), horizontalalignment='left', transform=ax2.transAxes)
        ax1.plot([0, np.max(N_obs)],[0, np.max(N_obs)],'k-')
        ax2.plot([0, np.max(N_obs)],[0, np.max(N_obs)],'k-')
        
        ax1.set_ylabel(r'$N_{predicted}$, out of sample')
        ax1.set_xlabel(r'$N_{observed}$, out of sample')
        ax1.set_title(r'including absent species',weight="bold")
        
        ax2.set_xlabel(r'$N_{observed}$, out of sample') 
        ax2.set_title(r'only present species',weight="bold")  
        if fold.startswith('/Users/ashish'): plt.tight_layout()
        else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.15, hspace=0.15)
        plt.savefig(destfold+file_suffix+"_performance_OutOfSample_from_pwise.png",dpi=200)       

        



fit_all_data_lsq()
fit_only_pwise_data_lsq()


