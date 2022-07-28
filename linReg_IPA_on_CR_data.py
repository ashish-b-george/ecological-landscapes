 #!/usr/bin/env python3
#$ -j y
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:59:11 2019
@author: ashish

Performs linear regression with/without regularization on abundance data from different initial conditions,
computes R^2 and out of sample error
"""
import numpy as np
import pickle # in python3, pickle uses cPickle automatically, there is no cPickle
import os
import argparse
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
import sys

##################  file location and such things  ########################

#fold="/usr3/graduate/ashishge/ecology/steady_state_simulation_data/halfresource/"
fold="/Users/ashish/Downloads/ecology_simulation_data/cluster/binaryIC_all/type2/l0.5_compiled/"
file_suffix='S10M10_P1_c4_l0.5_typeII_sigma20.0'

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
"legend.fontsize": 8, "xtick.labelsize": 8,"ytick.labelsize": 8, "lines.markersize": 4 }
mpl.rcParams.update(nice_fonts)

def abundance_function(r_matrix, sigma_vec):
    return  sigma_vec * np.exp(r_matrix@sigma_vec ) 

def abundance_function_linear(r_matrix, sigma_vec):
    return  sigma_vec * (r_matrix@sigma_vec) 
    

##################  reading and initializing data  ########################
pseudoCount=1e-6
toLog10=np.log10(np.e) #mutliplying by np.log10(np.e) in plots changes base of plots to 10 which is clearer to read
data = pickle.load(open(fold+file_suffix+'.dat', 'rb')) 
c_matrix=data['passed_params']['c']
S=len(c_matrix)
M=len(c_matrix[0])
print (S,M)

sigma_vectors= (data['initial_abundance'].T>0).astype(int)
N_obs=data['steady_state'].T
N_obs[N_obs<0.01]=pseudoCount

S=len(sigma_vectors[0])
n_exp=len(sigma_vectors)
lasso_object = Lasso()
linReg_object = LinearRegression()


##################  preparing data  ########################






#############     generating convenient subset matrices with only present species featuring   ############# 

X_all_i=[] # X_all_i[i] will have all the data with species i present
X_all_Order2_i=[]
Y_all_i=[]# Y_all_i[i] will have all the abundances of species i when it was present
Y_linear_all_i=[]
for i in range(S):  
    exp_i_present=np.where(sigma_vectors[:,i]==1)[0] 
    X_all_i.append( np.array( sigma_vectors[exp_i_present] ) )
    Y_all_i.append( np.log(N_obs[exp_i_present,i]) ) 
    Y_linear_all_i.append(N_obs[exp_i_present,i])
    Order2_temp=[]
    for j in exp_i_present:
        Order2_temp.append( np.ravel( np.outer(sigma_vectors[j],sigma_vectors[j]) )  )
    X_all_Order2_i.append( np.array( Order2_temp ) )
X_all_i=np.array(X_all_i)
Y_all_i=np.array(Y_all_i)
X_all_Order2_i=np.array(X_all_Order2_i)
Y_linear_all_i=np.array(Y_linear_all_i)

X_Orders12_i=np.concatenate( (X_all_i,X_all_Order2_i), axis=2)
'''
format for X_all_i is
X_all_i[species being fit][experimental well number][  species in that particular experimental well ]
'''

#############      Pairwise data     ###############
def split_into_pairwise(sigma_vectors,N_obs):        
        ##find pairwise values ## 
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

        return sigma_vectors_train, N_obs_train, sigma_vectors_test, N_obs_test
 

#############     fitting all the data to 1st order   ############# 

'''
We fit each species i separately and learn corresponding rij separately so that log can be taken and data passed without trouble.
'''
def do_linReg_full_data(X_all_i, Y_all_i, Y_linear_all_i):
        
        r_matrix=np.zeros((S,S))
        r_matrix_linear=np.zeros((S,S)) 
        sum_of_squares=np.zeros(S)
        residual_square=np.zeros(S)       
        sum_of_squares_linear=np.zeros(S)
        residual_square_linear=np.zeros(S)
        
        for i in range(S):
            X_train=X_all_i[i]
            ## log abundance
            Y_train=Y_all_i[i]
            linReg_object.fit(X_train,Y_train) # .reshape(-1, 1) is not reqd           
            r_matrix[i]=linReg_object.coef_
            r_matrix[i,i]=r_matrix[i,i]+linReg_object.intercept_
            sum_of_squares[i]=np.sum( np.square( Y_train.mean()-Y_train ))
            residual_square[i]=np.sum( np.square( Y_train-X_train@r_matrix[i] )  )    
            
            ## linear abundance
            Y_train=Y_linear_all_i[i]
            linReg_object.fit(X_train,Y_train)
            r_matrix_linear[i]=linReg_object.coef_
            r_matrix_linear[i,i]=r_matrix_linear[i,i]+linReg_object.intercept_
            sum_of_squares_linear[i]=np.sum( np.square( Y_train.mean()-Y_train ))
            residual_square_linear[i]=np.sum( np.square( Y_train-X_train@r_matrix_linear[i] )  )    
           
        Rsq=1-np.sum(residual_square)/np.sum(sum_of_squares)
        Rsq_linear=1-np.sum(residual_square_linear)/np.sum(sum_of_squares_linear)
        
        #############     plotting results      #############        
        ## log abundance
        fig = plt.figure(figsize=(3.5,3.5)) 
        ax = fig.add_subplot(111)
        for i in range(S):
            ax.plot(Y_all_i[i]*toLog10,X_all_i[i]@r_matrix[i]*toLog10,'bo')
        plt.text(0.05, 0.9,'nonzero $R^2=$'+ str( round(Rsq, 2) ) , horizontalalignment='left', transform=ax.transAxes) 
        ax.set_ylabel(r'log $N_{predicted}$')
        ax.set_xlabel(r'log $N_{observed}$')
        ax.plot(ax.get_xlim(),ax.get_xlim(),'k--')  
        if fold.startswith('/Users/ashish'): plt.tight_layout()
        else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
        plt.savefig(destfold+file_suffix+"_linReg.png",dpi=200)
        ## linear abundance
        fig = plt.figure(figsize=(3.5,3.5)) 
        ax = fig.add_subplot(111)
        for i in range(S):
            ax.plot(Y_linear_all_i[i],X_all_i[i]@r_matrix_linear[i],'bo')
        plt.text(0.05, 0.9,'nonzero $R^2=$'+ str( round(Rsq_linear, 2) ) , horizontalalignment='left', transform=ax.transAxes) 
        ax.set_ylabel(r' $N_{predicted}$')
        ax.set_xlabel(r' $N_{observed}$')
        ax.plot(ax.get_xlim(),ax.get_xlim(),'k--')  
        if fold.startswith('/Users/ashish'): plt.tight_layout()
        else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
        plt.savefig(destfold+file_suffix+"_linReg_linearModel.png",dpi=200)


        return r_matrix ,r_matrix_linear, Rsq, Rsq_linear



def do_linReg_train_test(sigma_vectors_train, N_obs_train, sigma_vectors_test, N_obs_test):  
        #############     fitting pairwise data    ############### 
        
        
        r_matrix=np.zeros((S,S))
        r_matrix_linear=np.zeros((S,S)) 
        sum_of_squares=np.zeros(S)
        residual_square=np.zeros(S)       
        sum_of_squares_linear=np.zeros(S)
        residual_square_linear=np.zeros(S)
        
        sum_of_squares_test=np.zeros(S)
        residual_square_test=np.zeros(S)
        sum_of_squares_linear_test=np.zeros(S)
        residual_square_linear_test=np.zeros(S)
        
        
        for i in range(S):
            exp_i_present=np.where(sigma_vectors_train[:,i]==1)[0] # wells where species i were present          
            X_train=np.array( sigma_vectors_train[exp_i_present] )
            ## log abundance
            Y_train=np.array(np.log(N_obs_train[exp_i_present,i]) )    
            linReg_object.fit(X_train,Y_train)
            r_matrix[i]=linReg_object.coef_
            r_matrix[i,i]=r_matrix[i,i]+linReg_object.intercept_ 
            sum_of_squares[i]=np.sum( np.square( Y_train.mean()-Y_train ))
            residual_square[i]=np.sum( np.square( Y_train-X_train@r_matrix[i] )  )   
            ## linear abundance
            Y_train=np.array(N_obs_train[exp_i_present,i])
            linReg_object.fit(X_train,Y_train)
            r_matrix_linear[i]=linReg_object.coef_
            r_matrix_linear[i,i]=r_matrix_linear[i,i]+linReg_object.intercept_  
            sum_of_squares_linear[i]=np.sum( np.square( Y_train.mean()-Y_train ))
            residual_square_linear[i]=np.sum( np.square( Y_train-X_train@r_matrix_linear[i] )  ) 
            
            ## test data
            exp_i_present=np.where(sigma_vectors_test[:,i]==1)[0]
            X_test=np.array( sigma_vectors_test[exp_i_present] )           
            ## log abundance
            Y_test=np.array(np.log(N_obs_test[exp_i_present,i]) )
            sum_of_squares_test[i]=np.sum( np.square( Y_test.mean()-Y_test ))
            residual_square_test[i]=np.sum( np.square( Y_test-X_test@r_matrix[i] )  )    
            ## linear abundance
            Y_test=np.array( N_obs_test[exp_i_present,i] )
            sum_of_squares_linear_test[i]=np.sum( np.square( Y_test.mean()-Y_test ))
            residual_square_linear_test[i]=np.sum( np.square( Y_test-X_test@r_matrix_linear[i] )  ) 
            
        
        Rsq=1-np.sum(residual_square)/np.sum(sum_of_squares)
        Rsq_linear=1-np.sum(residual_square_linear)/np.sum(sum_of_squares)
        Rsq_test=1-np.sum(residual_square_test)/np.sum(sum_of_squares_test)
        Rsq_linear_test=1-np.sum(residual_square_linear_test)/np.sum(sum_of_squares_test)   
        #############     plotting  pairwise fit results    ###############      
        ## log abundance
        fig = plt.figure(figsize=(7,3.5)) 
        ax1 = fig.add_subplot(1,2,1) 
        ax2 = fig.add_subplot(1, 2 ,2)
        for i in range(S):
            exp_i_present=np.where(sigma_vectors_train[:,i]==1)[0] # wells where species i were present
            X_train=np.array( sigma_vectors_train[exp_i_present] )
            Y_train=np.array(np.log(N_obs_train[exp_i_present,i]) )
            ax1.plot(Y_train*toLog10,X_train@r_matrix[i]*toLog10,'bo')
            
            exp_i_present=np.where(sigma_vectors_test[:,i]==1)[0]
            X_test=np.array( sigma_vectors_test[exp_i_present] )
            Y_test=np.array(np.log(N_obs_test[exp_i_present,i]) )
            ax2.plot(Y_test,X_test@r_matrix[i],'bo')
            
        ax1.plot(ax1.get_xlim(),ax1.get_xlim(),'k--')   
        ax2.plot(ax2.get_xlim(),ax2.get_xlim(),'k--')   
        ax1.set_title(r' training data (pairwise) fit',weight="bold")
        ax2.set_title(r'out of sample performance',weight="bold") 
        ax1.text(0.05, 0.9,'nonzero $R^2=$'+ str( round(Rsq, 2) ) , horizontalalignment='left', transform=ax1.transAxes)  
        ax2.text(0.05, 0.9,'nonzero $R^2=$'+ str( round(Rsq_test, 2) ) , horizontalalignment='left', transform=ax2.transAxes) 
        ax1.set_ylabel(r'log $N_{predicted}$')
        ax1.set_xlabel(r'log $N_{observed}$')
        ax2.set_xlabel(r'log $N_{observed}$') 
        ax2.set_ylabel(r'log $N_{predicted}$')    
        if fold.startswith('/Users/ashish'): plt.tight_layout()
        else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)        
        plt.savefig(destfold+file_suffix+"_linReg_pwise.png",dpi=200)
        
        ## linear abundance
        fig = plt.figure(figsize=(7,3.5)) 
        ax1 = fig.add_subplot(1,2,1) 
        ax2 = fig.add_subplot(1, 2 ,2)
        for i in range(S):
            exp_i_present=np.where(sigma_vectors_train[:,i]==1)[0] # wells where species i were present
            X_train=np.array( sigma_vectors_train[exp_i_present] )
            Y_train=np.array(N_obs_train[exp_i_present,i])
            ax1.plot(Y_train,X_train@r_matrix_linear[i],'bo')
            
            exp_i_present=np.where(sigma_vectors_test[:,i]==1)[0]
            X_test=np.array( sigma_vectors_test[exp_i_present] )
            Y_test=np.array(N_obs_test[exp_i_present,i] )
            ax2.plot(Y_test,X_test@r_matrix_linear[i],'bo')
        
        ax1.plot(ax1.get_xlim(),ax1.get_xlim(),'k--')   
        ax2.plot(ax2.get_xlim(),ax2.get_xlim(),'k--')  
        ax1.set_ylabel(r'$N_{predicted}$')
        ax1.set_xlabel(r'$N_{observed}$')
        ax2.set_xlabel(r'$N_{observed}$') 
        ax2.set_ylabel(r'$N_{predicted}$') 
         
        ax1.set_title(r'training data (pairwise) fit',weight="bold")
        ax2.set_title(r'out of sample performance',weight="bold")    
        ax1.text(0.05, 0.9,'nonzero $R^2=$'+ str( round(Rsq_linear, 2) ) , horizontalalignment='left', transform=ax1.transAxes)  
        ax2.text(0.05, 0.9,'nonzero $R^2=$'+ str( round(Rsq_linear_test, 2) ) , horizontalalignment='left', transform=ax2.transAxes) 
        if fold.startswith('/Users/ashish'): plt.tight_layout()
        else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)   
        plt.savefig(destfold+file_suffix+"__linReg_pwise_LinearModel.png",dpi=200)

        return r_matrix ,r_matrix_linear, Rsq, Rsq_linear

#############     fitting all data to 2nd order   ############# 
def do_linReg_Order2(X_all_i, Y_all_i, X_all_Order2_i, Y_linear_all_i):
        if n_exp< S**3: # there are enough independent parameters
            print ("not enough data for fitting order 2, S needs to be larger.")
            return 0
        '''
        actually number of independent fitting parameters are less than S^3, but we are being cautious
        It is is actually:
        S * (1 + S-1 +S(S-1)/2 ), where the term in the brackets is number of parameters for each species at each order
        ''' 
        r_matrix=np.zeros((S,S))
        r_matrix_linear=np.zeros((S,S)) 
        sum_of_squares=np.zeros(S)
        residual_square=np.zeros(S)       
        sum_of_squares_linear=np.zeros(S)
        residual_square_linear=np.zeros(S)       
        J_matrix=np.zeros( (S,S*S) )
        J_matrix_linear=np.zeros( (S,S*S) )
        linReg_object = LinearRegression()
        for i in range(S):
            X_train=np.hstack((X_all_i[i],X_all_Order2_i[i]))
            ##log abundance
            Y_train=Y_all_i[i]
            linReg_object.fit(X_train,Y_train)
            r_matrix[i]=linReg_object.coef_[:S]
            r_matrix[i,i]=r_matrix[i,i]+linReg_object.intercept_     
            J_matrix[i]=linReg_object.coef_[S:]            
            sum_of_squares[i]=np.sum( np.square( Y_train.mean()-Y_train ))
            residual_square[i]=np.sum( np.square( Y_train-X_train@linReg_object.coef_ -linReg_object.intercept_ )  ) 
            ##linear abundance
            Y_train=Y_linear_all_i[i]
            linReg_object.fit(X_train,Y_train)
            r_matrix_linear[i]=linReg_object.coef_[:S]
            r_matrix_linear[i,i]=r_matrix_linear[i,i]+linReg_object.intercept_     
            J_matrix_linear[i]=linReg_object.coef_[S:]
            sum_of_squares_linear[i]=np.sum( np.square( Y_train.mean()-Y_train ))
            residual_square_linear[i]=np.sum( np.square( Y_train-X_train@linReg_object.coef_ -linReg_object.intercept_   )  ) 

        Rsq=1-np.sum(residual_square)/np.sum(sum_of_squares)
        Rsq_linear=1-np.sum(residual_square_linear)/np.sum(sum_of_squares_linear)
        
        #############     plotting results      #############        
        ## log abundance
        fig = plt.figure(figsize=(3.5,3.5)) 
        ax = fig.add_subplot(111)
        for i in range(S):
            ax.plot(Y_all_i[i]*toLog10, (X_all_i[i]@r_matrix[i] +X_all_Order2_i[i] @ J_matrix[i]) *toLog10,'bo')
        plt.text(0.05, 0.9,'nonzero $R^2=$'+ str( round(Rsq, 2) ) , horizontalalignment='left', transform=ax.transAxes) 
        ax.set_ylabel(r'log $N_{predicted}$')
        ax.set_xlabel(r'log $N_{observed}$')
        ax.set_title(r'fit including $J_{ijk}$',weight="bold")       
        ax.plot(ax.get_xlim(),ax.get_xlim(),'k--')  
        if fold.startswith('/Users/ashish'): plt.tight_layout()
        else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
        plt.savefig(destfold+file_suffix+"_linReg_Order2.png",dpi=200)
        ## linear abundance
        fig = plt.figure(figsize=(3.5,3.5)) 
        ax = fig.add_subplot(111)
        for i in range(S):
            ax.plot(Y_linear_all_i[i],X_all_i[i]@r_matrix_linear[i] +X_all_Order2_i[i] @ J_matrix_linear[i] ,'bo')
        plt.text(0.05, 0.9,'nonzero $R^2=$'+ str( round(Rsq_linear, 2) ) , horizontalalignment='left', transform=ax.transAxes) 
        ax.set_ylabel(r' $N_{predicted}$')
        ax.set_xlabel(r' $N_{observed}$')
        ax.plot(ax.get_xlim(),ax.get_xlim(),'k--')  
        ax.set_title(r'fit including $J_{ijk}$',weight="bold")  
        if fold.startswith('/Users/ashish'): plt.tight_layout()
        else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)    
        plt.savefig(destfold+file_suffix+"_linReg_linearModel_Order2.png",dpi=200)     
            
            
            

#############      LASSO finding best regularizer, via grid search and 3-fold cross validation     ###############
def do_cross_validation(X_all_i, Y_all_i, suffix='', title_suffix=''):
        Fraction_test=0.2
        n_params_per_species=len(X_all_i[0][0]) # + one for intercept       
        coeff_matrix_cv=np.zeros((S, n_params_per_species))
        sum_of_squares=np.zeros(S)
        residual_square=np.zeros(S)
        sum_of_squares_test=np.zeros(S)
        residual_square_test=np.zeros(S)
        N_CV=[]
        
        if n_params_per_species==S:
            effective_total_parameters=S*S
        elif n_params_per_species==S*(S+1):
            effective_total_parameters=S*( S*(S-1)/2. +S )   
            print ("warning actual test for fitting: ", (1.-Fraction_test)/3. * n_exp/2. ,'>', effective_total_parameters,"is not applied!")
        if (1.-Fraction_test)/3. * n_exp/2. > S*S:            
#            print("S is large enough for us to do 3-fold crossvalidation ",(1.-Fraction_test)/4. * n_exp/2. , S**2 )
            fig = plt.figure(figsize=(7,3.5)) 
            ax1 = fig.add_subplot(1,2,1) 
            ax2 = fig.add_subplot(1, 2 ,2)
            for i in range(S):                   
                X_train, X_test, Y_train, Y_test = train_test_split(X_all_i[i], Y_all_i[i], test_size=Fraction_test)        
                alphas= np.logspace(-10, -1, 100)

                lasso_cv = LassoCV(cv=3, alphas=alphas).fit(X_train, Y_train) #random_state=0 can be used to fix the random permutation
                ## checking to see that both cross-validations give you similar answers.
#                lasso_object = Lasso()
#                lasso_on_grid = GridSearchCV(estimator=lasso_object, param_grid=dict(alpha=alphas),cv=3, scoring='r2',return_train_score=True, iid=False)
#                lasso_on_grid.fit(X_train, Y_train)                
#                if (lasso_on_grid.best_estimator_.alpha-lasso_cv.alpha_) !=0:
#                    print('S='+str(i)+suffix)
#                    print("alphas not equal",lasso_on_grid.best_estimator_.alpha,lasso_cv.alpha_)
#                    print (lasso_cv.coef_, lasso_on_grid.best_estimator_.coef_)
                    
                coeff_matrix_cv[i]=lasso_cv.coef_
                coeff_matrix_cv[i,i]=coeff_matrix_cv[i,i]+lasso_cv.intercept_  
#                ax1.plot(Y_train,X_train@coeff_matrix_cv[i],'o', label='S='+str(i) )
#                ax2.plot(Y_test,X_test@coeff_matrix_cv[i],'o', label='S='+str(i) )
                ax1.plot(Y_train,X_train@coeff_matrix_cv[i],'bo')
                ax2.plot(Y_test,X_test@coeff_matrix_cv[i],'bo')
                
                sum_of_squares[i]=np.sum( np.square( Y_train.mean()-Y_train ))
                residual_square[i]=np.sum( np.square( Y_train-X_train@coeff_matrix_cv[i] )  ) 
                
                ##### for testing purposes ######
#                X_test=X_all_i[i]
#                Y_test=Y_all_i[i]
#                print("CV is modified for testing purposes..")
                ###################
                sum_of_squares_test[i]=np.sum( np.square( Y_test.mean()-Y_test ))
                residual_square_test[i]=np.sum( np.square( Y_test-X_test@coeff_matrix_cv[i] )  ) 
                N_CV.append(X_test@coeff_matrix_cv[i])
            N_CV=np.array(N_CV)
            Rsq=1-np.sum(residual_square)/np.sum(sum_of_squares)
            Rsq_test=1-np.sum(residual_square_test)/np.sum(sum_of_squares_test)    
            ax1.plot(ax1.get_xlim(),ax1.get_xlim(),'k--')   
            ax2.plot(ax2.get_xlim(),ax2.get_xlim(),'k--')  
            ax1.set_ylabel(r'$N_{predicted}$')
            ax1.set_xlabel(r'$N_{observed}$')
            ax2.set_xlabel(r'$N_{observed}$') 
            ax2.set_ylabel(r'$N_{predicted}$') 
             
            ax1.set_title(r'training cross-validation'+title_suffix,weight="bold")
            ax2.set_title(r'test data'+title_suffix,weight="bold") 
            ax1.text(0.05, 0.9,'nonzero $R^2=$'+ str( round(Rsq, 2) ) , horizontalalignment='left', transform=ax1.transAxes)  
            ax2.text(0.05, 0.9,'nonzero $R^2=$'+ str( round(Rsq_test, 2) ) , horizontalalignment='left', transform=ax2.transAxes) 
            if fold.startswith('/Users/ashish'): plt.tight_layout()
            else:plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.25, hspace=0.1)   
            plt.savefig(destfold+file_suffix+'_LASSOcv'+suffix+'.png',dpi=200)
            
#            N_PA=[]
#            print(len(X_all_i[0]))
#            for i in range(int(n_exp/2 +1)):
#                if suffix=='_log':
#                    N_PA.append(abundance_function(coeff_matrix_cv, X_all_i[:,i,:])) 
#                elif suffix=='_linear':
#                    N_PA.append(abundance_function_linear(coeff_matrix_cv, X_all_i[:,i,:])) 
#            N_PA=np.array(N_PA) 
#            print(np.shape(Y_all_i),np.shape(N_PA)  )
#            print ("Score LASSOCV is ", r2_score( Y_all_i, N_PA )  )  

            return coeff_matrix_cv, N_CV
        else:
            print("S is too small for 3-fold crossvalidation ",(1.-Fraction_test)/4. * n_exp/2. , S*n_params_per_species)        
            return 0









r_matrix ,r_matrix_linear, Rsq, Rsq_linear=do_linReg_full_data(X_all_i, Y_all_i, Y_linear_all_i)
do_linReg_Order2(X_all_i, Y_all_i, X_all_Order2_i, Y_linear_all_i)

sigma_vectors_train, N_obs_train, sigma_vectors_test, N_obs_test=split_into_pairwise(sigma_vectors, N_obs)
r_matrix ,r_matrix_linear, Rsq, Rsq_linear=do_linReg_train_test(sigma_vectors_train, N_obs_train, sigma_vectors_test, N_obs_test)

r_matrix_EPA_cv, NEPA_cv=do_cross_validation(X_all_i, Y_all_i, suffix='_log')
r_matrix_LPA_cv, NLPA_cv=do_cross_validation(X_all_i, Y_linear_all_i, suffix='_linear')

#_=do_cross_validation(X_Orders12_i, Y_all_i, suffix='_log_Jijk',title_suffix='with $J_{ijk}$')
#_=do_cross_validation(X_Orders12_i, Y_linear_all_i, suffix='_Linear_Jijk',title_suffix='with  $J_{ijk}$')
sys.exit(1)
if isinstance(r_matrix_EPA_cv, np.ndarray): 
    data.update({'r_matrix_EPA_cv':r_matrix_EPA_cv})
    N_EPA=[]
    
    for i in range(n_exp):
        N_EPA.append(abundance_function(r_matrix_EPA_cv, sigma_vectors[i]))   
    
        
        
    N_EPA=np.array(N_EPA)    
    print ("EPA score is ", r2_score( N_obs, N_EPA )  )  
    print ("EPA score is ", r2_score( np.log(N_obs[np.where(N_obs>0.01)]), np.log(N_EPA[np.where(N_obs>0.01)]) )  )  
    data.update({'EPA_abundance':np.array(N_EPA)})
    
    
    N_EPA2=[]
    N_EPA3=[]
    for n in range(n_exp):
        
#        N_EPA2.append(sigma_vectors[n]@r_matrix_EPA_cv.T )    
        N_EPA3.append(r_matrix_EPA_cv@sigma_vectors[n])    
        #for i in range(S):
    N_EPA2=sigma_vectors@r_matrix_EPA_cv.T              
    N_EPA2=np.array(N_EPA2)
    
    
    N_EPA3=np.array(N_EPA3)  
    #print ("EPA score is ", r2_score( N_obs, N_EPA )  )  
    print ("EPA2 score is ", r2_score( np.log(N_obs[np.where(N_obs>0.01)]), N_EPA2[np.where(N_obs>0.01)] )  ) 
    print ("EPA3 score is ", r2_score( np.log(N_obs[np.where(N_obs>0.01)]), N_EPA3[np.where(N_obs>0.01)] )  ) 
    data.update({'EPA_abundance':np.array(N_EPA)})
    
    logN_by_species=[]
    for i in range(S):
        logN_by_species.append (sigma_vectors@r_matrix_EPA_cv[i])
    
    logN_by_species=np.array(logN_by_species)
    swapped_logN=np.swapaxes(logN_by_species,0,1)
    print ("swapped logN score is ", r2_score( np.log(N_obs[np.where(N_obs>0.01)]), swapped_logN[np.where(N_obs>0.01)] )  ) 
    
    
    '''
    the above was prediciting for data where Nobs>0, and did not account for the extinction data that we do try and fit in the mode
    to actually test R2, we need to use the  exact same data !
    '''
    
    Nobs_used_for_fitting=[]
    N_EPA_actually_predicted=[]
    N_EPA2_actually_predicted=[]
    N_EPA3_actually_predicted=[]
    
    for i in range(S):  
        exp_i_present=np.where(sigma_vectors[:,i]==1)[0] 
        
        Nobs_used_for_fitting.append(N_obs[exp_i_present,i])
        N_EPA_actually_predicted.append(N_EPA[exp_i_present,i])
        N_EPA2_actually_predicted.append(N_EPA2[exp_i_present,i])
        N_EPA3_actually_predicted.append(N_EPA3[exp_i_present,i])
        
    print ("actualEPA score is ", r2_score( Nobs_used_for_fitting, N_EPA_actually_predicted )  ) 
    print ("actual EPA score is ", r2_score( np.log(Nobs_used_for_fitting), np.log(N_EPA_actually_predicted) )  )
    print ("actualEPA2 score is ", r2_score( np.log(Nobs_used_for_fitting), N_EPA2_actually_predicted )  ) 
    print ("actualEPA3 score is ", r2_score( np.log(Nobs_used_for_fitting), N_EPA3_actually_predicted )  ) 
    
    for i in range(S): 
    
        temp=X_all_i[i]@r_matrix_EPA_cv[i]-NEPA_cv[i]
        if np.all(temp==0): print ("all were 0 for S="+str(i))
        else: print (temp)    
        
        exp_i_present=np.where(sigma_vectors[:,i]==1)[0] 
        temp2=N_EPA2[exp_i_present,i]-NEPA_cv[i]
        if np.all(temp2==0): print ("all were 0 for S="+str(i))
        else:  print ("NEPA2 was incprrecy",temp2) 
        
        exp_i_present=np.where(sigma_vectors[:,i]==1)[0] 
        temp3=logN_by_species[i,exp_i_present]-NEPA_cv[i]
        if np.all(temp3==0): print ("LogN= 0 for S="+str(i))
        else:  print ("logN was incprrecy",temp3) 
        
        exp_i_present=np.where(sigma_vectors[:,i]==1)[0] 
        temp3=logN_by_species[i,exp_i_present]-np.log(N_obs[exp_i_present,i])
        if np.all(temp3==0): print ("LogNOBS= 0 for S="+str(i))
        else:  print ("logN was incprrecy",temp3) 
        
        exp_i_present=np.where(sigma_vectors[:,i]==1)[0] 
        temp3=logN_by_species[i,exp_i_present]-Y_all_i[i]
        if np.all(temp3==0): print ("LogY= 0 for S="+str(i))
        else:  print ("logN was incprrecy",temp3) 
        
        
        print ("score for Species "+str(i)+"="+ str(r2_score( np.log(N_obs[exp_i_present,i]), logN_by_species[i,exp_i_present] ) ) ) 
    
        print ("The correct to multiply the data is done in logN_by_species abd N_EPA2 but the order is tricky")
#  Y_all_i.append( np.log(N_obs[exp_i_present,i]) )   

if isinstance(r_matrix_LPA_cv, np.ndarray): 
    data.update({'r_matrix_LPA_cv':r_matrix_LPA_cv})
    N_LPA=[]
    for i in range(n_exp):
        N_LPA.append(abundance_function_linear(r_matrix_LPA_cv, sigma_vectors[i]))       
    print ("LPA score is ", r2_score(N_obs,N_LPA)  )  
    data.update({'LPA_abundance':np.array(N_LPA)})

#pickle.dump( data, open(destfold+'dataF'+file_suffix+'.dat', 'wb') )      
    
############## to understand when species go extinct  #########
'''
a species goes extinct in a resource model  with almost  equal coefficients in  the following way.
2 species scenario intuition
 if c=np.array([[1.0, 0.],[1., 1.]])
 then at steady state abundance of first species=0 and second species takes all the resources because it has resource 2 to itself 
 which allows it to support a high abundance.
 In fact if     c=np.array([[1.0, 0.],[1., .1]]), this would still hold..
 The same mechanism can be extended to higher species number:
     if species i eats two resources: 1&2, and species j , k consume  resource 1 &2 respectively, i has to compete with j&k for the resources.
     now if species j &k have another resource they can eat as well say 3 &4 respectively, and they do not have any competition for 3&4, then
     if i,j,k are put together i goes extinct because j grows to a high abundance at steady state and eats all of resource 1, and  similarly
     k eats all of resource 2. This is because the resources 3&4 allow j and k to grow to a higher abundance than i and outcompete i.
'''
#extinct_indices=np.where(Y_all_i<-13)
#extinct_species=list( set(extinct_indices[0]) )
#for e in extinct_species:
#    print (e,c_matrix[e])
#    print('\n')
#    indx=np.where(extinct_indices[0]==e)[0]
#    for i in indx:
#        competing_species=X_all_i[e][ extinct_indices[1][i] ]
#        print (competing_species)
#    print('\n')
#    break
#sys.exit(1)