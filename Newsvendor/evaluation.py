import pickle
from docplex.mp.model import Model
import numpy as np
import random
from Models import *

def eval_single(Xsol, test_samples, MParam):
    #load MParam
    p = len(MParam['mu_1'])
    d = MParam['d']
    h = MParam['h']
    v = MParam['Retail Prices']
    b = MParam['Stockout Costs']
    
    # Calculate the expected loss over the test samples
    total_loss = 0
    for xi in test_samples:
        loss = d.dot(Xsol) + b.dot(xi) + h.dot(np.maximum(Xsol - xi, 0)).sum()
        total_loss += loss

    # Compute the average loss over all test instances
    average_loss = total_loss / len(test_samples)
    
    return average_loss


def saa_eval_group(N, Nsim, MParam, test_samples):
    
    print(f'############## Start SAA Evaluation for N = {N} ################')
    average_loss_list = []
    realibility_list = []
    
    
    for i in range(1,Nsim+1):
        # Load the xi samples
        with open(f'./xi/xi_{N}/group_{i}.pkl', 'rb') as file:
            xi_train = pickle.load(file)
        print(f'------------- Group {i}/{Nsim} -----------')
        Xsol,certificate = saa(MParam,xi_train)
        
        average_loss = eval_single(Xsol, test_samples, MParam)
        average_loss_list.append(average_loss)
        #print(average_loss)
        realibility_list.append(certificate>=average_loss)
    
    
    return average_loss_list, np.mean(realibility_list)


def bs_cvar_eval_group(N, Nsim, alpha, M, MParam, test_samples):
    
    average_loss_list = []
    realibility_list = []
    
    print(f'############## Start BS_CVaR Evaluation for alpha={alpha}, N ={N} ################')
    for i in range(1,Nsim+1):
        # Load the xi samples
        with open(f'./xi/xi_{N}/group_{i}.pkl', 'rb') as file:
            xi_train = pickle.load(file)
        print(f'------------- Group {i}/{Nsim} -----------')
        Xsol,certificate = bs_cvar(MParam,xi_train,M,alpha)
        
        average_loss = eval_single(Xsol, test_samples, MParam)
        average_loss_list.append(average_loss)
        #print(average_loss)
        realibility_list.append(certificate>=average_loss)
    
    
    return average_loss_list, np.mean(realibility_list)

def wass_eval_group(N, Nsim, eps, MParam, test_samples):
    
    average_loss_list = []
    realibility_list = []
    
    print('############## Start Wass Evaluation ################')
    for i in range(1,Nsim+1):
        # Load the xi samples
        with open(f'./xi/xi_{N}/group_{i}.pkl', 'rb') as file:
            xi_train = pickle.load(file)
        print(f'------------- Group {i}/{Nsim} -----------')
        Xsol,certificate = wass(MParam,xi_train,eps)
        
        average_loss = eval_single(Xsol, test_samples, MParam)
        average_loss_list.append(average_loss)
        #print(average_loss)
        realibility_list.append(certificate>=average_loss)
    
    
    return average_loss_list, np.mean(realibility_list)
