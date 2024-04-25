import os
import json
import numpy as np
import pickle
from docplex.mp.model import Model
import time
from Models import *
from TParam import TParam




def Stage2ObjVal(q, W, b):
    try:
        Ny = len(q)  # number of Y's
        model = Model(name='Stage2')
        
        # Add variables
        y = {k: model.continuous_var(name=f'y_{k}') for k in range(Ny)}

        # Claim objective function
        Objective = model.sum(q[j] * y[j] for j in range(Ny))
        model.minimize(Objective)

        # Add constraints
        NcWy_b = len(W)
        cWy_b = {i: model.add_constraint(ct=model.sum(W[i][j] * y[j] for j in range(Ny)) >= b[i], ctname=f'Stage2_{i}') \
                 for i in range(NcWy_b)}

        # Sove the model
        val = model.solve()
        
        # check = [v.objective_value for k,v in y.items()]
        # print(check)
        
        return val.objective_value

    except Exception as e:
        print("Received exception.")
        print(" - Reason        : {}".format(e))
    finally:
        pass
    
    

# In[]
def eval_single(Xsol, FParam,test_data ):
    try:
        # Record the start time
        start_time = time.time()
        
        c = np.array(FParam['c'])
        S1ObjVal = np.dot(Xsol, -c)

        N = len(test_data['q'])
        ObjVal = np.zeros(N)
        
        Hx = sum(Xsol)*TParam['fa'] + TParam['fb']
        
        for j in range(N):
            q = test_data['q'][j]
            W = test_data['W'][j]
            h = test_data['h'][j]
            T = test_data['T'][j]
            b = np.dot(T, Xsol) - Hx*h
            ObjVal[j] = Stage2ObjVal(q, W, b) + S1ObjVal


        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"This group evaluation time: {elapsed_time} seconds")
        
        return np.mean(ObjVal)

    except Exception as e:
        print("Received exception.")
        print(" - Reason        : {}".format(e))
    finally:
        pass
    
    
    
    
    
def saa_eval_group(FParam, N, Nsim, test_data):
    
    average_loss_list = []
    realibility_list = []
    solution_list = []
    certificate_list = []
    
    
    print(f'############## Start SAA Evaluation for N = {N} ################')
    for i in range(1,Nsim+1):
        # Load the xi samples
        with open(f'./xi/xi_{N}/group_{i}.pkl', 'rb') as file:
            xi_train = pickle.load(file)
        print(f'------------- Group {i}/{Nsim} -----------')
        Xsol,certificate = saa(FParam, xi_train)
        solution_list.append(Xsol)
        average_loss = eval_single(Xsol, FParam,test_data )
        average_loss_list.append(average_loss)
        #print(average_loss)
        realibility_list.append(certificate>=average_loss)
        certificate_list.append(certificate)
    
    #return average_loss_list, np.mean(realibility_list), solution_list, certificate_list
    return average_loss_list, np.mean(realibility_list)


def bs_cvar_eval_group(FParam, N, Nsim, alpha, M, test_data):
    
    average_loss_list = []
    realibility_list = []
    solution_list = []
    certificate_list = []
        
        
    print(f'############## Start BS_CVaR Evaluation for alpha={alpha}, N ={N} ################')
    for i in range(1,Nsim+1):
        # Load the xi samples
        with open(f'./xi/xi_{N}/group_{i}.pkl', 'rb') as file:
            xi_train = pickle.load(file)
        print(f'------------- Group {i}/{Nsim} -----------')
        Xsol,certificate = bs_cvar(FParam, xi_train, M,alpha)
        solution_list.append(Xsol)
        average_loss = eval_single(Xsol, FParam,test_data )
        average_loss_list.append(average_loss)
        #print(average_loss)
        realibility_list.append(certificate>=average_loss)
        certificate_list.append(certificate)
    
    #return average_loss_list, np.mean(realibility_list), solution_list, certificate_list
    return average_loss_list, np.mean(realibility_list)


def wass_eval_group(FParam, N, Nsim, eps, test_data):
    
    average_loss_list = []
    realibility_list = []
    solution_list = []
    certificate_list = []
        
    print(f'############## Start Wass Evaluation for eps={eps} ################')
    for i in range(1,Nsim+1):
        # Load the xi samples
        with open(f'./xi/xi_{N}/group_{i}.pkl', 'rb') as file:
            xi_train = pickle.load(file)
        print(f'------------- Group {i}/{Nsim} -----------')
        Xsol,certificate = wass(FParam, xi_train, eps)
        solution_list.append(Xsol)
        average_loss = eval_single(Xsol, FParam,test_data )
        print(f'solution:{Xsol}    loss:{average_loss}')
        average_loss_list.append(average_loss)
        #print(average_loss)
        realibility_list.append(certificate>=average_loss)
        certificate_list.append(certificate)
    
    #return average_loss_list, np.mean(realibility_list), solution_list, certificate_list
    return average_loss_list, np.mean(realibility_list)