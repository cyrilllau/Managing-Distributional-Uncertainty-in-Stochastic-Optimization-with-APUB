import os
import json
import numpy as np
import matplotlib.pyplot as plt
# import ModelEvaluation as meval
# import Parameters
import csv
import pickle
from docplex.mp.model import Model
import time
from Models import *
from TParam import TParam




# In[36]:
def kth_unit_vector(k, dimension):
    """Create the k-th unit vector of a given dimension."""
    if k < 0 or k >= dimension:
        raise ValueError("k must be within the range of the dimension")
    
    # Initialize a zero vector of the given dimension
    vector = np.zeros(dimension)
    
    # Set the k-th element to 1
    vector[k] = 1
    
    return vector
def all_kth_unit_vectors(dimension):
    """Create a list of all k-th unit vectors for a given dimension."""
    return [kth_unit_vector(k, dimension) for k in range(dimension)]



# In[]
def saa(FParam, train_data):
    
    # Record the start time
    start_time = time.time()
    
    # Set Param
    ## First Stage Param
    c = np.array(FParam['c'])
    A = np.array(FParam['A'])
    b = np.array(FParam['b'])
    
    ## Second Stage Para (xi)
    q = train_data['q']
    W = train_data['W']
    h = train_data['h']
    T = train_data['T']
    
    # Set Dimensions
    ## Dimensions for Sample Size
    N = len(train_data['q'])
    
    ## Dimensions for Decison Variables
    Nx = len(c)  # number of X's
    Ny = len(q[0])  # number of Y's
    
    ## Dimension for Constraints (Matrices)
    NcWy = len(W[0])
    
    # Create Model
    model = Model()

    # Claim Decision Variables
    x = {i: model.continuous_var(name=f'x_{i}') for i in range(Nx)}
    # y = {}
    # for n in range(N):
    #     for k in range(Ny):
    #         if k in [4,5]:
    #             y[(n,k)] = model.binary_var(name=f'y_{n}_{k}')
    #         else:
    #             y[(n,k)] = model.continuous_var(name=f'y_{n}_{k}')
    
    
    y = {(n, k): model.continuous_var(name=f'y_{n}_{k}') for n in range(N)          for k in range(Ny)}

    # Claim objective function
    obj1 = model.sum(c[i] * x[i] for i in range(Nx))
    obj2 = (1/N) * model.sum(
                            model.sum(q[n][j]* y[(n,j)] for j in range(Ny))
                            for n in range(N)
                            )
    Objective = obj1 + obj2
    model.minimize(Objective)

    
    
    
    # Claim H(x)
    Hx = 1
    
    
    # Claim constraints
    ## First Stage constraints
    NcAx_b = len(A)
    cAx_b = {j: model.add_constraint(ct=model.sum(A[j][i] * x[i] for i in range(Nx)) == b[j], ctname=f'bs_a_{j}')                   for j in range(NcAx_b)}
   
    ## Second Stage constraints
    cWy = []
    for n in range(N):
        cWy.append({j: model.add_constraint(ct=
                                                model.sum(W[n][j][k] * y[n, k] for k in range(Ny))
                                            == model.sum(-T[n][j][i] * x[i] for i in range(Nx)) + h[n][j],
                                            ctname=f'saa_b_{n}_{j}') 
                    for j in range(NcWy)})

    # Sove the model
    model.context.solver.lpmethod = 4  # 4 corresponds to the barrier method
    model.solve()
    
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"------- SAA solution time: {elapsed_time} seconds")



    return [v.solution_value for k, v in x.items()],model.objective_value,[v.solution_value for k, v in y.items()]


# In[43]:


# In[]
def wass(FParam, train_data, eps):
    
    # Record the start time
    start_time = time.time()
    
    # Set Param
    ## First Stage Param
    c = np.array(FParam['c'])
    
    ## Second Stage Para (xi)
    q = train_data['q'][0]
    W = train_data['W'][0]
    h = train_data['h']
    T = train_data['T'][0]
    
    
    # Set Dimensions
    ## Dimensions for Sample Size
    N = len(train_data['h'])
    K = len(train_data['h'][0])
    
    ## Dimensions for Decison Variables
    Nx = len(c)  # number of X's
    Ny = len(q)  # number of Y's
    Nphi = len(W[0])
    Npsi = len(W[0])
    
    ## Dimension for Constraints (Matrices)
    NcWy = len(W)
    NcWphi = len(W)
    NcWpsi = len(W)
    
    
    ## Unit Vectors
    e = all_kth_unit_vectors(Ny)
    
    # Create Model
    model = Model()
    
    # Claim Decision Variables
    x = {i: model.continuous_var(name=f'x_{i}') for i in range(Nx)}
    y = {(n, k): model.continuous_var(name=f'y_{n}_{k}') for n in range(N)          for k in range(Ny)}
    gamma = model.continuous_var(lb=0, name='gamma')
    psi = {(k,i): model.continuous_var(lb=float('-inf'), name=f'psi_{k}_{i}') for k in range(K)
              for i in range(Npsi)}
    phi = {(k,i): model.continuous_var(lb=float('-inf'), name=f'phi_{k}_{i}') for k in range(K)
              for i in range(Nphi)}
    
    

    # Claim objective function
    obj1 = model.sum(c[i] * x[i] for i in range(Nx))
    obj2 = (1/N) * model.sum(
                            model.sum(q[j]* y[(n,j)] for j in range(Ny))
                            for n in range(N)
                            )
    obj3 = eps*gamma
    Objective = obj1 + obj2 + obj3
    
    model.minimize(Objective)

    
    
    # Claim H(x)
    Hx =  1
    
    # Claim constraints
    ## First Stage constraints
    cGammaPhi = []
    cGammaPsi = []
    cWphi = []
    cWpsi = []
    for k in range(K):
        cGammaPhi.append(
            model.add_constraint(ct=
                            gamma >= model.sum(q[i]*phi[k,i] for i in range(Nphi))
                                ) 
        )
        
        cGammaPsi.append(
            model.add_constraint(ct=
                            gamma >= model.sum(q[i]*psi[k,i] for i in range(Npsi))
                                ) 
        )
        
        cWphi.append({j:
            model.add_constraint(ct=
                model.sum(W[j][i] * phi[k,i] for i in range(Nphi)) 
                                 >= Hx*e[k][j]
                                )  for j in range(NcWphi)})
        
        
        cWpsi.append({j:
            model.add_constraint(ct=
                model.sum(W[j][i] * psi[k,i] for i in range(Npsi)) >= -Hx*e[k][j]
                                )  for j in range(NcWpsi)})
        
        
                                                                
    
    ## Second Stage constraints
    cWy = []
    for n in range(N):
        cWy.append({j: model.add_constraint(ct=
                                                model.sum(W[j][k] * y[n, k] for k in range(Ny))
                                            >= model.sum(T[j][i] * x[i] for i in range(Nx)) - Hx*h[n][j],
                                            ctname=f'wass_b_{n}_{j}') 
                    for j in range(NcWy)})

        
        
    # Sove the model
    model.context.solver.lpmethod = 4  # 4 corresponds to the barrier method
    model.solve()
    
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"------- Wass solution time: {elapsed_time} seconds")



    return [v.solution_value for k, v in x.items()],model.objective_value




# In[]
def bootstrapGenerator(train_data, NumBS, RandSeed = 1):
    np.random.seed(RandSeed)
    N = len(train_data)
    BSParamGroup = []

    for i in range(NumBS):
        TempParam = []
        for j in range(N):
            k = np.random.randint(0, N - 1)
            TempParam.append(train_data[k])
        BSParamGroup.append(TempParam)

    return np.array(BSParamGroup)


# In[46]:
def bs_cvar(FParam, train_data, M, alpha):
    try:
         # Record the start time
        start_time = time.time()
        r = 1
        A = np.array(FParam['A'])
        b = np.array(FParam['b'])
        c = np.array(FParam['c'])
        
        N = len(train_data['q'])
        q_0 = train_data['q']
        W_0 = train_data['W']
        h_0 = train_data['h']
        T_0 = train_data['T']
    
        q = bootstrapGenerator(q_0, M, RandSeed = 1)
        W = bootstrapGenerator(W_0, M, RandSeed = 1)
        h = bootstrapGenerator(h_0, M, RandSeed = 1)
        T = bootstrapGenerator(T_0, M, RandSeed = 1)

        # Use CPLEX to solve the problem
        Nx = len(c)  # number of X's
        Ny = len(q[0][0])  # number of Y's
        Nz = Ny

        model = Model(name='BS_CVAR')
        # Claim variables
        x = {i: model.continuous_var(name=f'x_{i}') for i in range(Nx)}
        # z = {}
        # for m in range(M):
        #     for n in range(N):
        #         for k in range(Ny):
        #             if k in [4,5]:
        #                 z[(m, n, k)]=model.binary_var(name=f'z_{m}_{n}_{k}')
        #             else:
        #                 z[(m, n, k)] = model.continuous_var(name=f'z_{m}_{n}_{k}')
        
        z = {(m, n, k): model.continuous_var(name=f'z_{m}_{n}_{k}') for m in range(M) for n in range(N)               for k in range(Nz)}
        s = {m: model.continuous_var(name=f's_{m}') for m in range(M)}
        t = model.continuous_var(lb=-1.E+20, name='t')

        # Claim objective function
        Objective =  t  + 1 / (alpha * M) * model.sum(s[m] for m in range(M))
        model.minimize(Objective)
        
        
        
        # Claim H(x)
        Hx =  1
        
        
        # Claim constraints
        # first stage
        NcAx_b = len(A)
        cAx_b = {j: model.add_constraint(ct=model.sum(A[j][i] * x[i] for i in range(Nx)) == b[j], ctname=f'bs_a_{j}')                   for j in range(NcAx_b)}

        cSm = {m: model.add_constraint(ct=s[m] >= -t  + model.sum(model.sum(q[m][n][k] * z[m, n, k] for k in range(Nz))  + model.sum(c[i] * x[i] for i in range(Nx))                                                                  for n in range(N)) / N, ctname=f'bs_b_{m}') for m in range(M)}

        cWy = []
        cWz = []

        NcWyz = len(W[0][0])
        for m in range(M):
            for n in range(N):

                cWy.append({j: model.add_constraint(ct=
                        model.sum(W[m][n][j][k] * z[m, n, k] for k in range(Ny)) 
                    == Hx*h[m][n][j] - 
                            model.sum( T[m][n][j][i] * x[i] for i in range(Nx)), 
                                                    ctname=f'bs_d_{m}_{n}_{j}') 
                            for j in range(NcWyz)})

        # Sove the model
        model.context.solver.lpmethod = 4  # 4 corresponds to the barrier method
        Rlt = model.solve()
        #print(Rlt.objective_value)
        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"------- BS_CVaR solution time: {elapsed_time} seconds")
        return [v.solution_value for k, v in x.items()],model.objective_value

    except Exception as e:
        print("Received exception.")
        print(" - Reason        : {}".format(e))
    finally:
        pass











