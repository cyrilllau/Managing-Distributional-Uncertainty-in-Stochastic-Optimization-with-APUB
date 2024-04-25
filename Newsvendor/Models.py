from docplex.mp.model import Model
import numpy as np
import random






# In[]
def saa(MParam, train_data):
    mdl = Model("SAA_Newsvendor")
    
    #load MParam
    p = len(MParam['mu_1'])
    d = MParam['d']
    h = MParam['h']
    v = MParam['Retail Prices']
    b = MParam['Stockout Costs']
    
    #extract train_data
    N = len(train_data)
    # Define decision variables
    x = mdl.continuous_var_list(p, lb=0, name='x')
    s = mdl.continuous_var_matrix(N, p, lb=0, name='s')

    # Define the objective function
    objective_expr = d.dot(x)
    for n in range(N):
        xi_n = train_data[n]
        objective_expr += (1/N) * (b.dot(xi_n) + h.dot(np.array([s[n, i] for i in range(p)])))
    mdl.minimize(objective_expr)

    # Define constraints
    for n in range(N):
        xi_n = train_data[n]
        for i in range(p):
            mdl.add_constraint(s[n, i] >= x[i] - xi_n[i])

    # Solve the model
    solution = mdl.solve()
    #s_values = [[solution.get_value(s[n, i]) for i in range(p)] for n in range(N)]
    if solution:
        return solution.get_values(x), mdl.objective_value
    else:
        print("No solution found")
        
        
        
        
        
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



# In[]
def bs_cvar(MParam, train_data, M, alpha):
    model = Model()
    
    # Load MParam
    N = len(train_data)
    p = len(MParam['mu_1'])
    d = MParam['d']
    h = MParam['h']
    b = MParam['Stockout Costs']

    # Generate tilde_xi samples (resampling with replacement)
    xi = train_data
    tilde_xi = bootstrapGenerator(xi,M,RandSeed=1)

    # Define Decision variables
    x = {i: model.continuous_var(lb=0, name=f'x_{i}') for i in range(p)}
    #s = {(n,i): model.continuous_var(lb=0, name=f's_{n}_{i}') for n in range(N) for i in range(p)}
    tilde_s = {(m,n,i): model.continuous_var(lb=0, name=f't_s_{m}_{n}_{i}') for m in range(M) for n in range(N) for i in range(p)}
    w = {m: model.continuous_var(lb=0, name=f'w_{m}') for m in range(M)}
    t = model.continuous_var(lb=-float('inf'), name='t')

    # Define the objective function
    objective_expr =  t
    objective_expr += (1/(alpha * M)) * model.sum(w[n] for n in range(M))

    model.minimize(objective_expr)


    
    # Claim constraints
    
    # Wm
    for m in range(M):

        dx = model.sum(d[i]*x[i] for i in range(p))
        b_tildexi = (1/N)*model.sum( model.sum(b[i]*tilde_xi[m,n,i] for i in range(p)) for n in range(N)  )
        h_tildes = (1/N)*model.sum( model.sum(h[i]*tilde_s[m,n,i] for i in range(p)) for n in range(N)  )
        Wm = {m: model.add_constraint(ct= w[m] >= -t + dx + b_tildexi + h_tildes, ctname=f'BS_CVaR_w_{m}')}
    
    #TSmn
    TSmn = {(m,n,i): model.add_constraint(ct=tilde_s[m,n,i] >= x[i] - tilde_xi[m,n,i],ctname=f'BS_CVaR_ts_{m}_{n}_{i}') for i in range(p) for n in range(N) for m in range(M)  }
        
    

    # Solve the model
    solution = model.solve()
    if solution:
        return [l.solution_value for k, l in x.items()], model.objective_value
    else:
        print("No solution found")
        
        

# In[]
def wass(MParam, train_data, eps):
    model = Model()
    
    # Load MParam
    N = len(train_data)
    K = len(MParam['fa'])
    p = len(MParam['fa'][0])
    fa = MParam['fa']
    fb = MParam['fb']
    xi = train_data
    #e = [1] * p  # Adjusted this to p-dimensional vector
    
    # Define Decision variables
    x = model.continuous_var_list(p, lb=0, name='x')
    s = model.continuous_var_list(N, lb=-float('inf'), name='s')  # Adjusted this to a list
    lbd = model.continuous_var(name='lbd')
    
    # Objective function
    total_s = model.sum(s[i] for i in range(N))
    model.minimize(total_s/N + lbd * eps)
    
    # Constraints
    for n in range(N):
        for k in range(K):
            model.add_constraint(s[n] >= fa[k].dot(xi[n]) + fb[k].dot(x))

    for k in range(K):
        for j in range(p):
            model.add_constraint(fa[k][j] <= lbd)
            model.add_constraint(fa[k][j] >= -lbd)

    # Solve the model
    solution = model.solve()
    #s_values = [[solution.get_value(s[n, i]) for i in range(p)] for n in range(N)]
    if solution:
        return solution.get_values(x), model.objective_value
    else:
        print("No solution found")
    







