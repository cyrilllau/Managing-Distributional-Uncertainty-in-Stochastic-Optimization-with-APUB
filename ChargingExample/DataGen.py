import numpy as np
import pickle
import os
# Set a random seed for reproducibility
np.random.seed(12345)




num_groups = 500
test_size = 10000




''' First Stage'''
################   First Stage    ###############

B = 10

A = [[1,1]]
b = [B]
c = [1, 0]
c_coef = 1
q_coef = 1
d_coef = 1



# Save the parameters to a file using pickle
# Store parameters in a dictionary
parameters = {
    'A': A,
    'b': b,
    'c': c
}
with open('FParam.pkl', 'wb') as file:
    pickle.dump(parameters, file)
    
    
    
    
    
    
''' Second Stage'''

cut_prob = 0.7





################ Train Stage ##################
xi_sizes = [10,20,30,40,50,80,320,640]
xi_sizes = [10,20]
# xi_sizes = [10,20,30,40,60,80,90,160,240,320,480,640]
#xi_sizes = [10,20,30,40,60,80,90]




# Create the main demand folder if it doesn't exist
if not os.path.exists('./xi'):
    os.mkdir('./xi')
    
if not os.path.exists('./xi/xi_{size}'):
    os.mkdir('./xi/xi_{size}')

# Create subfolders for each demand size inside the main demand folder
for size in xi_sizes:
    subfolder_path = f'./xi/xi_{size}'
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
        

D_multi = 2
P_multi = 2.5
P = 2



for xi_size in xi_sizes:
    for group in range(num_groups):
        q = []
        W = []
        h = []
        T = []
        
        
        for i in range(xi_size):
            
            case = np.random.rand()
            if case < cut_prob:
                price = np.random.uniform(0.5,1.5)
                #price = np.random.normal(0.85,0.2)
                D = np.random.uniform(5,15)

            else:
                price = np.random.uniform(0.5,1.5)*P_multi
                #price = np.random.normal(0.85,0.2)*P_multi
                D = np.random.uniform(5,15)*D_multi

                
                

        

            
            
            #D = np.random.multivariate_normal(mean_state0, cov_state1) if np.random.rand() > cut_prob else (np.random.multivariate_normal(mean_state1, cov_state1) if np.random.rand() > cut_prob_2 else np.random.multivariate_normal(mean_state2, cov_state2))
            #D_2 = np.random.multivariate_normal(mean_state2, cov_state2) if np.random.rand() > cut_prob else np.random.multivariate_normal(mean_state1, cov_state1)
            
            q_curr = [P,0,price,0]

            
            
            
            W_curr = [
                [0,0,price,1],
                [1,-1,1,0]
                ]
            
            # W_curr = [
            #     [0,0,price,1],
            #     [1,-1,0,0]
            #     ]
            
            
            h_curr = [B,D ]

            
            
            
            T_curr = [[1,0],[1,0]]
            
            
            
            q.append(q_curr)
            W.append(W_curr)  
            h.append(h_curr)
            T.append(T_curr)
            
        
        q = np.array(q)
        W = np.array(W)
        h = np.array(h)
        T = np.array(T)
        
        # Save the bimodal data to corresponding subfolder
        data = {
            'q':q,
            'W':W,
            'h':h,
            'T':T
        }
        with open(f'./xi/xi_{xi_size}/group_{group+1}.pkl', 'wb') as file:
            pickle.dump(data, file)
            
            
            
            
            
            


################ Test Stage ##################



for group in range(test_size):
    q = []
    W = []
    h = []
    T = []
    
    
    for i in range(xi_size):
    

    
            
        case = np.random.rand()
        if case < cut_prob:
            price = np.random.uniform(0.5,1.5)
            #price = np.random.normal(0.85,0.2)
            D = np.random.uniform(5,15)

        else:
            price = np.random.uniform(0.5,1.5)*P_multi
            #price = np.random.normal(0.85,0.2)*P_multi
            D = np.random.uniform(5,15)*D_multi
     
            
        # case = np.random.rand()
        # if case < cut_prob:
        #     #price = np.random.uniform(0.5,1.2)
        #     price = np.random.normal(1,0.05)
        #     #D = np.random.uniform(5,15)
        #     D = np.random.normal(10,1)
        #     #P = np.random.uniform(1.2-0.1,1.2+0.1)
        #     #P = np.random.normal(1.2,0.1)
        #     P = 1.2
        # else:
        #     #price = np.random.uniform(0.5,1.2)
        #     price = np.random.normal(1,0.05)*P_multi
        #     #D = np.random.uniform(5,15)
        #     D = np.random.normal(10,1)*D_multi
        #     #P = np.random.uniform(1.2-0.1,1.2+0.1)
        #     #P = np.random.normal(1.2,0.1)*P_multi
        #     P = 1.2*P_multi

    

        
        q_curr = [P,0,price,0]
        
        
        W_curr = [
            [0,0,price,1],
            [1,-1,1,0]
            ]
        
        # W_curr = [
        #     [0,0,price,1],
        #     [1,-1,0,0]
        #     ]
        
        
        
        h_curr = [B,D ]

        
        
        
        T_curr = [[1,0],[1,0]]
        
        q.append(q_curr)
        W.append(W_curr)
        h.append(h_curr)
        T.append(T_curr)
        
    
    q = np.array(q)
    W = np.array(W)
    h = np.array(h)
    T = np.array(T)



#Save the bimodal data to corresponding subfolder
# test_data = {
#     'q':q,
#     'W':W,
#     'h':h,
#     'T':T
# }


# test_folder_path = './xi10000/test'
# if not os.path.exists(test_folder_path):
#     os.makedirs(test_folder_path)
    
# # Save the bimodal test data to the test folder
# with open(f'./xi10000/test/test_data.pkl', 'wb') as file:
#     pickle.dump(test_data, file)


