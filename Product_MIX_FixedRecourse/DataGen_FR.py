import numpy as np
import pickle
import os
# Set a random seed for reproducibility
np.random.seed(12345)




num_groups =1000
test_size = 10000




''' First Stage'''
################   First Stage    ###############
A = []
b = []
c = np.array([12, 20, 18, 40])
c_coef = 1
q_coef = 1.2

c = c/c_coef
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


q = [5, 10]
W = [[1, 0], [0, 1]]
h = [np.random.normal(6000, 100), np.random.normal(4000, 50)]
T = [[4, 9, 7, 10],
     [1,1,3,40]]

cut_prob = 0.3






# Define means and covariance matrices for both states
mu_1 = np.array([12,8])
sd_1 = 0.2*mu_1
var_1 = sd_1**2
corr = 0.5

cov_state1 = [[var_1[0],                corr*sd_1[0]*sd_1[1]],
              [corr*sd_1[0]*sd_1[1],    var_1[1]]]

mean_state1 = [mu_1[0] , mu_1[1]]
# mean_state1 = [12, 8]
# cov_state1 = [[9, 1.5], [1.5, 4]]  # Covariance is 1.5 for state 1

mu_2 =np.array( [2, 1])
sd_2 = 0.2*mu_2

var_2 = sd_2**2


cov_state2 = [[var_2[0],                corr*sd_2[0]*sd_2[1]],
              [corr*sd_2[0]*sd_2[1],    var_2[1]]]

mean_state2 = [mu_2[0],mu_2[1]]

# cov_state2 = [[0.04, 0.01], [0.01, 0.01]]  # Covariance is 0.01 for state 2





################ Train Stage ##################

# Update the demand sizes
# xi_sizes_10_100 = list(range(10, 101, 10))
# xi_sizes_100_1000 = list(range(100, 1001, 100))
# xi_sizes = xi_sizes_10_100 + xi_sizes_100_1000

#xi_sizes = [10000]
#xi_sizes = [30,100,300]
xi_sizes = [30,60,120,240,480,960,1920,3840]


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
        
        
for xi_size in xi_sizes:
    for group in range(num_groups):
        q =  np.array([[5, 10] for i in range(xi_size)] )
        q = q_coef*q
        
        
        W =  np.array([[[0.9, 0], [0, 0.9]]for i in range(xi_size) ])
        
        
        
        
        
        # h = np.array([
        #              [np.random.normal(12, 1.2) if np.random.rand() > cut_prob else np.random.normal(2, 0.2),
        #               np.random.normal(8, 0.8) if np.random.rand() > cut_prob else np.random.normal(1, 0.1)] 
        #              for i in range(xi_size)
        #              ])
        
        # h = np.array([
        #            [np.random.normal(10, 2) if np.random.rand() > cut_prob else np.random.normal(2, 0.2),
        #              np.random.normal(8, 1.6) if np.random.rand() > cut_prob else np.random.normal(1, 0.1)] 
        #               for i in range(xi_size)
        #               ])
        
        # h = np.array([
        #              [np.random.normal(12, 3) if np.random.rand() > cut_prob else np.random.normal(2, 0.2),
        #               np.random.normal(8, 2) if np.random.rand() > cut_prob else np.random.normal(1, 0.1)] 
        #              for i in range(xi_size)
        #              ])
        
        h = np.array([
                    np.random.multivariate_normal(mean_state1, cov_state1) 
                    if np.random.rand() > cut_prob 
                    else np.random.multivariate_normal(mean_state2, cov_state2)
                    for i in range(xi_size)
                ])
                        
        # h = np.array([
        #             [np.random.normal(12000, 2400) if np.random.rand() > 0.1 else np.random.normal(2000, 400),
        #              np.random.normal(8000, 1600) if np.random.rand() > 0.1 else np.random.normal(1000, 200)] 
        #              for i in range(xi_size)])
        
        
        
        
        # T = np.array([
        #                 [[4, 9, 7, 10],
        #                       [1,1,3,40]]
        #     for i in range(xi_size) 
        #     ])
        
        T = np.array([
                        [[4, 9, 7, 10],
                              [3,1,3,6]]
            for i in range(xi_size) 
            ])
        
        
        
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

q =  np.array([[5, 10] for i in range(test_size)] )
q = q_coef*q
W =  np.array([[[0.9, 0], [0, 0.9]]for i in range(test_size) ])



# h = np.array([
#              [np.random.normal(12 if np.random.rand() > cut_prob else 2, 1.2 if np.random.rand() > cut_prob else 0.2), 
#               np.random.normal(8 if np.random.rand() > cut_prob else 1, 0.8 if np.random.rand() > cut_prob else 0.1)]
#              for i in range(test_size)
#              ])

# h = np.array([
#             [np.random.normal(12, 1.2) if np.random.rand() > cut_prob else np.random.normal(2, 0.2),
#              np.random.normal(8, 0.8) if np.random.rand() > cut_prob else np.random.normal(1, 0.1)] 
#              for i in range(test_size)])

# h = np.array([
#             [np.random.normal(10, 2) if np.random.rand() > cut_prob else np.random.normal(2, 0.2),
#               np.random.normal(8, 1.6) if np.random.rand() > cut_prob else np.random.normal(1, 0.1)] 
#               for i in range(test_size)])



# h = np.array([
#             [np.random.normal(12, 3) if np.random.rand() > cut_prob else np.random.normal(2, 0.2),
#              np.random.normal(8, 2) if np.random.rand() > cut_prob else np.random.normal(1, 0.1)] 
#              for i in range(test_size)])

h = np.array([
            np.random.multivariate_normal(mean_state1, cov_state1) 
            if np.random.rand() > cut_prob 
            else np.random.multivariate_normal(mean_state2, cov_state2)
            for i in range(test_size)
        ])

# h = np.array([
#             [np.random.normal(12000, 2400) if np.random.rand() > 0.1 else np.random.normal(2000, 400),
#              np.random.normal(8000, 1600) if np.random.rand() > 0.1 else np.random.normal(1000, 200)] 
#              for i in range(test_size)])

# T = np.array([
#                 [[4, 9, 7, 10],
#                      [1,1,3,40]]
#             for i in range(test_size) 
#             ])


T = np.array([
                [[4, 9, 7, 10],
                      [3,1,3,6]]
            for i in range(test_size) 
            ])



# Save the bimodal data to corresponding subfolder
test_data = {
    'q':q,
    'W':W,
    'h':h,
    'T':T
}


# Create the test folder if it doesn't exist
test_folder_path = './xi10000/test'
if not os.path.exists(test_folder_path):
    os.makedirs(test_folder_path)
    
# Save the bimodal test data to the test folder
with open(f'./xi10000/test/test_data.pkl', 'wb') as file:
    pickle.dump(test_data, file)


