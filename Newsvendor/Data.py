import numpy as np
import pickle
import os


#myseed = 1234
myseed = 123
num_groups = 300  # Number of simulation


# Set a random seed for reproducibility
np.random.seed(myseed)

# Number of products
p = 10

# Set of states
cT = [1, 2]
p_tau = [0.5, 0.5]

p_tau = [1, 0]

# Draw mean xi vectors from uniform distribution over [5, 100]
mu_1 = np.random.uniform(10,20, p)

# mu_1 = np.array([46.96469186, 42.86139335, 42.26851454, 45.51314769, 47.1946897 ,
#        44.2310646 , 49.80764198, 46.84829739, 44.80931901, 43.92117518])

mu_2 = np.random.uniform(10, 20, p)
# mu_2 = np.array([53.43178016, 57.29049707, 54.38572245, 50.59677897, 53.98044255,
#        57.37995406, 51.8249173 , 51.75451756, 55.31551374, 55.31827587])



# mu_1 = np.random.uniform(10, 50, p)

# mu_2 = np.random.uniform(10, 50, p)

# # Original vectors
# mu_1 = np.array([37.66077802, 54.88435084, 47.50910956, 61.41434335, 61.19903232,
#                  40.90370421, 41.05857021, 62.0748871, 68.32557415, 65.03730539])

# mu_2 = np.array([47.1563454, 50.01990251, 53.6692587, 54.25404054, 47.4050151,
#                  51.22392372, 50.06166331, 40.27536899, 55.45653243, 57.65282381])


# # Calculate midpoints
# midpoints = (mu_1 + mu_2) / 2

# # Separate each component double from its midpoint
# new_mu_1 = midpoints - 2 * (midpoints - mu_1)
# new_mu_2 = midpoints + 2 * (mu_2 - midpoints)

# mu_1 = new_mu_1
# mu_2 = new_mu_2



# Define standard deviations
sigma_1 = 0.1 * mu_1
sigma_2 = 0.1 * mu_2
delta_U = 10

# Generate the random matrix S
S = np.random.randn(p, p)

# Compute the matrix U
U = np.matmul(S.T, S)

# Compute the vector u
u = 1 / np.sqrt(np.diag(U))

# Define the correlation matrix C
C = np.diag(u) @ U @ np.diag(u)


# C =np.array([[ 1.        ,  0.38440508, -0.00955798,  0.13882873, -0.05102385,
#          0.17410306,  0.2712801 ,  0.44069259, -0.1689124 , -0.24877427],
#        [ 0.38440508,  1.        , -0.4979222 ,  0.03005204,  0.30394682,
#         -0.37425353, -0.10534804,  0.57416417,  0.5189533 ,  0.44294725],
#        [-0.00955798, -0.4979222 ,  1.        , -0.04473734,  0.12482724,
#          0.57381096, -0.55475082, -0.34614647, -0.45674818, -0.81772125],
#        [ 0.13882873,  0.03005204, -0.04473734,  1.        , -0.11717806,
#         -0.17100546, -0.01563753, -0.18082449, -0.22028756, -0.04908322],
#        [-0.05102385,  0.30394682,  0.12482724, -0.11717806,  1.        ,
#         -0.00970036, -0.3453014 ,  0.29056381, -0.0820479 , -0.04325996],
#        [ 0.17410306, -0.37425353,  0.57381096, -0.17100546, -0.00970036,
#          1.        , -0.13813795, -0.34266407,  0.00150234, -0.76112176],
#        [ 0.2712801 , -0.10534804, -0.55475082, -0.01563753, -0.3453014 ,
#         -0.13813795,  1.        , -0.01564285, -0.01647271,  0.27113254],
#        [ 0.44069259,  0.57416417, -0.34614647, -0.18082449,  0.29056381,
#         -0.34266407, -0.01564285,  1.        , -0.09300477,  0.24768336],
#        [-0.1689124 ,  0.5189533 , -0.45674818, -0.22028756, -0.0820479 ,
#          0.00150234, -0.01647271, -0.09300477,  1.        ,  0.31789631],
#        [-0.24877427,  0.44294725, -0.81772125, -0.04908322, -0.04325996,
#         -0.76112176,  0.27113254,  0.24768336,  0.31789631,  1.        ]])

# Compute the covariance matrices for the two states
Sigma_1 = 1*np.diag(sigma_1) @ C @ np.diag(sigma_1)
Sigma_2 = 1*np.diag(sigma_2) @ C @ np.diag(sigma_2)

# Assign a uniform retail price for each product
# v = np.full(p, 10)
# c = np.full(p, 3)
v = np.full(p, 0)
c = np.full(p, 0)

# Set the salvage value and stockout cost
# g = v*0.2
# b = v*0.5

# osp 达到效果了 但是reliability不行
g = np.full(p, -4)
b = np.full(p, 6)



# Store parameters in a dictionary
parameters = {
    'mu_1': mu_1,
    'mu_2': mu_2,
    'Sigma_1': Sigma_1,
    'Sigma_2': Sigma_2,
    'Retail Prices': v,
    'Salvage Values': g,
    'Stockout Costs': b
}

# Save the parameters to a file using pickle
with open('MParam.pkl', 'wb') as file:
    pickle.dump(parameters, file)

print("Parameters saved to 'MParam.pkl'")





# Calculate d and h
d = c - v - b
h = v + b - g

def powerset(iterable):
    """Generate the power set of the input iterable."""
    from itertools import chain, combinations
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def generate_h_vectors(h):
    """Generate the desired vectors based on the given vector h."""
    p = len(h)
    all_subsets = list(powerset(range(1, p + 1)))
    h_k = []
    for subset in all_subsets:
        h_new = np.zeros(p)
        for idx in subset:
            h_new[idx - 1] = h[idx - 1]  # subtracting 1 because Python indices start from 0
        h_k.append(h_new)
    return h_k

h_vectors = generate_h_vectors(h)

fa = [b-h_vectors[k] for k in range(len(h_vectors))] 
fb = [d+h_vectors[k] for k in range(len(h_vectors))] 


# Load previously stored parameters
with open('MParam.pkl', 'rb') as file:
    parameters = pickle.load(file)

# Add the new parameters to the dictionary
parameters['d'] = d
parameters['h'] = h
parameters['h_vectors'] = h_vectors
parameters['fa'] = fa
parameters['fb'] = fb

# Store the updated parameters
with open('MParam.pkl', 'wb') as file:
    pickle.dump(parameters, file)

print("Parameters 'd' and 'h' have been added to parameters.pkl.")
print(b)
print(d)
print(h)



# Update the xi sizes
xi_sizes_10_100 = list(range(10, 101, 10))
xi_sizes_100_1000 = list(range(100, 1001, 100))
xi_sizes = [20,40,120]




# Create the main xi folder if it doesn't exist
if not os.path.exists('./xi'):
    os.mkdir('./xi')

# Create subfolders for each xi size inside the main xi folder
for size in xi_sizes:
    subfolder_path = f'./xi/xi_{size}'
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

# Load previously stored parameters
with open('MParam.pkl', 'rb') as file:
    parameters = pickle.load(file)

mu_values = [parameters['mu_1'], parameters['mu_2']]
Sigma_values = [parameters['Sigma_1'], parameters['Sigma_2']]
p1 = 0.5

p1 = 1


scale_values = [mu_1,mu_2]
# Generate xi instances and store them
for size in xi_sizes:
    for group in range(num_groups):
        # Generate uncorrelated data from both modes
        uncorrelated_data1 = np.random.multivariate_normal(mu_values[0], Sigma_values[0], size)
        uncorrelated_data2 = np.random.multivariate_normal(mu_values[1], Sigma_values[1], size)
        
        ##########################################
        # Generate uncorrelated EXP datas
        dim = len(scale_values[0])
        uncorrelated_data1 = np.random.exponential(scale=scale_values[0], size=(size, dim))
        uncorrelated_data2 = np.random.exponential(scale=scale_values[1], size=(size, dim))
        ##########################################
        
        
        # Create bimodal data using Bernoulli distribution
        bernoulli_choices = np.random.binomial(1, p1, size)[:, np.newaxis]
        bimodal_data = bernoulli_choices * uncorrelated_data1 + (1 - bernoulli_choices) * uncorrelated_data2
        
        random_errors = np.random.uniform(- delta_U, delta_U, bimodal_data.shape)
        bimodal_data += random_errors
        
        
        bimodal_data = np.maximum(bimodal_data, 0)
        # Save the bimodal data to corresponding subfolder
        with open(f'./xi/xi_{size}/group_{group+1}.pkl', 'wb') as file:
            pickle.dump(bimodal_data, file)

print("All xi instances saved in their respective subfolders.")



# Set the random seed for reproducibility
np.random.seed(42)

# Number of instances for the test data
test_data_size = 10000

# Load previously stored parameters
with open('MParam.pkl', 'rb') as file:
    parameters = pickle.load(file)

mu_values = [parameters['mu_1'], parameters['mu_2']]
Sigma_values = [parameters['Sigma_1'], parameters['Sigma_2']]



# Generate uncorrelated data from both modes for the test data
uncorrelated_data1_test = np.random.multivariate_normal(mu_values[0], Sigma_values[0], test_data_size)
uncorrelated_data2_test = np.random.multivariate_normal(mu_values[1], Sigma_values[1], test_data_size)



##########################################
# Generate uncorrelated EXP datas
dim = len(scale_values[0])
uncorrelated_data1_test = np.random.exponential(scale=scale_values[0], size=(test_data_size, dim))
uncorrelated_data2_test = np.random.exponential(scale=scale_values[1], size=(test_data_size, dim))
##########################################


# Create bimodal data for the test data using Bernoulli distribution
bernoulli_choices_test = np.random.binomial(1, p1, test_data_size)[:, np.newaxis]
bimodal_data_test = bernoulli_choices_test * uncorrelated_data1_test + (1 - bernoulli_choices_test) * uncorrelated_data2_test
random_errors = np.random.uniform(- delta_U, delta_U, bimodal_data_test.shape)
bimodal_data_test += random_errors
bimodal_data_test = np.maximum(bimodal_data_test, 0)




# Create the test folder if it doesn't exist
test_folder_path = './xi/test'
if not os.path.exists(test_folder_path):
    os.makedirs(test_folder_path)

# Save the bimodal test data to the test folder
with open(f'./xi/test/test_data.pkl', 'wb') as file:
    pickle.dump(bimodal_data_test, file)

print("Test data with 10,000 instances saved in './xi/test/test_data.pkl'.")








