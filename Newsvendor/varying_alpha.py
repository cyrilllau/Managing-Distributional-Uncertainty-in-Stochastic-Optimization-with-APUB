import pickle
from docplex.mp.model import Model
import numpy as np
import random
import matplotlib.pyplot as plt
from Models import *
from evaluation import *
from matplotlib.font_manager import FontProperties
import os
import concurrent.futures

# Enable LaTeX for the entire script
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"



###############  Set N  ##############
sample_size = 30
sim_size = 300
bs_size = 300
# Load the MParam
with open('MParam.pkl', 'rb') as file:
    MParam = pickle.load(file)


# Load the test xi data
with open('./xi/test/test_data.pkl', 'rb') as file:
    xi_test = pickle.load(file)
    



def compute_bs_cvar(alpha):
    """Wrapper function for bs_cvar_eval_group to use with multiprocessing."""
    return bs_cvar_eval_group(sample_size, sim_size, alpha, bs_size, MParam, xi_test)


if __name__ == '__main__':
    
    
    SAA_result,SAA_reliability = saa_eval_group(sample_size, sim_size, MParam, xi_test)
    # Your alpha values
    
    #one_alpha_list = [0.03*i for i in range(1,34)]
    
    #alpha_list = 1 - one_alpha_list
    
    alpha_list = [0.0001]+[0.05*i for i in range(1,20)]

    
    BS_CVaR_results = []
    BS_CVaR_reliabilities = []
    q1_list = []
    q3_list = []
    lower_fence_list = []
    upper_fence_list = []
    
    
    # Generate the average_loss_list and reliability for each alpha
    # for alpha in alpha_list:
    #     average_loss_list, reliability = bs_cvar_eval_group(sample_size, sim_size, alpha, bs_size, MParam, xi_test)
    #     BS_CVaR_results.append(average_loss_list)
    #     BS_CVaR_reliabilities.append(reliability)
    

    
    # Use multiprocessing to compute bs_cvar_eval_group for each alpha
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     results = executor.map(compute_bs_cvar, alpha_list)
    
    
    
    num_processes = 10  # Adjust based on your machine's capability
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = executor.map(compute_bs_cvar, alpha_list)
    
    
    # Unpack results
    for result in results:
        average_loss_list, reliability = result
        BS_CVaR_results.append(average_loss_list)
        BS_CVaR_reliabilities.append(reliability)
          
    
    
    
    ##### Save Result #############
    data_to_save = {
        "BS_CVaR_results": BS_CVaR_results,
        "BS_CVaR_reliabilities": BS_CVaR_reliabilities,
        "SAA_result": SAA_result,
        "SAA_reliability": SAA_reliability,
        'alpha_list': alpha_list
    }
    save_path = os.path.join('result_varying_alpha', f'results_N{sample_size}.pkl')
    # Save the results_data using pickle
    with open(save_path, 'wb') as file:
        pickle.dump(data_to_save, file)