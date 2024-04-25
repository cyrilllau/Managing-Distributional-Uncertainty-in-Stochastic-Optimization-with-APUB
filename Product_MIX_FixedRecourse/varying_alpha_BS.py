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




###############  Set N  ##############
sample_size =  480
sim_size = 500
bs_size = 200
# Load the MParam
with open('FParam.pkl', 'rb') as file:
    FParam = pickle.load(file)


# Load the test xi data
with open('./xi10000/test/test_data.pkl', 'rb') as file:
    xi_test = pickle.load(file)
    
# SAA_result,SAA_reliability = saa_eval_group(FParam111111111111111111111````````````````````ram, sample_size, sim_size, xi_test)


def compute_bs_cvar(alpha):
    """Wrapper function for bs_cvar_eval_group to use with multiprocessing."""
    return  bs_cvar_eval_group(FParam, sample_size, sim_size,alpha,bs_size, xi_test)

def compute_wass(eps):
    """Wrapper function for bs_cvar_eval_group to use with multiprocessing."""
    return  wass_eval_group(FParam, sample_size, sim_size,eps, xi_test)

if __name__ == '__main__':
    
    # SAA_result,SAA_reliability = saa_eval_group(FParam, sample_size, sim_size, xi_test)
    
    # # Your alpha values
    # one_alpha_list = np.concatenate(
    #     [np.arange(1, 10)*10.0**(i) for i in range(-3, 0)]
    # )
    
    alpha_list =[0.00001]+ [0.05*i for i in range(1,20)] 
    alpha_list = alpha_list[10:]
    
    # #eps_list = 10*np.concatenate([np.arange(1, 10)*10.0**(i) for i in range(-2, 1)])
    # eps_list = np.append(10*np.concatenate([np.arange(1, 10)*10.0**(i) for i in range(-3, 0)]), [12, 14, 16, 18, 20])
    
    # # Values to be transformed
    # values = np.array([0.01, 10])
    
    # # Applying log10 transformation
    # log_transformed = np.log10(values)
    
    # # Generating equal spaced numbers after log transformation
    # equal_spaced_numbers = np.linspace(log_transformed[0], log_transformed[1], num=20)
    
    # # Transforming the equal spaced numbers back from log10
    # original_values = 10 ** equal_spaced_numbers
    # eps_list = original_values
    
    
    BS_CVaR_results = []
    BS_CVaR_reliabilities = []
    # Wass_results = []
    # Wass_reliabilities = []
    
    
    # Generate the average_loss_list and reliability for each alpha
    # for alpha in alpha_list:
    #     average_loss_list, reliability = bs_cvar_eval_group(FParam, sample_size, sim_size,alpha,bs_size, xi_test)
    #     BS_CVaR_results.append(average_loss_list)
    #     BS_CVaR_reliabilities.append(reliability)
    

    
    # Use multiprocessing to compute bs_cvar_eval_group for each alpha
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     BS_CVaR_temp_results = executor.map(compute_bs_cvar, alpha_list)
    #     Wass_temp_results = executor.map(compute_wass, eps_list)
        
    num_processes = 10  # Adjust based on your machine's capability
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        BS_CVaR_temp_results = executor.map(compute_bs_cvar, alpha_list)
        #Wass_temp_results = executor.map(compute_wass, eps_list)
        
    
    
    # Unpack results
    for result in BS_CVaR_temp_results:
        average_loss_list, reliability = result
        BS_CVaR_results.append(average_loss_list)
        BS_CVaR_reliabilities.append(reliability)
        
        
    # for result in Wass_temp_results:
    #     average_loss_list, reliability = result
    #     Wass_results.append(average_loss_list)
    #     Wass_reliabilities.append(reliability)
          
    
    
    # save_path = os.path.join('result_varying_alpha', f'results_N{sample_size}.pkl')
    # # Load the initial saved data
    # with open(save_path, 'rb') as file:
    #     data_to_save = pickle.load(file)
    
    # # Add Wass results to the data
    # data_to_save["Wass_results"] = Wass_results
    # data_to_save["Wass_reliabilities"] = Wass_reliabilities
    # data_to_save['eps_list'] = eps_list
    
    # # Save the updated data
    # updated_save_path = os.path.join('result_varying_alpha', f'results_N{sample_size}_updated.pkl')
    # with open(updated_save_path, 'wb') as file:
    #     pickle.dump(data_to_save, file)

    
    
    # ##### Save Result #############
    data_to_save = {
        "BS_CVaR_results": BS_CVaR_results,
        "BS_CVaR_reliabilities": BS_CVaR_reliabilities,
        # "Wass_results": Wass_results,
        # "Wass_reliabilities": Wass_reliabilities,
        # "SAA_result": SAA_result,
        # "SAA_reliability": SAA_reliability,
        'alpha_list': alpha_list,
        # 'eps_list': eps_list
    }
    save_path = os.path.join('result_varying_alpha', f'BS_results_N{sample_size}_2.pkl')
    # Save the results_data using pickle
    with open(save_path, 'wb') as file:
        pickle.dump(data_to_save, file)