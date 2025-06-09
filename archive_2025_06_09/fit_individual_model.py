import math
import os
import torch
import time
import argparse
import isingmodel
from isingmodel import IsingModel

parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the expected values of their observables. Compare to expected values from data. Do an Euler step update. Repeat.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-f", "--data_subset", type=str, default='training', help="the subset of subjects over which to search for unique states")
parser.add_argument("-s", "--sim_length", type=int, default=12000, help="number of simulation steps between Euler updates")
parser.add_argument("-d", "--num_updates_beta", type=int, default=3, help="number of updates used when optimizing beta")
parser.add_argument("-u", "--num_updates", type=int, default=10, help="number of Euler updates to do")
parser.add_argument("-p", "--num_parallel", type=int, default=4, help="number of parallel simulations to run")
parser.add_argument("-b", "--beta", type=str, default='0.01', help="inverse temperature at which to run model")
parser.add_argument("-l", "--learning_rate", type=float, default=0.1, help="amount by which to multiply updates to the model parameters during Boltzmann learning")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
data_subset = args.data_subset
print(f'data_subset={data_subset}')
sim_length = args.sim_length
print(f'sim_length={sim_length}')
num_updates_beta = args.num_updates_beta
print(f'num_updates_beta={num_updates_beta}')
num_updates = args.num_updates
print(f'num_updates={num_updates}')
num_parallel = args.num_parallel
print(f'num_parallel={num_parallel}')
beta = args.beta
print(f'beta={beta}')
learning_rate = args.learning_rate
print(f'learning_rate={learning_rate}')

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')
    
    # Load the mean values and FC of the data.
    individual_state_mean_file = os.path.join(output_directory, f'mean_state_individual_{data_subset}.pt')
    individual_state_mean = torch.load(individual_state_mean_file)
    print(f'loaded {individual_state_mean_file}, time {time.time() - code_start_time:.3f}')
    individual_product_mean_file = os.path.join(output_directory, f'mean_state_product_individual_{data_subset}.pt')
    individual_product_mean = torch.load(individual_product_mean_file)
    print(f'loaded {individual_product_mean_file}, time {time.time() - code_start_time:.3f}')
    # Compute the FC of the data.
    individual_fc_file = os.path.join(output_directory, f'fc_individual_{data_subset}.pt')
    individual_fc = torch.load(individual_fc_file)
    print(f'loaded {individual_fc_file}, time {time.time() - code_start_time:.3f}')

    file_suffix = f'individual_{data_subset}_parallel_{num_parallel}_sims_{num_updates_beta}_steps_{sim_length}.pt'
    # Load the file with optimized beta.
    individual_ising_model_file = os.path.join(output_directory, f'ising_model_{file_suffix}')
    individual_ising_model = torch.load(individual_ising_model_file)
    print(f'loaded {individual_ising_model_file}, time {time.time() - code_start_time:.3f}')
    previous_fc_rmse_mean = 2.0# Set to the maximum possible value.
    for update in range(num_updates):
        print(f'update {update}')
        # Do a round of fitting using Boltzmann learning.
        sim_state_mean, sim_product_mean = individual_ising_model.fit(target_mean=individual_state_mean, target_product_mean=individual_product_mean, num_updates=1, steps_per_update=sim_length, learning_rate=learning_rate)
        # Use the means to find the FC.
        sim_individual_fc = isingmodel.get_fc_binary(s_mean=sim_state_mean, s_product_mean=sim_product_mean)
        # Compare the sim FC to the data FC.
        fc_rmse = isingmodel.get_pairwise_rmse(mat1=individual_fc, mat2=sim_individual_fc)
        fc_rmse_mean = fc_rmse.mean()
        # print( 'RMSE', fc_rmse.tolist() )
        print(f'FC RMSE min {fc_rmse.min():.3g}, mean {fc_rmse_mean:.3g}, max {fc_rmse.max():.3g}, time {time.time() - code_start_time:.3f}')
        fc_correlation = isingmodel.get_pairwise_correlation(mat1=individual_fc, mat2=sim_individual_fc)
        # print( 'correlation', fc_correlation.tolist() )
        print(f'FC correlation min {fc_correlation.min():.3g}, mean {fc_correlation.mean():.3g}, max {fc_correlation.max():.3g}, time {time.time() - code_start_time:.3f}')
        # If the RMSE increases, that is usually a sign that the learning rate is too large, making us overshoot the target.
        # Try decreasing it.
        if fc_rmse_mean > previous_fc_rmse_mean:
            learning_rate = 0.1*learning_rate
            print(f'mean FC RMSE increased. Decreasing learning rate to {learning_rate:.3g}.')
    # Save the fitted model.
    file_suffix = f'individual_{data_subset}_parallel_{num_parallel}_beta_sims_{num_updates_beta}_fitting_sims_{num_updates}_steps_{sim_length}_learning_rate_{learning_rate:.3g}.pt'
    individual_ising_model_file = os.path.join(output_directory, f'ising_model_{file_suffix}')
    torch.save(obj=individual_ising_model, f=individual_ising_model_file)
    print(f'loaded {individual_ising_model_file}, time {time.time() - code_start_time:.3f}')
    # Save the simulation FC, FC RMSE, and FC correlation.
    sim_group_fc_file = os.path.join(output_directory, f'sim_fc_{file_suffix}')
    torch.save(obj=sim_individual_fc, f=sim_group_fc_file)
    print(f'saved {sim_group_fc_file}, time {time.time() - code_start_time:.3f}')
    fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{file_suffix}')
    torch.save(obj=fc_rmse, f=fc_rmse_file)
    print(f'saved {fc_rmse_file}, time {time.time() - code_start_time:.3f}')
    fc_correlation_file = os.path.join(output_directory, f'fc_correlation_{file_suffix}')
    torch.save(obj=fc_correlation, f=fc_correlation_file)
    print(f'saved {fc_correlation_file}, time {time.time() - code_start_time:.3f}')

print(f'done, time {time.time() - code_start_time:.3f}')