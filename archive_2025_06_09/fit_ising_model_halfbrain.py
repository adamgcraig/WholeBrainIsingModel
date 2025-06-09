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
parser.add_argument("-f", "--data_subset", type=str, default='validation', help="the subset of subjects over which to search for unique states")
parser.add_argument("-m", "--model_type", type=str, default='group', help="either 'group' for a group model or 'individual' for an individual model")
parser.add_argument("-b", "--part", type=str, default='halfbrain', help="'halfbrain' or 'wholebrain'")
parser.add_argument("-l", "--sim_length", type=int, default=4800, help="number of simulation steps between Euler updates")
parser.add_argument("-s", "--sims_per_save", type=int, default=3, help="determines how frequently to save snapshots of the model")
parser.add_argument("-d", "--num_updates_beta", type=int, default=5, help="number of updates used when optimizing beta")
parser.add_argument("-u", "--num_updates", type=int, default=6, help="number of Euler updates to do")
parser.add_argument("-p", "--num_parallel", type=int, default=1000, help="number of parallel simulations to run")
parser.add_argument("-r", "--learning_rate", type=float, default=0.01, help="amount by which to multiply updates to the model parameters during Boltzmann learning")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
data_subset = args.data_subset
print(f'data_subset={data_subset}')
part = args.part
print(f'part={part}')
model_type = args.model_type
print(f'model_type={model_type}')
sim_length = args.sim_length
print(f'sim_length={sim_length}')
sims_per_save = args.sims_per_save
print(f'sims_per_save={sims_per_save}')
num_updates_beta = args.num_updates_beta
print(f'num_updates_beta={num_updates_beta}')
num_updates = args.num_updates
print(f'num_updates={num_updates}')
num_parallel = args.num_parallel
print(f'num_parallel={num_parallel}')
learning_rate = args.learning_rate
print(f'learning_rate={learning_rate}')

def get_quarter_connections(triu_pairs:torch.Tensor, num_nodes:int):
    batch_size = triu_pairs.size(dim=0)
    triu_rows, triu_cols = isingmodel.get_triu_indices_for_products(num_nodes=num_nodes, device=triu_pairs.device)
    square_mat = torch.zeros( size=(batch_size, num_nodes, num_nodes), dtype=triu_pairs.dtype, device=triu_pairs.device )
    square_mat[:,triu_rows,triu_cols] = triu_pairs
    square_mat[:,triu_cols,triu_rows] = triu_pairs
    half_nodes = num_nodes//2
    first_quad = square_mat[:,:half_nodes,:half_nodes]
    last_quad = square_mat[:,half_nodes:,half_nodes:]
    half_triu_rows, half_triu_cols = isingmodel.get_triu_indices_for_products(num_nodes=half_nodes, device=triu_pairs.device)
    first_triu = first_quad[:,half_triu_rows,half_triu_cols]
    last_triu = last_quad[:,half_triu_rows,half_triu_cols]
    return torch.cat( tensors=(first_triu, last_triu), dim=0 )

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')
    
    # Load the mean values and FC of the data.
    file_suffix = f'{model_type}_{data_subset}.pt'
    state_mean_file = os.path.join(output_directory, f'mean_state_{file_suffix}')
    mean_state = torch.load(state_mean_file)
    print(f'loaded {state_mean_file}, time {time.time() - code_start_time:.3f}')
    product_mean_file = os.path.join(output_directory, f'mean_state_product_{file_suffix}')
    mean_state_product = torch.load(product_mean_file)
    print(f'loaded {product_mean_file}, time {time.time() - code_start_time:.3f}')
    # Compute the FC of the data.
    fc_file = os.path.join(output_directory, f'fc_{file_suffix}')
    data_fc = torch.load(fc_file)
    print(f'loaded {fc_file}, time {time.time() - code_start_time:.3f}')
    num_subjects, num_nodes = mean_state.size()
    if part == 'halfbrain':
        num_models = 2*num_subjects*num_parallel
        mean_state = torch.unflatten( input=mean_state, dim=-1, sizes=(2,num_nodes//2) ).flatten(start_dim=0, end_dim=1)
        mean_state_product = get_quarter_connections(triu_pairs=mean_state_product, num_nodes=num_nodes)
        data_fc = get_quarter_connections(triu_pairs=data_fc, num_nodes=num_nodes)
    else:
        num_models = num_subjects*num_parallel
    mean_state = mean_state.repeat( (num_parallel, 1) )
    mean_state_product = mean_state_product.repeat( (num_parallel, 1) )
    data_fc = data_fc.repeat( (num_parallel,1) )

    file_suffix = f'{part}_{model_type}_{data_subset}_parallel_{num_parallel}_sims_{num_updates_beta}_steps_{sim_length}.pt'
    # Load the file with optimized beta.
    ising_model_file = os.path.join(output_directory, f'ising_model_{file_suffix}')
    ising_model = torch.load(ising_model_file)
    print(f'loaded {ising_model_file}, time {time.time() - code_start_time:.3f}')
    num_models = ising_model.s.size(dim=0)
    num_subjects = mean_state.size(dim=0)
    models_per_subject = num_models//num_subjects
    num_models_to_use = models_per_subject * num_subjects
    mean_state = mean_state.repeat( (models_per_subject, 1) )
    mean_state_product = mean_state_product.repeat( (models_per_subject, 1) )
    data_fc = data_fc.repeat( (models_per_subject, 1) )
    ising_model.s = ising_model.s[:num_models_to_use,:]
    ising_model.h = ising_model.h[:num_models_to_use,:]
    ising_model.J = ising_model.J[:num_models_to_use,:,:]
    print(f'We have {num_models} group models and {num_subjects} subjects. Give each subject {models_per_subject} models, and truncate the rest. time {time.time() - code_start_time:.3f}')
    fc_rmse_mean = 2.0# Set to the maximum possible value.
    for update in range(num_updates):
        print(f'update {update}')
        # Do a round of fitting using Boltzmann learning.
        sim_state_mean, sim_product_mean = ising_model.fit(target_mean=mean_state, target_product_mean=mean_state_product, num_updates=1, steps_per_update=sim_length, learning_rate=learning_rate)
        # Use the means to find the FC.
        sim_fc = isingmodel.get_fc_binary(s_mean=sim_state_mean, s_product_mean=sim_product_mean)
        # Compare the sim FC to the data FC.
        fc_rmse = isingmodel.get_pairwise_rmse(mat1=data_fc, mat2=sim_fc)
        previous_fc_rmse_mean = fc_rmse_mean
        fc_rmse_mean = fc_rmse.mean()
        # print( 'RMSE', fc_rmse.tolist() )
        print(f'FC RMSE min {fc_rmse.min():.3g}, mean {fc_rmse_mean:.3g}, max {fc_rmse.max():.3g}, time {time.time() - code_start_time:.3f}')
        fc_correlation = isingmodel.get_pairwise_correlation(mat1=data_fc, mat2=sim_fc)
        # print( 'correlation', fc_correlation.tolist() )
        print(f'FC correlation min {fc_correlation.min():.3g}, mean {fc_correlation.mean():.3g}, max {fc_correlation.max():.3g}, time {time.time() - code_start_time:.3f}')
        # If the RMSE increases, that is usually a sign that the learning rate is too large, making us overshoot the target.
        # Try decreasing it.
        if fc_rmse_mean > previous_fc_rmse_mean:
            learning_rate = 0.1*learning_rate
            print(f'mean FC RMSE increased. Decreasing learning rate to {learning_rate:.3g}.')
        num_updates_so_far = update+1
        if (num_updates_so_far % sims_per_save == 0) or (num_updates_so_far == num_updates):
            # Save the fitted model.
            file_suffix = f'{model_type}_{data_subset}_parallel_{num_parallel}_beta_sims_{num_updates_beta}_fitting_sims_{num_updates_so_far}_steps_{sim_length}_learning_rate_{learning_rate:.3g}.pt'
            ising_model_file = os.path.join(output_directory, f'ising_model_{file_suffix}')
            torch.save(obj=ising_model, f=ising_model_file)
            print(f'loaded {ising_model_file}, time {time.time() - code_start_time:.3f}')
            # Save the simulation FC, FC RMSE, and FC correlation.
            sim_fc_file = os.path.join(output_directory, f'sim_fc_{file_suffix}')
            torch.save(obj=sim_fc, f=sim_fc_file)
            print(f'saved {sim_fc_file}, time {time.time() - code_start_time:.3f}')
            fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{file_suffix}')
            torch.save(obj=fc_rmse, f=fc_rmse_file)
            print(f'saved {fc_rmse_file}, time {time.time() - code_start_time:.3f}')
            fc_correlation_file = os.path.join(output_directory, f'fc_correlation_{file_suffix}')
            torch.save(obj=fc_correlation, f=fc_correlation_file)
            print(f'saved {fc_correlation_file}, time {time.time() - code_start_time:.3f}')

print(f'done, time {time.time() - code_start_time:.3f}')