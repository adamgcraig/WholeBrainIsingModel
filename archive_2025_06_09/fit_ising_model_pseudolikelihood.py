import os
import torch
import time
import argparse
import hcpdatautils as hcp
import isingmodel
from isingmodel import IsingModel

parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the expected values of their observables. Compare to expected values from data. Do an Euler step update. Repeat.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-d", "--data_set", type=str, default='individual_all', help="the set of state and state product means to which to fit models")
parser.add_argument("-f", "--num_folds", type=int, default=1, help="number of unconnected parts into which to partition the brain")
parser.add_argument("-r", "--num_betas_per_target", type=int, default=1, help="number of different beta values to test at a time for each set of target means")
parser.add_argument("-l", "--learning_rate", type=float, default=0.1, help="amount by which to multiply updates to the model parameters during an Euler step")
parser.add_argument("-e", "--num_param_saves", type=int, default=5000000, help="number of times to save the model during parameter optimization")
parser.add_argument("-p", "--param_updates_per_save", type=int, default=10000, help="number of updates between saves during parameter optimization")
parser.add_argument("-a", "--random_init", action='store_true', default=False, help="Include this flag to start h and J from a random configuration instead of from the sample state and state product means.")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
data_set = args.data_set
print(f'data_set={data_set}')
num_folds = args.num_folds
print(f'num_folds={num_folds}')
num_betas_per_target = args.num_betas_per_target
print(f'num_replicas={num_betas_per_target}')
learning_rate = args.learning_rate
print(f'learning_rate={learning_rate}')
num_param_saves = args.num_param_saves
print(f'num_param_saves={num_param_saves}')
param_updates_per_save = args.param_updates_per_save
print(f'param_updates_per_save={param_updates_per_save}')
random_init = args.random_init
print(f'random_init={random_init}')

def init_from_data_mean_files(output_directory:str, data_set:str, num_folds:int, num_betas_per_target:int, random_init:bool=False):
    file_suffix = f'{data_set}.pt'
    state_mean_file = os.path.join(output_directory, f'mean_state_{file_suffix}')
    target_state_means = torch.load(state_mean_file)
    batch_size, num_nodes = target_state_means.size()
    target_state_means = isingmodel.fold_nodes(node_params=target_state_means, num_folds=num_folds)
    folded_batch_size, folded_num_nodes = target_state_means.size()
    print(f'time {time.time() - code_start_time:.3f},\t loaded {state_mean_file} and folded from {batch_size} x {num_nodes} to {folded_batch_size} x {folded_num_nodes}')
    product_mean_file = os.path.join(output_directory, f'mean_state_product_{file_suffix}')
    target_state_product_means = torch.load(product_mean_file)
    batch_size, num_pairs = target_state_product_means.size()
    target_state_product_means = isingmodel.fold_node_pairs(pair_params=target_state_product_means, num_nodes=num_nodes, num_folds=num_folds)
    folded_batch_size, folded_num_pairs = target_state_product_means.size()
    print(f'time {time.time() - code_start_time:.3f},\t loaded {product_mean_file} and folded from {batch_size} x {num_pairs} to {folded_batch_size} x {folded_num_pairs}')
    return IsingModel(target_state_means=target_state_means, target_state_product_means=target_state_product_means, num_betas_per_target=num_betas_per_target, random_init=random_init)

def load_and_binarize_time_series(data_directory:str, data_subset:str):
    subject_list = hcp.load_subject_subset(directory_path=data_directory, subject_subset=data_subset, require_sc=True)
    num_subjects = len(subject_list)
    print(f'allocating memory for time series of {num_subjects} subjects, time {time.time() - code_start_time:.3f}')
    num_nodes = hcp.num_brain_areas
    num_steps = hcp.time_series_per_subject * hcp.num_time_points
    binary_data_ts = torch.zeros( size=(num_subjects, num_nodes, num_steps), dtype=float_type, device=device )
    print(f'loading and binarizing time series data, time {time.time() - code_start_time:.3f}')
    for subject in range(num_subjects):
        subject_id = subject_list[subject]
        # We initially load the data time series with dimensions time series x step x node,
        # but we want to work with them as time series x node x step.
        data_ts = hcp.load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject_id, dtype=float_type, device=device).transpose(dim0=-2, dim1=-1)
        # Once we have binarized each time series, we want to concatenate together the time series for a single subject.
        binary_data_ts[subject,:,:] = isingmodel.binarize_data_ts(data_ts).transpose(dim0=0, dim1=1).flatten(start_dim=-2, end_dim=-1)
    return binary_data_ts

def print_and_save_tensor(values:torch.Tensor, name:str, output_directory:str, file_suffix:str):
    file_name = os.path.join(output_directory, f'{name}_{file_suffix}')
    torch.save(obj=values, f=file_name)
    print(f'time {time.time() - code_start_time:.3f},\t {name} with min {values.min():.3g},\t mean {values.mean():.3g},\t max {values.max():.3g} saved to\t {file_name}')

def print_and_save_goodness(sim_mean_state:torch.Tensor, sim_mean_state_product:torch.Tensor, target_mean_state:torch.Tensor, target_mean_state_product:torch.Tensor, output_directory:str, file_suffix:str):
    mean_state_rmse = isingmodel.get_pairwise_rmse(sim_mean_state, target_mean_state)
    print_and_save_tensor(values=mean_state_rmse, name='mean_state_rmse', output_directory=output_directory, file_suffix=file_suffix)
    mean_state_product_rmse = isingmodel.get_pairwise_rmse(sim_mean_state_product, target_mean_state_product)
    print_and_save_tensor(values=mean_state_product_rmse, name='mean_state_product_rmse', output_directory=output_directory, file_suffix=file_suffix)
    combined_mean_state_rmse = isingmodel.get_pairwise_rmse(  torch.cat( (sim_mean_state, sim_mean_state_product), dim=-1 ), torch.cat( (target_mean_state, target_mean_state_product), dim=-1 )  )
    print_and_save_tensor(values=combined_mean_state_rmse, name='combined_mean_state_rmse', output_directory=output_directory, file_suffix=file_suffix)
    sim_fc = isingmodel.get_fc_binary(s_mean=sim_mean_state, s_product_mean=sim_mean_state_product)
    target_fc = isingmodel.get_fc_binary(s_mean=target_mean_state, s_product_mean=target_mean_state_product)
    fc_rmse = isingmodel.get_pairwise_rmse(sim_fc, target_fc)
    print_and_save_tensor(values=fc_rmse, name='fc_rmse', output_directory=output_directory, file_suffix=file_suffix)
    fc_correlation = isingmodel.get_pairwise_correlation(sim_fc, target_fc)
    print_and_save_tensor(values=fc_correlation, name='fc_correlation', output_directory=output_directory, file_suffix=file_suffix)
    return 0

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')
    # If we have a saved set of binarized time series, then load it.
    # Otherwise, load the individual non-binarized time series from their respective files and binarize them.
    target_ts_file = os.path.join(output_directory, f'binary_data_ts_{data_set}.pt')
    if os.path.exists(target_ts_file):
        target_ts = torch.load(target_ts_file)
    else:
        target_ts = load_and_binarize_time_series(data_directory=data_directory, data_subset='all')
        torch.save(obj=target_ts, f=target_ts_file)
        print(f'time {time.time() - code_start_time:.3f},\t binarized the time series data and saved it to {target_ts_file}')
    # If we have a saved model, then load it and resume from where we left off.
    # Otherwise, create a new one initialized with the data mean states and state products.
    file_suffix = f'{data_set}_fold_{num_folds}_betas_{num_betas_per_target}_pseudolikelihood_lr_{learning_rate}.pt'
    model_file = os.path.join(output_directory, f'ising_model_{file_suffix}')
    if os.path.exists(model_file):
        model = torch.load(model_file)
        print(f'time {time.time() - code_start_time:.3f},\t loaded Ising model from {model_file}')
    else:
        model = init_from_data_mean_files(output_directory=output_directory, data_set=data_set, num_folds=num_folds, num_betas_per_target=num_betas_per_target, random_init=random_init)
        torch.save(obj=model, f=model_file)
        print(f'time {time.time() - code_start_time:.3f},\t initialized Ising model from target mean data and saved to {model_file}')
    target_mean_state, target_mean_state_product = model.get_target_means()
    # Optimize the model parameters h and J using the pseudolikelihood method.
    num_param_updates = num_param_saves * param_updates_per_save
    while model.num_param_updates < num_param_updates:
        sim_mean_state, sim_mean_state_product = model.fit_pseudolikelihood_to_stored_means(target_ts=target_ts, num_updates=param_updates_per_save, learning_rate=learning_rate)
        if os.path.exists(path=model_file):
            os.remove(path=model_file)
        torch.save(obj=model, f=model_file)
        print(f'time {time.time() - code_start_time:.3f},\t completed param optimization steps {model.num_param_updates - param_updates_per_save} to {model.num_param_updates} and saved to {model_file}')
        print_and_save_goodness(sim_mean_state=sim_mean_state, sim_mean_state_product=sim_mean_state_product, target_mean_state=target_mean_state, target_mean_state_product=target_mean_state_product, output_directory=output_directory, file_suffix=f'beta_updates_{model.num_beta_updates}_param_updates_{model.num_param_updates}_{file_suffix}')
print(f'time {time.time() - code_start_time:.3f},\t done')