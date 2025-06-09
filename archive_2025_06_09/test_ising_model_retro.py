import os
import torch
import time
import argparse
# import hcpdatautils as hcp
import isingmodel
from isingmodel import IsingModel
# params_simple_euler_from_expected_group_training_threshold_median_parallel_10000_updates_1000_sim_length_12000_learning_rate_0.01_beta_0.0122000000000000007743805596760466869.pt
parser = argparse.ArgumentParser(description="Run an Ising model from parameters saved in a file, and compute the FIM.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-s", "--data_subset", type=str, default='training', help="the subset of subjects over which to search for unique states")
parser.add_argument("-e", "--num_epochs", type=int, default=50, help="number of times to iterate over all starting states")
parser.add_argument("-b", "--batch_size", type=int, default=8000, help="number of starting states to use in a single training step")
parser.add_argument("-l", "--learning_rate", type=str, default='0.01', help="learning rate used when training")
parser.add_argument("-t", "--sim_length", type=int, default=12000, help="number of steps for which we simulated when fitting")
parser.add_argument("-d", "--new_sim_length", type=int, default=1200, help="number of steps for which to simulate in this script")
parser.add_argument("-n", "--num_sims", type=int, default=1000, help="number of updates")
parser.add_argument("-p", "--num_parallel", type=int, default=10000, help="number of simulations to run in parallel")
parser.add_argument("-z", "--beta", type=str, default='0.0122000000000000007743805596760466869', help="beta, the inverse temperature parameter")
parser.add_argument("-r", "--rows_per_file", type=int, default=360, help="number of rows of the FIM to save in each file")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
data_subset = args.data_subset
print(f'data_subset={data_subset}')
num_epochs = args.num_epochs
print(f'num_epochs={num_epochs}')
batch_size = args.batch_size
print(f'batch_size={batch_size}')
learning_rate = args.learning_rate
print(f'learning_rate={learning_rate}')
sim_length = args.sim_length
print(f'sim_length={sim_length}')
new_sim_length = args.new_sim_length
print(f'new_sim_length={new_sim_length}')
num_sims = args.num_sims
print(f'num_sims={num_sims}')
num_parallel = args.num_parallel
print(f'num_parallel={num_parallel}')
beta_str = args.beta
beta_float = float(beta_str)
print(f'beta={beta_str}')
rows_per_file = args.rows_per_file
print(f'rows_per_file={rows_per_file}')

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')
with torch.no_grad():
    # group_fc_file = os.path.join(output_directory, f'fc_group_{data_subset}.pt')
    # group_fc = torch.load(group_fc_file)
    # print(f'loaded {group_fc_file}, time {time.time() - code_start_time:.3f}')
    file_suffix = f'group_{data_subset}_threshold_median_parallel_{num_parallel}_updates_{num_sims}_sim_length_{sim_length}_learning_rate_{learning_rate}_beta_{beta_str}.pt'
    ising_model_file = os.path.join(output_directory, f'params_simple_euler_from_expected_{file_suffix}')
    params = torch.unsqueeze( input=torch.load(ising_model_file), dim=0 )
    print(f'loaded {ising_model_file}, time {time.time() - code_start_time:.3f}')
    num_params = params.numel()
    num_nodes = isingmodel.num_params_to_num_nodes(num_params)
    model = IsingModel(batch_size=1, num_nodes=num_nodes, beta=beta_float, dtype=params.dtype, device=params.device)
    model.init_with_target_means(target_mean=params[:,:num_nodes], target_product_mean=params[:,num_nodes:])
    model.s = -1*torch.ones_like(model.s)
    # print('sim,\tRMSE,\tcorrelation,\ttime')
    # for sim in range(num_sims):
    #     sim_fc = model.simulate_and_record_fc(num_steps=new_sim_length)
    #     fc_rmse = isingmodel.get_pairwise_rmse(group_fc, sim_fc)
    #     fc_correlation = isingmodel.get_pairwise_correlation(group_fc, sim_fc)
    #     print(f'{sim},\t{fc_rmse.item():.3g},\t{fc_correlation.item():.3g},\t{time.time()-code_start_time:.3f}')
    print(f'starting sim, time {time.time() - code_start_time:.3f}')
    fim = model.simulate_and_record_fim_square(num_steps=new_sim_length)
    print(f'finished sim, time {time.time() - code_start_time:.3f}')
    num_full_files = num_params//rows_per_file
    has_partial_file = (num_params % rows_per_file) > 0
    num_files = num_full_files + int(has_partial_file)
    for file_index in range(num_full_files):
        start_row = file_index * rows_per_file
        end_row = start_row + rows_per_file
        # fim_file = os.path.join(output_directory, f'fim_row_{start_row}_to_{end_row}_of_{num_params}_{file_suffix}')
        fim_file = os.path.join(output_directory, f'fim_file_{file_index+1}_of_{num_files}_{file_suffix}')
        # Have to clone. Slicing creates a view. When we save a view, it will save all the underlying data.
        torch.save( obj=fim[0,start_row:end_row,:].clone(), f=fim_file )
        print(f'saved file {file_index+1} of {num_files}, {fim_file}, rows {start_row} to {end_row} of {num_params}, time {time.time() - code_start_time:.3f}')
    if has_partial_file:
        start_row = num_full_files * rows_per_file
        end_row = num_params
        # fim_file = os.path.join(output_directory, f'fim_row_{start_row}_to_{end_row}_of_{num_params}_{file_suffix}')
        fim_file = os.path.join(output_directory, f'fim_file_{num_files}_of_{num_files}_{file_suffix}')
        # Have to clone. Slicing creates a view. When we save a view, it will save all the underlying data.
        torch.save( obj=fim[0,start_row:end_row,:].clone(), f=fim_file )
        print(f'saved file {num_files} of {num_files}, rows {start_row} to {end_row} of {num_params}, {fim_file}, time {time.time() - code_start_time:.3f}')
print(f'done, time {time.time() - code_start_time:.3f}')