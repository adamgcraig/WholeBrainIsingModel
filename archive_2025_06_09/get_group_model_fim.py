import os
import torch
import time
import argparse
import hcpdatautils as hcp
import isingmodel
from isingmodel import IsingModel
# ising_model_group_training_epochs_10_batch_8000_lr_1e-09.pt
parser = argparse.ArgumentParser(description="Train an Ising model to predict the probability of a transition based on empirical transition counts.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-s", "--data_subset", type=str, default='training', help="the subset of subjects over which to search for unique states")
parser.add_argument("-e", "--num_epochs", type=int, default=50, help="number of times to iterate over all starting states")
parser.add_argument("-b", "--batch_size", type=int, default=8000, help="number of starting states to use in a single training step")
parser.add_argument("-l", "--learning_rate", type=str, default='1e-09', help="learning rate used when training")
parser.add_argument("-t", "--sim_length", type=int, default=12000, help="number of steps for which to simulate")
parser.add_argument("-n", "--num_sims", type=int, default=1, help="number of simulations")
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
num_sims = args.num_sims
print(f'num_sims={num_sims}')

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')
with torch.no_grad():
    group_fc_file = os.path.join(output_directory, f'fc_group_{data_subset}.pt')
    group_fc = torch.load(group_fc_file)
    print(f'loaded {group_fc_file}, time {time.time() - code_start_time:.3f}')
    file_suffix = f'group_{data_subset}_epochs_{num_epochs}_batch_{batch_size}_lr_{learning_rate}.pt'
    ising_model_file = os.path.join(output_directory, f'ising_model_{file_suffix}')
    params = torch.unsqueeze( input=torch.load(ising_model_file), dim=0 )
    print(f'loaded {ising_model_file}, time {time.time() - code_start_time:.3f}')
    num_params = params.numel()
    num_nodes = isingmodel.num_params_to_num_nodes(num_params)
    model = IsingModel(batch_size=1, num_nodes=num_nodes, beta=1, dtype=params.dtype, device=params.device)
    model.init_with_target_means(target_mean=params[:,:num_nodes], target_product_mean=params[:,num_nodes:])
    model.s = -1*torch.ones_like(model.s)
    print('sim,\tRMSE,\tcorrelation,\ttime')
    for sim in range(num_sims):
        sim_fc = model.simulate_and_record_fc(num_steps=sim_length)
        fc_rmse = isingmodel.get_pairwise_rmse(group_fc, sim_fc)
        fc_correlation = isingmodel.get_pairwise_correlation(group_fc, sim_fc)
        print(f'{sim},\t{fc_rmse.item():.3g},\t{fc_correlation.item():.3g},\t{time.time()-code_start_time:.3f}')
    fim = model.simulate_and_record_fim(num_steps=sim_length)
    fim_file = os.path.join(output_directory, f'fim_{file_suffix}')
    torch.save(obj=fim, f=fim_file)
    print(f'saved {fim_file}, time {time.time() - code_start_time:.3f}')
print(f'done, time {time.time() - code_start_time:.3f}')