import os
import torch
import time
import argparse
import hcpdatautils as hcp
import isingmodel
from isingmodel import IsingModel
parser = argparse.ArgumentParser(description="Train an Ising model to predict the probability of a transition based on empirical transition counts.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-s", "--data_subset", type=str, default='validation', help="the subset of subjects over which to search for unique states")
parser.add_argument("-n", "--file_suffix", type=str, default='group_training_epochs_50_batch_12000_lr_1e-07.pt', help="part of the file name after ising_model_")
parser.add_argument("-j", "--model_index", type=int, default=0, help="model to select")
parser.add_argument("-t", "--sim_length", type=int, default=1200, help="number of steps for which to simulate")
parser.add_argument("-v", "--num_sims", type=int, default=10, help="number of simulations")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
data_subset = args.data_subset
print(f'data_subset={data_subset}')
file_suffix = args.file_suffix
print(f'file_suffix={file_suffix}')
model_index = args.model_index
print(f'model_index={model_index}')
sim_length = args.sim_length
print(f'sim_length={sim_length}')
num_sims = args.num_sims
print(f'num_sims={num_sims}')


# Run a simulation, and record and return the Fisher Information Matrix (FIM) estimated from the observed states.
# Since the FIM is symmetric, we only retain the upper triangular part, including the diagonal.
def simulate_and_record_fim(model:torch.nn.Module, num_steps:int):
    batch_size, num_nodes = model.s.size()
    state_triu_rows, state_triu_cols = model.get_triu_indices_for_products()
    # Unlike with the products matrix, the diagonal of the FIM is meaningful.
    num_observables = num_nodes + state_triu_rows.numel()
    # fim_triu_indices = torch.triu_indices(row=num_observables, col=num_observables, offset=0, dtype=int_type, device=model.s.device)
    # fim_triu_rows = fim_triu_indices[0]
    # fim_triu_cols = fim_triu_indices[1]
    # num_fim_triu_elements = fim_triu_indices.size(dim=-1)
    observables = torch.zeros( (batch_size, num_observables), dtype=model.s.dtype, device=model.s.device )
    observables_mean = torch.zeros_like(observables)
    observable_product_mean = torch.zeros( size=(batch_size, num_observables, num_observables), dtype=model.s.dtype, device=model.s.device )
    for _ in range(num_steps):
        model.do_balanced_metropolis_step()
        observables[:,:num_nodes] = model.s
        observables[:,num_nodes:] = model.s[:,state_triu_rows] * model.s[:,state_triu_cols]
        observables_mean += observables
        observable_product_mean.addcmul_(tensor1=observables[:,:,None], tensor2=observables[:,None,:])
    observables_mean /= num_steps
    observable_product_mean /= num_steps
    observable_product_mean.addcmul_(tensor1=observables_mean[:,:,None], tensor2=observables_mean[:,None,:], value=-1)
    return observable_product_mean

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')
with torch.no_grad():
    # group_fc_file = os.path.join(output_directory, f'fc_group_{data_subset}.pt')
    # group_fc = torch.load(group_fc_file)
    # print(f'loaded {group_fc_file}, time {time.time() - code_start_time:.3f}')
    ising_model_file = os.path.join(output_directory, f'ising_model_{file_suffix}')
    model = torch.load(ising_model_file)
    model.h = model.h[model_index:model_index+1,:].clone()
    model.J = model.J[model_index:model_index+1,:,:].clone()
    model.s = model.s[model_index:model_index+1,:].clone()
    print(f'loaded {ising_model_file}, selected instance at index {model_index}, time {time.time() - code_start_time:.3f}')
    # print('sim,\tRMSE,\tcorrelation,\ttime')
    # for sim in range(num_sims):
    #     sim_fc = model.simulate_and_record_fc(num_steps=sim_length)
    #     fc_rmse = isingmodel.get_pairwise_rmse(group_fc, sim_fc)
    #     fc_correlation = isingmodel.get_pairwise_correlation(group_fc, sim_fc)
    #     print(f'{sim},\t{fc_rmse.item():.3g},\t{fc_correlation.item():.3g},\t{time.time()-code_start_time:.3f}')
    fim = simulate_and_record_fim(model=model, num_steps=sim_length)
    fim_file = os.path.join(output_directory, f'fim_{file_suffix}')
    torch.save(obj=fim, f=fim_file)
    print(f'saved {fim_file}, time {time.time() - code_start_time:.3f}')
print(f'done, time {time.time() - code_start_time:.3f}')