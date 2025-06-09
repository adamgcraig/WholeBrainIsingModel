import os
import torch
import time
import argparse
import isingmodellight
from isingmodellight import IsingModelLight

code_start_time = time.time()
float_type = torch.float
int_type = torch.int

parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the expected values of their observables. Compare to expected values from data. Do an Euler step update. Repeat.")
parser.add_argument("-a", "--data_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the Ising model file")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
parser.add_argument("-e", "--model_file_fragment", type=str, default='ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_63000', help="Ising model file name, except .pt extension")
parser.add_argument("-f", "--sim_length", type=int, default=120000, help="number of simulation steps to run")
parser.add_argument("-o", "--rep_index", type=int, default=89, help="index of selected replica")
parser.add_argument("-p", "--target_index", type=int, default=10, help="index of selected threshold or subject")
parser.add_argument("-q", "--model_device", type=str, default='cuda', help="device on which the model Tensors existed when the other script passed it to torch.save()")
parser.add_argument("-r", "--device", type=str, default='cuda', help="device to which we want to load the model now")
parser.add_argument("-s", "--fim_file_part", type=str, default='fim_ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_63000_test_length_24000', help="file from which to load the FIM")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
model_file_fragment = args.model_file_fragment
print(f'model_file_fragment={model_file_fragment}')
sim_length = args.sim_length
print(f'sim_length={sim_length}')
rep_index = args.rep_index
print(f'rep_index={rep_index}')
target_index = args.target_index
print(f'target_index={target_index}')
model_device_str = args.model_device
print(f'model_device={model_device_str}')
model_device = torch.device(model_device_str)
device_str = args.device
print(f'device={device_str}')
device = torch.device(device_str)
fim_file_part = args.fim_file_part
print(f'fim_file_part={fim_file_part}')

output_file_fragment = f'{model_file_fragment}_test_length_{sim_length}'

def save_and_print(mat:torch.Tensor, mat_name:str):
    mat_file = os.path.join(output_directory, f'{mat_name}_{output_file_fragment}.pt')
    torch.save(obj=mat, f=mat_file)
    print( f'time {time.time()-code_start_time:.3f}, saved {mat_file}, size', mat.size(), f'min {mat.min():.3g}, mean {mat.mean():.3g}, max {mat.max():.3g}' )
    return 0
    
def get_one_model():
    model_file = os.path.join(data_directory, f'{model_file_fragment}.pt')
    model = torch.load(f=model_file, weights_only=False, map_location={model_device_str:device_str})
    # model = torch.load(f=model_file, weights_only=False)
    print(f'time {time.time()-code_start_time:.3f}, loaded {model_file}')
    # Select one model, but create singleton dimensions, since the simulation code assumes that these dimensions exist.
    # Make copies instead of just slices so that we can free the memory from the larger ensemble of models.
    num_nodes = model.h.size(dim=-1)
    model_dtype = model.h.dtype
    new_h = torch.zeros( size=(1, 1, num_nodes), dtype=model_dtype, device=device )
    new_h[0, 0, :] = model.h[rep_index, target_index, :]
    model.h = new_h
    print( 'selected h, size', model.h.size(), f'min {model.h.min():.3g}, mean {model.h.mean():.3g}, max {model.h.max():.3g}' )
    new_J = torch.zeros( size=(1, 1, num_nodes, num_nodes), dtype=model_dtype, device=device )
    new_J[0, 0, :, :] = model.J[rep_index, target_index, :, :]
    model.J = new_J
    print( 'selected J, size', model.J.size(), f'min {model.J.min():.3g}, mean {model.J.mean():.3g}, max {model.J.max():.3g}' )
    new_beta = torch.zeros( size=(1, 1), dtype=model_dtype, device=device )
    new_beta[0, 0] = model.beta[rep_index, target_index]
    model.beta = new_beta
    print( 'selected beta, size', model.beta.size(), f'min {model.beta.min():.3g}, mean {model.beta.mean():.3g}, max {model.beta.max():.3g}' )
    new_s = torch.zeros( size=(1, 1, num_nodes), dtype=model_dtype, device=device )
    new_s[0, 0, :] = model.s[rep_index, target_index, :]
    model.s = new_s
    print( 'selected s, size', model.s.size(), f'min {model.s.min():.3g}, mean {model.s.mean():.3g}, max {model.s.max():.3g}' )
    return model
    
def get_time_series():
    # Squeeze out the singleton dimensions from the time series, since we no longer need them.
    ts = torch.squeeze( input=get_one_model().simulate_and_record_time_series_pmb(num_steps=sim_length), dim=0 ).squeeze(dim=0)
    save_and_print(mat=ts, mat_name='ts')
    return ts

def get_augmented_time_series():
    ts = get_time_series()
    num_nodes = ts.size(dim=0)
    triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products(num_nodes=num_nodes, device=ts.device)
    augmented_ts = torch.cat( tensors=(ts, ts[triu_rows,:] * ts[triu_cols,:]), dim=0 )
    print( f'time {time.time()-code_start_time:.3f}, computed augmented time series, size', augmented_ts.size(), f'min {augmented_ts.min():.3g}, mean {augmented_ts.mean():.3g}, max {augmented_ts.max():.3g}' )
    # Do not save this. It is too big.
    return augmented_ts

def get_fim():
    fim_file = os.path.join(output_directory, f'{fim_file_part}.pt')
    fim = torch.load(f=fim_file)
    is_not_nan = torch.logical_not( torch.isnan(fim) )
    num_non_nans = 0
    num_rows = fim.size(dim=0)
    num_values = fim.numel()
    fim_min = -1.0*torch.inf
    fim_sum = 0.0
    fim_max = torch.inf
    for row_index in range(num_rows):
        fim_row = fim[row_index,:]
        is_not_nan_row = is_not_nan[row_index,:]
        row_non_nans = fim_row[is_not_nan_row]
        num_non_nans += row_non_nans.numel()
        fim_min = min( fim_min, row_non_nans.min().item() )
        fim_max = max( fim_max, row_non_nans.max().item() )
        fim_sum += row_non_nans.sum()
    # fim_non_nan = fim[is_not_nan]
    # num_non_nans = fim_non_nan.numel()
    num_nans = num_values - num_non_nans
    fim_mean = fim_sum/num_non_nans
    print( f'time {time.time()-code_start_time:.3f}, saved {fim_file}, size', fim.size(), f'num NaN {num_nans}, num non-NaN {num_non_nans}, min {fim_min:.3g}, mean {fim_mean:.3g}, max {fim_max:.3g}' )
    return fim

def get_fim_eigs():
    eigenvalues, eigenvectors = torch.linalg.eigh( get_fim() )
    eigenvalues_file = os.path.join(output_directory, f'eigenvalues_{output_file_fragment}.pt')
    torch.save(obj=eigenvalues, f=eigenvalues_file)
    print( f'time {time.time()-code_start_time:.3f}, saved {eigenvalues_file}, size', eigenvalues.size(), f'min {eigenvalues.min():.3g}, mean {eigenvalues.mean():.3g}, max {eigenvalues.max():.3g}' )
    eigenvectors_file = os.path.join(output_directory, f'eigenvectors_{output_file_fragment}.pt')
    torch.save(obj=eigenvectors, f=eigenvectors_file)
    print( f'time {time.time()-code_start_time:.3f}, saved {eigenvectors_file}, size', eigenvectors.size(), f'min {eigenvectors.min():.3g}, mean {eigenvectors.mean():.3g}, max {eigenvectors.max():.3g}' )
    return 0

with torch.no_grad():
    get_fim_eigs()
    print(f'time {time.time()-code_start_time:.3f}, done')