import os
import torch
import time
import argparse
import isingmodel
from isingmodel import IsingModel

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')

parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the number of times each node flips.")
parser.add_argument("-f", "--file_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we read and write files")
parser.add_argument("-m", "--model_file_suffix", type=str, default='beta_updates_100_param_updates_3100_group_training_and_individual_all_fold_1_betas_5_steps_1200_lr_0.01', help="part of the Ising model file name")
parser.add_argument("-s", "--sim_length", type=int, default=12000, help="number of simulation steps between beta or parameter optimization updates")
parser.add_argument("-z", "--zero_h", action='store_true', default=False, help="Set this flag in order to 0 out the h parameters.")
args = parser.parse_args()
print('getting arguments...')
file_directory = args.file_directory
print(f'file_directory={file_directory}')
model_file_suffix = args.model_file_suffix
print(f'model_file_suffix={model_file_suffix}')
sim_length = args.sim_length
print(f'sim_length={sim_length}')
zero_h = args.zero_h
print(f'zero_h={zero_h}')

def print_and_save_tensor(values:torch.Tensor, name:str, output_directory:str, file_suffix:str):
    file_name = os.path.join(output_directory, f'{name}_{file_suffix}')
    torch.save(obj=values, f=file_name)
    print(f'time {time.time() - code_start_time:.3f},\t {name} with min {values.min():.3g},\t mean {values.mean():.3g},\t max {values.max():.3g} saved to\t {file_name}')

with torch.no_grad():
    # device = torch.device('cpu')
    # If we have a saved model, then load it and resume from where we left off.
    # Otherwise, create a new one initialized with the data mean states and state products.
    model_file = os.path.join(file_directory, f'ising_model_{model_file_suffix}.pt')
    model = torch.load(f=model_file)
    if zero_h:
        model.h.zero_()
        zero_h_str = 'h_no'
    else:
        zero_h_str = 'h_yes'
    # Run the simulation.
    s_previous = model.s.clone()
    flip_count = torch.zeros_like(s_previous)
    for _ in range(sim_length):
        model.do_balanced_metropolis_step()
        flip_count += torch.abs(model.s - s_previous)/2.0
        s_previous[:,:] = model.s[:,:]
    flip_count_file = os.path.join(file_directory, f'flip_count_sim_length_{sim_length}_{model_file_suffix}_{zero_h_str}.pt')
    torch.save(obj=flip_count, f=flip_count_file)
print(f'time {time.time() - code_start_time:.3f},\t done')