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

parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the expected values of their observables. Compare to expected values from data.")
parser.add_argument("-f", "--file_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we read and write files")
parser.add_argument("-m", "--model_file_suffix", type=str, default='beta_updates_100_param_updates_3100_group_training_and_individual_all_fold_1_betas_5_steps_1200_lr_0.01', help="part of the Ising model file name")
parser.add_argument("-o", "--output_file_suffix", type=str, default='retest', help="additional string to include in the file name to help distinguish save files from different test runs")
parser.add_argument("-s", "--sim_length", type=int, default=12000, help="number of simulation steps between beta or parameter optimization updates")
parser.add_argument("-z", "--zero_h", action='store_true', default=False, help="Set this flag in order to 0 out the h parameters.")
parser.add_argument("-r", "--reset_J", action='store_true', default=False, help="Set this flag in order to reset J to the target state product means.")
args = parser.parse_args()
print('getting arguments...')
file_directory = args.file_directory
print(f'file_directory={file_directory}')
model_file_suffix = args.model_file_suffix
print(f'model_file_suffix={model_file_suffix}')
output_file_suffix = args.output_file_suffix
print(f'output_file_suffix={output_file_suffix}')
sim_length = args.sim_length
print(f'sim_length={sim_length}')
zero_h = args.zero_h
print(f'zero_h={zero_h}')
reset_J = args.reset_J
print(f'reset_J={reset_J}')

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
    fim_diag = 1 - torch.cat( (sim_mean_state, sim_mean_state_product), dim=-1 ).square()
    print_and_save_tensor(values=fim_diag, name='fim_diag', output_directory=output_directory, file_suffix=file_suffix)
    return 0

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
    if reset_J:
        model.J = model.target_state_product_means
        reset_J_str = 'J_fc'
    else:
        reset_J_str = 'J_fit'
    # Run the simulation.
    target_mean_state, target_mean_state_product = model.get_target_means()
    sim_mean_state, sim_mean_state_product = model.simulate_and_record_means(num_steps=sim_length)
    print_and_save_goodness(sim_mean_state=sim_mean_state, sim_mean_state_product=sim_mean_state_product, target_mean_state=target_mean_state, target_mean_state_product=target_mean_state_product, output_directory=file_directory, file_suffix=f'sim_length_{sim_length}_{model_file_suffix}_{zero_h_str}_{reset_J_str}_{output_file_suffix}.pt')
print(f'time {time.time() - code_start_time:.3f},\t done')