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
parser.add_argument("-s", "--sim_length", type=int, default=1200, help="number of simulation steps between beta or parameter optimization updates")
parser.add_argument("-n", "--num_beta", type=int, default=10, help="number of beta values to try")
parser.add_argument("-a", "--min_beta", type=float, default=0, help="min beta value to try")
parser.add_argument("-b", "--max_beta", type=float, default=0.2, help="max beta value to try")
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
num_beta = args.num_beta
print(f'num_beta={num_beta}')
min_beta = args.min_beta
print(f'min_beta={min_beta}')
max_beta = args.max_beta
print(f'max_beta={max_beta}')

def test_model(model:IsingModel, suffix:str):
    s_previous = model.s.clone()
    flip_count = torch.zeros_like(s_previous)
    s_sum = torch.zeros_like(model.s)
    s_product = torch.zeros_like(model.J)
    s_product_sum = torch.zeros_like(model.J)
    print(f'time {time.time() - code_start_time:.3f}, simulating original beta values...')
    for _ in range(sim_length):
        model.do_balanced_metropolis_step()
        flip_count += torch.abs(model.s - s_previous)/2.0
        s_previous[:,:] = model.s[:,:]
        model.do_balanced_metropolis_step()
        s_sum += model.s# B x N x 1
        torch.mul(input=model.s[:,:,None], other=model.s[:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
        s_product_sum += s_product
    s_mean = s_sum/sim_length
    s_product_mean = isingmodel.square_to_triu_pairs(s_product_sum/sim_length)
    sim_fc = isingmodel.get_fc_binary(s_mean=s_mean, s_product_mean=s_product_mean)
    target_product_mean = isingmodel.triu_to_square_pairs(model.target_state_product_means, diag_fill=0)
    target_fc = isingmodel.get_fc_binary( s_mean=model.target_state_means, s_product_mean= target_product_mean)
    fc_rmse = isingmodel.get_pairwise_rmse(mat1=sim_fc, mat2=target_fc)
    fc_corr = isingmodel.get_pairwise_correlation(mat1=sim_fc, mat2=target_fc)
    flip_count_file = os.path.join(file_directory, f'flip_count_{suffix}.pt')
    torch.save(obj=flip_count, f=flip_count_file)
    print(f'time {time.time() - code_start_time:.3f},\t saved {flip_count_file}')
    fc_rmse_file = os.path.join(file_directory, f'fc_rmse_{suffix}.pt')
    torch.save(obj=fc_rmse, f=fc_rmse_file)
    print(f'time {time.time() - code_start_time:.3f},\t saved {fc_rmse_file}')
    fc_corr_file = os.path.join(file_directory, f'fc_corr_{suffix}.pt')
    torch.save(obj=fc_corr, f=fc_corr_file)
    print(f'time {time.time() - code_start_time:.3f},\t saved {fc_corr_file}')
    std_flip_count, mean_flip_count = torch.std_mean(flip_count)
    std_fc_rmse, mean_fc_rmse = torch.std_mean(fc_rmse)
    std_fc_corr, mean_fc_corr = torch.std_mean(fc_rmse)
    return std_flip_count, mean_flip_count, std_fc_rmse, mean_fc_rmse, std_fc_corr, mean_fc_corr

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
    file_suffix = f'sim_length_{sim_length}_{model_file_suffix}_{zero_h_str}'
    test_model(model=model, suffix=file_suffix)
    beta_choices = torch.linspace(start=min_beta, end=max_beta, steps=num_beta, dtype=model.beta.dtype, device=model.beta.device)
    std_flip_count = torch.zeros_like(beta_choices)
    mean_flip_count = torch.zeros_like(beta_choices)
    std_fc_rmse = torch.zeros_like(beta_choices)
    mean_fc_rmse = torch.zeros_like(beta_choices)
    std_fc_corr = torch.zeros_like(beta_choices)
    mean_fc_corr = torch.zeros_like(beta_choices)
    for beta_index in range(num_beta):
        beta_value = beta_choices[beta_index]
        model.beta.fill_(value=beta_value)
        file_suffix = f'sim_length_{sim_length}_{model_file_suffix}_{zero_h_str}_beta_{beta_value:.3g}'
        std_flip_count[beta_index], mean_flip_count[beta_index], std_fc_rmse[beta_index], mean_fc_rmse[beta_index], std_fc_corr[beta_index], mean_fc_corr[beta_index] = test_model(model=model, suffix=file_suffix)
summary_file = os.path.join(file_directory, f'beta_test_summary_sim_length_{sim_length}_{model_file_suffix}_{zero_h_str}.pt')
torch.save( obj=(beta_choices, std_flip_count, mean_flip_count, std_fc_rmse, mean_fc_rmse, std_fc_corr, mean_fc_corr), f=summary_file )
print(f'time {time.time() - code_start_time:.3f}, saved {summary_file}')
print(f'time {time.time() - code_start_time:.3f},\t done')