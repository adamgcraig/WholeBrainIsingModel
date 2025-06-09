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
parser.add_argument("-s", "--sim_length", type=int, default=12000, help="number of simulation steps between beta or parameter optimization updates")
args = parser.parse_args()
print('getting arguments...')
file_directory = args.file_directory
print(f'file_directory={file_directory}')
model_file_suffix = args.model_file_suffix
print(f'model_file_suffix={model_file_suffix}')
sim_length = args.sim_length
print(f'sim_length={sim_length}')

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
    # device = torch.device('cpu')
    # If we have a saved model, then load it and resume from where we left off.
    # Otherwise, create a new one initialized with the data mean states and state products.
    model_file = os.path.join(file_directory, f'ising_model_{model_file_suffix}.pt')
    model = torch.load(f=model_file)
    # Run the simulation.
    target_mean_state, target_mean_state_product = model.get_target_means()
    s_sum = torch.zeros_like(model.s)
    s_product = torch.zeros_like(model.J)
    s_product_sum = torch.zeros_like(model.J)
    # Track a few values at each time step.
    rmse_time_points = torch.zeros( (10,sim_length), dtype=float_type, device=device )
    quantile_cutoffs = torch.tensor([0.025, 0.5, 0.975], dtype=float_type, device=device)
    for step in range(sim_length):
        model.do_balanced_metropolis_step()
        s_sum += model.s# B x N x 1
        torch.mul(input=model.s[:,:,None], other=model.s[:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
        s_product_sum += s_product
        num_steps = step+1
        mean_state_rmse = isingmodel.get_pairwise_rmse(mat1=target_mean_state, mat2=s_sum/num_steps)
        rmse_time_points[0,step] = torch.min(mean_state_rmse)
        quantiles = torch.quantile(mean_state_rmse, quantile_cutoffs)
        rmse_time_points[1,step] = quantiles[0]
        rmse_time_points[2,step] = quantiles[1]
        rmse_time_points[3,step] = quantiles[2]
        rmse_time_points[4,step] = torch.max(mean_state_rmse)
        mean_state_rmse_product = isingmodel.get_pairwise_rmse( mat1=target_mean_state_product, mat2=isingmodel.square_to_triu_pairs(s_product_sum)/num_steps )
        rmse_time_points[5,step] = torch.min(mean_state_rmse_product)
        quantiles_product = torch.quantile(mean_state_rmse_product, quantile_cutoffs)
        rmse_time_points[6,step] = quantiles_product[0]
        rmse_time_points[7,step] = quantiles_product[1]
        rmse_time_points[8,step] = quantiles_product[2]
        rmse_time_points[9,step] = torch.max(mean_state_rmse_product)
        if (num_steps % 1200) == 0:
            print(f'time {time.time() - code_start_time:.3f}, step {step}, state RMSE min {rmse_time_points[0,step]:.3g}, 2.5%-ile {rmse_time_points[1,step]:.3g}, median {rmse_time_points[2,step]:.3g}, 97.5%-ile {rmse_time_points[3,step]:.3g}, max {rmse_time_points[4,step]:.3g}, state product RMSE min {rmse_time_points[5,step]:.3g}, 2.5%-ile {rmse_time_points[6,step]:.3g}, median {rmse_time_points[7,step]:.3g}, 97.5%-ile {rmse_time_points[8,step]:.3g}, max {rmse_time_points[9,step]:.3g}')
rmse_time_points_file = os.path.join(file_directory, f'convergence_test_time_points_sim_length_{sim_length}_{model_file_suffix}.pt')
torch.save(obj=rmse_time_points, f=rmse_time_points_file)
print(f'time {time.time() - code_start_time:.3f}, done')