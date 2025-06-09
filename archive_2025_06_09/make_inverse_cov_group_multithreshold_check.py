import os
import torch
import time
import argparse
import hcpdatautilsnopandas as hcp
import isingmodellight
from isingmodellight import IsingModelLight

code_start_time = time.time()
float_type = torch.float32
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')
epsilon = 0.0

parser = argparse.ArgumentParser(description="Test the inverse covariance as an initial value for group model J.")
parser.add_argument("-a", "--input_directory", type=str, default='D:\\Ising_model_results_daai', help="containing folder of fMRI_ts_binaries/ts_[6-digit subject ID]_[1|2]_[RL|LR].bin")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the mean state and mean state product PyTorch pickle files")
parser.add_argument("-c", "--data_mean_file_part", type=str, default='thresholds_31_min_0_max_3', help="file name part for the mean state and mean state product files")
parser.add_argument("-d", "--comparison_model_file_part", type=str, default='ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_63000', help="file name of Ising models minus the .pt file extension")
parser.add_argument("-e", "--num_betas", type=int, default=101, help="number of beta (inverse temperature) values to try")
parser.add_argument("-f", "--min_beta", type=float, default=1e-10, help="minimum beta")
parser.add_argument("-g", "--max_beta", type=float, default=1.0, help="maximum beta")
parser.add_argument("-i", "--test_sim_length", type=int, default=120000, help="number of sim steps in beta test")
parser.add_argument("-j", "--beta_opt_length", type=int, default=120000, help="number of sim steps in beta optimization")
parser.add_argument("-k", "--param_opt_length", type=int, default=1200, help="number of sim steps in Boltzmann learning")
parser.add_argument("-l", "--max_beta_updates", type=int, default=1000000, help="maximum number of beta updates to allow")
parser.add_argument("-m", "--param_updates_per_save", type=int, default=1000, help="number of Boltzmann learning steps to perform between saves")
parser.add_argument("-n", "--total_num_saves", type=int, default=1000, help="number of Boltzmann learning model snapshots to save")
parser.add_argument("-o", "--learning_rate", type=float, default=0.0001, help="learning rate for Boltzmann learning")
# parser.add_argument("-p", "--device", type=str, default='cuda', help="device on which to run PyTorch Tensor operations")

args = parser.parse_args()
print('getting arguments...')
input_directory = args.input_directory
print(f'input_directory={input_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
data_mean_file_part = args.data_mean_file_part
print(f'data_mean_file_part={data_mean_file_part}')
comparison_model_file_part = args.comparison_model_file_part
print(f'comparison_model_file_part={comparison_model_file_part}')
num_betas = args.num_betas
print(f'num_betas={num_betas}')
min_beta = args.min_beta
print(f'min_beta={min_beta}')
max_beta = args.max_beta
print(f'max_beta={max_beta}')
test_sim_length = args.test_sim_length
print(f'test_sim_length={test_sim_length}')
beta_opt_length = args.beta_opt_length
print(f'beta_opt_length={beta_opt_length}')
param_opt_length = args.param_opt_length
print(f'param_opt_length={param_opt_length}')
max_beta_updates = args.max_beta_updates
print(f'max_beta_updates={max_beta_updates}')
param_updates_per_save = args.param_updates_per_save
print(f'param_updates_per_save={param_updates_per_save}')
total_num_saves = args.total_num_saves
print(f'total_num_saves={total_num_saves}')
learning_rate = args.learning_rate
print(f'learning_rate={learning_rate}')

def get_fc_in_place(mean_state:torch.Tensor, mean_state_product:torch.Tensor):
    mean_state_product -= ( mean_state.unsqueeze(dim=-1) * mean_state.unsqueeze(dim=-2) )
    std_state = torch.sqrt( mean_state_product.diagonal(offset=0, dim1=-2, dim2=-1) )
    mean_state_product /= ( std_state.unsqueeze(dim=-1) * std_state.unsqueeze(dim=-2) )
    return mean_state_product
    
def count_nans(m:torch.Tensor):
    return torch.count_nonzero( torch.isnan(m) )
    
def save_and_print(m:torch.Tensor, m_file_part:str):
    m_file = os.path.join(output_directory, f'{m_file_part}.pt')
    torch.save(obj=m, f=m_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {m_file}, size', m.size(), f'NaNs {count_nans(m=m)}, min {m.max():.3g}, mean {m.mean():.3g}, max {m.min():.3g}')
    return 0
    
def save_and_print_ising_model(m:IsingModelLight, m_file_part:str):
    m_file = os.path.join(output_directory, f'ising_model_{m_file_part}.pt')
    torch.save(obj=m, f=m_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {m_file}')
    print(f'beta size', m.beta.size(), f'NaNs {count_nans(m=m.beta)}, min {m.beta.max():.3g}, mean {m.beta.mean():.3g}, max {m.beta.min():.3g}')
    print(f'h size', m.h.size(), f'NaNs {count_nans(m=m.h)}, min {m.h.max():.3g}, mean {m.h.mean():.3g}, max {m.h.min():.3g}')
    print(f'J size', m.J.size(), f'NaNs {count_nans(m=m.J)}, min {m.J.max():.3g}, mean {m.J.mean():.3g}, max {m.J.min():.3g}')
    print(f's size', m.s.size(), f'NaNs {count_nans(m=m.s)}, min {m.s.max():.3g}, mean {m.s.mean():.3g}, max {m.s.min():.3g}')
    return 0

def get_inv_covariance(data_mean_file_part:str):
    target_mean_state_file = os.path.join(input_directory, f'mean_state_{data_mean_file_part}.pt')
    target_mean_state = torch.load(f=target_mean_state_file, weights_only=False)
    print( f'time {time.time()-code_start_time:.3f}, loaded {target_mean_state_file}, size', target_mean_state.size() )
    target_mean_state_product_file = os.path.join(input_directory, f'mean_state_product_{data_mean_file_part}.pt')
    target_mean_state_product = torch.load(f=target_mean_state_product_file, weights_only=False)
    print( f'time {time.time()-code_start_time:.3f}, loaded {target_mean_state_product_file}, size', target_mean_state_product.size() )
    # Subtract the means in-place to save memory.
    target_cov = isingmodellight.get_cov(state_mean=target_mean_state, state_product_mean=target_mean_state_product)
    print( f'time {time.time()-code_start_time:.3f}, computed covariance, size', target_cov.size() )
    inv_covariance = torch.linalg.inv(target_cov)
    print( f'time {time.time()-code_start_time:.3f}, computed inverse covariance, size', inv_covariance.size() )
    inv_covariance_file = os.path.join(output_directory, f'inv_cov_{data_mean_file_part}.pt')
    torch.save(obj=inv_covariance, f=inv_covariance_file)
    print( f'time {time.time()-code_start_time:.3f}, saved {inv_covariance_file}' )
    return target_mean_state, target_mean_state_product, inv_covariance

def do_inv_cov_J_comparison(inv_covariance:torch.Tensor, comparison_model_file_part:str):
    model_file = os.path.join(input_directory, f'{comparison_model_file_part}.pt')
    model = torch.load(f=model_file, weights_only=False)
    print( f'time {time.time()-code_start_time:.3f}, loaded {model_file}, J size', model.J.size() )
    inv_covariance_diagonal = torch.diagonal(input=inv_covariance, offset=0, dim1=-2, dim2=-1)
    inv_covariance_diagonal_h_size = isingmodellight.get_h_from_means( models_per_subject=model.h.size(dim=0), mean_state=inv_covariance_diagonal )
    print( f'time {time.time()-code_start_time:.3f}, expanded inverse covariance diagonal to size', inv_covariance_diagonal_h_size.size() )
    h_inv_cov_diag_corr = isingmodellight.get_pairwise_correlation(mat1=model.h, mat2=inv_covariance_diagonal_h_size, epsilon=0.0, dim=-1)
    save_and_print(m=h_inv_cov_diag_corr, m_file_part=f'h_inv_cov_diag_corr_{comparison_model_file_part}')
    inv_covariance_J_size = isingmodellight.get_J_from_means( models_per_subject=model.J.size(dim=0), mean_state_product=inv_covariance )
    print( f'time {time.time()-code_start_time:.3f}, expanded inverse covariance to size', inv_covariance_J_size.size() )
    J_inv_cov_corr = isingmodellight.get_pairwise_correlation_ut( mat1=model.J, mat2=inv_covariance_J_size, epsilon=epsilon )
    save_and_print(m=J_inv_cov_corr, m_file_part=f'J_inv_cov_corr_{comparison_model_file_part}')
    return 0

def get_inv_cov_J_model(inv_covariance:torch.Tensor, min_beta:float, max_beta:float, num_betas:int):
    dtype = inv_covariance.dtype
    device = inv_covariance.device
    num_thresholds, num_nodes, _ = inv_covariance.size()
    beta = isingmodellight.get_linspace_beta(models_per_subject=num_betas, num_subjects=num_thresholds, dtype=dtype, device=device, min_beta=min_beta, max_beta=max_beta)
    J = isingmodellight.get_J_from_means(models_per_subject=num_betas, mean_state_product=-1.0*inv_covariance)
    h = isingmodellight.get_zero_h(models_per_subject=num_betas, num_subjects=num_thresholds, num_nodes=num_nodes, dtype=dtype, device=device)
    s = isingmodellight.get_neg_state_like(input=h)
    return IsingModelLight(beta=beta, J=J, h=h, s=s)

def test_model(model:IsingModelLight, target_fc:torch.Tensor, model_file_fragment:str):
    print(f'time {time.time()-code_start_time:.3f}, starting test simulation...')
    sim_mean_state, sim_mean_state_product, flip_rate = model.simulate_and_record_means_and_flip_rate_pmb(num_steps=test_sim_length)
    print(f'time {time.time()-code_start_time:.3f}, simulation complete')
    test_file_fragment = f'{model_file_fragment}_sim_steps_{test_sim_length}'
    save_and_print(m=sim_mean_state, m_file_part=f'sim_mean_state_{test_file_fragment}')
    save_and_print(m=sim_mean_state_product, m_file_part=f'sim_mean_state_product_{test_file_fragment}')
    save_and_print(m=flip_rate, m_file_part=f'flip_rate_{test_file_fragment}')
    sim_fc = get_fc_in_place(mean_state=sim_mean_state, mean_state_product=sim_mean_state_product)
    fc_corr = isingmodellight.get_pairwise_correlation_ut(mat1=sim_fc, mat2=target_fc, epsilon=epsilon)
    save_and_print(m=fc_corr, m_file_part=f'fc_corr_{test_file_fragment}')
    return 0

with torch.no_grad():
    target_mean_state, target_mean_state_product, inv_covariance = get_inv_covariance(data_mean_file_part=data_mean_file_part)
    do_inv_cov_J_comparison(inv_covariance=inv_covariance, comparison_model_file_part=comparison_model_file_part)
    model = get_inv_cov_J_model(inv_covariance=inv_covariance, min_beta=min_beta, max_beta=max_beta, num_betas=num_betas)
    model_init_file_fragment = f'group_J_inv_cov_h_0_{data_mean_file_part}'
    save_and_print_ising_model(m=model, m_file_part=model_init_file_fragment)
    target_fc = isingmodellight.get_fc(state_mean=target_mean_state, state_product_mean=target_mean_state_product)
    beta_test_model_file_part = f'group_J_inv_cov_h_0_{data_mean_file_part}_beta_num_{num_betas}_min_{min_beta:.3g}_max_{max_beta:.3g}'
    print('beta test')
    test_model(model=model, target_fc=target_fc, model_file_fragment=beta_test_model_file_part)
    num_beta_updates = model.optimize_beta_for_flip_rate_pmb(target_flip_rate=0.5, num_updates=max_beta_updates, num_steps=beta_opt_length, min_beta=min_beta, max_beta=max_beta, epsilon=0.0, verbose=True)
    new_model_file_part = f'{beta_test_model_file_part}_sim_length_{beta_opt_length}_updates_{num_beta_updates}_param_sim_length_{param_opt_length}_updates'
    beta_optimized_model_file_part = f'{new_model_file_part}_0'
    save_and_print_ising_model(m=model, m_file_part=beta_optimized_model_file_part)
    print('post beta-optimization test')
    test_model(model=model, target_fc=target_fc, model_file_fragment=beta_optimized_model_file_part)
    adjusted_learning_rate = learning_rate/model.beta
    for num_saves in range(1,total_num_saves+1):
        model.fit_by_simulation_pmb_multi_learning_rate(target_state_mean=target_mean_state, target_state_product_mean=target_mean_state_product, num_updates=param_updates_per_save, steps_per_update=param_opt_length, learning_rate=adjusted_learning_rate, verbose=True)
        num_updates = num_saves*param_updates_per_save
        save_and_print_ising_model(m=model, m_file_part=f'{new_model_file_part}_{num_updates}')
    print(f'time {time.time()-code_start_time:.3f}, done')