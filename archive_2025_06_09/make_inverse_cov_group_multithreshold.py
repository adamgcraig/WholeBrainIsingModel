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

parser = argparse.ArgumentParser(description="Test the effect of binarization threshold on our ability to fit a group Ising model.")
parser.add_argument("-a", "--input_directory", type=str, default='D:\\Ising_model_results_daai', help="containing folder of fMRI_ts_binaries/ts_[6-digit subject ID]_[1|2]_[RL|LR].bin")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the mean state and mean state product PyTorch pickle files")
parser.add_argument("-c", "--mean_state_file_part", type=str, default='mean_state_thresholds_31_min_0_max_3', help="file name of the mean states minus the .pt file extension")
parser.add_argument("-d", "--mean_state_product_file_part", type=str, default='mean_state_product_thresholds_31_min_0_max_3', help="file name of the mean state products minus the .pt file extension")
parser.add_argument("-e", "--model_file_part", type=str, default='ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000', help="file name of Ising models minus the .pt file extension")
parser.add_argument("-f", "--training_subject_start", type=int, default=0, help="first training subject index")
parser.add_argument("-g", "--training_subject_end", type=int, default=670, help="1 past last training subject index")
parser.add_argument("-i", "--num_thresholds", type=int, default=31, help="number of binarization thresholds in std. dev.s above the mean to try")
parser.add_argument("-j", "--min_threshold", type=float, default=0, help="minimum threshold in std. dev.s")
parser.add_argument("-k", "--max_threshold", type=float, default=3, help="maximum threshold in std. dev.s")
parser.add_argument("-l", "--num_betas", type=int, default=101, help="number of beta (inverse temperature) values to try")
parser.add_argument("-m", "--min_beta", type=float, default=1e-10, help="minimum beta")
parser.add_argument("-n", "--max_beta", type=float, default=1.0, help="maximum beta")
parser.add_argument("-o", "--as_is", action='store_true', default=False, help="Set this flag to indicate that the mean state and mean state product are for the unbinarized data.")
parser.add_argument("-p", "--target_cov_threshold", type=float, default=-1, help="Set this to a non-negative number to use that threshold (or the closest match) for inverse covariance instead of matching the data and model thresholds.")
parser.add_argument("-q", "--target_model_threshold", type=float, default=-1, help="Set this to a non-negative number to use that threshold (or the closest match) for model J instead of matching the data and model thresholds.")
parser.add_argument("-r", "--test_sim_length", type=int, default=120000, help="number of sim steps in beta test")
parser.add_argument("-s", "--beta_opt_length", type=int, default=1200, help="number of sim steps in beta optimization")
parser.add_argument("-t", "--param_opt_length", type=int, default=1200, help="number of sim steps in Boltzmann learning")
parser.add_argument("-u", "--zero_h", action='store_true', default=False, help="Set this flag to indicate that we should use 0 as the initial guess for h instead of -1/inv_cov_diag.")
parser.add_argument("-v", "--max_beta_updates", type=int, default=1000000, help="maximum number of beta updates to allow")
parser.add_argument("-w", "--param_updates_per_save", type=int, default=1000, help="number of Boltzmann learning steps to perform between saves")
parser.add_argument("-x", "--total_num_saves", type=int, default=1000, help="number of post-Boltzmann learning model snapshots to save")

args = parser.parse_args()
print('getting arguments...')
input_directory = args.input_directory
print(f'input_directory={input_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
mean_state_file_part = args.mean_state_file_part
print(f'mean_state_file_part={mean_state_file_part}')
mean_state_product_file_part = args.mean_state_product_file_part
print(f'mean_state_product_file_part={mean_state_product_file_part}')
model_file_part = args.model_file_part
print(f'model_file_part={model_file_part}')
training_subject_start = args.training_subject_start
print(f'training_subject_start={training_subject_start}')
training_subject_end = args.training_subject_end
print(f'training_subject_end={training_subject_end}')
num_thresholds = args.num_thresholds
print(f'num_thresholds={num_thresholds}')
min_threshold = args.min_threshold
print(f'min_threshold={min_threshold}')
max_threshold = args.max_threshold
print(f'max_threshold={max_threshold}')
num_betas = args.num_betas
print(f'num_betas={num_betas}')
min_beta = args.min_beta
print(f'min_beta={min_beta}')
max_beta = args.max_beta
print(f'max_beta={max_beta}')
as_is = args.as_is
print(f'as_is={as_is}')
target_cov_threshold = args.target_cov_threshold
print(f'target_cov_threshold={target_cov_threshold}')
model_threshold = args.target_model_threshold
print(f'target_model_threshold={model_threshold}')
test_sim_length = args.test_sim_length
print(f'test_sim_length={test_sim_length}')
beta_opt_length = args.beta_opt_length
print(f'beta_opt_length={beta_opt_length}')
param_opt_length = args.param_opt_length
print(f'param_opt_length={param_opt_length}')
zero_h = args.zero_h
print(f'zero_h={zero_h}')
max_beta_updates = args.max_beta_updates
print(f'max_beta_updates={max_beta_updates}')
param_updates_per_save = args.param_updates_per_save
print(f'param_updates_per_save={param_updates_per_save}')
total_num_saves = args.total_num_saves
print(f'total_num_saves={total_num_saves}')

def get_J():
    model_file = os.path.join(input_directory, f'{model_file_part}.pt')
    J = torch.load(f=model_file, weights_only=False)
    print( f'loaded J from {model_file}, size', J.size() )
    return J

def get_index_for_threshold(target_threshold:float):
    return torch.argmin(  torch.abs( target_threshold - torch.linspace(start=min_threshold, end=max_threshold, steps=num_thresholds, dtype=float_type, device=device) )  )

with torch.no_grad():
    mean_state_file = os.path.join(input_directory, f'{mean_state_file_part}.pt')
    mean_state = torch.load(f=mean_state_file, weights_only=False)
    print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_file}, size', mean_state.size() )
    mean_state_product_file = os.path.join(input_directory, f'{mean_state_product_file_part}.pt')
    mean_state_product = torch.load(f=mean_state_product_file, weights_only=False)
    print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_product_file}, size', mean_state_product.size() )
    model_file = os.path.join(input_directory, f'{model_file_part}.pt')
    model = torch.load(f=model_file, weights_only=False)
    print( f'time {time.time()-code_start_time:.3f}, loaded {model_file}, J size', model.J.size() )
    models_per_threshold, num_thresholds, _ = model.h.size()
    if as_is:
        # The as-is data is prior to z-scoring.
        # To avoid bias due to different overall scales of data, we can convert the mean and uncentered covariance to those of the z-scored data prior to taking the means over scans and subjects.
        # Taking the z-score converts every mean to 0 and every variance to 1 so that mean_state == 0 and mean_state_product is equivalent to the Pearson correlation.
        mean_state_product = isingmodellight.get_fc(state_mean=mean_state, state_product_mean=mean_state_product, epsilon=epsilon)
        mean_state = torch.zeros_like(input=mean_state)
        # The as-is versions have dimensions (scans, subjects) over which we need to take the mean.
        # We then need to repeat the result for each model along dimensions (replicas, thresholds).
        # Both are the first 2 dimensions, with the third and fourth being nodes.
        mean_state = mean_state.mean( dim=(0,1), keepdim=False ).unsqueeze(dim=0).repeat( (num_thresholds, 1) )
        mean_state_product = mean_state_product.mean( dim=(0,1), keepdim=False ).unsqueeze(dim=0).repeat( (num_thresholds, 1, 1) )
        data_threshold_str = 'as_is'
    elif target_cov_threshold >= 0:
        selected_threshold_index = get_index_for_threshold(target_threshold=target_cov_threshold)
        mean_state = torch.unsqueeze(input=mean_state[selected_threshold_index,:], dim=0).repeat(  ( mean_state.size(dim=0), 1 )  )
        mean_state_product = torch.unsqueeze(input=mean_state_product[selected_threshold_index,:,:], dim=0).repeat(  ( mean_state_product.size(dim=0), 1, 1 )  )
        data_threshold_str = f'thresh_{target_cov_threshold:.3g}'
    else:
        data_threshold_str = f'thresh_num_{num_thresholds}_min_{min_threshold:.3g}_max_{max_threshold:.3g}'
    if model_threshold >= 0:
        selected_threshold_index = get_index_for_threshold(target_threshold=model_threshold)
        model.J = torch.unsqueeze(input=model.J[:,selected_threshold_index,:,:], dim=1).repeat(  ( 1, model.J.size(dim=1), 1, 1 )  )
        thresh_str = f'_thresh_{model_threshold:.3g}'
    else:
        thresh_str = ''
    if zero_h:
        h_str = '0'
    else:
        h_str = 'neg_recip_inv_cov_diag'
    
    target_cov = isingmodellight.get_cov(state_mean=mean_state, state_product_mean=mean_state_product)
    print( f'time {time.time()-code_start_time:.3f}, computed covariance, size', target_cov.size() )
    inv_covariance = torch.linalg.inv(target_cov)
    print( f'time {time.time()-code_start_time:.3f}, computed inverse covariance, size', inv_covariance.size() )
    inv_covariance_file = os.path.join(output_directory, f'inv_cov_{data_threshold_str}.pt')
    torch.save(obj=inv_covariance, f=inv_covariance_file)
    print( f'time {time.time()-code_start_time:.3f}, saved {inv_covariance_file}' )
    inv_covariance_diagonal = torch.diagonal(input=inv_covariance, offset=0, dim1=-2, dim2=-1)
    inv_covariance -= torch.diag_embed(input=inv_covariance_diagonal, offset=0, dim1=-2, dim2=-1)
    inv_covariance_diagonal_h_size = isingmodellight.get_h_from_means( models_per_subject=model.h.size(dim=0), mean_state=inv_covariance_diagonal )
    print( f'time {time.time()-code_start_time:.3f}, expanded inverse covariance diagonal to size', inv_covariance_diagonal_h_size.size() )
    h_inv_cov_diag_corr = isingmodellight.get_pairwise_correlation(mat1=model.h, mat2=inv_covariance_diagonal_h_size, epsilon=0.0, dim=-1)
    print( f'time {time.time()-code_start_time:.3f}, computed correlations between h and inverse covariance diagonal, size', h_inv_cov_diag_corr.size() )
    h_inv_cov_daig_corr_file = os.path.join(output_directory, f'corr_h_{model_file_part}_inv_cov_diag_{data_threshold_str}.pt')
    torch.save(obj=h_inv_cov_diag_corr, f=h_inv_cov_daig_corr_file)
    print( f'time {time.time()-code_start_time:.3f}, saved {h_inv_cov_daig_corr_file}' )
    inv_covariance_J_size = isingmodellight.get_J_from_means( models_per_subject=model.J.size(dim=0), mean_state_product=inv_covariance )
    print( f'time {time.time()-code_start_time:.3f}, expanded inverse covariance to size', inv_covariance_J_size.size() )
    J_inv_cov_corr = isingmodellight.get_pairwise_correlation_ut( mat1=model.J, mat2=inv_covariance_J_size, epsilon=epsilon )
    print( f'time {time.time()-code_start_time:.3f}, computed correlations between J and inverse covariance, size', J_inv_cov_corr.size() )
    J_inv_cov_corr_file = os.path.join(output_directory, f'corr_J_{model_file_part}_inv_cov_{data_threshold_str}.pt')
    torch.save(obj=J_inv_cov_corr, f=J_inv_cov_corr_file)
    print( f'time {time.time()-code_start_time:.3f}, saved {J_inv_cov_corr_file}' )
    num_thresholds = model.J.size(dim=1)
    model.beta = isingmodellight.get_linspace_beta(models_per_subject=num_betas, num_subjects=num_thresholds, dtype=model.J.dtype, device=model.J.device, min_beta=min_beta, max_beta=max_beta)
    if zero_h:
        model.h = torch.zeros_like(model.h)
    else:
        model.h = -1.0/inv_covariance_diagonal_h_size
    model.J = inv_covariance_J_size
    model.J *= -1.0
    # if mean_h:
    #     model.h = isingmodellight.get_h_from_means( models_per_subject=model.h.size(dim=0), mean_state=mean_state )
    # else:
    #     model.h = torch.zeros_like( input=model.h )
    print(f'time {time.time()-code_start_time:.3f}, starting beta test simulation...')
    sim_state_mean, sim_state_product_mean, flip_rate = model.simulate_and_record_means_and_flip_rate_pmb(num_steps=test_sim_length)
    print(f'time {time.time()-code_start_time:.3f}, simulation complete')

    model_init_file_fragment = f'ising_model_group_J_inv_cov_h_{h_str}_thresh_num_{num_thresholds}_min_{min_threshold:.3g}_max_{max_threshold:.3g}'

    beta_test_file_fragment = f'{model_init_file_fragment}_test_beta_min_{min_beta:.3g}_max_{max_beta:.3g}_sim_steps_{test_sim_length}'
    sim_state_mean_file = os.path.join(output_directory, f'sim_state_mean_{beta_test_file_fragment}.pt')
    torch.save(obj=sim_state_mean, f=sim_state_mean_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {sim_state_mean_file}')
    sim_state_product_mean_file = os.path.join(output_directory, f'sim_state_product_mean_{beta_test_file_fragment}.pt')
    torch.save(obj=sim_state_product_mean, f=sim_state_product_mean_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {sim_state_product_mean_file}')
    flip_rate_file = os.path.join(output_directory, f'flip_rate_{beta_test_file_fragment}.pt')
    torch.save(obj=flip_rate, f=flip_rate_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {flip_rate_file}')
    sim_cov = isingmodellight.get_cov(state_mean=sim_state_mean, state_product_mean=sim_state_product_mean)
    cov_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_cov, mat2=target_cov)
    cov_rmse_file = os.path.join(output_directory, f'cov_rmse_{beta_test_file_fragment}.pt')
    torch.save(obj=cov_rmse, f=cov_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {cov_rmse_file}')
    cov_corr = isingmodellight.get_pairwise_correlation_ut(mat1=sim_cov, mat2=target_cov, epsilon=epsilon)
    cov_corr_file = os.path.join(output_directory, f'cov_corr_{beta_test_file_fragment}.pt')
    torch.save(obj=cov_corr, f=cov_corr_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {cov_corr_file}')
    target_fc = isingmodellight.get_fc_binary(state_mean=mean_state, state_product_mean=mean_state_product, epsilon=epsilon)
    sim_fc = isingmodellight.get_fc_binary(state_mean=sim_state_mean, state_product_mean=sim_state_product_mean, epsilon=epsilon)
    fc_rmse = isingmodellight.get_pairwise_correlation_ut(mat1=sim_fc, mat2=target_fc, epsilon=epsilon)
    fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{beta_test_file_fragment}.pt')
    torch.save(obj=fc_rmse, f=fc_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {fc_rmse_file}')
    fc_corr = isingmodellight.get_pairwise_correlation_ut(mat1=sim_fc, mat2=target_fc, epsilon=epsilon)
    fc_corr_file = os.path.join(output_directory, f'fc_corr_{beta_test_file_fragment}.pt')
    torch.save(obj=fc_corr, f=fc_corr_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {fc_corr_file}')

    model = IsingModelLight(beta=model.beta, J=model.J, h=model.h, s=model.s)
    num_beta_updates = model.optimize_beta_for_fc_corr_pmb(target_fc=target_fc, num_updates=max_beta_updates, num_steps=test_sim_length, min_beta=min_beta, max_beta=max_beta, epsilon=0.0, verbose=True)
    new_model_file_part = f'{model_init_file_fragment}_beta_num_{num_betas}_min_{min_beta:.3g}_max_{max_beta:.3g}_updates_{num_beta_updates}_beta_length_{beta_opt_length}_param_length_{param_opt_length}_param_updates'
    model_file = os.path.join( output_directory, f'{new_model_file_part}_0.pt' )
    torch.save(obj=model, f=model_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {model_file}')
    for num_saves in range(1,total_num_saves+1):
        model.fit_by_simulation_pmb(target_state_mean=mean_state, target_state_product_mean=mean_state_product, num_updates=param_updates_per_save, steps_per_update=param_opt_length, learning_rate=0.01, verbose=True)
        num_updates = num_saves*param_updates_per_save
        model_file = os.path.join( output_directory, f'{new_model_file_part}_{num_updates}.pt')
        torch.save(obj=model, f=model_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {model_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')