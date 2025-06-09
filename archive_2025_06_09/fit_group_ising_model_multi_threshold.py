import os
import torch
import time
import argparse
import hcpdatautilsnopandas as hcp
import isingmodellight
from isingmodellight import IsingModelLight

def get_binarized_info(data_ts:torch.Tensor, threshold:torch.Tensor, data_cov:torch.Tensor, data_fc:torch.Tensor):
    binarized_ts = 2*(data_ts > threshold).float() - 1
    flip_rate = (binarized_ts[:,:,:,1:] != binarized_ts[:,:,:,:-1]).to(float_type).mean(dim=-1)
    print( 'flip_rate dtype', flip_rate.dtype )
    state_mean = binarized_ts.mean(dim=-1)
    state_product_mean = torch.matmul( binarized_ts, binarized_ts.transpose(dim0=-2, dim1=-1) )/binarized_ts.size(dim=-1)
    cov_binary = isingmodellight.get_cov(state_mean=state_mean, state_product_mean=state_product_mean)
    cov_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=cov_binary, mat2=data_cov)
    fc_binary = isingmodellight.get_fc_binary(state_mean=state_mean, state_product_mean=state_product_mean)
    fc_corr = isingmodellight.get_pairwise_correlation_ut(mat1=fc_binary, mat2=data_fc, epsilon=0)
    return flip_rate, cov_rmse, fc_corr, state_mean.mean( dim=(0,1) ), state_product_mean.mean( dim=(0,1) )

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float32
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Test the effect of binarization threshold on our ability to fit a group Ising model.")
    parser.add_argument("-a", "--input_directory", type=str, default='E:\\HCP_data', help="containing folder of fMRI_ts_binaries/ts_[6-digit subject ID]_[1|2]_[RL|LR].bin")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the mean state and mean state product PyTorch pickle files")
    parser.add_argument("-d", "--data_file_name", type=str, default='data_ts_all_as_is.py', help="name of file with data time series")
    parser.add_argument("-c", "--training_subject_start", type=int, default=0, help="first training subject index")
    parser.add_argument("-e", "--training_subject_end", type=int, default=670, help="1 past last training subject index")
    parser.add_argument("-f", "--num_thresholds", type=int, default=1000, help="number of binarization thresholds in std. dev.s above the mean to try")
    parser.add_argument("-g", "--min_threshold", type=float, default=-4, help="minimum threshold in std. dev.s")
    parser.add_argument("-i", "--max_threshold", type=float, default=4, help="minimum threshold in std. dev.s")
    parser.add_argument("-j", "--models_per_threshold", type=int, default=5, help="number of instances of the group Ising model to train for each threshold")
    parser.add_argument("-k", "--sim_length", type=int, default=1200, help="number of simulation steps between updates")
    parser.add_argument("-l", "--num_updates_beta", type=int, default=1000000, help="maximum number of updates within which to find the optimal inverse temperature beta (We stop if we find it to within machine precision.)")
    parser.add_argument("-m", "--updates_per_save", type=int, default=1000, help="number of fitting updates of individual parameters between re-optimizations of beta (In practice, these never perfectly converge, so we do not set any stopping criterion.)")
    parser.add_argument("-n", "--num_saves", type=int, default=1000, help="number of times we save a model after doing updates_per_save parameter updates")
    parser.add_argument("-o", "--learning_rate", type=float, default=0.01, help="amount by which to multiply updates to the model parameters during the Euler step")
    parser.add_argument("-p", "--min_beta", type=float, default=1e-10, help="low end of initial beta search interval")
    parser.add_argument("-q", "--max_beta", type=float, default=1.0, help="high end of initial beta search interval")
    parser.add_argument("-r", "--randomize_uniform", action='store_true', default=False, help="Set this flag to start from random parameters uniformly distributed between -1 and +1.")
    parser.add_argument("-s", "--randomize_normal", action='store_true', default=False, help="Set this flag to start from random parameters normally distributed with mean and variance based on the data.")
    parser.add_argument("-t", "--log_init", action='store_true', default=False, help="Set this flag to start from h_i=-log( 1/mean(i) - 1 ) and J_ij = log( 1 + cov(i,j) ).")
    args = parser.parse_args()
    print('getting arguments...')
    input_directory = args.input_directory
    print(f'input_directory={input_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    data_file_name = args.data_file_name
    print(f'data_file_name={data_file_name}')
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
    models_per_threshold = args.models_per_threshold
    print(f'models_per_threshold={models_per_threshold}')
    sim_length = args.sim_length
    print(f'sim_length={sim_length}')
    num_updates_beta = args.num_updates_beta
    print(f'num_updates_beta={num_updates_beta}')
    updates_per_save = args.updates_per_save
    print(f'updates_per_save={updates_per_save}')
    num_saves = args.num_saves
    print(f'num_saves={num_saves}')
    learning_rate = args.learning_rate
    print(f'learning_rate={learning_rate}')
    min_beta = args.min_beta
    print(f'min_beta={min_beta}')
    max_beta = args.max_beta
    print(f'min_beta={max_beta}')
    randomize_uniform = args.randomize_uniform
    print(f'randomize_uniform={randomize_uniform}')
    randomize_normal = args.randomize_normal
    print(f'randomize_normal={randomize_normal}')
    log_init = args.log_init
    print(f'log_init={log_init}')

    data_ts_file = os.path.join(output_directory, data_file_name)
    data_ts = torch.load(data_ts_file)[:,training_subject_start:training_subject_end,:,:].to(float_type)
    # float_type = data_ts.dtype
    print( 'data_ts dtype', float_type )
    num_ts, num_subjects, num_nodes, num_steps = data_ts.size()
    print(f'time {time.time()-code_start_time:.3f}, computing unbinarized means and flip rates...')
    # data_ts /= data_ts.abs().max(dim=-1).values# Divide the whole time series by the largest absolute value so that the FC stays the same but the means and mean products are <= 1.
    target_state_mean = data_ts.mean(dim=-1, keepdim=False)
    print( f'state mean', target_state_mean.size() )
    state_mean_product = torch.matmul( data_ts, data_ts.transpose(dim0=-2, dim1=-1) )/data_ts.size(dim=-1)
    print( f'state mean', state_mean_product.size() )
    data_cov = isingmodellight.get_cov(state_mean=target_state_mean, state_product_mean=state_mean_product)
    data_fc = isingmodellight.get_fc(state_mean=target_state_mean, state_product_mean=state_mean_product, epsilon=0)
    data_ts_std, data_ts_mean = torch.std_mean(data_ts, dim=-1, keepdim=True)
    threshold_choices = torch.linspace(start=min_threshold, end=max_threshold, steps=num_thresholds, dtype=float_type, device=device)
    num_choices = threshold_choices.numel()
    quantiles = torch.tensor(data=[0.025, 0.5, 0.975], dtype=float_type, device=device)
    num_quantiles = len(quantiles)
    flip_rate_quantiles = torch.zeros( size=(num_choices, num_quantiles), dtype=float_type, device=device )
    cov_rmse_quantiles = torch.zeros( size=(num_choices, num_quantiles), dtype=float_type, device=device )
    fc_corr_quantiles = torch.zeros( size=(num_choices, num_quantiles), dtype=float_type, device=device )
    target_state_mean = torch.zeros( size=(num_choices, num_nodes), dtype=float_type, device=data_ts.device )
    target_state_product_mean = torch.zeros( size=(num_choices, num_nodes, num_nodes), dtype=float_type, device=data_ts.device )
    target_cov = isingmodellight.get_cov(state_mean=target_state_mean, state_product_mean=target_state_product_mean)
    for choice_index in range(num_choices):
        choice = threshold_choices[choice_index]
        print(f'threshold {choice_index+1} of {num_choices}: {choice:.3g}')
        threshold = data_ts_mean + choice*data_ts_std
        flip_rate, cov_rmse, fc_corr, target_state_mean[choice_index,:], target_state_product_mean[choice_index,:,:] = get_binarized_info(data_ts=data_ts, threshold=threshold, data_cov=data_cov, data_fc=data_fc)
        flip_rate_quantiles[choice_index,:] = torch.quantile(input=flip_rate, q=quantiles)
        cov_rmse_quantiles[choice_index,:] = torch.quantile(input=cov_rmse, q=quantiles)
        fc_corr_quantiles[choice_index,:] = torch.quantile(input=fc_corr, q=quantiles)
    summary_file = os.path.join(output_directory, f'data_flip_rate_and_fc_rmse_ci_thresholds_{num_choices}_min_{min_threshold:.3g}_max_{max_threshold:.3g}.pt')
    torch.save( obj=(threshold_choices, flip_rate_quantiles, cov_rmse_quantiles, fc_corr_quantiles), f=summary_file )
    print(f'time {time.time()-code_start_time:.3f}, saved {summary_file}')
    
    print('initializing Ising model...')
    beta = isingmodellight.get_linspace_beta(models_per_subject=models_per_threshold, num_subjects=num_thresholds, dtype=float_type, device=device)
    s = isingmodellight.get_neg_state(models_per_subject=models_per_threshold, num_subjects=num_thresholds, num_nodes=num_nodes, dtype=float_type, device=device)
    if randomize_uniform:
        h = torch.rand_like(input=s)
        J = 2.0*torch.rand( size=(models_per_threshold, num_thresholds, num_nodes, num_nodes), dtype=s.dtype, device=s.device ) - 1.0
        init_mode = 'uniform'
    elif randomize_normal:
        h_std, h_mean = torch.std_mean(input=target_state_mean, dim=-1, keepdim=True)
        h = h_mean.unsqueeze(dim=0) + h_std.unsqueeze(dim=0) * torch.randn_like(input=s)
        J_std, J_mean = torch.std_mean( input=target_state_product_mean, dim=(-2, -1), keepdim=True )
        J = J_mean.unsqueeze(dim=0) + J_std.unsqueeze(dim=0) * torch.randn( size=(models_per_threshold, num_thresholds, num_nodes, num_nodes), dtype=s.dtype, device=s.device )
        init_mode = 'normal'
    elif log_init:
        target_outer_product_mean = target_state_mean.unsqueeze(dim=-1) * target_state_mean.unsqueeze(dim=-2)
        h = ( -1 * (1/target_state_mean - 1).log() ).unsqueeze(dim=0).repeat( (models_per_threshold,1,1) )
        J = ( (target_state_product_mean-target_outer_product_mean)/target_outer_product_mean + 1 ).log().unsqueeze(dim=0).repeat( (models_per_threshold, 1, 1, 1) )
        init_mode = 'log_means'
    else:
        h = target_state_mean.unsqueeze(dim=0).repeat( (models_per_threshold,1,1) )
        J = target_state_product_mean.unsqueeze(dim=0).repeat( (models_per_threshold, 1, 1, 1) )
        init_mode = 'means'
    # Make sure J is symmetric.
    J = ( J + J.transpose(dim0=-2, dim1=-1) )/2.0
    # Make sure the diagonal is 0.
    J -= torch.diag_embed( input=torch.diagonal(input=J, offset=0, dim1=-2, dim2=-1), offset=0, dim1=-2, dim2=-1 )
    model = IsingModelLight(beta=beta, J=J, h=h, s=s)
    print( f'time {time.time()-code_start_time:.3f}, initialized model with h of size', model.h.size(), ' J of size', model.J.size(), 'and state of size', model.s.size() )
    
    print('optimizing beta...')
    num_beta_updates_completed = model.optimize_beta_pmb( target_cov=isingmodellight.get_cov(state_mean=target_state_mean, state_product_mean=target_state_product_mean), num_updates=num_updates_beta, num_steps=sim_length, verbose=True, min_beta=min_beta, max_beta=max_beta )
    print( f'time {time.time()-code_start_time:.3f}, done optimizing beta after {num_beta_updates_completed} iterations' )
    model_file_fragment = f'ising_model_light_group_init_{init_mode}_thresholds_{num_thresholds}_min_{min_threshold:.3g}_max_{max_threshold:.3g}_betas_{models_per_threshold}_min_{min_beta:.3g}_max_{max_beta:.3g}_steps_{sim_length}_lr_{learning_rate:.3g}_beta_updates_{num_beta_updates_completed}'
    model_file = os.path.join(output_directory, f'{model_file_fragment}.pt')
    torch.save(obj=model, f=model_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {model_file}')
    print('fitting parameters h and J...')
    num_param_updates_total = 0
    for save_index in range(num_saves):
        model.fit_by_simulation_pmb(target_state_mean=target_state_mean, target_state_product_mean=target_state_product_mean, num_updates=updates_per_save, steps_per_update=sim_length, learning_rate=learning_rate, verbose=True)
        num_param_updates_total += updates_per_save
        model_file = os.path.join(output_directory, f'{model_file_fragment}_param_updates_{num_param_updates_total}.pt')
        torch.save(obj=model, f=model_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {model_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')
