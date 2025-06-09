import os
import torch
import time
import argparse
import isingmodellight
from isingmodellight import IsingModelLight

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the expected values of their observables. Compare to expected values from data. Do an Euler step update. Repeat.")
    parser.add_argument("-a", "--data_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--data_file_name_part", type=str, default='all_aal_mean_std_1', help="part of the data mean state and state product file names between mean_state_ or mean_state_product_ and .pt")
    parser.add_argument("-d", "--init_params_file_name_part", type=str, default='all_aal_mean_std_1', help="similar to data_file_name_part but used along with reset_params to select the data mean state and state product files to use as parameter values .pt")
    parser.add_argument("-e", "--model_file_fragment", type=str, default='all_aal_mean_std_1_medium_init_uncentered_reps_5_beta_min_1e-09_max_0.03_steps_12000_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_64_popt_steps_10000', help="additional identifier to append to output file names to distinguish different runs with the same arguments")
    parser.add_argument("-f", "--sim_length", type=int, default=120000, help="number of simulation steps between updates")
    parser.add_argument("-g", "--combine_scans", action='store_true', default=False, help="Set this flag in order to take the mean over scans of the target mean state and state product. Otherwise, we just flatten the scan (0) and subject (1) dimensions together.")
    parser.add_argument("-i", "--reset_params", action='store_true', default=False, help="Set this flag in order to do a simulation with h and J reset to the mean states and mean state products, respectively.")
    parser.add_argument("-j", "--zero_h", action='store_true', default=False, help="Set this flag to zero out the h values.")
    parser.add_argument("-k", "--area_mean_h", action='store_true', default=False, help="Set this flag to replace h with the mean over areas for the individual.")
    parser.add_argument("-l", "--group_mean_h", action='store_true', default=False, help="Set this flag to replace h with the mean over individuals for the area.")
    parser.add_argument("-m", "--training_subject_start", type=int, default=0, help="first training subject index")
    parser.add_argument("-n", "--training_subject_end", type=int, default=670, help="1 past last training subject index")
    parser.add_argument("-o", "--device", type=str, default='cuda', help="string to pass to torch.device(device)")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    data_file_name_part = args.data_file_name_part
    print(f'data_file_name_part={data_file_name_part}')
    init_params_file_name_part = args.init_params_file_name_part
    print(f'init_params_file_name_part={init_params_file_name_part}')
    model_file_fragment = args.model_file_fragment
    print(f'model_file_fragment={model_file_fragment}')
    sim_length = args.sim_length
    print(f'sim_length={sim_length}')
    combine_scans = args.combine_scans
    print(f'combine_scans={combine_scans}')
    reset_params = args.reset_params
    print(f'reset_params={reset_params}')
    zero_h = args.zero_h
    print(f'zero_h={zero_h}')
    area_mean_h = args.area_mean_h
    print(f'area_mean_h={area_mean_h}')
    group_mean_h = args.group_mean_h
    print(f'group_mean_h={group_mean_h}')
    training_subject_start = args.training_subject_start
    print(f'training_subject_start={training_subject_start}')
    training_subject_end = args.training_subject_end
    print(f'training_subject_end={training_subject_end}')
    device_str = args.device
    print(f'device={device_str}')
    device = torch.device(device_str)

    def load_data_means(data_file_name_part:str):
        print('loading data time series state and state product means')
        state_mean_file = os.path.join(data_directory, f'mean_state_{data_file_name_part}.pt')
        state_mean = torch.load(state_mean_file, weights_only=False)
        # On load, the dimensions of target_state_mean should be subject x node or scan x subject x node.
        print( f'time {time.time()-code_start_time:.3f}, loaded state_mean with size', state_mean.size() )
        state_product_mean_file = os.path.join(data_directory, f'mean_state_product_{data_file_name_part}.pt')
        state_product_mean = torch.load(state_product_mean_file, weights_only=False)
        # On load, the dimensions of target_state_product_mean should be subject x node-pair, subject x node x node, scan x subject x node-pair, or scan x subject x node x node.
        print( f'time {time.time()-code_start_time:.3f}, loaded target_state_product_mean with size', state_product_mean.size() )
        # Assume that either both target Tensors have a scan dimension, or neither does.
        # If they have a scan dimension, then first remove it, either by averaging over scans or flattening together the subject and scan dimensions.
        state_mean_size = state_mean.size()
        # target_state_product_mean_size = target_state_product_mean.size()
        num_batch_dims = len(state_mean_size) - 1
        if num_batch_dims > 1:
            if combine_scans:
                extra_dim_range = tuple( range(num_batch_dims-1) )# average over all extra batch dimensions
                print( 'averaging over extra batch dimensions', extra_dim_range )
                state_mean = torch.mean(state_mean, dim=extra_dim_range, keepdim=False)
                state_product_mean = torch.mean(state_product_mean, dim=extra_dim_range, keepdim=False)
            else:
                print('flattening extra batch dimensions')
                state_mean = torch.flatten(state_mean, start_dim=0, end_dim=-2)
                state_product_mean = torch.flatten(state_product_mean, start_dim=0, end_dim=-3)
        # Regardless of whether the data originally had a scan dimension, we add in a singleton model replica dimension so that we can broadcast with h and J.
        print('prepending singleton model replica dimension')
        state_mean = state_mean.unsqueeze(dim=0)
        state_product_mean = state_product_mean.unsqueeze(dim=0)
        # This was getting too complicated.
        # Just assume we are reading in batches of square matrices.
        # # We want to work with the mean state products as square matrices, not upper triangular part vectors.
        # if len( state_product_mean.size() ) < 4:
        #     print('converting mean state products from upper triangular parts to square matrices')
        #     state_product_mean = isingmodellight.triu_to_square_pairs(triu_pairs=state_product_mean, diag_fill=0)
        return state_mean, state_product_mean
    
    print('loading target data time series state and state product means')
    target_state_mean, target_state_product_mean = load_data_means(data_file_name_part=data_file_name_part)
    print( f'time {time.time()-code_start_time:.3f}, target_state_mean size', target_state_mean.size() )
    print( f'time {time.time()-code_start_time:.3f}, target_state_product_mean size', target_state_product_mean.size() )
    model_file = os.path.join(data_directory, f'ising_model_light_{model_file_fragment}.pt')
    model = torch.load(f=model_file, weights_only=False)
    if reset_params:
        print('loading initial parameter state and state product means')
        param_state_mean, param_state_product_mean = load_data_means(data_file_name_part=init_params_file_name_part)
        print( f'time {time.time()-code_start_time:.3f}, param_state_mean size', param_state_mean.size() )
        print( f'time {time.time()-code_start_time:.3f}, param_state_product_mean size', param_state_product_mean.size() )
        model.h[:,:,:] = param_state_mean
        model.J[:,:,:,:] = param_state_product_mean
        reset_str = '_reset'
    else:
        reset_str = ''
    if group_mean_h or area_mean_h:
        mean_str = '_h_mean'
    else:
        mean_str = ''
    if group_mean_h:
        num_reps = model.h.size(dim=0)
        num_subjects = model.h.size(dim=1)
        model.h = model.h[:,training_subject_start:training_subject_end,:].mean( dim=(0,1), keepdim = True ).repeat( (num_reps, num_subjects, 1) )
        mean_str += '_group'
    if area_mean_h:
        num_nodes = model.h.size(dim=2)
        model.h = model.h.mean(dim=2, keepdim = True).repeat( (1, 1, num_nodes) )
        mean_str += '_area'
    if zero_h:
        model.h.zero_()
        zero_h_str = '_no_h'
    else:
        zero_h_str = ''
    print( f'time {time.time()-code_start_time:.3f}, loaded {model_file}, simulating...' )
    sim_state_mean, sim_state_product_mean, flip_rate = model.simulate_and_record_means_and_flip_rate_pmb(num_steps=sim_length)
    sim_file_fragment = f'{model_file_fragment}{reset_str}{mean_str}{zero_h_str}_test_length_{sim_length}'
    print( 'sim_state_mean size', sim_state_mean.size() )
    print( 'sim_state_product_mean size', sim_state_product_mean.size() )
    print( 'flip_rate size', flip_rate.size() )
    print( f'time {time.time()-code_start_time:.3f}, done simulating {sim_length} steps' )
    print(f'flip rate  min {flip_rate.min():.3g}, mean {flip_rate.mean():.3g}, max {flip_rate.max():.3g}')
    flip_rate_file = os.path.join(output_directory, f'flip_rate_{sim_file_fragment}.pt')
    torch.save(obj=flip_rate, f=flip_rate_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {flip_rate_file}')
    print(f'sim state mean min {sim_state_mean.min():.3g}, mean {sim_state_mean.mean():.3g}, max {sim_state_mean.max():.3g}')
    sim_state_mean_file = os.path.join(output_directory, f'sim_state_mean_{sim_file_fragment}.pt')
    torch.save(obj=sim_state_mean, f=sim_state_mean_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {sim_state_mean_file}')
    print(f'sim state product mean min {sim_state_product_mean.min():.3g}, mean {sim_state_product_mean.mean():.3g}, max {sim_state_product_mean.max():.3g}')
    sim_state_product_mean_file = os.path.join(output_directory, f'sim_state_product_mean_{sim_file_fragment}.pt')
    torch.save(obj=sim_state_product_mean, f=sim_state_product_mean_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {sim_state_product_mean_file}')
    state_mean_rmse = isingmodellight.get_pairwise_rmse(mat1=sim_state_mean, mat2=target_state_mean)
    print(f'state mean RMSE  min {state_mean_rmse.min():.3g}, mean {state_mean_rmse.mean():.3g}, max {state_mean_rmse.max():.3g}')
    state_mean_rmse_file = os.path.join(output_directory, f'state_mean_rmse_{sim_file_fragment}.pt')
    torch.save(obj=state_mean_rmse, f=state_mean_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {state_mean_rmse_file}')
    state_product_mean_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_state_product_mean, mat2=target_state_product_mean)
    print(f'state product mean RMSE  min {state_product_mean_rmse.min():.3g}, mean {state_product_mean_rmse.mean():.3g}, max {state_product_mean_rmse.max():.3g}')
    state_product_mean_rmse_file = os.path.join(output_directory, f'state_product_mean_rmse_{sim_file_fragment}.pt')
    torch.save(obj=state_product_mean_rmse, f=state_product_mean_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {state_product_mean_rmse_file}')
    target_cov = target_state_product_mean
    target_cov -= ( target_state_mean.unsqueeze(dim=-2) * target_state_mean.unsqueeze(dim=-1) )
    # target_cov = isingmodellight.get_cov(state_mean=target_state_mean, state_product_mean=target_state_product_mean)
    sim_cov = sim_state_product_mean
    sim_cov -= ( sim_state_mean.unsqueeze(dim=-2) * sim_state_mean.unsqueeze(dim=-1) )
    # sim_cov = isingmodellight.get_cov(state_mean=sim_state_mean, state_product_mean=sim_state_product_mean)
    cov_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_cov, mat2=target_cov)
    print(f'covariance RMSE  min {cov_rmse.min():.3g}, mean {cov_rmse.mean():.3g}, max {cov_rmse.max():.3g}')
    cov_rmse_file = os.path.join(output_directory, f'cov_rmse_{sim_file_fragment}.pt')
    torch.save(obj=cov_rmse, f=cov_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {cov_rmse_file}')
    target_std = torch.sqrt( torch.diagonal(input=target_cov, offset=0, dim1=-2, dim2=-1) )
    target_fc = target_cov
    target_fc /= ( target_std.unsqueeze(dim=-2) * target_std.unsqueeze(dim=-1) )
    # target_fc = isingmodellight.get_fc_binary(state_mean=target_state_mean, state_product_mean=target_state_product_mean)
    sim_std = torch.sqrt( torch.diagonal(input=sim_cov, dim1=-2, dim2=-1) )
    sim_fc = sim_cov
    sim_fc /= ( sim_std.unsqueeze(dim=-2) * sim_std.unsqueeze(dim=-1) )
    # sim_fc = isingmodellight.get_fc_binary(state_mean=sim_state_mean, state_product_mean=sim_state_product_mean)
    fc_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_fc, mat2=target_fc)
    print(f'FC RMSE min {fc_rmse.min():.3g}, mean {fc_rmse.mean():.3g}, max {fc_rmse.max():.3g}')
    fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{sim_file_fragment}.pt')
    torch.save(obj=fc_rmse, f=fc_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {fc_rmse_file}')
    fc_corr = isingmodellight.get_pairwise_correlation_ut(mat1=sim_fc, mat2=target_fc)
    print(f'FC correlation min {fc_corr.min():.3g}, mean {fc_corr.mean():.3g}, max {fc_corr.max():.3g}')
    fc_corr_file = os.path.join(output_directory, f'fc_corr_{sim_file_fragment}.pt')
    torch.save(obj=fc_corr, f=fc_corr_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {fc_corr_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')