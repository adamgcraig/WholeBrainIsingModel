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
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the expected values of their observables. Compare to expected values from data. Do an Euler step update. Repeat.")
    parser.add_argument("-a", "--data_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--data_file_name_part", type=str, default='all_quantile_0.5', help="part of the output file name between mean_state_ or mean_state_product_ and .pt")
    parser.add_argument("-d", "--model_file_fragment", type=str, default='all_quantile_0.5_medium_init_uncentered_reps_10_steps_1200_beta_updates_31_lr_0.01_param_updates_3000', help="additional identifier to append to output file names to distinguish different runs with the same arguments")
    parser.add_argument("-f", "--sim_length", type=int, default=120000, help="number of simulation steps between updates")
    parser.add_argument("-g", "--combine_scans", action='store_true', default=False, help="Set this flag in order to take the mean over scans of the target mean state and state product. Otherwise, we just flatten the scan (0) and subject (1) dimensions together.")
    parser.add_argument("-i", "--reset_params", action='store_true', default=False, help="Set this flag in order to do a simulation with h and J reset to the mean states and mean state products, respectively.")
    parser.add_argument("-j", "--zero_h", action='store_true', default=False, help="Set this flag to zero out the h values.")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    data_file_name_part = args.data_file_name_part
    print(f'data_file_name_part={data_file_name_part}')
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

    print('loading data time series state and state product means')
    target_state_mean_file = os.path.join(data_directory, f'mean_state_{data_file_name_part}.pt')
    target_state_mean = torch.load(target_state_mean_file)
    # On load, the dimensions of target_state_mean should be scan x subject x node or scan x subject x node-pair.
    print( f'time {time.time()-code_start_time:.3f}, loaded target_state_mean with size', target_state_mean.size() )
    if combine_scans:
        target_state_mean = torch.mean(target_state_mean, dim=0, keepdim=False)
    else:
        target_state_mean = torch.flatten(target_state_mean, start_dim=0, end_dim=1)
    target_state_mean = target_state_mean.unsqueeze(dim=0)
    print( f'time {time.time()-code_start_time:.3f}, flattened scan and subject dimensions', target_state_mean.size() )
    target_state_product_mean_file = os.path.join(data_directory, f'mean_state_product_{data_file_name_part}.pt')
    target_state_product_mean = torch.load(target_state_product_mean_file)
    # On load, the dimensions of target_state_product_mean should be scan x subject x node x node or scan x subject x node-pair.
    print( f'time {time.time()-code_start_time:.3f}, loaded target_state_product_mean with size', target_state_product_mean.size() )
    if len( target_state_product_mean.size() ) < 4:
        target_state_product_mean = isingmodellight.triu_to_square_pairs( triu_pairs=target_state_product_mean, diag_fill=0 )
    if combine_scans:
        target_state_product_mean = torch.mean(target_state_product_mean, dim=0, keepdim=False)
    else:
        target_state_product_mean = torch.flatten(target_state_product_mean, start_dim=0, end_dim=1)
    target_state_product_mean = target_state_product_mean.unsqueeze(dim=0)
    print( f'time {time.time()-code_start_time:.3f}, converted to square format and flattened scan and subject dimensions', target_state_product_mean.size() )
    model_file = os.path.join(data_directory, f'ising_model_light_{model_file_fragment}.pt')
    model = torch.load(f=model_file)
    if reset_params:
        model.h[:,:,:] = target_state_mean
        model.J[:,:,:,:] = target_state_product_mean
        reset_str = '_reset'
    else:
        reset_str = ''
    if zero_h:
        model.h.zero_()
        zero_h_str = '_no_h'
    else:
        zero_h_str = ''
    print( f'time {time.time()-code_start_time:.3f}, loaded {model_file}' )
    avalanche_counts = model.simulate_and_record_avalanche_counts(num_steps=sim_length)
    sim_file_fragment = f'{model_file_fragment}{reset_str}{zero_h_str}_test_length_{sim_length}'
    print( 'avalanche_counts size', avalanche_counts.size() )
    print( f'time {time.time()-code_start_time:.3f}, done simulating {sim_length} steps' )
    avalanche_counts_file = os.path.join(output_directory, f'avalanche_counts_{sim_file_fragment}.pt')
    torch.save(obj=avalanche_counts, f=avalanche_counts_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {avalanche_counts_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')