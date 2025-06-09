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
    parser.add_argument("-c", "--data_file_name_part", type=str, default='all_as_is', help="part of the output file name between mean_state_ or mean_state_product_ and .pt")
    parser.add_argument("-d", "--output_file_name_part", type=str, default='short', help="additional identifier to append to output file names to distinguish different runs with the same arguments")
    parser.add_argument("-e", "--beta_sim_length", type=int, default=120000, help="number of simulation steps between updates in the beta optimization loop")
    parser.add_argument("-f", "--param_sim_length", type=int, default=1200, help="number of simulation steps between updates in the main parameter-fitting loop")
    parser.add_argument("-g", "--num_updates_beta", type=int, default=1000000, help="number of beta optimization steps performed")
    parser.add_argument("-i", "--updates_per_save", type=int, default=1000, help="number of fitting updates of individual parameters between re-optimizations of beta (In practice, these never perfectly converge, so we do not set any stopping criterion.)")
    parser.add_argument("-j", "--saves_per_beta_opt", type=int, default=1000, help="number of times we save a model to a file")
    parser.add_argument("-k", "--last_saved_popt", type=int, default=1000, help="parameter optimization step number of the saved model")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.01, help="amount by which to multiply updates to the model parameters during the Euler step")
    parser.add_argument("-m", "--combine_scans", action='store_true', default=False, help="Set this flag in order to take the mean over scans of the target mean state and state product. Otherwise, we just flatten the scan (0) and subject (1) dimensions together.")    
    parser.add_argument("-n", "--device", type=str, default='cuda', help="string to pass to torch.device(device)")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    data_file_name_part = args.data_file_name_part
    print(f'data_file_name_part={data_file_name_part}')
    output_file_name_part = args.output_file_name_part
    print(f'output_file_name_part={output_file_name_part}')
    param_sim_length = args.param_sim_length
    print(f'param_sim_length={param_sim_length}')
    updates_per_save = args.updates_per_save
    print(f'updates_per_save={updates_per_save}')
    saves_per_beta_opt = args.saves_per_beta_opt
    print(f'saves_per_beta_opt={saves_per_beta_opt}')
    last_saved_popt = args.last_saved_popt
    print(f'last_saved_popt={last_saved_popt}')
    learning_rate = args.learning_rate
    print(f'learning_rate={learning_rate}')
    combine_scans = args.combine_scans
    print(f'combine_scans={combine_scans}')
    device_str = args.device
    print(f'device={device_str}')
    device = torch.device(device_str)
    
    print('loading data time series state and state product means')
    target_state_mean_file = os.path.join(data_directory, f'mean_state_{data_file_name_part}.pt')
    target_state_mean = torch.load(target_state_mean_file)
    # On load, the dimensions of target_state_mean should be either
    # scan x subject x node (individual data)
    # or
    # 1 x node-pair (group data).
    print( f'time {time.time()-code_start_time:.3f}, loaded target_state_mean with size', target_state_mean.size() )
    if len( target_state_mean.size() ) > 2:
        if combine_scans:
            target_state_mean = torch.mean(target_state_mean, dim=0, keepdim=False)
            print( f'time {time.time()-code_start_time:.3f}, averaged over scan dimension', target_state_mean.size() )
        else:
            target_state_mean = torch.flatten(target_state_mean, start_dim=0, end_dim=1)
            print( f'time {time.time()-code_start_time:.3f}, flattened scan and subject dimensions', target_state_mean.size() )
    target_state_product_mean_file = os.path.join(data_directory, f'mean_state_product_{data_file_name_part}.pt')
    target_state_product_mean = torch.load(target_state_product_mean_file)
    # On load, the dimensions of target_state_product_mean should be either
    # scan x subject x node x node (individual data)
    # or
    # 1 x node x node (group data)
    print( f'time {time.time()-code_start_time:.3f}, loaded target_state_product_mean with size', target_state_product_mean.size() )
    if len( target_state_product_mean.size() ) > 3:
        # target_state_product_mean = isingmodellight.triu_to_square_pairs( triu_pairs=target_state_product_mean, diag_fill=0 )
        # print( f'time {time.time()-code_start_time:.3f}, converted to square format and flattened scan and subject dimensions', target_state_product_mean.size() )
        if combine_scans:
            target_state_product_mean = torch.mean(target_state_product_mean, dim=0, keepdim=False)
            print( f'time {time.time()-code_start_time:.3f}, averaged over scan dimension', target_state_product_mean.size() )
        else:
            target_state_product_mean = torch.flatten(target_state_product_mean, start_dim=0, end_dim=1)
            print( f'time {time.time()-code_start_time:.3f}, flattened scan and subject dimensions', target_state_product_mean.size() )
    
    model_file = os.path.join(output_directory, f'{output_file_name_part}_{last_saved_popt}.pt')
    model = torch.load(f=model_file)
    print(f'time {time.time()-code_start_time:.3f}, loaded {model_file}')
    
    num_updates_params_total = last_saved_popt
    for save_index in range(saves_per_beta_opt):
        print('fitting parameters h and J...')
        model.fit_by_simulation_pmb(target_state_mean=target_state_mean, target_state_product_mean=target_state_product_mean, num_updates=updates_per_save, steps_per_update=param_sim_length, learning_rate=learning_rate, verbose=True)
        num_updates_params_total += updates_per_save
        model_file = os.path.join(output_directory, f'{output_file_name_part}_{num_updates_params_total}.pt')
        torch.save(obj=model, f=model_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {model_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')