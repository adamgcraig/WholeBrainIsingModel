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

    parser = argparse.ArgumentParser(description="Simulate several Ising models and make a Fisher information matrix for one.")
    parser.add_argument("-a", "--data_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--data_file_name_part", type=str, default='all_quantile_0.5', help="part of the output file name between mean_state_ or mean_state_product_ and .pt")
    parser.add_argument("-d", "--model_file_fragment", type=str, default='all_quantile_0.5_medium_init_uncentered_reps_10_steps_1200_beta_updates_31_lr_0.01_param_updates_3000', help="additional identifier to append to output file names to distinguish different runs with the same arguments")
    parser.add_argument("-f", "--sim_length", type=int, default=120000, help="number of simulation steps between updates")
    parser.add_argument("-s", "--subject_index", type=int, default=0, help="index of subject for which to compute the FIM")
    parser.add_argument("-m", "--model_index", type=int, default=0, help="index of model for which to compute the FIM")
    parser.add_argument("-l", "--save_side_length", type=int, default=16245, help="length of individual squares into which we divide the FIM when saving")
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
    subject_index = args.subject_index
    print(f'subject_index={subject_index}')
    model_index = args.model_index
    print(f'model_index={model_index}')
    save_side_length = args.save_side_length
    print(f'save_side_length={save_side_length}')
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

    print(f'simulating and recording FIM for model {model_index} of subject {subject_index}...')
    demo_fim = model.simulate_and_record_one_fim(num_steps=sim_length, model_index=model_index, subject_index=subject_index)
    print( f'time {time.time()-code_start_time:.3f}, done simulating, demo_fim size', demo_fim.size() )
    fim_length = demo_fim.size(dim=-1)
    saves_per_side = fim_length//save_side_length
    save_square = torch.zeros( size=(save_side_length, save_side_length), dtype=demo_fim.dtype, device=demo_fim.device )
    sim_file_fragment = f'{model_file_fragment}{reset_str}{zero_h_str}_test_length_{sim_length}'
    for row_save_index in range(saves_per_side):
        row_start = row_save_index*save_side_length
        row_end = row_start + save_side_length
        for col_save_index in range(saves_per_side):
            col_start = col_save_index*save_side_length
            col_end = col_start + save_side_length
            save_square[:,:] = demo_fim[row_start:row_end,col_start:col_end]
            save_file = os.path.join(output_directory, f'fim_part_{sim_file_fragment}_row_{row_start}_to_{row_end}_col_{col_start}_to_{col_end}.pt')
            torch.save(obj=save_square, f=save_file)
            print(f'time {time.time()-code_start_time:.3f}, saved {save_file}')
    
    print('computing just eigenvalues...')
    eigvals = torch.linalg.eigvalsh(demo_fim)
    print( f'time {time.time()-code_start_time:.3f}, eigvals size', eigvals.size() )
    eigvals_file = os.path.join(output_directory, f'fim_eigvals_{sim_file_fragment}.pt')
    torch.save(obj=eigvals, f=eigvals_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {eigvals_file}')
    
    print('computing eigenvalues along with eigenvectors...')
    eigvals, eigvecs = torch.linalg.eigh(demo_fim)
    print( f'time {time.time()-code_start_time:.3f}, eigvals size', eigvals.size() )
    eigvals_file = os.path.join(output_directory, f'fim_eigvals_2_{sim_file_fragment}.pt')
    torch.save(obj=eigvals, f=eigvals_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {eigvals_file}')
    for row_save_index in range(saves_per_side):
        row_start = row_save_index*save_side_length
        row_end = row_start + save_side_length
        for col_save_index in range(saves_per_side):
            col_start = col_save_index*save_side_length
            col_end = col_start + save_side_length
            save_square[:,:] = eigvecs[row_start:row_end,col_start:col_end]
            save_file = os.path.join(output_directory, f'fim_eigvecs_part_{sim_file_fragment}_row_{row_start}_to_{row_end}_col_{col_start}_to_{col_end}.pt')
            torch.save(obj=save_square, f=save_file)
            print(f'time {time.time()-code_start_time:.3f}, saved {save_file}')
    
    print(f'time {time.time()-code_start_time:.3f}, done')