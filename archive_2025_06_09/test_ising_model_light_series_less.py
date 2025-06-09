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
    parser.add_argument("-a", "--data_directory", type=str, default='D:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--data_file_name_part", type=str, default='all_mean_std_1', help="part of the output file name between mean_state_ or mean_state_product_ and .pt")
    parser.add_argument("-d", "--model_file_fragment", type=str, default='ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_to_thresh_1_reps_5_subj_837_individual_updates', help="the part of the Ising model file between ising_model_ and _[the number of parameter update steps].pt.")
    parser.add_argument("-f", "--sim_length", type=int, default=120000, help="number of simulation steps between updates")
    parser.add_argument("-m", "--update_increment", type=int, default=1000, help="number of updates between models to test")
    parser.add_argument("-k", "--min_updates", type=int, default=0, help="first number of updates to test")
    parser.add_argument("-l", "--max_updates", type=int, default=65000, help="last number of updates to test")
    parser.add_argument("-g", "--combine_scans", action='store_true', default=True, help="Set this flag in order to take the mean over scans of the target mean state and state product. Otherwise, we just flatten the scan (0) and subject (1) dimensions together.")
    parser.add_argument("-i", "--reset_params", action='store_true', default=False, help="Set this flag in order to do a simulation with h and J reset to the mean states and mean state products, respectively.")
    parser.add_argument("-j", "--zero_h", action='store_true', default=False, help="Set this flag to zero out the h values.")
    parser.add_argument("-n", "--training_subject_start", type=int, default=0, help="first training subject index")
    parser.add_argument("-e", "--training_subject_end", type=int, default=837, help="1 past last training subject index")
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
    update_increment = args.update_increment
    print(f'update_increment={update_increment}')
    min_updates = args.min_updates
    print(f'min_updates={min_updates}')
    max_updates = args.max_updates
    print(f'max_updates={max_updates}')
    combine_scans = args.combine_scans
    print(f'combine_scans={combine_scans}')
    reset_params = args.reset_params
    print(f'reset_params={reset_params}')
    zero_h = args.zero_h
    print(f'zero_h={zero_h}')
    training_subject_start = args.training_subject_start
    print(f'training_subject_start={training_subject_start}')
    training_subject_end = args.training_subject_end
    print(f'training_subject_end={training_subject_end}')

    epsilon = 0.0

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
        print(f'time {time.time()-code_start_time:.3f}, saved {m_file}, size', m.size(), f'NaNs {count_nans(m=m)}, min {m.min():.3g}, mean {m.mean():.3g}, max {m.max():.3g}')
        return 0

    def load_target_fc(data_file_name_part:str):
        print('loading data time series state and state product means')
        target_state_mean_file = os.path.join(data_directory, f'mean_state_{data_file_name_part}.pt')
        target_state_mean = torch.load(target_state_mean_file, weights_only=False)
        # On load, the dimensions of target_state_mean should be subject x node or scan x subject x node.
        print( f'time {time.time()-code_start_time:.3f}, loaded target_state_mean with size', target_state_mean.size() )
        target_state_product_mean_file = os.path.join(data_directory, f'mean_state_product_{data_file_name_part}.pt')
        target_state_product_mean = torch.load(target_state_product_mean_file, weights_only=False)
        # On load, the dimensions of target_state_product_mean should be subject x node-pair, subject x node x node, scan x subject x node-pair, or scan x subject x node x node.
        print( f'time {time.time()-code_start_time:.3f}, loaded target_state_product_mean with size', target_state_product_mean.size() )
        # Assume that either both target Tensors have a scan dimension, or neither does.
        # If they have a scan dimension, then first remove it, either by averaging over scans or flattening together the subject and scan dimensions.
        target_state_mean_size = target_state_mean.size()
        # target_state_product_mean_size = target_state_product_mean.size()
        num_batch_dims = len(target_state_mean_size) - 1
        if num_batch_dims > 1:
            if combine_scans:
                extra_dim_range = tuple( range(num_batch_dims-1) )# average over all extra batch dimensions
                print( 'averaging over extra batch dimensions', extra_dim_range )
                target_state_mean = torch.mean(target_state_mean, dim=extra_dim_range, keepdim=False)
                target_state_product_mean = torch.mean(target_state_product_mean, dim=extra_dim_range, keepdim=False)
            else:
                print('flattening extra batch dimensions')
                target_state_mean = torch.flatten(target_state_mean, start_dim=0, end_dim=-2)
                target_state_product_mean = torch.flatten(target_state_product_mean, start_dim=0, end_dim=-3)
        # Regardless of whether the data originally had a scan dimension, we add in a singleton model replica dimension so that we can broadcast with h and J.
        print('prepending singleton model replica dimension')
        target_state_mean = target_state_mean.unsqueeze(dim=0)
        target_state_product_mean = target_state_product_mean.unsqueeze(dim=0)
        print( f'time {time.time()-code_start_time:.3f}, target_state_mean size', target_state_mean.size() )
        print( f'time {time.time()-code_start_time:.3f}, target_state_product_mean size', target_state_product_mean.size() )
        target_fc = get_fc_in_place(mean_state=target_state_mean, mean_state_product=target_state_product_mean)
        print( f'time {time.time()-code_start_time:.3f}, target_fc size', target_fc.size(), f'NaNs {count_nans(target_fc)}, min {target_fc.min():.3g}, mean {target_fc.mean():.3g}, max {target_fc.max():.3g}' )
        return target_fc

    def test_model(model_file_fragment_with_updates:str, target_fc:torch.Tensor):
        model_file = os.path.join(data_directory, f'{model_file_fragment_with_updates}.pt')
        model = torch.load(f=model_file, weights_only=False)
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file}' )
        print('running simulation...')
        sim_state_mean, sim_state_product_mean = model.simulate_and_record_means_pmb(num_steps=sim_length)
        sim_file_fragment = f'{model_file_fragment_with_updates}_test_length_{sim_length}'
        print( f'time {time.time()-code_start_time:.3f}, done simulating {sim_length} steps' )
        save_and_print(m=sim_state_mean, m_file_part=f'sim_state_mean_{sim_file_fragment}')
        save_and_print(m=sim_state_product_mean, m_file_part=f'sim_state_product_mean_{sim_file_fragment}')
        sim_fc = get_fc_in_place(mean_state=sim_state_mean, mean_state_product=sim_state_product_mean)
        fc_corr = isingmodellight.get_pairwise_correlation_ut(mat1=sim_fc, mat2=target_fc, epsilon=epsilon)
        save_and_print(m=fc_corr, m_file_part=f'fc_corr_{sim_file_fragment}')
        return fc_corr
    
    def quantiles_and_min_max(input:torch.Tensor, quantiles:torch.Tensor):
        input = input[:,training_subject_start:training_subject_end]
        return torch.cat(  ( torch.quantile(input=input, q=quantiles), torch.min(input).unsqueeze(dim=0), torch.max(input).unsqueeze(dim=0) ), dim=0  )
    
    target_fc = load_target_fc(data_file_name_part=data_file_name_part)
    update_counts = torch.arange(start=min_updates, end=max_updates+1, step=update_increment, dtype=int_type, device=device)
    num_model_files = update_counts.numel()
    quantiles = torch.tensor(data=[0.5, 0.025, 0.975], dtype=float_type, device=device)
    num_quants = len(quantiles)
    fc_corr_summary = torch.zeros(  size=( num_model_files, num_quants+2 ), dtype=float_type, device=device  )
    for model_file_index in range(num_model_files):
        num_updates = update_counts[model_file_index]
        print(f'model file {model_file_index+1} of {num_model_files}, update count {num_updates}')
        model_file_fragment_with_updates = f'{model_file_fragment}_{num_updates}'
        fc_corr = test_model(model_file_fragment_with_updates=model_file_fragment_with_updates, target_fc=target_fc)
        fc_corr_summary[model_file_index,:] = quantiles_and_min_max(input=fc_corr, quantiles=quantiles)
    save_and_print(m=fc_corr_summary, m_file_part=f'fc_corr_summary_{model_file_fragment}_updates_min_{min_updates}_max_{max_updates}_increment_{update_increment}')
print(f'time {time.time()-code_start_time:.3f}, done')