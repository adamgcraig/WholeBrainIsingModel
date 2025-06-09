import os
import torch
import time
import argparse
import isingmodellight
from isingmodellight import IsingModelLight

def get_binarized_info(data_ts:torch.Tensor, threshold_z:float):
    data_ts_std, data_ts_mean = torch.std_mean(data_ts, dim=-1, keepdim=True)
    binarized_ts = 2*( data_ts > (data_ts_mean + threshold_z*data_ts_std) ).float() - 1
    return binarized_ts.mean(dim=-1).mean(dim=0), torch.matmul( binarized_ts, binarized_ts.transpose(dim0=-2, dim1=-1) ).mean(dim=0)/binarized_ts.size(dim=-1)

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Test the effect of binarization threshold on our ability to fit a group Ising model.")
    parser.add_argument("-a", "--input_directory", type=str, default='E:\\HCP_data', help="containing folder of fMRI_ts_binaries/ts_[6-digit subject ID]_[1|2]_[RL|LR].bin")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the mean state and mean state product PyTorch pickle files")
    parser.add_argument("-c", "--target_data_file_fragment", type=str, default='all_aal_mean_std_1', help="part of the file name of the fitting target files after mean_state_ or mean_state_product_ .pt")
    parser.add_argument("-d", "--group_model_file_fragment", type=str, default='ising_model_light_group_init_means_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_1_steps_1200_lr_0.01_beta_updates_8_v2_param_updates_40000_to_thresh_1_reps_5_subj_837', help="part of the file name of the group model after ising_model_ and before .pt")
    parser.add_argument("-e", "--param_sim_length", type=int, default=1200, help="number of simulation steps between parameter updates")
    parser.add_argument("-f", "--updates_per_save", type=int, default=1000, help="number of fitting updates of individual parameters between re-optimizations of beta (In practice, these never perfectly converge, so we do not set any stopping criterion.)")
    parser.add_argument("-g", "--num_saves", type=int, default=1000, help="number of times we save a model after doing updates_per_save parameter updates")
    parser.add_argument("-i", "--learning_rate", type=float, default=0.01, help="amount by which to multiply updates to the model parameters during the Euler step")
    parser.add_argument("-j", "--separate_scans", action='store_true', default=False, help="Set this flag to make a separate fitting target for each scan instead of averaging over them.")
    args = parser.parse_args()
    print('getting arguments...')
    input_directory = args.input_directory
    print(f'input_directory={input_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    target_data_file_fragment = args.target_data_file_fragment
    print(f'target_data_file_fragment={target_data_file_fragment}')
    group_model_file_fragment = args.group_model_file_fragment
    print(f'group_model_file_fragment={group_model_file_fragment}')
    param_sim_length = args.param_sim_length
    print(f'param_sim_length={param_sim_length}')
    updates_per_save = args.updates_per_save
    print(f'updates_per_save={updates_per_save}')
    num_saves = args.num_saves
    print(f'num_saves={num_saves}')
    learning_rate = args.learning_rate
    print(f'learning_rate={learning_rate}')
    separate_scans = args.separate_scans
    print(f'separate_scans={separate_scans}')

    def load_target(target_prefix:str):
        target_file = os.path.join(input_directory, f'{target_prefix}_{target_data_file_fragment}.pt')
        target = torch.load(f=target_file, weights_only=False)
        if separate_scans:
            target = torch.flatten(input=target, start_dim=0, end_dim=1)
        else:
            target = torch.mean(input=target, dim=0, keepdim=False)
        print( f'time {time.time()-code_start_time:.3f}, loaded {target_file}, converted to size', target.size() )
        return target
    
    target_state_mean = load_target(target_prefix='mean_state')
    target_state_product_mean = load_target(target_prefix='mean_state_product')
    
    print('loading Ising model...')
    group_model_file = os.path.join( output_directory, f'{group_model_file_fragment}.pt' )
    model = torch.load(f=group_model_file)
    print( f'time {time.time()-code_start_time:.3f}, loaded group model from {group_model_file} with h of size', model.h.size(), ' J of size', model.J.size(), 'and state of size', model.s.size() )
    
    print('fitting parameters h and J to individual data...')
    num_param_updates_total = 0
    for save_index in range(num_saves):
        model.fit_by_simulation_pmb(target_state_mean=target_state_mean, target_state_product_mean=target_state_product_mean, num_updates=updates_per_save, steps_per_update=param_sim_length, learning_rate=learning_rate, verbose=True)
        num_param_updates_total += updates_per_save
        model_file = os.path.join(output_directory, f'{group_model_file_fragment}_individual_updates_{num_param_updates_total}.pt')
        torch.save(obj=model, f=model_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {model_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')
