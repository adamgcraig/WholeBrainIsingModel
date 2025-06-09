import os
import torch
from scipy import stats
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
    epsilon = 0.0

    parser = argparse.ArgumentParser(description="Summarize train-test-partitioned correlations between model parameters and predictions from structure.")
    parser.add_argument("-a", "--data_directory", type=str, default='D:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--file_name_part", type=str, default='ising_model_light_group_threshold_0_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_70_param_updates_10000_individual_updates_10000', help="part of the output file name before .pt")
    parser.add_argument("-e", "--num_node_perms", type=int, default=1000000, help="number of permutations for node correlations")
    parser.add_argument("-f", "--num_edge_perms", type=int, default=1000, help="number of permutations for edge correlations")
    parser.add_argument("-g", "--num_nodes", type=int, default=360, help="number of regions")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    file_name_part = args.file_name_part
    print(f'file_name_part={file_name_part}')
    num_node_perms = args.num_node_perms
    print(f'num_node_perms={num_node_perms}')
    num_edge_perms = args.num_edge_perms
    print(f'num_edge_perms={num_edge_perms}')
    num_nodes = args.num_nodes
    print(f'num_nodes={num_nodes}')
    num_edges = ( num_nodes*(num_nodes-1) )//2
    print(f'num_edges={num_edges}')

    def save_and_print(obj:torch.Tensor, file_name_part:str):
        file_name = os.path.join(output_directory, f'{file_name_part}.pt')
        torch.save(obj=obj, f=file_name)
        num_nan = torch.count_nonzero( torch.isnan(obj) )
        print( f'time {time.time()-code_start_time:.3f}, saved {file_name}, size', obj.size(), f'num NaN {num_nan}, min {obj.min():.3g} mean {obj.mean():.3g} max {obj.max():.3g}' )
        return 0

    def load_and_save_std_mean(corr_file_name_part:str):
        corr_file = os.path.join(output_directory, f'{corr_file_name_part}.pt')
        corr = torch.load(f=corr_file, weights_only=False)
        is_nan = torch.isnan(corr)
        num_nan = torch.count_nonzero(is_nan)
        print( f'time {time.time()-code_start_time:.3f}, loaded {corr_file}, size', corr.size(), f'num NaN {num_nan}, min {corr.min():.3g} mean {corr.mean():.3g} max {corr.max():.3g}' )
        # The correlations should be NaN only when one of the variances is 0.
        # We know that the original dependent values have non-0 variance, so the slope of the regression line must be 0.
        # It is reasonable to fill in 0 for the correlation in this case.
        corr[is_nan] = 0.0
        std_train_corr, mean_train_corr = torch.std_mean(input=corr, dim=-1)
        save_and_print(obj=std_train_corr, file_name_part=f'std_{corr_file_name_part}')
        save_and_print(obj=mean_train_corr, file_name_part=f'mean_{corr_file_name_part}')
        return corr

    def save_means_and_fractions(corr_file_name_part:str):
        train_corr = load_and_save_std_mean(corr_file_name_part=f'lstsq_corr_train_{corr_file_name_part}')
        test_corr = load_and_save_std_mean(corr_file_name_part=f'lstsq_corr_test_{corr_file_name_part}')
        fraction_test_ge = torch.count_nonzero(test_corr >= train_corr, dim=-1)/test_corr.size(dim=-1)
        save_and_print(obj=fraction_test_ge, file_name_part=f'fraction_test_ge_{corr_file_name_part}')
        return 0

    for param in ['h', 'mean_state']:
        lstsq_corr_file_name_part = f'all_{param}_{file_name_part}_perms_{num_node_perms}'
        save_means_and_fractions(corr_file_name_part=lstsq_corr_file_name_part)
        for feature in ['thickness', 'myelination', 'curvature', 'sulcus_depth']:
            corr_file_name_part = f'{feature}_{param}_{file_name_part}_perms_{num_node_perms}'
            save_means_and_fractions(corr_file_name_part=corr_file_name_part)
    for param in ['J', 'FC']:
        corr_file_name_part = f'SC_{param}_{file_name_part}_perms_{num_edge_perms}'
        save_means_and_fractions(corr_file_name_part=corr_file_name_part)
print('done')