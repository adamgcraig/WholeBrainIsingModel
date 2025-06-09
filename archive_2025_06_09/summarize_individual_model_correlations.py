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

    parser = argparse.ArgumentParser(description="Find linear regressions to predict individual differences in Ising model parameters from individual differences in structural features.")
    parser.add_argument("-a", "--data_directory", type=str, default='D:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--file_name_part", type=str, default='ising_model_light_group_threshold_1_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_68_param_updates_10000_individual_updates_45000', help="part of the output file name before .pt")
    parser.add_argument("-e", "--num_node_perms", type=int, default=1000000, help="number of permutations for node correlations")
    parser.add_argument("-f", "--num_edge_perms", type=int, default=20000, help="number of permutations for edge correlations")
    parser.add_argument("-g", "--num_nodes", type=int, default=360, help="number of regions")
    parser.add_argument("-i", "--alpha", type=float, default=0.05, help="alpha for the statistical tests, used to find the critical values")
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
    alpha = args.alpha
    print(f'alpha={alpha}')
    node_alpha_quantile = torch.tensor(data=[1.0-alpha/num_nodes], dtype=float_type, device=device)
    edge_alpha_quantile = torch.tensor(data=[1.0-alpha/num_edges], dtype=float_type, device=device)

    def make_p_and_crit_files(corr_file_name_part:str, num_perms:int, alpha_quantile:torch.Tensor):
        corr_file = os.path.join(output_directory, f'{corr_file_name_part}.pt')
        corr = torch.load(f=corr_file, weights_only=False)
        print( f'time {time.time()-code_start_time:.3f}, loaded {corr_file}, size', corr.size() )
        perm_corr_file = os.path.join(output_directory, f'perm_{corr_file_name_part}_perms_{num_perms}.pt')
        perm_corr = torch.load(f=perm_corr_file, weights_only=False)
        print( f'time {time.time()-code_start_time:.3f}, loaded {perm_corr_file}, size', perm_corr.size() )
        p_value = torch.count_nonzero(  input=( perm_corr >= corr.unsqueeze(dim=-1) ), dim=-1  )/num_perms
        p_value_file = os.path.join(output_directory, f'p_value_{corr_file_name_part}_perms_{num_perms}.pt')
        torch.save(obj=p_value, f=p_value_file)
        print( f'time {time.time()-code_start_time:.3f}, saved {p_value_file}, size', p_value.size(), f'min {p_value.min():.3g} mean {p_value.mean():.3g} max {p_value.max():.3g}' )
        crit_value = torch.quantile(input=perm_corr, q=alpha_quantile, dim=-1)
        crit_value_file = os.path.join(output_directory, f'crit_value_{corr_file_name_part}_perms_{num_perms}.pt')
        torch.save(obj=crit_value, f=crit_value_file)
        print( f'time {time.time()-code_start_time:.3f}, saved {crit_value_file}, size', crit_value.size(), f'min {crit_value.min():.3g} mean {crit_value.mean():.3g} max {crit_value.max():.3g}' )
        return 0

    # for param in ['h', 'mean_state']:
    #     lstsq_corr_file_name_part = f'lstsq_corr_all_{param}_{file_name_part}'
    #     make_p_and_crit_files(corr_file_name_part=lstsq_corr_file_name_part, num_perms=num_node_perms, alpha_quantile=node_alpha_quantile)
    #     for feature in ['thickness', 'myelination', 'curvature', 'sulcus_depth']:
    #         corr_file_name_part = f'corr_{feature}_{param}_{file_name_part}'
    #         make_p_and_crit_files(corr_file_name_part=corr_file_name_part, num_perms=num_node_perms, alpha_quantile=node_alpha_quantile)
    for param in ['J', 'FC']:
        corr_file_name_part = f'corr_SC_{param}_{file_name_part}'
        make_p_and_crit_files(corr_file_name_part=corr_file_name_part, num_perms=num_edge_perms, alpha_quantile=edge_alpha_quantile)
print('done')