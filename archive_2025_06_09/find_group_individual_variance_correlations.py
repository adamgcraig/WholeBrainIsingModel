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

    parser = argparse.ArgumentParser(description="Find correlations between group model variance over replicas and individual model variance over subjects.")
    parser.add_argument("-a", "--data_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--region_feature_file_name_part", type=str, default='node_features_all_as_is', help="part of the output file name before .pt")
    parser.add_argument("-d", "--region_pair_feature_file_name_part", type=str, default='edge_features_all_as_is', help="part of the output file name before .pt")
    parser.add_argument("-e", "--group_model_file_name_part", type=str, default='ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000', help="the part of the Ising model file name before .pt.")
    parser.add_argument("-f", "--individual_model_file_name_part", type=str, default='light_group_threshold_1_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_68_param_updates_10000_individual_updates_45000', help="the part of the Ising model file name before .pt.")
    parser.add_argument("-g", "--data_file_name_part", type=str, default='all_mean_std_1', help="the data mean state and state product file name after mean_state_ or mean_state_product_ and before .pt.")
    parser.add_argument("-i", "--num_region_permutations", type=int, default=100, help="number of permutations to use in each permutation test for region features")
    parser.add_argument("-j", "--num_region_pair_permutations", type=int, default=100, help="number of permutations to use in each permutation test for region pairs (SC v. J or SC v. FC)")
    parser.add_argument("-k", "--alpha", type=float, default=0.001, help="alpha for the statistical tests, used to find the critical values")
    parser.add_argument("-l", "--threshold_index", type=int, default=10, help="index of threshold to select out of group model")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    region_feature_file_name_part = args.region_feature_file_name_part
    print(f'region_feature_file_name_part={region_feature_file_name_part}')
    region_pair_feature_file_name_part = args.region_pair_feature_file_name_part
    print(f'region_pair_feature_file_name_part={region_pair_feature_file_name_part}')
    group_model_file_name_part = args.group_model_file_name_part
    print(f'group_model_file_name_part={group_model_file_name_part}')
    individual_model_file_name_part = args.individual_model_file_name_part
    print(f'individual_model_file_name_part={individual_model_file_name_part}')
    data_file_name_part = args.data_file_name_part
    print(f'data_file_name_part={data_file_name_part}')
    num_region_permutations = args.num_region_permutations
    print(f'num_region_permutations={num_region_permutations}')
    num_region_pair_permutations = args.num_region_pair_permutations
    print(f'num_region_pair_permutations={num_region_pair_permutations}')
    alpha = args.alpha
    print(f'alpha={alpha}')
    threshold_index = args.threshold_index
    print(f'threshold_index={threshold_index}')

    def get_p_value(corr:torch.Tensor, perm_corr:torch.Tensor):
        return torch.count_nonzero(  input=( perm_corr > corr.unsqueeze(dim=-1) ), dim=-1  )/perm_corr.size(dim=-1)
    
    def get_crit_value(perm_corr:torch.Tensor, alpha:float):
        # Squeeze to get rid of the singleton q-wise dimension that quantile leaves in.
        return torch.quantile( input=perm_corr, q=torch.tensor(data=[1.0-alpha], dtype=perm_corr.dtype, device=perm_corr.device), dim=-1 ).squeeze(dim=0)
    
    def print_and_save(values:torch.Tensor, value_name:str, file_part:str, print_all:bool=False):
        print( f'time {time.time()-code_start_time:.3f}, {value_name} min {values.min():.3g} mean {values.mean():.3g} max {values.max():.3g}, size', values.size() )
        if print_all:
            print( values.tolist() )
        crit_val_file_name = os.path.join(output_directory, f'{value_name}_{file_part}.pt')
        torch.save(obj=values, f=crit_val_file_name)
        print(f'time {time.time()-code_start_time:.3f}, saved {crit_val_file_name}')
        return 0
    
    def save_corr_p_and_crit_value(group_param:torch.Tensor, group_param_name:str, individual_param:torch.Tensor, individual_param_name:str, num_perms:int):
        num_regions = group_param.size(dim=0)

        print_and_save(values=group_param, value_name=group_param_name, file_part=f'{group_model_file_name_part}_threshold_index_{threshold_index}', print_all=False)
        print_and_save(values=individual_param, value_name=individual_param_name, file_part=individual_model_file_name_part, print_all=False)

        corr = isingmodellight.get_pairwise_correlation(mat1=group_param, mat2=individual_param, epsilon=epsilon, dim=0)
        print(f'time {time.time()-code_start_time:.3f}, {individual_param_name}-{group_param_name} corr {corr:.3g}')
        print_and_save(values=corr, value_name='corr', file_part=f'{group_param_name}_{individual_param_name}_{individual_model_file_name_part}', print_all=True)
        abs_corr = torch.abs(corr)

        abs_perm_corr = torch.zeros( size=(num_perms,), dtype=corr.dtype, device=corr.device )
        for perm_index in range(num_perms):
            abs_perm_corr[perm_index] = isingmodellight.get_pairwise_correlation( mat1=individual_param[ torch.randperm(n=num_regions, dtype=int_type, device=individual_param.device)], mat2=group_param, epsilon=epsilon, dim=0 ).abs()
        print( f'time {time.time()-code_start_time:.3f}, {individual_param_name}-{group_param_name} abs perm corr min {abs_perm_corr.min():.3g} mean {abs_perm_corr.mean():.3g} max {abs_perm_corr.max():.3g}, size', abs_perm_corr.size() )

        perm_file_part = f'{group_param_name}_{individual_param_name}_{individual_model_file_name_part}_perms_{num_perms}'

        p = get_p_value(corr=abs_corr, perm_corr=abs_perm_corr)
        print_and_save(values=p, value_name='p_value', file_part=perm_file_part, print_all=True)

        crit_val = get_crit_value(perm_corr=abs_perm_corr, alpha=alpha)
        print_and_save(values=crit_val, value_name='crit_val', file_part=perm_file_part, print_all=True)

        return 0
    
    def do_param_variance_correlation_analysis(group_param:torch.Tensor, group_param_name:str, individual_param:torch.Tensor, individual_param_name:str, num_perms:int):
        save_corr_p_and_crit_value( group_param=torch.std(input=group_param, dim=1, keepdim=False), group_param_name=f'std_{group_param_name}', individual_param=torch.std(input=individual_param, dim=1, keepdim=False), individual_param_name=f'std_{individual_param_name}', num_perms=num_perms )
        return 0
    
    def get_group_h():
        # Take the mean over replicas.
        # Transpose so that dim 0 is regions and dim 1 is subjects.
        # We will take one correlation over inter-subject differences for each region.
        model_file_name = os.path.join(data_directory, f'{group_model_file_name_part}.pt')
        h = torch.load(f=model_file_name, weights_only=False).h[:,threshold_index,:].transpose(dim0=0, dim1=1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name} h size', h.size() )
        return h
    
    def get_individual_h():
        # Take the mean over replicas.
        # Transpose so that dim 0 is regions and dim 1 is subjects.
        # We will take one correlation over inter-subject differences for each region.
        model_file_name = os.path.join(data_directory, f'{individual_model_file_name_part}.pt')
        h = torch.mean( input=torch.load(f=model_file_name, weights_only=False).h, dim=0 ).transpose(dim0=0, dim1=1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name} h size', h.size() )
        return h
    
    def get_group_J():
        # Take the mean over replicas.
        # Transpose so that dim 0 is regions and dim 1 is subjects.
        # We will take one correlation over inter-subject differences for each region.
        model_file_name = os.path.join(data_directory, f'{group_model_file_name_part}.pt')
        model = torch.load(f=model_file_name, weights_only=False)
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=model.J.size(dim=-1), device=model.J.device )
        J = model.J[:,threshold_index,triu_rows,triu_cols].transpose(dim0=0, dim1=1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name} J size', J.size() )
        return J
    
    def get_individual_J():
        # Take the part above the diagonal, and then take the mean over replicas.
        # This gives us a smaller Tensor with which to work.
        # Transpose so that dim 0 is region pairs and dim 1 is subjects.
        # We will take one correlation over inter-subject differences for each region pair.
        model_file_name = os.path.join(data_directory, f'{individual_model_file_name_part}.pt')
        model = torch.load(f=model_file_name, weights_only=False)
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=model.J.size(dim=-1), device=model.J.device )
        J = torch.mean(input=model.J[:,:,triu_rows,triu_cols], dim=0).transpose(dim0=0, dim1=1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name} J size', J.size() )
        return J

    def do_node_correlation_analyses():
        do_param_variance_correlation_analysis( group_param=get_group_h(), group_param_name='group_h', individual_param=get_individual_h(), individual_param_name='individual_h', num_perms=num_region_permutations )
        return 0
    
    def do_edge_correlation_analyses():
        do_param_variance_correlation_analysis( group_param=get_group_J(), group_param_name='group_J', individual_param=get_individual_J(), individual_param_name='individual_J', num_perms=num_region_pair_permutations )
        return 0
    
    do_node_correlation_analyses()
    do_edge_correlation_analyses()
print('done')