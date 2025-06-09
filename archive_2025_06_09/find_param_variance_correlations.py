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
    parser.add_argument("-c", "--region_feature_file_name_part", type=str, default='node_features_all_as_is', help="part of the output file name before .pt")
    parser.add_argument("-d", "--region_pair_feature_file_name_part", type=str, default='edge_features_all_as_is', help="part of the output file name before .pt")
    parser.add_argument("-e", "--model_file_name_part", type=str, default='light_group_threshold_1_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_68_param_updates_10000_individual_updates_45000', help="the part of the Ising model file name before .pt.")
    parser.add_argument("-f", "--data_file_name_part", type=str, default='all_mean_std_1', help="the data mean state and state product file name after mean_state_ or mean_state_product_ and before .pt.")
    parser.add_argument("-g", "--num_region_permutations", type=int, default=1000000, help="number of permutations to use in each permutation test for region features")
    parser.add_argument("-i", "--num_region_pair_permutations", type=int, default=1000000, help="number of permutations to use in each permutation test for region pairs (SC v. J or SC v. FC)")
    parser.add_argument("-j", "--alpha", type=float, default=0.001, help="alpha for the statistical tests, used to find the critical values")
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
    model_file_name_part = args.model_file_name_part
    print(f'model_file_name_part={model_file_name_part}')
    data_file_name_part = args.data_file_name_part
    print(f'data_file_name_part={data_file_name_part}')
    num_region_permutations = args.num_region_permutations
    print(f'num_region_permutations={num_region_permutations}')
    num_region_pair_permutations = args.num_region_pair_permutations
    print(f'num_region_pair_permutations={num_region_pair_permutations}')
    alpha = args.alpha
    print(f'alpha={alpha}')

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
    
    def save_corr_p_and_crit_value(feature:torch.Tensor, feature_name:str, param:torch.Tensor, param_name:str, num_perms:int):
        num_regions, num_features = feature.size()

        print_and_save(values=feature, value_name=feature_name, file_part='all_as_is', print_all=False)
        print_and_save(values=param, value_name=param_name, file_part=model_file_name_part, print_all=False)

        corr = isingmodellight.get_pairwise_correlation(mat1=feature, mat2=param, epsilon=epsilon, dim=0)
        print_and_save(values=corr, value_name='corr', file_part=f'{feature_name}_{param_name}_{model_file_name_part}', print_all=True)
        abs_corr = torch.abs(corr)
        print(f'time {time.time()-code_start_time:.3f}, {param_name}-{feature_name} abs corr min {abs_corr.min():.3g} mean {abs_corr.mean():.3g} max {abs_corr.max():.3g}')

        abs_perm_corr = torch.zeros( size=(num_features, num_perms), dtype=corr.dtype, device=corr.device )
        for perm_index in range(num_perms):
            abs_perm_corr[:, perm_index] = isingmodellight.get_pairwise_correlation( mat1=param[ torch.randperm(n=num_regions, dtype=int_type, device=param.device), : ], mat2=feature, epsilon=epsilon, dim=0 ).abs()
        print( f'time {time.time()-code_start_time:.3f}, {param_name}-{feature_name} abs perm corr min {abs_perm_corr.min():.3g} mean {abs_perm_corr.mean():.3g} max {abs_perm_corr.max():.3g}, size', abs_perm_corr.size() )

        perm_file_part = f'{feature_name}_{param_name}_{model_file_name_part}_perms_{num_perms}'

        p = get_p_value(corr=abs_corr, perm_corr=abs_perm_corr)
        print_and_save(values=p, value_name='p_value', file_part=perm_file_part, print_all=True)

        crit_val = get_crit_value(perm_corr=abs_perm_corr, alpha=alpha)
        print_and_save(values=crit_val, value_name='crit_val', file_part=perm_file_part, print_all=True)

        return 0
    
    def do_param_variance_correlation_analysis(feature:torch.Tensor, feature_name:str, param:torch.Tensor, param_name:str, num_perms:int):
        feature_std, feature_mean = torch.std_mean(feature, dim=1, keepdim=False)
        save_corr_p_and_crit_value( feature=feature_mean, feature_name=f'mean_{feature_name}', param=torch.std(param, dim=1), param_name=f'std_{param_name}', num_perms=num_perms )
        save_corr_p_and_crit_value( feature=feature_std, feature_name=f'std_{feature_name}', param=torch.std(param, dim=1), param_name=f'std_{param_name}', num_perms=num_perms )
        return 0

    def get_lstsq_correlation(independent:torch.Tensor, dependent:torch.Tensor):
        return isingmodellight.get_pairwise_correlation(  mat1=dependent, mat2=torch.matmul( input=independent, other=torch.linalg.lstsq(independent, dependent).solution ), epsilon=epsilon, dim=1  ).squeeze(dim=-1)
    
    def compare_lstsq_corr_to_mean_and_std(feature:torch.Tensor, feature_name:str, lstsq_corr:torch.Tensor, lstsq_corr_name:str, num_perms:int):
        feature_std, feature_mean = torch.std_mean(feature, dim=1, keepdim=False)
        save_corr_p_and_crit_value( feature=feature_mean, feature_name=f'mean_{feature_name}', param=lstsq_corr, param_name=lstsq_corr_name, num_perms=num_perms )
        save_corr_p_and_crit_value( feature=feature_std, feature_name=f'std_{feature_name}', param=lstsq_corr, param_name=lstsq_corr_name, num_perms=num_perms )
        return 0
    
    def do_lstsq_correlation_versus_mean_and_perm_test(feature:torch.Tensor, feature_name:str, param:torch.Tensor, param_name:str, num_perms:int):
        num_regions, num_subjects, _ = feature.size()
        lstsq_corr = get_lstsq_correlation( independent=torch.cat(   tensors=(  feature, torch.ones( size=(num_regions, num_subjects, 1), dtype=feature.dtype, device=feature.device )  ), dim=-1   ), dependent=param ).unsqueeze(dim=-1)
        lstsq_corr_name = f'lstsq_corr_{feature_name}_{param_name}'
        compare_lstsq_corr_to_mean_and_std(feature=feature, feature_name=feature_name, lstsq_corr=lstsq_corr, lstsq_corr_name=lstsq_corr_name, num_perms=num_perms)
        compare_lstsq_corr_to_mean_and_std(feature=param, feature_name=param_name, lstsq_corr=lstsq_corr, lstsq_corr_name=lstsq_corr_name, num_perms=num_perms)
        return 0
    
    def do_lstsq_correlation_versus_z_abs_and_perm_test(feature:torch.Tensor, feature_name:str, param:torch.Tensor, param_name:str, num_perms:int):
        num_regions, num_subjects, _ = feature.size()
        lstsq_corr = get_lstsq_correlation( independent=torch.cat(   tensors=(  feature, torch.ones( size=(num_regions, num_subjects, 1), dtype=feature.dtype, device=feature.device )  ), dim=-1   ), dependent=param ).unsqueeze(dim=-1)
        feature_mean = torch.mean(feature, dim=1, keepdim=False)
        total_std, total_mean = torch.std_mean(input=feature_mean, dim=0, keepdim=True)
        save_corr_p_and_crit_value( feature=torch.abs( (feature_mean-total_mean)/total_std ), feature_name=f'feature_z_abs_{feature_name}', param=lstsq_corr, param_name=f'lstsq_corr_{feature_name}_{param_name}', num_perms=num_perms )
        return 0
    
    def get_region_features(num_region_features:int=4):
        # Transpose so that dim 0 is regions and dim 1 is subjects.
        # We will take one correlation over inter-subject differences for each region.
        # clone() so that we can deallocate the larger Tensor of which it is a view.
        region_feature_file_name = os.path.join(data_directory, f'{region_feature_file_name_part}.pt')
        region_features = torch.transpose( input=torch.load(f=region_feature_file_name, weights_only=False)[:,:,:num_region_features], dim0=0, dim1=1 ).clone()
        print( f'time {time.time()-code_start_time:.3f}, loaded {region_feature_file_name} region features size', region_features.size() )
        return region_features
    
    def get_mean_state():
        # Take the mean over scans.
        # Transpose so that dim 0 is regions and dim 1 is subjects.
        # We will take one correlation over inter-subject differences for each region.
        mean_state_file_name = os.path.join(data_directory, f'mean_state_{data_file_name_part}.pt')
        mean_state = torch.mean( input=torch.load(f=mean_state_file_name, weights_only=False), dim=0 ).transpose(dim0=0, dim1=1).unsqueeze(dim=-1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_file_name} mean state size', mean_state.size() )
        return mean_state
    
    def get_h():
        # Take the mean over replicas.
        # Transpose so that dim 0 is regions and dim 1 is subjects.
        # We will take one correlation over inter-subject differences for each region.
        model_file_name = os.path.join(data_directory, f'{model_file_name_part}.pt')
        h = torch.mean( input=torch.load(f=model_file_name, weights_only=False).h, dim=0 ).transpose(dim0=0, dim1=1).unsqueeze(dim=-1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name} h size', h.size() )
        return h
    
    def get_sc():
        # Transpose so that dim 0 is region pairs and dim 1 is subjects.
        # We will take one correlation over inter-subject differences for each region pair.
        # clone() so that we can de-allocate the larger Tensor of which this is a view.
        region_pair_feature_file_name = os.path.join(data_directory, f'{region_pair_feature_file_name_part}.pt')
        sc = torch.transpose( input=torch.load(f=region_pair_feature_file_name, weights_only=False)[:,:,0], dim0=0, dim1=1 ).unsqueeze(dim=-1).clone()
        print( f'time {time.time()-code_start_time:.3f}, loaded {region_pair_feature_file_name} SC size', sc.size() )
        return sc
    
    def get_J():
        # Take the part above the diagonal, and then take the mean over replicas.
        # This gives us a smaller Tensor with which to work.
        # Transpose so that dim 0 is region pairs and dim 1 is subjects.
        # We will take one correlation over inter-subject differences for each region pair.
        model_file_name = os.path.join(data_directory, f'{model_file_name_part}.pt')
        model = torch.load(f=model_file_name, weights_only=False)
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=model.J.size(dim=-1), device=model.J.device )
        J = torch.mean(input=model.J[:,:,triu_rows,triu_cols], dim=0).transpose(dim0=0, dim1=1).unsqueeze(dim=-1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name} J size', J.size() )
        return J
    
    def get_fc():
        # Before computing the FC, pool subject data from individual scans with mean().
        # After computing the FC, take the part above the diagonal.
        # This gives us a smaller Tensor with which to work.
        # Transpose so that dim 0 is region pairs and dim 1 is subjects.
        # We will take one correlation over inter-subject differences for each region pair.
        # clone() so that we can deallocate the larger Tensor of which it is a view.
        mean_state_file_name = os.path.join(data_directory, f'mean_state_{data_file_name_part}.pt')
        mean_state = torch.load(f=mean_state_file_name, weights_only=False)
        print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_file_name}, mean state size', mean_state.size() )
        mean_state_product_file_name = os.path.join(data_directory, f'mean_state_product_{data_file_name_part}.pt')
        mean_state_product = torch.load(f=mean_state_product_file_name, weights_only=False)
        print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_product_file_name}, mean state product size', mean_state_product.size() )
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=mean_state_product.size(dim=-1), device=mean_state_product.device )
        fc = torch.transpose(  input=isingmodellight.get_fc( state_mean=torch.mean(mean_state, dim=0), state_product_mean=torch.mean(mean_state_product, dim=0), epsilon=0.0 )[:,triu_rows,triu_cols], dim0=0, dim1=1  ).unsqueeze(dim=-1).clone()
        print( f'time {time.time()-code_start_time:.3f}, computed FC size', fc.size() )
        return fc
    
    def do_both_analyses(feature:torch.Tensor, feature_name:str, param:torch.Tensor, param_name:str, num_perms:int):
        do_param_variance_correlation_analysis(feature=feature, feature_name=feature_name, param=param, param_name=param_name, num_perms=num_perms)
        do_lstsq_correlation_versus_mean_and_perm_test(feature=feature, feature_name=feature_name, param=param, param_name=param_name, num_perms=num_perms)
        # do_lstsq_correlation_versus_z_abs_and_perm_test(feature=feature, feature_name=feature_name, param=param, param_name=param_name, num_perms=num_perms)
        return 0

    def do_node_correlation_analyses():
        region_features = get_region_features()
        do_both_analyses( feature=region_features, feature_name='all', param=get_h(), param_name='h', num_perms=num_region_permutations )
        do_both_analyses( feature=region_features, feature_name='all', param=get_mean_state(), param_name='mean_state', num_perms=num_region_permutations )
        return 0
    
    def do_edge_correlation_analyses():
        sc = get_sc()
        do_both_analyses( feature=sc, feature_name='SC', param=get_J(), param_name='J', num_perms=num_region_pair_permutations )
        do_both_analyses( feature=sc, feature_name='SC', param=get_fc(), param_name='FC', num_perms=num_region_pair_permutations )
        return 0
    
    do_node_correlation_analyses()
    do_edge_correlation_analyses()
print('done')