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
    parser.add_argument("-e", "--model_file_name_part", type=str, default='ising_model_light_all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps_40000', help="the part of the Ising model file name before .pt.")
    parser.add_argument("-f", "--data_file_name_part", type=str, default='all_mean_std_1', help="the data mean state and state product file name after mean_state_ or mean_state_product_ and before .pt.")
    parser.add_argument("-g", "--num_region_permutations", type=int, default=100, help="number of permutations to use in each permutation test for region features")
    parser.add_argument("-i", "--num_region_pair_permutations", type=int, default=100, help="number of permutations to use in each permutation test for region pairs (SC v. J or SC v. FC)")
    parser.add_argument("-j", "--alpha", type=float, default=0.05, help="alpha for the statistical tests, used to find the critical values")
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
    alpha_quantile = torch.tensor(data=[1.0-alpha], dtype=float_type, device=device)

    def get_p_value(corr:torch.Tensor, perm_corr:torch.Tensor):
        return torch.count_nonzero(  input=( perm_corr > corr.unsqueeze(dim=-1) ), dim=-1  )/perm_corr.size(dim=-1)
    
    def get_crit_value(perm_corr:torch.Tensor, adjusted_alpha:float):
        return torch.quantile( input=perm_corr, q=torch.tensor(data=[1.0-adjusted_alpha], dtype=perm_corr.dtype, device=perm_corr.device), dim=-1 )

    def do_direct_correlation_analysis_and_perm_test(independent:torch.Tensor, independent_name:str, dependent:torch.Tensor, dependent_name:str, num_perms:int):
        obsrv_dim = 1
        num_vars, num_observations = dependent.size()

        corr = isingmodellight.get_pairwise_correlation(mat1=dependent, mat2=independent, epsilon=epsilon, dim=obsrv_dim)
        print(f'time {time.time()-code_start_time:.3f}, {dependent_name}-{independent_name} corr min {corr.min():.3g} mean {corr.mean():.3g} max {corr.max():.3g}')
        corr_file_name = os.path.join(output_directory, f'corr_{independent_name}_{dependent_name}_{model_file_name_part}.pt')
        torch.save(obj=corr, f=corr_file_name)
        print(f'time {time.time()-code_start_time:.3f}, saved {corr_file_name}')
        abs_corr = torch.abs(corr)
        print(f'time {time.time()-code_start_time:.3f}, {dependent_name}-{dependent_name}({independent_name}) abs. corr min {abs_corr.min():.3g} mean {abs_corr.mean():.3g} max {abs_corr.max():.3g}')

        perm_file_part = f'{independent_name}_{dependent_name}_{model_file_name_part}_perms_{num_perms}'

        abs_perm_corr = torch.zeros( size=(num_vars, num_perms), dtype=corr.dtype, device=corr.device )
        for perm_index in range(num_perms):
            abs_perm_corr[:, perm_index] = isingmodellight.get_pairwise_correlation( mat1=dependent[ :, torch.randperm(n=num_observations, dtype=int_type, device=dependent.device) ], mat2=independent, epsilon=epsilon, dim=obsrv_dim ).abs()
        print(f'time {time.time()-code_start_time:.3f}, {dependent_name}-{independent_name} perm corr min {abs_perm_corr.min():.3g} mean {abs_perm_corr.mean():.3g} max {abs_perm_corr.max():.3g}')
        # perm_corr_file_name = os.path.join(output_directory, f'perm_corr_{perm_file_part}.pt')
        # torch.save(obj=perm_corr, f=perm_corr_file_name)
        # print(f'time {time.time()-code_start_time:.3f}, saved {perm_corr_file_name}')
        # abs_perm_corr = torch.abs(abs_perm_corr)
        # print(f'time {time.time()-code_start_time:.3f}, {dependent_name}-{dependent_name}({independent_name}) abs. perm corr min {abs_perm_corr.min():.3g} mean {abs_perm_corr.mean():.3g} max {abs_perm_corr.max():.3g}')

        p = get_p_value(corr=abs_corr, perm_corr=abs_perm_corr)
        print(f'time {time.time()-code_start_time:.3f}, {dependent_name}-{independent_name} p-value min {p.min():.3g} mean {p.mean():.3g} max {p.max():.3g}')
        p_file_name = os.path.join(output_directory, f'p_value_{perm_file_part}.pt')
        torch.save(obj=p, f=p_file_name)
        print(f'time {time.time()-code_start_time:.3f}, saved {p_file_name}')

        crit_val = get_crit_value(perm_corr=abs_perm_corr, adjusted_alpha=alpha/num_vars)
        print(f'time {time.time()-code_start_time:.3f}, {dependent_name}-{independent_name} critical value min {crit_val.min():.3g} mean {crit_val.mean():.3g} max {crit_val.max():.3g}')
        crit_val_file_name = os.path.join(output_directory, f'crit_val_{perm_file_part}.pt')
        torch.save(obj=crit_val, f=crit_val_file_name)
        print(f'time {time.time()-code_start_time:.3f}, saved {crit_val_file_name}')

        return 0

    def do_lstsq_correlation_analysis(independent:torch.Tensor, dependent:torch.Tensor):
        return isingmodellight.get_pairwise_correlation(  mat1=dependent, mat2=torch.matmul( input=independent, other=torch.linalg.lstsq(independent, dependent).solution ), epsilon=epsilon, dim=1  ).squeeze(dim=-1)
    
    def do_lstsq_correlation_analysis_and_perm_test(independent:torch.Tensor, independent_name:str, dependent:torch.Tensor, dependent_name:str, num_perms:int):
        num_vars, num_observations, _ = dependent.size()
        independent = torch.cat(  tensors=( independent, torch.ones_like(dependent) ), dim=-1  )

        corr = do_lstsq_correlation_analysis(independent=independent, dependent=dependent)
        print(f'time {time.time()-code_start_time:.3f}, {dependent_name}-{dependent_name}({independent_name}) corr min {corr.min():.3g} mean {corr.mean():.3g} max {corr.max():.3g}')
        corr_file_name = os.path.join(output_directory, f'corr_lstsq_{independent_name}_{dependent_name}_{model_file_name_part}.pt')
        torch.save(obj=corr, f=corr_file_name)
        print(f'time {time.time()-code_start_time:.3f}, saved {corr_file_name}')

        perm_file_part = f'lstsq_{independent_name}_{dependent_name}_{model_file_name_part}_perms_{num_perms}'

        perm_corr = torch.zeros( size=(num_vars, num_perms), dtype=corr.dtype, device=corr.device )
        for perm_index in range(num_perms):
            perm_corr[:, perm_index] = do_lstsq_correlation_analysis( independent=independent, dependent=dependent[ :, torch.randperm(n=num_observations, dtype=int_type, device=dependent.device), : ] )
        print(f'time {time.time()-code_start_time:.3f}, {dependent_name}-{dependent_name}({independent_name}) perm corr min {perm_corr.min():.3g} mean {perm_corr.mean():.3g} max {perm_corr.max():.3g}')
        # perm_corr_file_name = os.path.join(output_directory, f'perm_corr_{perm_file_part}.pt')
        # torch.save(obj=perm_corr, f=perm_corr_file_name)
        # print(f'time {time.time()-code_start_time:.3f}, saved {perm_corr_file_name}')

        p = get_p_value(corr=corr, perm_corr=perm_corr)
        print(f'time {time.time()-code_start_time:.3f}, {dependent_name}-{independent_name} p-value min {p.min():.3g} mean {p.mean():.3g} max {p.max():.3g}')
        p_file_name = os.path.join(output_directory, f'p_value_{perm_file_part}.pt')
        torch.save(obj=p, f=p_file_name)
        print(f'time {time.time()-code_start_time:.3f}, saved {p_file_name}')

        crit_val = get_crit_value(perm_corr=perm_corr, adjusted_alpha=alpha/num_vars)
        print(f'time {time.time()-code_start_time:.3f}, {dependent_name}-{independent_name} critical value min {crit_val.min():.3g} mean {crit_val.mean():.3g} max {crit_val.max():.3g}')
        crit_val_file_name = os.path.join(output_directory, f'crit_val_{perm_file_part}.pt')
        torch.save(obj=crit_val, f=crit_val_file_name)
        print(f'time {time.time()-code_start_time:.3f}, saved {crit_val_file_name}')

        return corr, perm_corr
    
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
        mean_state = torch.mean( input=torch.load(f=mean_state_file_name, weights_only=False), dim=0 ).transpose(dim0=0, dim1=1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_file_name} mean state size', mean_state.size() )
        return mean_state
    
    def get_h():
        # Take the mean over replicas.
        # Transpose so that dim 0 is regions and dim 1 is subjects.
        # We will take one correlation over inter-subject differences for each region.
        model_file_name = os.path.join(data_directory, f'{model_file_name_part}.pt')
        h = torch.mean( input=torch.load(f=model_file_name, weights_only=False).h, dim=0 ).transpose(dim0=0, dim1=1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name} h size', h.size() )
        return h
    
    def get_sc():
        # Transpose so that dim 0 is region pairs and dim 1 is subjects.
        # We will take one correlation over inter-subject differences for each region pair.
        # clone() so that we can de-allocate the larger Tensor of which this is a view.
        region_pair_feature_file_name = os.path.join(data_directory, f'{region_pair_feature_file_name_part}.pt')
        sc = torch.transpose( input=torch.load(f=region_pair_feature_file_name, weights_only=False)[:,:,0], dim0=0, dim1=1 ).clone()
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
        J = torch.mean(input=model.J[:,:,triu_rows,triu_cols], dim=0).transpose(dim0=0, dim1=1)
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
        fc = torch.transpose(  input=isingmodellight.get_fc( state_mean=torch.mean(mean_state, dim=0), state_product_mean=torch.mean(mean_state_product, dim=0), epsilon=0.0 )[:,triu_rows,triu_cols], dim0=0, dim1=1  ).clone()
        print( f'time {time.time()-code_start_time:.3f}, computed FC size', fc.size() )
        return fc
    
    def do_node_correlation_analyses():
        feature_names = ['thickness', 'myelination', 'curvature', 'sulcus_depth']
        region_features = get_region_features()
        h = get_h()
        mean_state = get_mean_state()
        # For least squares regression, region features and h need to have the same number of dimensions:
        # region (batch dimension) x subject (observation dimension) x feature (feature dimension)
        h_for_lstsq = h.unsqueeze(dim=-1)
        mean_state_for_lstsq = mean_state.unsqueeze(dim=-1)
        do_lstsq_correlation_analysis_and_perm_test(independent=region_features, independent_name='all', dependent=h_for_lstsq, dependent_name='h', num_perms=num_region_permutations)
        do_lstsq_correlation_analysis_and_perm_test(independent=region_features, independent_name='all', dependent=mean_state_for_lstsq, dependent_name='mean_state', num_perms=num_region_permutations)
        num_features = region_features.size(dim=-1)
        for feature_index in range(num_features):
            do_direct_correlation_analysis_and_perm_test(independent=region_features[:,:,feature_index], independent_name=feature_names[feature_index], dependent=h, dependent_name='h', num_perms=num_region_permutations)
            do_direct_correlation_analysis_and_perm_test(independent=region_features[:,:,feature_index], independent_name=feature_names[feature_index], dependent=mean_state, dependent_name='mean_state', num_perms=num_region_permutations)
        return 0
    
    def do_edge_correlation_analyses():
        sc = get_sc()
        do_direct_correlation_analysis_and_perm_test( independent=sc, independent_name='SC', dependent=get_J(), dependent_name='J', num_perms=num_region_pair_permutations )
        do_direct_correlation_analysis_and_perm_test( independent=sc, independent_name='SC', dependent=get_fc(), dependent_name='FC', num_perms=num_region_pair_permutations )
        return 0
    
    do_node_correlation_analyses()
    do_edge_correlation_analyses()
print('done')