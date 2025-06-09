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
    # Set epsilon to a small non-0 number to prevent NaNs in correlations.
    # The corelations may still be nonsense values.
    epsilon = 0.0

    parser = argparse.ArgumentParser(description="Find correlations between group model variance over replicas and individual model variance over subjects.")
    parser.add_argument("-a", "--data_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--region_feature_file_part", type=str, default='node_features_all_as_is', help='region feature file name except for the .pt file extension')
    parser.add_argument("-d", "--sc_file_part", type=str, default='edge_features_all_as_is', help='SC file name except for the .pt file extension')
    parser.add_argument("-e", "--group_model_file_part", type=str, default='ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000', help='the part of the Ising model file name before .pt.')
    parser.add_argument("-f", "--fmri_file_name_part", type=str, default='thresholds_31_min_0_max_3', help="the data mean state and state product file name after mean_state_ or mean_state_product_ and before .pt.")
    parser.add_argument("-g", "--num_training_subjects", type=int, default=670, help="uses the first this many subjects to for the group features, SC, mean state, and FC.")
    parser.add_argument("-i", "--num_training_regions", type=int, default=288, help="uses the first this many regions to fit the least squares model.")
    parser.add_argument("-j", "--num_training_region_pairs", type=int, default=51696, help="uses the first this many region pairs to fit the least squares model.")
    parser.add_argument("-k", "--num_permutations", type=int, default=1000, help="number of train-test splits to try")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    region_feature_file_part = args.region_feature_file_part
    print(f'region_feature_file_part={region_feature_file_part}')
    sc_file_part = args.sc_file_part
    print(f'sc_file_part={sc_file_part}')
    group_model_file_part = args.group_model_file_part
    print(f'group_model_file_part={group_model_file_part}')
    fmri_file_name_part = args.fmri_file_name_part
    print(f'fmri_file_name_part={fmri_file_name_part}')
    num_training_subjects = args.num_training_subjects
    print(f'num_training_subjects={num_training_subjects}')
    num_training_regions = args.num_training_regions
    print(f'num_training_regions={num_training_regions}')
    num_training_region_pairs = args.num_training_region_pairs
    print(f'num_training_region_pairs={num_training_region_pairs}')
    num_permutations = args.num_permutations
    print(f'num_permutations={num_permutations}')

    def count_non_nan(m:torch.Tensor):
        return (   torch.count_nonzero(  torch.logical_not( torch.isnan(m) )  )/m.numel()   ).item()
    
    def get_region_features():
        # Select out the actual structural features, omitting the region coordinates from the Atlas.
        # Take the mean over subjects, keeping the dimension as a singleton to align with the thresholds dimension for Ising model parameters.
        # region_features has size (1, num_nodes, num_features).
        region_features_file = os.path.join(data_directory, f'{region_feature_file_part}.pt')
        region_features = torch.mean( torch.load(region_features_file, weights_only=False)[:num_training_subjects,:,:4], dim=0, keepdim=True )
        print( f'time {time.time()-code_start_time:.3f}, loaded {region_features_file}, region features size', region_features.size() )
        return region_features
    
    def get_sc():
        # Select out only the SC.
        # Take the mean over subjects, keeping the dimension as a singleton to align with the thresholds dimension for Ising model parameters.
        # Unsqueeze so that we have a singleton dimension that matches up with the feature dimension.
        # sc has size (1, num_pairs, 1).
        sc_file = os.path.join(data_directory, f'{sc_file_part}.pt')
        sc = torch.mean( torch.load(sc_file, weights_only=False)[:num_training_subjects,:,0], dim=0, keepdim=True ).unsqueeze(dim=-1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {sc_file}, SC size', sc.size() )
        return sc

    def get_group_parameters():
        # Take the elements of J above the diagonal.
        # Take the mean over replicas.
        # Unsqueeze so that we have singleton dimensions that match up with feature dimensions.
        # Group h has size (num_thresholds, num_nodes, 1).
        # Group J has size (num_thresholds, num_pairs, 1).
        model_file_name = os.path.join(data_directory, f'{group_model_file_part}.pt')
        model = torch.load(f=model_file_name, weights_only=False)
        h = torch.mean(input=model.h, dim=0).unsqueeze(dim=-1)
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=model.J.size(dim=-1), device=model.J.device )
        J = torch.mean(input=model.J[:,:,triu_rows,triu_cols], dim=0).unsqueeze(dim=-1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name} group h size', h.size(), 'group J size', J.size() )
        return h, J
    
    def get_mean_state_and_fc():
        # Take the mean over scans.
        # Compute FC from mean state and mean state product.
        # Take the part of FC above the diagonal.
        # Clone so that we do not retain memory of the larger Tensor.
        # Unsqueeze so that we have a dimension that matches up with the feature dimension.
        # group_mean_state has size (num_thresholds, num_nodes, 1).
        # group_fc has size (num_thresholds, num_pairs, 1).

        # Get the individual mean state and FC.
        mean_state_file = os.path.join(data_directory, f'mean_state_{fmri_file_name_part}.pt')
        mean_state_product_file = os.path.join(data_directory, f'mean_state_product_{fmri_file_name_part}.pt')
        # Get the group mean state and FC.
        group_mean_state = torch.clone( input=torch.load(f=mean_state_file, weights_only=False) )
        group_mean_state_product = torch.clone( input=torch.load(f=mean_state_product_file, weights_only=False) )
        group_fc = isingmodellight.get_fc(state_mean=group_mean_state, state_product_mean=group_mean_state_product, epsilon=epsilon)
        # Do additional adjustments of the dimensions.
        group_mean_state = torch.unsqueeze(input=group_mean_state, dim=-1)
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=group_fc.size(dim=-1), device=group_fc.device )
        group_fc = torch.clone( input=group_fc[:,triu_rows,triu_cols].unsqueeze(dim=-1) )
        print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_file} and {mean_state_product_file}, group mean state size', group_mean_state.size(), 'group FC size', group_fc.size() )
        return group_mean_state, group_fc
    
    def append_ones(m:torch.Tensor):
        num_thresholds, num_parts, _ = m.size()
        return torch.cat(   (  m, torch.ones( size=(num_thresholds, num_parts, 1), dtype=m.dtype, device=m.device )  ), dim=-1   )
    
    def get_linear_regression_correlations_train_test(params:torch.Tensor, features:torch.Tensor, num_parts_train:int):
        features = append_ones(features)
        num_thresholds, num_parts, _ = params.size()
        corrs_train = torch.zeros( size=(num_thresholds, num_permutations), dtype=features.dtype, device=features.device )
        corrs_test = torch.zeros( size=(num_thresholds, num_permutations), dtype=features.dtype, device=features.device )
        for perm_index in range(num_permutations):
            permutation = torch.randperm(n=num_parts, dtype=int_type, device=features.device)
            indices_train = permutation[:num_parts_train]
            indices_test = permutation[num_parts_train:]
            features_train = features[:,indices_train,:]
            params_train = params[:,indices_train,:]
            features_test = features[:,indices_test,:]
            params_test = params[:,indices_test,:]
            coeffs = torch.linalg.lstsq(features_train, params_train).solution
            # print( f'time {time.time()-code_start_time:.3f}, coeffs size', coeffs.size(), f'fraction non-NaN {count_non_nan(coeffs):.3g}' )
            # print( 'coefficient values:', torch.flatten(coeffs).tolist() )
            predictions_train = torch.matmul(features_train, coeffs)
            predictions_test = torch.matmul(features_test, coeffs)
            # print( f'time {time.time()-code_start_time:.3f}, predictions train size', predictions_train.size(), f'fraction non-NaN {count_non_nan(predictions_train):.3g}, test size', predictions_test.size(), f'fraction non-NaN {count_non_nan(predictions_test):.3g}' )
            # ctr = isingmodellight.get_pairwise_correlation(mat1=params_train, mat2=predictions_train, epsilon=epsilon, dim=1).squeeze(dim=-1)
            # cte = isingmodellight.get_pairwise_correlation(mat1=params_test, mat2=predictions_test, epsilon=epsilon, dim=1).squeeze(dim=-1)
            # print( f'time {time.time()-code_start_time:.3f}, correlations train size', ctr.size(), f'fraction non-NaN {count_non_nan(ctr):.3g}, test size', cte.size(), f'fraction non-NaN {count_non_nan(cte):.3g}' )
            # Take the correlation over nodes or pairs (dim=1).
            corrs_train[:,perm_index] = isingmodellight.get_pairwise_correlation(mat1=params_train, mat2=predictions_train, epsilon=epsilon, dim=1).squeeze(dim=-1)
            corrs_test[:,perm_index] = isingmodellight.get_pairwise_correlation(mat1=params_test, mat2=predictions_test, epsilon=epsilon, dim=1).squeeze(dim=-1)
        print( f'time {time.time()-code_start_time:.3f}, correlations train size', corrs_train.size(), f'fraction non-NaN {count_non_nan(corrs_train):.3g}, test size', corrs_test.size(), f'fraction non-NaN {count_non_nan(corrs_test):.3g}' )
        return corrs_train, corrs_test
    
    def save_linear_regression_train_test_correlations(params:torch.Tensor, features:torch.Tensor, param_name:str, feature_name:str, num_parts_train:int):
        print(f'time {time.time()-code_start_time:.3f}, starting {param_name}-{feature_name} correlations...')
        # Get correlations for different train-test splits.
        corrs_train, corrs_test = get_linear_regression_correlations_train_test(params=params, features=features, num_parts_train=num_parts_train)
        # Save the correlations.
        output_file_suffix = f'from_{feature_name}_to_{param_name}_model_{group_model_file_part}_perms_{num_permutations}'
        corrs_train_file_name = os.path.join( output_directory, f'lstsq_corrs_train_{output_file_suffix}.pt')
        torch.save(obj=corrs_train, f=corrs_train_file_name)
        print(f'time {time.time()-code_start_time:.3f}, saved {corrs_train_file_name}')
        corrs_test_file_name = os.path.join( output_directory, f'lstsq_corr_test_{output_file_suffix}.pt')
        torch.save(obj=corrs_test, f=corrs_test_file_name)
        print(f'time {time.time()-code_start_time:.3f}, saved {corrs_test_file_name}')
        return 0
    
    def save_individual_region_feature_regression_correlations(region_params:torch.Tensor, region_features:torch.Tensor, param_name:str):
        region_feature_names = ['thickness', 'myelination', 'curvature', 'sulcus_depth']
        num_region_features = len(region_feature_names)
        for region_feature_index in range(num_region_features):
            save_linear_regression_train_test_correlations( params=region_params, features=region_features[:,:,region_feature_index].unsqueeze(dim=-1), param_name=param_name, feature_name=region_feature_names[region_feature_index], num_parts_train=num_training_regions )
        return 0
    
    def get_and_apply_all_linear_regressions():
        region_features = get_region_features()
        sc = get_sc()
        group_h, group_J = get_group_parameters()
        group_mean_state, group_fc = get_mean_state_and_fc()
        save_linear_regression_train_test_correlations(params=group_h, features=region_features, param_name='h', feature_name='all', num_parts_train=num_training_regions)
        save_linear_regression_train_test_correlations(params=group_mean_state, features=region_features, param_name='mean_state', feature_name='all', num_parts_train=num_training_regions)
        save_individual_region_feature_regression_correlations(region_params=group_h, region_features=region_features, param_name='h')
        save_individual_region_feature_regression_correlations(region_params=group_mean_state, region_features=region_features, param_name='mean_state')
        save_linear_regression_train_test_correlations(params=group_J, features=sc, param_name='J', feature_name='SC', num_parts_train=num_training_region_pairs)
        save_linear_regression_train_test_correlations(params=group_fc, features=sc, param_name='FC', feature_name='SC', num_parts_train=num_training_region_pairs)
        return 0
    
with torch.no_grad():
    get_and_apply_all_linear_regressions()

print('done')