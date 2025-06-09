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
    parser.add_argument("-f", "--individual_model_file_part", type=str, default='ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_to_thresh_1_reps_5_subj_837_individual_updates_40000', help='the part of the individual Ising model file name before .pt.')
    parser.add_argument("-g", "--fmri_file_name_part", type=str, default='all_mean_std_1', help="the data mean state and state product file name after mean_state_ or mean_state_product_ and before .pt.")
    parser.add_argument("-i", "--num_training_subjects", type=int, default=670, help="uses the first this many subjects for the group data.")
    parser.add_argument("-j", "--threshold_index", type=int, default=10, help="index of threshold to select out of group model")
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
    individual_model_file_part = args.individual_model_file_part
    print(f'individual_model_file_part={individual_model_file_part}')
    fmri_file_name_part = args.fmri_file_name_part
    print(f'fmri_file_name_part={fmri_file_name_part}')
    num_training_subjects = args.num_training_subjects
    print(f'num_training_subjects={num_training_subjects}')
    threshold_index = args.threshold_index
    print(f'threshold_index={threshold_index}')

    def count_non_nan(m:torch.Tensor):
        return (   torch.count_nonzero(  torch.logical_not( torch.isnan(m) )  )/m.numel()   ).item()
    
    def get_region_features():
        # Select out the actual structural features, omitting the region coordinates from the Atlas.
        # Clone so that we do not retain memory of the larger Tensor after exiting the function.
        # region_features has size (num_subjects, num_nodes, num_features).
        region_features_file = os.path.join(data_directory, f'{region_feature_file_part}.pt')
        region_features = torch.clone( torch.load(region_features_file, weights_only=False)[:,:,:4] )
        print( f'time {time.time()-code_start_time:.3f}, loaded {region_features_file}, region features size', region_features.size() )
        return region_features
    
    def get_sc():
        # Select out only the SC.
        # Clone so that we do not retain memory of the larger Tensor after exiting the function.
        # Unsqueeze so that we have a singleton dimension that matches up with the feature dimension.
        # sc has size (num_subjects, num_pairs, 1).
        sc_file = os.path.join(data_directory, f'{sc_file_part}.pt')
        sc = torch.clone( torch.load(sc_file, weights_only=False)[:,:,0] ).unsqueeze(dim=-1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {sc_file}, SC size', sc.size() )
        return sc

    def get_group_parameters():
        # Select the same threshold used for the individual models.
        # Take the elements of J above the diagonal.
        # Take the mean over replicas.
        # Unsqueeze twice so that we have singleton dimensions that match up with both subject and feature dimensions.
        # Group h has size (1, num_nodes, 1).
        # Group J has size (1, num_pairs, 1).
        model_file_name = os.path.join(data_directory, f'{group_model_file_part}.pt')
        model = torch.load(f=model_file_name, weights_only=False)
        h = torch.mean(input=model.h[:,threshold_index,:], dim=0).unsqueeze(dim=0).unsqueeze(dim=-1)
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=model.J.size(dim=-1), device=model.J.device )
        J = torch.mean(input=model.J[:,threshold_index,triu_rows,triu_cols], dim=0).unsqueeze(dim=0).unsqueeze(dim=-1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name} group h size', h.size(), 'group J size', J.size() )
        return h, J
    
    def get_individual_parameters():
        # Take the elements of J above the diagonal.
        # Take the mean over replicas.
        # Unsqueeze so that we have a dimension that matches up with the feature dimension.
        # Individual h has size (num_subjects, num_nodes, 1).
        # Individual J has size (num_subjects, num_pairs, 1).
        model_file_name = os.path.join(data_directory, f'{individual_model_file_part}.pt')
        model = torch.load(f=model_file_name, weights_only=False)
        h = torch.mean(input=model.h, dim=0).unsqueeze(dim=-1)
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=model.J.size(dim=-1), device=model.J.device )
        J = torch.mean(input=model.J[:,:,triu_rows,triu_cols], dim=0).unsqueeze(dim=-1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name} individual h size', h.size(), 'individual J size', J.size() )
        return h, J
    
    def get_mean_state_and_fc():
        # Take the mean over scans.
        # Compute FC from mean state and mean state product.
        # For the group mean state and FC, take the mean over subjects but retain the singleton subject dimension.
        # Take the part of FC above the diagonal.
        # Clone so that we do not retain memory of the larger Tensor.
        # Unsqueeze so that we have a dimension that matches up with the feature dimension.
        # group_mean_state has size (1, num_nodes, 1).
        # group_fc has size (1, num_pairs, 1)
        # individual_mean_state has size (num_subjects, num_nodes, 1).
        # individual_fc has size (num_subjects, num_pairs, 1).

        # Get the individual mean state and FC.
        mean_state_file = os.path.join(data_directory, f'mean_state_{fmri_file_name_part}.pt')
        individual_mean_state = torch.mean( input=torch.load(f=mean_state_file, weights_only=False), dim=0 )
        mean_state_product_file = os.path.join(data_directory, f'mean_state_product_{fmri_file_name_part}.pt')
        individual_mean_state_product = torch.mean( input=torch.load(f=mean_state_product_file, weights_only=False), dim=0 )
        individual_fc = isingmodellight.get_fc(state_mean=individual_mean_state, state_product_mean=individual_mean_state_product, epsilon=epsilon)
        # Get the group mean state and FC.
        group_mean_state = torch.mean(input=individual_mean_state[:num_training_subjects,:], dim=0, keepdim=True)
        group_mean_state_product = torch.mean(input=individual_mean_state_product[:num_training_subjects,:,:], dim=0, keepdim=True)
        group_fc = isingmodellight.get_fc(state_mean=group_mean_state, state_product_mean=group_mean_state_product, epsilon=epsilon)
        # Do additional adjustments of the dimensions.
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=individual_fc.size(dim=-1), device=individual_fc.device )
        group_mean_state = group_mean_state.unsqueeze(dim=-1)
        group_fc = torch.clone( group_fc[:,triu_rows,triu_cols].unsqueeze(dim=-1) )
        individual_fc = torch.clone( individual_fc[:,triu_rows,triu_cols].unsqueeze(dim=-1) )
        individual_mean_state = individual_mean_state.unsqueeze(dim=-1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_file} and {mean_state_product_file}, group mean state size', group_mean_state.size(), 'group FC size', group_fc.size(), 'individual mean state size', individual_mean_state.size(), 'individual FC size', individual_fc.size() )
        return group_mean_state, group_fc, individual_mean_state, individual_fc
    
    def append_ones(m:torch.Tensor):
        num_subjects, num_parts, _ = m.size()
        return torch.cat(   (  m, torch.ones( size=(num_subjects, num_parts, 1), dtype=m.dtype, device=m.device )  ), dim=-1   )
    
    def get_linear_regression(params:torch.Tensor, features:torch.Tensor):
        features = append_ones(features)
        coeffs = torch.linalg.lstsq(features, params).solution
        print( f'time {time.time()-code_start_time:.3f}, coeffs size', coeffs.size(), f'fraction non-NaN {count_non_nan(coeffs):.3g}' )
        print( 'coefficient values:', torch.flatten(coeffs).tolist() )
        predictions = torch.matmul(features, coeffs)
        print( f'time {time.time()-code_start_time:.3f}, predictions size', predictions.size(), f'fraction non-NaN {count_non_nan(predictions):.3g}' )
        # Take the correlation over nodes or pairs (dim=1).
        corrs = isingmodellight.get_pairwise_correlation(mat1=params, mat2=predictions, epsilon=epsilon, dim=1)
        print( f'time {time.time()-code_start_time:.3f}, correlations size', corrs.size(), f'fraction non-NaN {count_non_nan(corrs):.3g}' )
        print( 'correlation values:', corrs.flatten().tolist() )
        return coeffs
    
    def apply_linear_regression(params:torch.Tensor, features:torch.Tensor, coeffs:torch.Tensor):
        features = append_ones(features)
        predictions = torch.matmul(features, coeffs)
        print( f'time {time.time()-code_start_time:.3f}, predictions size', predictions.size(), f'fraction non-NaN {count_non_nan(predictions):.3g}' )
        # Take the correlation over nodes or pairs (dim=1).
        corrs = isingmodellight.get_pairwise_correlation(mat1=params, mat2=predictions, epsilon=epsilon, dim=1)
        print( f'time {time.time()-code_start_time:.3f}, correlations size', corrs.size(), f'fraction non-NaN {count_non_nan(corrs):.3g}, min {corrs.min():.3g}, median {corrs.median():.3g}, max {corrs.max():.3g}' )
        return corrs
    
    def get_apply_save_linear_regression(group_params:torch.Tensor, individual_params:torch.Tensor, features:torch.Tensor, param_name:str, feature_name:str):
        print(f'time {time.time()-code_start_time:.3f}, starting {param_name}-{feature_name} correlations...')
        # Get coefficients using the group model and group mean features.
        coeffs = get_linear_regression( params=group_params, features=torch.mean(features[:num_training_subjects,:,:], dim=0, keepdim=True) )
        # Apply the linear model to individual subjects.
        corrs = apply_linear_regression(params=individual_params, features=features, coeffs=coeffs)
        # Save the correlations.
        corrs_file_name = os.path.join( output_directory, f'lstsq_corr_from_{feature_name}_to_{param_name}_model_{individual_model_file_part}.pt')
        torch.save(obj=corrs, f=corrs_file_name)
        print(f'time {time.time()-code_start_time:.3f}, saved {corrs_file_name}')
        return 0
    
    def get_and_apply_individual_region_feature_regressions(group_params:torch.Tensor, individual_params:torch.Tensor, region_features:torch.Tensor, param_name:str):
        region_feature_names = ['thickness', 'myelination', 'curvature', 'sulcus_depth']
        num_region_features = len(region_feature_names)
        for region_feature_index in range(num_region_features):
            get_apply_save_linear_regression( group_params=group_params, individual_params=individual_params, features=region_features[:,:,region_feature_index].unsqueeze(dim=-1), param_name=param_name, feature_name=region_feature_names[region_feature_index] )
        return 0
    
    def get_and_apply_all_linear_regressions():
        region_features = get_region_features()
        sc = get_sc()
        group_h, group_J = get_group_parameters()
        individual_h, individual_J = get_individual_parameters()
        group_mean_state, group_fc, individual_mean_state, individual_fc = get_mean_state_and_fc()
        get_apply_save_linear_regression(group_params=group_h, individual_params=individual_h, features=region_features, param_name='h', feature_name='all')
        get_apply_save_linear_regression(group_params=group_mean_state, individual_params=individual_mean_state, features=region_features, param_name='mean_state', feature_name='all')
        get_and_apply_individual_region_feature_regressions(group_params=group_h, individual_params=individual_h, region_features=region_features, param_name='h')
        get_and_apply_individual_region_feature_regressions(group_params=group_mean_state, individual_params=individual_mean_state, region_features=region_features, param_name='mean_state')
        get_apply_save_linear_regression(group_params=group_J, individual_params=individual_J, features=sc, param_name='J', feature_name='SC')
        get_apply_save_linear_regression(group_params=group_fc, individual_params=individual_fc, features=sc, param_name='FC', feature_name='SC')
        return 0
    
with torch.no_grad():
    get_and_apply_all_linear_regressions()

print('done')