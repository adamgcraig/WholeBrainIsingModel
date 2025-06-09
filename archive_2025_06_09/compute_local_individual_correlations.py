import os
import torch
import isingmodellight
from isingmodellight import IsingModelLight
import time
import argparse

with torch.no_grad():
    code_start_time = time.time()

    parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the expected values of their observables. Compare to expected values from data. Do an Euler step update. Repeat.")
    parser.add_argument("-a", "--file_directory", type=str, default='E:\\Ising_model_results_daai', help="directory from which to read and to which to write files")
    parser.add_argument("-b", "--data_file_name_part", type=str, default='all_mean_std_0', help="part of the file name shared between model and data files")
    parser.add_argument("-c", "--model_file_name_part", type=str, default='medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_66_popt_steps_10000', help="part of the file name specific to the model")
    parser.add_argument("-d", "--original_one_over_alpha", type=int, default=20, help="integer-value representing 1/alpha for the permutation test")
    parser.add_argument("-e", "--num_largest_values", type=int, default=10, help="multiplier such that the total number of permutations we test is 2*num_largest_values*num_degrees_of_freedom*original_one_over_alpha where num_degrees_of_freedom = num_nodes or num_edges")

    args = parser.parse_args()
    print('getting arguments...')
    file_directory = args.file_directory
    print(f'file_directory={file_directory}')
    data_file_name_part = args.data_file_name_part
    print(f'data_file_name_part={data_file_name_part}')
    model_file_name_part = args.model_file_name_part
    print(f'model_file_name_part={model_file_name_part}')
    original_one_over_alpha = args.original_one_over_alpha
    print(f'original_one_over_alpha={original_one_over_alpha}')
    num_largest_values = args.num_largest_values
    print(f'num_largest_values={num_largest_values}')

    int_type = torch.int
    float_type = torch.float
    device = torch.device('cuda')
    param_string = f'{data_file_name_part}_{model_file_name_part}'

    def get_corr_and_crit_corr(param:torch.Tensor, feature:torch.Tensor, original_one_over_alpha:int, num_largest_values:int, param_string:str, param_name:str, feature_name:str):
        original_alpha = 1.0/original_one_over_alpha
        param_feature_corr = isingmodellight.get_pairwise_correlation( mat1=param, mat2=feature, epsilon=0, dim=0 )
        abs_param_feature_corr = param_feature_corr.abs()
        print( f'{param_name}-{feature_name} correlation size ', param_feature_corr.size(), f'range of absolute values [{abs_param_feature_corr.min():.3g}, {abs_param_feature_corr.max():.3g}]' )
        param_feature_corr_file = os.path.join(file_directory, f'{param_name}_{feature_name}_corr_{param_string}.pt')
        torch.save(obj=param_feature_corr, f=param_feature_corr_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {param_feature_corr_file}')
        num_subjects, num_deg_freedom = param.size()
        # We are looking for crit_corr such that P(crit_corr) < 1/(2*num_samples*original_one_over_alpha) = adjusted_alpha.
        # To find it, we could sample 2*num_samples*original_one_over_alpha permutations and pick the largest single value.
        # However, this would not be very robust.
        # Instead, we sample 2*num_samples*original_one_over_alpha*sample_buffer_size permutations and pick the largest sample_buffer_size values.
        # The least of these is our critical value.
        adjusted_one_over_alpha = 2 * original_one_over_alpha * num_deg_freedom# 2 because it is a 2-tailed test
        adjusted_alpha = 1.0/adjusted_one_over_alpha
        num_permutations = num_largest_values * adjusted_one_over_alpha
        param_feature_corr_permutations = torch.zeros( size=(num_largest_values, num_deg_freedom), dtype=param_feature_corr.dtype, device=param_feature_corr.device )
        print(f'time {time.time()-code_start_time:.3f}, running {num_permutations} permutations...')
        deg_freedom_indices = torch.arange(end=num_deg_freedom, dtype=int_type, device=param_feature_corr_permutations.device)
        for _ in range(num_permutations):
            permutation = torch.randperm(n=num_subjects, dtype=int_type, device=param.device)
            new_permuted_corrs = isingmodellight.get_pairwise_correlation(mat1=param[permutation,:], mat2=feature, epsilon=0, dim=0).abs()
            # Out of the num_largest_values largest values, pick the lowest one.
            # If it is less than the new value, replace it with the new value.
            # This lowest value is our critical value, since only num_largest_values out of (num_largest_values * adjusted_one_over_alpha) values are greater than or equal to it.
            current_crit_corrs, current_crit_corr_indices = torch.min(param_feature_corr_permutations, dim=0)
            current_crit_corrs = torch.maximum(input=current_crit_corrs, other=new_permuted_corrs)
            param_feature_corr_permutations[current_crit_corr_indices,deg_freedom_indices] = current_crit_corrs
        param_feature_corr_critical = current_crit_corrs
        print( f'time {time.time()-code_start_time:.3f}, {param_name}-{feature_name} critical correlation size ', param_feature_corr_critical.size(), f'range [{param_feature_corr_critical.min():.3g}, {param_feature_corr_critical.max():.3g}]' )
        param_feature_corr_critical_file = os.path.join(file_directory, f'{param_name}_{feature_name}_corr_critical_{param_string}_perms_{num_permutations}_alpha_{adjusted_alpha:.3g}.pt')
        torch.save(obj=param_feature_corr_critical, f=param_feature_corr_critical_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {param_feature_corr_critical_file}')

    # data_string = 'all_mean_std_1'
    # param_string = f'{data_string}_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps_40000'

    def load_mean_params(param_string:str):
        model_file = os.path.join( file_directory, f'ising_model_light_{param_string}.pt' )
        print(f'time {time.time()-code_start_time:.3f}, loading {model_file}...')
        model = torch.load(model_file, weights_only=False)
        print(f'time {time.time()-code_start_time:.3f}, loaded {model_file}')
        num_nodes = model.J.size(dim=-1)
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=num_nodes, device=model.J.device )
        return model.h.mean(dim=0), model.J[:,:,triu_rows,triu_cols].mean(dim=0)

    h, J = load_mean_params(param_string=param_string)
    print( 'h size', h.size() )
    print( 'J size', J.size() )
    
    node_features = torch.load( os.path.join(file_directory, 'node_features_all_as_is.pt'), weights_only=False )[:,:,:4].clone()
    print( 'node features size', node_features.size() )
    num_features = node_features.size(dim=-1)
    for feature_index, feature_name in zip( range(num_features), ['thickness', 'myelination', 'curvature', 'sulcus_depth'] ):
        get_corr_and_crit_corr(param=h, feature=node_features[:,:,feature_index], original_one_over_alpha=original_one_over_alpha, num_largest_values=num_largest_values, param_string=param_string, param_name='h', feature_name=feature_name)
    
    sc = torch.load( os.path.join(file_directory, 'edge_features_all_as_is.pt'), weights_only=False )[:,:,0].clone()# so we can drop the others from memory.
    print( 'SC size', sc.size() )
    get_corr_and_crit_corr(param=J, feature=sc, original_one_over_alpha=original_one_over_alpha, num_largest_values=num_largest_values, param_string=param_string, param_name='J', feature_name='sc')

    def get_data_fc_triu(data_string:str):
        mean_state = torch.load( os.path.join(file_directory, f'mean_state_{data_string}.pt'), weights_only=False ).mean(dim=0)
        print( 'mean state size', mean_state.size() )
        mean_state_product = torch.load( os.path.join(file_directory, f'mean_state_product_{data_string}.pt'), weights_only=False ).mean(dim=0)
        print( 'mean state product size', mean_state_product.size() )
        fc = isingmodellight.get_fc(state_mean=mean_state, state_product_mean=mean_state_product, epsilon=0)
        print( 'fc size', fc.size() )
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=mean_state.size(dim=-1), device=mean_state.device )
        return fc[:,triu_rows,triu_cols]
    fc = get_data_fc_triu(data_string=data_file_name_part)
    print( 'triu fc size', fc.size() )
    get_corr_and_crit_corr(param=fc, feature=sc, original_one_over_alpha=original_one_over_alpha, num_largest_values=num_largest_values, param_string=param_string, param_name='fc', feature_name='sc')

print(f'time {time.time()-code_start_time:.3f}, done')