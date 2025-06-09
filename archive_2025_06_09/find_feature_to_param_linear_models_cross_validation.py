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
    parser.add_argument("-j", "--num_training_subjects", type=int, default=760, help="number of subjects to use when fitting the linear model. We use the remaining for testing.")
    parser.add_argument("-k", "--save_all", action='store_true', default=False, help="Set this flag to save resutls from individual permutations.")
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
    num_training_subjects = args.num_training_subjects
    print(f'num_training_subjects={num_training_subjects}')
    save_all = args.save_all
    print(f'save_all={save_all}')

    def save_and_print(obj:torch.Tensor, file_name_part:str):
        file_name = os.path.join(output_directory, f'{file_name_part}.pt')
        torch.save(obj=obj, f=file_name)
        num_nan = torch.count_nonzero( torch.isnan(obj) )
        print( f'time {time.time()-code_start_time:.3f}, saved {file_name}, size', obj.size(), f'num NaN {num_nan}, min {obj.min():.3g} mean {obj.mean():.3g} max {obj.max():.3g}' )
        return 0
    
    def count_invalid(input:torch.Tensor):
        return torch.count_nonzero(   torch.logical_or(  torch.logical_or( torch.isnan(input), torch.isinf(input) ), torch.abs(input) > 1  )   ).item()
    
    def get_min_max_std(input:torch.Tensor):
        # Find the region or pair where the largest SD over subjects of any feature is smallest.
        return torch.min(  torch.max( torch.std(input, dim=1), dim=-1 ).values  ).item()
    
    def get_min_abs_slope(input:torch.Tensor):
        return torch.min( torch.abs(input[:,0,0]) ).item()
    
    def get_max_abs_slope(input:torch.Tensor):
        return torch.max( torch.abs(input[:,0,0]) ).item()
    
    def do_lstsq_correlation_and_validation(independent:torch.Tensor, independent_name:str, dependent:torch.Tensor, dependent_name:str, num_perms:int):
        num_vars, num_observations, _ = dependent.size()
        num_test_subjects = num_observations - num_training_subjects
        independent = torch.cat(  tensors=( independent, torch.ones_like(dependent) ), dim=-1  )
        lstsq_file_part = f'{independent_name}_{dependent_name}_{model_file_name_part}_perms_{num_perms}'
        
        corr_train = torch.zeros( size=(num_vars, num_perms), dtype=independent.dtype, device=independent.device )
        corr_test = torch.zeros( size=(num_vars, num_perms), dtype=independent.dtype, device=independent.device )
        train_corr_file_part = f'lstsq_corr_train_{num_training_subjects}_{lstsq_file_part}'
        test_corr_file_part = f'lstsq_corr_test_{num_test_subjects}_{lstsq_file_part}'
        
        rmse_train = torch.zeros( size=(num_vars, num_perms), dtype=independent.dtype, device=independent.device )
        rmse_test = torch.zeros( size=(num_vars, num_perms), dtype=independent.dtype, device=independent.device )
        train_rmse_file_part = f'lstsq_rmse_train_{num_training_subjects}_{lstsq_file_part}'
        test_rmse_file_part = f'lstsq_rmse_test_{num_test_subjects}_{lstsq_file_part}'
        
        rsqd_train = torch.zeros( size=(num_vars, num_perms), dtype=independent.dtype, device=independent.device )
        rsqd_test = torch.zeros( size=(num_vars, num_perms), dtype=independent.dtype, device=independent.device )
        train_rsqd_file_part = f'lstsq_rsquared_train_{num_training_subjects}_{lstsq_file_part}'
        test_rsqd_file_part = f'lstsq_rsquared_test_{num_test_subjects}_{lstsq_file_part}'

        for perm_index in range(num_perms):
            perm = torch.randperm(n=num_observations, dtype=int_type, device=dependent.device)
            indep_perm = independent[:,perm,:]
            indep_perm_train = indep_perm[:,:num_training_subjects,:]
            indep_perm_test = indep_perm[:,num_training_subjects:,:]
            dep_perm = dependent[:,perm,:]
            dep_perm_train = dep_perm[:,:num_training_subjects,:]
            dep_perm_test = dep_perm[:,num_training_subjects:,:]
            coeffs = torch.linalg.lstsq(indep_perm_train, dep_perm_train).solution
            train_pred = torch.matmul(input=indep_perm_train, other=coeffs)
            test_pred = torch.matmul(input=indep_perm_test, other=coeffs)

            ctrain = isingmodellight.get_pairwise_correlation( mat1=dep_perm_train, mat2=train_pred, epsilon=epsilon, dim=1 ).squeeze(dim=-1)
            corr_train[:,perm_index] = ctrain
            ctest = isingmodellight.get_pairwise_correlation( mat1=dep_perm_test, mat2=test_pred, epsilon=epsilon, dim=1 ).squeeze(dim=-1)
            corr_test[:,perm_index] = ctest
            rtrain = isingmodellight.get_pairwise_rmse( mat1=dep_perm_train, mat2=train_pred, dim=1 ).squeeze(dim=-1)
            rmse_train[:,perm_index] = rtrain
            rtest = isingmodellight.get_pairwise_rmse( mat1=dep_perm_test, mat2=test_pred, dim=1 ).squeeze(dim=-1)
            rmse_test[:,perm_index] = rtest
            strain = isingmodellight.get_pairwise_r_squared( target=dep_perm_train, prediction=train_pred, dim=1 ).squeeze(dim=-1)
            rsqd_train[:,perm_index] = strain
            stest = isingmodellight.get_pairwise_r_squared( target=dep_perm_test, prediction=test_pred, dim=1 ).squeeze(dim=-1)
            rsqd_test[:,perm_index] = stest

            # print( f'perm {perm_index+1} invalid correlation counts train {count_invalid(ctrain)}, test {count_invalid(ctest)}, train pred min SD {get_min_max_std(train_pred):.3g}, test pred min SD {get_min_max_std(test_pred):.3g}, train dep min SD {get_min_max_std(dep_perm_train):.3g}, test dep min SD {get_min_max_std(dep_perm_test):.3g}, train indep min SD {get_min_max_std(indep_perm_train):.3g}, test indep min SD {get_min_max_std(indep_perm_test):.3g}, min abs slope {get_min_abs_slope(coeffs):.3g}, max abs slope {get_max_abs_slope(coeffs):.3g}' )
        print(f'time {time.time()-code_start_time:.3f}, {dependent_name}-{dependent_name}({independent_name})')
        print(f'train corr min {corr_train.min():.3g} mean {corr_train.mean():.3g} max {corr_train.max():.3g}')
        print(f'test  corr min {corr_test.min():.3g} mean {corr_test.mean():.3g} max {corr_test.max():.3g}')
        if save_all:
            save_and_print(obj=corr_train, file_name_part=train_corr_file_part)
            save_and_print(obj=corr_test, file_name_part=test_corr_file_part)
            save_and_print(obj=rmse_train, file_name_part=train_rmse_file_part)
            save_and_print(obj=rmse_test, file_name_part=test_rmse_file_part)
            save_and_print(obj=rsqd_train, file_name_part=train_rsqd_file_part)
            save_and_print(obj=rsqd_test, file_name_part=test_rsqd_file_part)
        else:
            std_corr_train, mean_corr_train = torch.std_mean(input=corr_train, dim=-1)
            save_and_print(obj=std_corr_train, file_name_part=f'std_{train_corr_file_part}')
            save_and_print(obj=mean_corr_train, file_name_part=f'mean_{train_corr_file_part}')
            std_corr_test, mean_corr_test = torch.std_mean(input=corr_test, dim=-1)
            save_and_print(obj=std_corr_test, file_name_part=f'std_{test_corr_file_part}')
            save_and_print(obj=mean_corr_test, file_name_part=f'mean_{test_corr_file_part}')
            fraction_test_ge = torch.count_nonzero( input=(corr_test >= corr_train), dim=-1 )/num_perms
            save_and_print(obj=fraction_test_ge, file_name_part=f'fraction_ge_{test_corr_file_part}')
        return 0
    
    def get_region_features(num_region_features:int=4):
        # Transpose so that dim 0 is regions and dim 1 is subjects.
        # We will take one correlation over inter-subject differences for each region.
        # clone() so that we can deallocate the larger Tensor of which it is a view.
        # region_features already has a feature dimension, so we do not need to unsqueeze():
        # region or region pair (batch dimension) x subject (observation dimension) x feature (feature dimension)
        region_feature_file_name = os.path.join(data_directory, f'{region_feature_file_name_part}.pt')
        region_features = torch.transpose( input=torch.load(f=region_feature_file_name, weights_only=False)[:,:,:num_region_features], dim0=0, dim1=1 ).clone()
        print( f'time {time.time()-code_start_time:.3f}, loaded {region_feature_file_name} region features size', region_features.size() )
        return region_features
    
    def get_mean_state():
        # Take the mean over scans.
        # Transpose so that dim 0 is regions and dim 1 is subjects.
        # We will take one correlation over inter-subject differences for each region.
        # unsqueeze() because, for least squares regression, both independent and dependent need to have the same number of dimensions:
        # region or region pair (batch dimension) x subject (observation dimension) x feature (feature dimension)
        mean_state_file_name = os.path.join(data_directory, f'mean_state_{data_file_name_part}.pt')
        mean_state = torch.mean( input=torch.load(f=mean_state_file_name, weights_only=False), dim=0 ).transpose(dim0=0, dim1=1).unsqueeze(dim=-1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_file_name} mean state size', mean_state.size() )
        return mean_state
    
    def get_h():
        # Take the mean over replicas.
        # Transpose so that dim 0 is regions and dim 1 is subjects.
        # We will take one correlation over inter-subject differences for each region.
        # unsqueeze() because, for least squares regression, both independent and dependent need to have the same number of dimensions:
        # region or region pair (batch dimension) x subject (observation dimension) x feature (feature dimension)
        model_file_name = os.path.join(data_directory, f'{model_file_name_part}.pt')
        h = torch.mean( input=torch.load(f=model_file_name, weights_only=False).h, dim=0 ).transpose(dim0=0, dim1=1).unsqueeze(dim=-1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name} h size', h.size() )
        return h
    
    def get_sc():
        # Transpose so that dim 0 is region pairs and dim 1 is subjects.
        # We will take one correlation over inter-subject differences for each region pair.
        # clone() so that we can de-allocate the larger Tensor of which this is a view.
        # unsqueeze() because, for least squares regression, both independent and dependent need to have the same number of dimensions:
        # region or region pair (batch dimension) x subject (observation dimension) x feature (feature dimension)
        region_pair_feature_file_name = os.path.join(data_directory, f'{region_pair_feature_file_name_part}.pt')
        sc = torch.transpose( input=torch.load(f=region_pair_feature_file_name, weights_only=False)[:,:,0], dim0=0, dim1=1 ).unsqueeze(dim=-1).clone()
        print( f'time {time.time()-code_start_time:.3f}, loaded {region_pair_feature_file_name} SC size', sc.size() )
        return sc
    
    def get_J():
        # Take the part above the diagonal, and then take the mean over replicas.
        # This gives us a smaller Tensor with which to work.
        # Transpose so that dim 0 is region pairs and dim 1 is subjects.
        # We will take one correlation over inter-subject differences for each region pair.
        # unsqueeze() because, for least squares regression, both independent and dependent need to have the same number of dimensions:
        # region or region pair (batch dimension) x subject (observation dimension) x feature (feature dimension)
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
    
    def do_node_correlation_analyses():
        feature_names = ['thickness', 'myelination', 'curvature', 'sulcus_depth']
        region_features = get_region_features()
        h = get_h()
        mean_state = get_mean_state()
        do_lstsq_correlation_and_validation(independent=region_features, independent_name='all', dependent=h, dependent_name='h', num_perms=num_region_permutations)
        do_lstsq_correlation_and_validation(independent=region_features, independent_name='all', dependent=mean_state, dependent_name='mean_state', num_perms=num_region_permutations)
        num_features = region_features.size(dim=-1)
        for feature_index in range(num_features):
            do_lstsq_correlation_and_validation( independent=region_features[:,:,feature_index].unsqueeze(dim=-1), independent_name=feature_names[feature_index], dependent=h, dependent_name='h', num_perms=num_region_permutations )
            do_lstsq_correlation_and_validation( independent=region_features[:,:,feature_index].unsqueeze(dim=-1), independent_name=feature_names[feature_index], dependent=mean_state, dependent_name='mean_state', num_perms=num_region_permutations )
        return 0
    
    def do_edge_correlation_analyses():
        sc = get_sc()
        do_lstsq_correlation_and_validation( independent=sc, independent_name='SC', dependent=get_J(), dependent_name='J', num_perms=num_region_pair_permutations )
        do_lstsq_correlation_and_validation( independent=sc, independent_name='SC', dependent=get_fc(), dependent_name='FC', num_perms=num_region_pair_permutations )
        return 0
    
    do_node_correlation_analyses()
    do_edge_correlation_analyses()
print('done')