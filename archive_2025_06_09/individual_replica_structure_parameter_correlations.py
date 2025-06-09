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
    parser.add_argument("-e", "--group_fmri_file_name_part", type=str, default='thresholds_31_min_0_max_3', help="the multi-threshold group data mean state and state product file name after mean_state_ or mean_state_product_ and before .pt.")
    parser.add_argument("-f", "--individual_fmri_file_name_part", type=str, default='all_mean_std_1', help="the single-threshold individual data mean state and state product file name after mean_state_ or mean_state_product_ and before .pt.")
    parser.add_argument("-g", "--group_model_file_part", type=str, default='ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000', help='the part of the Ising model file name before .pt.')
    parser.add_argument("-i", "--individual_model_file_part", type=str, default='ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_to_thresh_1_reps_5_subj_837_individual_updates_40000', help='the part of the individual Ising model file name before .pt.')
    parser.add_argument("-j", "--num_training_subjects", type=int, default=670, help="uses the first this many subjects for the group data.")
    parser.add_argument("-k", "--threshold_index", type=int, default=10, help="index of threshold to select out of group model")
    parser.add_argument("-l", "--num_perms_group_node", type=int, default=1000, help="number of permutations to use for group node-wise permutation tests")
    parser.add_argument("-m", "--num_perms_group_pair", type=int, default=1000, help="number of permutations to use for group node-pair-wise permutation tests")
    parser.add_argument("-n", "--num_perms_individual_node", type=int, default=1000, help="number of permutations to use for individual subject-wise permutation tests when we have one for each node")
    parser.add_argument("-o", "--num_perms_individual_pair", type=int, default=100, help="number of permutations to use for individual subject-wise permutation tests when we have one for each node pair")
    parser.add_argument("-p", "--num_perms_train_test_node", type=int, default=1000, help="number of train-test splits to use for individual subject-wise cross-validation tests when we have one correlation for each node")
    parser.add_argument("-q", "--num_perms_train_test_pair", type=int, default=100, help="number of train-test to use for individual subject-wise cross-validation tests when we have one correlation for each node pair")
    parser.add_argument("-r", "--base_alpha_group", type=float, default=0.05, help="alpha to use for node-wise or node-pair-wise permutation tests, which we will bonferroni-correct by the number of replica group models")
    parser.add_argument("-s", "--base_alpha_individual", type=float, default=0.05, help="alpha to use for subject-wise permutation tests, which we will bonferroni-correct by the number of nodes or node-pairs as appropriate")
    parser.add_argument("-t", "--group_model_short_identifier", type=str, default='group_thresholds_31_min_0_max_3', help='abbreviated name for the group model')
    parser.add_argument("-u", "--individual_model_short_identifier", type=str, default='individual_from_group_glasser_1', help='abbreviated name for the individual model')
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
    group_fmri_file_name_part = args.group_fmri_file_name_part
    print(f'group_fmri_file_name_part={group_fmri_file_name_part}')
    individual_fmri_file_name_part = args.individual_fmri_file_name_part
    print(f'individual_fmri_file_name_part={individual_fmri_file_name_part}')
    group_model_file_part = args.group_model_file_part
    print(f'group_model_file_part={group_model_file_part}')
    individual_model_file_part = args.individual_model_file_part
    print(f'individual_model_file_part={individual_model_file_part}')
    num_training_subjects = args.num_training_subjects
    print(f'num_training_subjects={num_training_subjects}')
    threshold_index = args.threshold_index
    print(f'threshold_index={threshold_index}')
    num_perms_group_node = args.num_perms_group_node
    print(f'num_perms_group_node={num_perms_group_node}')
    num_perms_group_pair = args.num_perms_group_pair
    print(f'num_perms_group_pair={num_perms_group_pair}')
    num_perms_individual_node = args.num_perms_individual_node
    print(f'num_perms_individual_node={num_perms_individual_node}')
    num_perms_individual_pair = args.num_perms_individual_pair
    print(f'num_perms_individual_pair={num_perms_individual_pair}')
    num_perms_train_test_node = args.num_perms_train_test_node
    print(f'num_perms_train_test_node={num_perms_train_test_node}')
    num_perms_train_test_pair = args.num_perms_train_test_pair
    print(f'num_perms_train_test_pair={num_perms_train_test_pair}')
    base_alpha_group = args.base_alpha_group
    print(f'base_alpha_group={base_alpha_group}')
    base_alpha_individual = args.base_alpha_individual
    print(f'base_alpha_individual={base_alpha_individual}')
    group_model_short_identifier = args.group_model_short_identifier
    print(f'group_model_short_identifier={group_model_short_identifier}')
    individual_model_short_identifier = args.individual_model_short_identifier
    print(f'individual_model_short_identifier={individual_model_short_identifier}')
    
    def get_node_features():
        # Select out the actual structural features, omitting the region coordinates from the Atlas.
        # Clone so that we do not retain memory of the larger Tensor after exiting the function.
        # region_features has size (num_subjects, num_nodes, num_features).
        node_features_file = os.path.join(data_directory, f'{region_feature_file_part}.pt')
        node_features = torch.clone( torch.load(node_features_file, weights_only=False)[:,:,:4] )
        print( f'time {time.time()-code_start_time:.3f}, loaded {node_features_file}, region features size', node_features.size() )
        return node_features
    
    def get_sc():
        # Select out only the SC.
        # Clone so that we do not retain memory of the larger Tensor after exiting the function.
        # Unsqueeze to get a dimension that aligns with the features dimension of region_features.
        # sc has size (num_subjects, num_pairs, 1).
        sc_file = os.path.join(data_directory, f'{sc_file_part}.pt')
        sc = torch.clone( torch.load(sc_file, weights_only=False)[:,:,0] ).unsqueeze(dim=-1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {sc_file}, SC size', sc.size() )
        return sc

    def get_group_parameters():
        # Take the elements of J above the diagonal.
        # Keep individual replicas.
        # Unsqueeze twice so that we have singleton dimensions that match up with both subject and feature dimensions.
        # Group h has size (num_replicas, num_thresholds, num_nodes).
        # Group J has size (num_replicas, num_thresholds, num_pairs).
        model_file_name = os.path.join(data_directory, f'{group_model_file_part}.pt')
        model = torch.load(f=model_file_name, weights_only=False)
        h = torch.clone(model.h)
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=model.J.size(dim=-1), device=model.J.device )
        J = torch.clone(input=model.J[:,:,triu_rows,triu_cols])
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name} group h size', h.size(), 'group J size', J.size() )
        return h, J
    
    def get_individual_parameters():
        # Take the elements of J above the diagonal.
        # Keep individual replicas.
        # Individual h has size (num_replicas, num_subjects, num_nodes).
        # Individual J has size (num_replicas, num_subjects, num_pairs).
        model_file_name = os.path.join(data_directory, f'{individual_model_file_part}.pt')
        model = torch.load(f=model_file_name, weights_only=False)
        h = torch.clone(input=model.h)
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=model.J.size(dim=-1), device=model.J.device )
        J = torch.clone(input=model.J[:,:,triu_rows,triu_cols])
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name} individual h size', h.size(), 'individual J size', J.size() )
        return h, J
    
    def get_group_mean_state_and_fc():
        # Compute FC from mean state and mean state product.
        # Take the part of FC above the diagonal.
        # Clone so that we do not retain memory of the larger Tensor.
        # unsqueeze() a singleton dimension to align with the replicas dimension of h and J.
        # group_mean_state has size (1, num_thresholds, num_nodes).
        # group_fc has size (1, num_thresholds, num_pairs).
        mean_state_file = os.path.join(data_directory, f'mean_state_{group_fmri_file_name_part}.pt')
        group_mean_state = torch.clone( torch.load(f=mean_state_file, weights_only=False) )
        mean_state_product_file = os.path.join(data_directory, f'mean_state_product_{group_fmri_file_name_part}.pt')
        group_mean_state_product = torch.clone( torch.load(f=mean_state_product_file, weights_only=False) )
        group_fc = isingmodellight.get_fc(state_mean=group_mean_state, state_product_mean=group_mean_state_product, epsilon=epsilon)
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=group_fc.size(dim=-1), device=group_fc.device )
        group_fc = torch.clone(group_fc[:,triu_rows,triu_cols])
        print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_file} and {mean_state_product_file}, group mean state size', group_mean_state.size(), 'group FC size', group_fc.size() )
        return group_mean_state.unsqueeze(dim=0), group_fc.unsqueeze(dim=0)
    
    def get_individual_mean_state_and_fc():
        # Take the mean over scans.
        # Compute FC from mean state and mean state product.
        # Take the part of FC above the diagonal.
        # Clone so that we do not retain memory of the larger Tensor.
        # unsqueeze() a singleton dimension to align with the replicas dimension of h and J.
        # individual_mean_state has size (1, num_subjects, num_nodes).
        # individual_fc has size (1, num_subjects, num_pairs).
        mean_state_file = os.path.join(data_directory, f'mean_state_{individual_fmri_file_name_part}.pt')
        individual_mean_state = torch.mean( input=torch.load(f=mean_state_file, weights_only=False), dim=0 )
        mean_state_product_file = os.path.join(data_directory, f'mean_state_product_{individual_fmri_file_name_part}.pt')
        individual_mean_state_product = torch.mean( input=torch.load(f=mean_state_product_file, weights_only=False), dim=0 )
        individual_fc = isingmodellight.get_fc(state_mean=individual_mean_state, state_product_mean=individual_mean_state_product, epsilon=epsilon)
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=individual_fc.size(dim=-1), device=individual_fc.device )
        individual_fc = torch.clone(individual_fc[:,triu_rows,triu_cols])
        print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_file} and {mean_state_product_file}, individual mean state size', individual_mean_state.size(), 'individual FC size', individual_fc.size() )
        return individual_mean_state.unsqueeze(dim=0), individual_fc.unsqueeze(dim=0)

    def get_fraction_non_nan(m:torch.Tensor):
        return (   torch.count_nonzero(  torch.logical_not( torch.isnan(m) )  )/m.numel()   ).item()
    
    # corrs: the Tensor of correlations, any dimensions
    # feature_name: the name of the feature(s) (thickness, myelination, curvature, sulcus_depth, all, SC)
    # param_name: the name of the parameter (h, mean_state, J, FC)
    # method: 'direct' or 'lstsq'
    # wise: 'region', 'pair'
    # identifier: any additional string to include last, usually the model file name part for h or J, fMRI file name part for mean_state or FC
    def save_corrs(corrs:torch.Tensor, feature_name:str, param_name:str, method:str, wise:str, identifier:str):
        corrs_file_name = os.path.join(output_directory, f'corr_{wise}wise_{method}_{param_name}_{feature_name}_{identifier}.pt')
        torch.save(obj=corrs, f=corrs_file_name)
        fraction_non_nan = get_fraction_non_nan(corrs)
        print( f'time {time.time()-code_start_time:.3f}, saved {corrs_file_name}, min {corrs.min().item():.3f}, mean {corrs.mean().item():.3f}, max {corrs.max().item():.3f}, fraction non-NaN {fraction_non_nan:.3g}' )
        return 0
    
    def save_coeffs(coeffs:torch.Tensor, feature_name:str, param_name:str, method:str, wise:str, identifier:str):
        coeffs_file_name = os.path.join(output_directory, f'coeffs_{wise}wise_{method}_{param_name}_{feature_name}_{identifier}.pt')
        torch.save(obj=coeffs, f=coeffs_file_name)
        fraction_non_nan = get_fraction_non_nan(coeffs)
        print(f'time {time.time()-code_start_time:.3f}, saved {coeffs_file_name}, fraction non-NaN {fraction_non_nan:.3g}')
        return 0
    
    def save_p(p:torch.Tensor, feature_name:str, param_name:str, method:str, wise:str, num_perms:int, identifier:str):
        p_file_name = os.path.join(output_directory, f'p_{method}_{wise}wise_{param_name}_{feature_name}_perms_{num_perms}_{identifier}.pt')
        torch.save(obj=p, f=p_file_name)
        print(f'time {time.time()-code_start_time:.3f}, saved {p_file_name}, min {p.min().item():.3f}, mean {p.mean().item():.3f}, max {p.max().item():.3f}')
        return 0
    
    def save_crit(crit:torch.Tensor, feature_name:str, param_name:str, method:str, wise:str, num_perms:int, alpha:float, identifier:str):
        crit_file_name = os.path.join(output_directory, f'crit_corr_{method}_{wise}wise_{param_name}_{feature_name}_perms_{num_perms}_alpha_{alpha:.3g}_{identifier}.pt')
        torch.save(obj=crit, f=crit_file_name)
        print(f'time {time.time()-code_start_time:.3f}, saved {crit_file_name}, min {crit.min().item():.3f}, mean {crit.mean().item():.3f}, max {crit.max().item():.3f}')
        return 0
    
    def get_p_and_crit(corrs:torch.Tensor, perm_corrs:torch.Tensor, alpha:float, two_way:bool=True, perm_dim:int=-1):
        if two_way:
            corrs = corrs.abs()
            perm_corrs = perm_corrs.abs()
            alpha = alpha/2
        p = torch.count_nonzero(  input=( perm_corrs >= corrs.unsqueeze(dim=perm_dim) ), dim=-1  )/perm_corrs.size(dim=perm_dim)
        crit = torch.quantile(input=perm_corrs, q=alpha, dim=perm_dim, keepdim=False)
        return p, crit
    
    # features and params should have aligned dimensions.
    # features (1, 1, num_nodes/pairs, num_features)
    # params (num_replicas, num_thresholds, num_nodes/pairs, 1)
    # The correlations are along the nodes/pairs dimension.
    # output (num_replicas, num_thresholds, num_features)
    def get_direct_group_correlations(features:torch.Tensor, params:torch.Tensor):
        return isingmodellight.get_pairwise_correlation(mat1=features, mat2=params, epsilon=epsilon, dim=2)
    
    def get_direct_group_correlations_p_and_crit(features:torch.Tensor, params:torch.Tensor, num_perms:int, alpha:float):
        num_replicas, num_thresholds, num_parts, _ = params.size()
        num_features = features.size(dim=-1)
        corrs = get_direct_group_correlations(features=features, params=params)
        perm_corrs = torch.zeros( size=(num_replicas, num_thresholds, num_features, num_perms), dtype=corrs.dtype, device=corrs.device )
        for perm_index in range(num_perms):
            perm = torch.randperm(n=num_parts, dtype=int_type, device=params.device)
            perm_corrs[:,:,:,perm_index] = get_direct_group_correlations( features=features, params=params[:,:,perm,:] )
        p, crit = get_p_and_crit(corrs=corrs, perm_corrs=perm_corrs, alpha=alpha, two_way=True)
        return corrs, p, crit
    
    def save_direct_group_correlations_p_and_crit(features:torch.Tensor, params:torch.Tensor, num_perms:int, alpha:float, feature_name:str, param_name:str, wise:str, identifer:str):
        corrs, p, crit = get_direct_group_correlations_p_and_crit(features=features, params=params, num_perms=num_perms, alpha=alpha)
        method = 'direct'
        save_corrs(corrs=corrs, feature_name=feature_name, param_name=param_name, method=method, wise=wise, identifier=identifer)
        save_p(p=p, feature_name=feature_name, param_name=param_name, method=method, wise=wise, num_perms=num_perms, identifier=identifer)
        save_crit(crit=crit, feature_name=feature_name, param_name=param_name, method=method, wise=wise, num_perms=num_perms, alpha=alpha, identifier=identifer)
        return 0
    
    # features and params should have aligned dimensions.
    # features (1, 1, num_nodes/pairs, num_features+1) (append a dimension of 1s)
    # params (num_replicas, num_thresholds, num_nodes/pairs, 1)
    # The correlations are along the nodes/pairs dimension.
    # We squeeze() coeffs at the end to get rid of the singleton feature dimension.
    # coeffs also has a singleton dimension, but it serves to make the stacked matrices of size num_features x 1.
    # We keep it so that we can use coeffs for future matrix multiplications.
    # coeffs (num_replicas, num_thresholds, num_features+1, 1)
    # corrs (num_replicas, num_thresholds)
    def get_lstsq_group_correlations(features:torch.Tensor, params:torch.Tensor):
        coeffs = torch.linalg.lstsq(features, params).solution
        predictions = torch.matmul(features, coeffs)
        corrs = get_direct_group_correlations(features=predictions, params=params).squeeze(dim=-1)
        return corrs, coeffs
    
    def get_lstsq_group_correlations_p_and_crit(features:torch.Tensor, params:torch.Tensor, num_perms:int, alpha:float):
        num_replicas, num_thresholds, num_parts, _ = params.size()
        features = torch.cat(   tensors=(  features, torch.ones( size=(1, 1, num_parts, 1), dtype=features.dtype, device=features.device )  ), dim=-1   )
        corrs, coeffs = get_lstsq_group_correlations(features=features, params=params)
        perm_corrs = torch.zeros( size=(num_replicas, num_thresholds, num_perms), dtype=corrs.dtype, device=corrs.device )
        for perm_index in range(num_perms):
            perm = torch.randperm(n=num_parts, dtype=int_type, device=params.device)
            perm_corrs[:,:,perm_index], _ = get_lstsq_group_correlations( features=features, params=params[:,:,perm,:] )
        p, crit = get_p_and_crit(corrs=corrs, perm_corrs=perm_corrs, alpha=alpha, two_way=False)
        return corrs, p, crit, coeffs
    
    def save_lstsq_group_correlations_p_and_crit(features:torch.Tensor, params:torch.Tensor, num_perms:int, alpha:float, feature_name:str, param_name:str, wise:str, identifer:str):
        corrs, p, crit, coeffs = get_lstsq_group_correlations_p_and_crit(features=features, params=params, num_perms=num_perms, alpha=alpha)
        method = 'lstsq'
        save_corrs(corrs=corrs, feature_name=feature_name, param_name=param_name, method=method, wise=wise, identifier=identifer)
        save_coeffs(coeffs=coeffs, feature_name=feature_name, param_name=param_name, method=method, wise=wise, identifier=identifer)
        save_p(p=p, feature_name=feature_name, param_name=param_name, method=method, wise=wise, num_perms=num_perms, identifier=identifer)
        save_crit(crit=crit, feature_name=feature_name, param_name=param_name, method=method, wise=wise, num_perms=num_perms, alpha=alpha, identifier=identifer)
        return coeffs
    
    # features is originally (num_nodes, num_features).
    # params is originally (num_replicas, num_thresholds, num_nodes)
    def save_group_corrs(features:torch.Tensor, params:torch.Tensor, num_perms:int, alpha:float, feature_name:str, param_name:str, wise:str, identifier:str):
        features = features.unsqueeze(dim=0).unsqueeze(dim=0)# Make singleton dimensions to align with replicas and thresholds.
        params = params.unsqueeze(dim=-1)# Make a singleton dimension to align with features.
        save_direct_group_correlations_p_and_crit(features=features, params=params, num_perms=num_perms, alpha=alpha, feature_name=feature_name, param_name=param_name, wise=wise, identifer=identifier)
        coeffs = save_lstsq_group_correlations_p_and_crit(features=features, params=params, num_perms=num_perms, alpha=alpha, feature_name=feature_name, param_name=param_name, wise=wise, identifer=identifier)
        return coeffs
    
    # features and params should have aligned dimensions.
    # features (1, num_subjects, num_nodes/pairs, num_features)
    # params (num_replicas, num_subjects, num_nodes/pairs, 1)
    # The correlations are along the subjects dimension.
    def get_direct_individual_correlations(features:torch.Tensor, params:torch.Tensor):
        return isingmodellight.get_pairwise_correlation(mat1=features, mat2=params, epsilon=epsilon, dim=1)
    
    def get_direct_individual_correlations_p_and_crit(features:torch.Tensor, params:torch.Tensor, num_perms:int, alpha:float):
        num_replicas, num_subjects, num_parts, _ = params.size()
        num_features = features.size(dim=-1)
        corrs = get_direct_individual_correlations(features=features, params=params)
        perm_corrs = torch.zeros( size=(num_replicas, num_parts, num_features, num_perms), dtype=corrs.dtype, device=corrs.device )
        for perm_index in range(num_perms):
            perm = torch.randperm(n=num_subjects, dtype=int_type, device=params.device)
            perm_corrs[:,:,:,perm_index] = get_direct_individual_correlations(features=features, params=params[:,perm,:,:])
        p, crit = get_p_and_crit(corrs=corrs, perm_corrs=perm_corrs, alpha=alpha, two_way=True)
        return corrs, p, crit
    
    def save_direct_individual_correlations_p_and_crit(features:torch.Tensor, params:torch.Tensor, num_perms:int, alpha:float, feature_name:str, param_name:str, identifer:str):
        corrs, p, crit = get_direct_individual_correlations_p_and_crit(features=features, params=params, num_perms=num_perms, alpha=alpha)
        method = 'direct'
        wise = 'subject'
        save_corrs(corrs=corrs, feature_name=feature_name, param_name=param_name, method=method, wise=wise, identifier=identifer)
        save_p(p=p, feature_name=feature_name, param_name=param_name, method=method, wise=wise, num_perms=num_perms, identifier=identifer)
        save_crit(crit=crit, feature_name=feature_name, param_name=param_name, method=method, wise=wise, num_perms=num_perms, alpha=alpha, identifier=identifer)
        return 0
    
    # features and params should have aligned dimensions.
    # features (1, num_nodes/pairs, num_subjects, num_features+1) (append a dimension of 1s)
    # params (num_replicas, num_nodes/pairs, num_subjects, 1)
    # The correlations are along the subjects dimension.
    # squeeze() out the singleton feature dimension of correlations.
    # Keep the singleton dimension of coeffs, which makes it a stack of num_features+1 x 1 matrices.
    # We need this dimension in order to do more matmul() calls with it.
    # corrs (num_replicas, num_nodes/pairs)
    # coeffs (num_replicas, num_nodes/pairs, num_features+1, 1)
    def get_lstsq_individual_correlations(features:torch.Tensor, params:torch.Tensor):
        coeffs = torch.linalg.lstsq(features, params).solution
        predictions = torch.matmul(features, coeffs)
        corrs = isingmodellight.get_pairwise_correlation(mat1=predictions, mat2=params, epsilon=epsilon, dim=2).squeeze(dim=-1)
        return corrs, coeffs
    
    def get_lstsq_individual_correlations_p_and_crit(features:torch.Tensor, params:torch.Tensor, num_perms:int, alpha:float):
        params = params.transpose(dim0=1, dim1=2)
        num_replicas, num_parts, num_subjects, _ = params.size()
        features = torch.cat(   tensors=(  features.transpose(dim0=1, dim1=2), torch.ones( size=(1, num_parts, num_subjects, 1), dtype=features.dtype, device=features.device )  ), dim=-1   )
        corrs, coeffs = get_lstsq_individual_correlations(features=features, params=params)
        perm_corrs = torch.zeros( size=(num_replicas, num_parts, num_perms), dtype=corrs.dtype, device=corrs.device )
        for perm_index in range(num_perms):
            perm = torch.randperm(n=num_subjects, dtype=int_type, device=params.device)
            perm_corrs[:,:,perm_index], _ = get_lstsq_individual_correlations(features=features, params=params[:,:,perm,:])
        p, crit = get_p_and_crit(corrs=corrs, perm_corrs=perm_corrs, alpha=alpha, two_way=False)
        return corrs, p, crit, coeffs
    
    def save_lstsq_individual_correlations_p_and_crit(features:torch.Tensor, params:torch.Tensor, num_perms:int, alpha:float, feature_name:str, param_name:str, identifer:str):
        corrs, p, crit, coeffs = get_lstsq_individual_correlations_p_and_crit(features=features, params=params, num_perms=num_perms, alpha=alpha)
        method = 'lstsq'
        wise = 'subject'
        save_corrs(corrs=corrs, feature_name=feature_name, param_name=param_name, method=method, wise=wise, identifier=identifer)
        save_coeffs(coeffs=coeffs, feature_name=feature_name, param_name=param_name, method=method, wise=wise, identifier=identifer)
        save_p(p=p, feature_name=feature_name, param_name=param_name, method=method, wise=wise, num_perms=num_perms, identifier=identifer)
        save_crit(crit=crit, feature_name=feature_name, param_name=param_name, method=method, wise=wise, num_perms=num_perms, alpha=alpha, identifier=identifer)
        return corrs
    
    def get_lstsq_individual_correlations_train_and_test(features:torch.Tensor, params:torch.Tensor, num_perms:int, num_training_subjects:int):
        params = params.transpose(dim0=1, dim1=2)
        num_replicas, num_parts, num_subjects, _ = params.size()
        features = torch.cat(   tensors=(  features.transpose(dim0=1, dim1=2), torch.ones( size=(1, num_parts, num_subjects, 1), dtype=features.dtype, device=features.device )  ), dim=-1   )
        train_corrs = torch.zeros( size=(num_replicas, num_parts, num_perms), dtype=features.dtype, device=features.device )
        test_corrs = torch.zeros( size=(num_replicas, num_parts, num_perms), dtype=features.dtype, device=features.device )
        for perm_index in range(num_perms):
            perm = torch.randperm(n=num_subjects, dtype=int_type, device=params.device)
            features_perm = features[:,:,perm,:]
            features_train = features_perm[:,:,:num_training_subjects,:]
            features_test = features_perm[:,:,num_training_subjects:,:]
            params_perm = params[:,:,perm,:]
            params_train = params_perm[:,:,:num_training_subjects,:]
            params_test = params_perm[:,:,num_training_subjects:,:]
            train_corrs[:,:,perm_index], coeffs = get_lstsq_individual_correlations(features=features_train, params=params_train)
            predictions = torch.matmul(features_test, coeffs)
            test_corrs[:,:,perm_index] = isingmodellight.get_pairwise_correlation(mat1=predictions, mat2=params_test, epsilon=epsilon, dim=2).squeeze(dim=-1)
        return train_corrs, test_corrs
    
    def save_lstsq_individual_correlations_train_and_test(features:torch.Tensor, params:torch.Tensor, num_perms:int, feature_name:str, param_name:str, identifer:str):
        train_corrs, test_corrs = get_lstsq_individual_correlations_train_and_test(features=features, params=params, num_perms=num_perms, num_training_subjects=num_training_subjects)
        method = 'lstsq'
        wise = 'subject'
        num_testing_subjects = params.size(dim=1) - num_training_subjects
        save_corrs(corrs=train_corrs, feature_name=feature_name, param_name=param_name, method=method, wise=wise, identifier=f'train_subj_{num_training_subjects}_perms_{num_perms}_{identifer}')
        save_corrs(corrs=test_corrs, feature_name=feature_name, param_name=param_name, method=method, wise=wise, identifier=f'test_subj_{num_testing_subjects}_perms_{num_perms}_{identifer}')
        return 0
    
    # features is (1, num_subjects, num_nodes/pairs, num_features).
    # params is (num_replicas, num_subjects, num_nodes/pairs, 1).
    # coeffs is (num_replicas, num_thresholds, num_features, 1).
    # threshold_index tells us which threshold to select out of coeffs.
    # Correlations are over nodes/pairs.
    # We squeeze() out the singleton feature dimension.
    # corrs (num_replicas, num_subjects)
    def apply_group_coeffs_to_individuals(features:torch.Tensor, params:torch.Tensor, coeffs:torch.Tensor):
        # Select the group models for the same threshold as the individual models.
        # Replace the threshold dimension with a singleton to align with the individual model subject dimension.
        coeffs = coeffs[:,threshold_index,:,:].unsqueeze(dim=1)
        singleton, num_subjects, num_parts, _ = features.size()
        features = torch.cat(   tensors=(  features, torch.ones( size=(singleton, num_subjects, num_parts, 1), dtype=features.dtype, device=features.device )  ), dim=-1   )
        predictions = torch.matmul(features, coeffs)
        # print( 'features', features.size(), 'params', params.size(), 'predictions', predictions.size() )
        # predictions (num_reps_group, num_subjects, num_parts, 1) -> (1, num_reps_group, num_subjects, num_parts, 1)
        # predictions (num_reps_individual, num_subjects, num_parts, 1) -> (num_reps_individual, 1, num_subjects, num_parts, 1)
        # corrs (num_reps_individual, num_reps_group, num_subjects, 1) -> (num_reps_individual, num_reps_group, num_subjects)
        corrs = isingmodellight.get_pairwise_correlation( mat1=predictions.unsqueeze(dim=0), mat2=params.unsqueeze(dim=1), epsilon=epsilon, dim=2 ).squeeze(dim=-1)
        return corrs
    
    def save_group_to_individual_corrs(features:torch.Tensor, params:torch.Tensor, coeffs:torch.Tensor, feature_name:str, param_name:str, wise:str, identifier:str):
        corrs = apply_group_coeffs_to_individuals(features=features, params=params, coeffs=coeffs)
        save_corrs(corrs=corrs, feature_name=feature_name, param_name=param_name, wise=wise, method='lstsq', identifier=f'group_to_indi_{identifier}')
        return 0
    
    # features is originally (num_subjects, num_nodes/pairs, num_features).
    # params is originally (num_replicas, num_subjects, num_nodes/pairs).
    def save_individual_corrs(features:torch.Tensor, params:torch.Tensor, group_coeffs:torch.Tensor, num_perms_p_value:int, num_perms_train_test:int, alpha:float, feature_name:str, param_name:str, wise:str, identifier:str):
        features = features.unsqueeze(dim=0)# Make a singleton dimension to align with replicas.
        params = params.unsqueeze(dim=-1)# Make a singleton dimension to align with features.
        save_direct_individual_correlations_p_and_crit(features=features, params=params, num_perms=num_perms_p_value, alpha=alpha, feature_name=feature_name, param_name=param_name, identifer=identifier)
        individidual_lstsq_corrs = save_lstsq_individual_correlations_p_and_crit(features=features, params=params, num_perms=num_perms_p_value, alpha=alpha, feature_name=feature_name, param_name=param_name, identifer=identifier)
        save_lstsq_individual_correlations_train_and_test(features=features, params=params, num_perms=num_perms_train_test, feature_name=feature_name, param_name=param_name, identifer=identifier)
        save_group_to_individual_corrs(features=features, params=params, coeffs=group_coeffs, feature_name=feature_name, param_name=param_name, wise=wise, identifier=identifier)
        return individidual_lstsq_corrs
    
    # features is (1, num_nodes/pairs, num_features).
    # params is (num_replicas, num_nodes/pairs, 1).
    # correlations are over nodes/pairs.
    # corr is (num_replicas, num_features).
    def get_presliced_partwise_correlations(features:torch.Tensor, params:torch.Tensor):
        return isingmodellight.get_pairwise_correlation(mat1=features, mat2=params, dim=1)
    
    # features is originally (num_nodes/pairs, num_features).
    # params is originally (num_replicas, num_nodes/pairs).
    def get_presliced_partwise_correlations_p_and_crit(features:torch.Tensor, params:torch.Tensor, num_perms:int, alpha:float):
        num_replicas, num_parts = params.size()
        params = params.unsqueeze(dim=-1)# unsqueeze a dimension to align with features
        num_features = features.size(dim=-1)
        features = features.unsqueeze(dim=0)# unsqueeze a dimension to align with replicas
        corrs = get_presliced_partwise_correlations(features=features, params=params)
        perm_corrs = torch.zeros( size=(num_replicas, num_features, num_perms), dtype=corrs.dtype, device=corrs.device )
        for perm_index in range(num_perms):
            perm = torch.randperm(n=num_parts, dtype=int_type, device=params.device)
            perm_corrs[:,:,perm_index] = get_presliced_partwise_correlations(features=features, params=params[:,perm,:])
        p, crit = get_p_and_crit(corrs=corrs, perm_corrs=perm_corrs, alpha=alpha, two_way=True)
        return corrs, p, crit
    
    def save_presliced_partwise_correlations_p_and_crit(features:torch.Tensor, params:torch.Tensor, num_perms:int, alpha:float, feature_name:str, param_name:str, wise:str, identifer:str):
        corrs, p, crit = get_presliced_partwise_correlations_p_and_crit(features=features, params=params, num_perms=num_perms, alpha=alpha)
        method = 'direct'
        save_corrs(corrs=corrs, feature_name=feature_name, param_name=param_name, method=method, wise=wise, identifier=identifer)
        save_p(p=p, feature_name=feature_name, param_name=param_name, method=method, wise=wise, num_perms=num_perms, identifier=identifer)
        save_crit(crit=crit, feature_name=feature_name, param_name=param_name, method=method, wise=wise, num_perms=num_perms, alpha=alpha, identifier=identifer)
        return 0
    
    def save_std_mean_correlations_p_and_crit(features:torch.Tensor, params:torch.Tensor, individual_lstsq_corrs:torch.Tensor, num_perms:int, alpha:int, feature_name:str, param_name:str, lstsq_feature_name:str, wise:str, identifier:str):
        std_features, mean_features = torch.std_mean(input=features, dim=0)
        std_params, mean_params = torch.std_mean(input=params, dim=1)
        std_feature_name = f'std_{feature_name}'
        mean_feature_name = f'mean_{feature_name}'
        std_param_name = f'std_{param_name}'
        save_presliced_partwise_correlations_p_and_crit(features=std_features, params=std_params, num_perms=num_perms, alpha=alpha, feature_name=std_feature_name, param_name=std_param_name, wise=wise, identifer=identifier)
        save_presliced_partwise_correlations_p_and_crit(features=mean_features, params=std_params, num_perms=num_perms, alpha=alpha, feature_name=mean_feature_name, param_name=std_param_name, wise=wise, identifer=identifier)
        mean_param_name = f'mean_{param_name}'
        save_presliced_partwise_correlations_p_and_crit(features=std_features, params=mean_params, num_perms=num_perms, alpha=alpha, feature_name=std_feature_name, param_name=mean_param_name, wise=wise, identifer=identifier)
        save_presliced_partwise_correlations_p_and_crit(features=mean_features, params=mean_params, num_perms=num_perms, alpha=alpha, feature_name=mean_feature_name, param_name=mean_param_name, wise=wise, identifer=identifier)
        lstsq_corr_name = f'indi_lstsq_corr_{param_name}_{lstsq_feature_name}'
        save_presliced_partwise_correlations_p_and_crit(features=std_features, params=individual_lstsq_corrs, num_perms=num_perms, alpha=alpha, feature_name=std_feature_name, param_name=lstsq_corr_name, wise=wise, identifer=identifier)
        save_presliced_partwise_correlations_p_and_crit(features=mean_features, params=individual_lstsq_corrs, num_perms=num_perms, alpha=alpha, feature_name=mean_feature_name, param_name=lstsq_corr_name, wise=wise, identifer=identifier)
        return 0
    
    # features (num_subjects, num_nodes/pairs, num_features)
    # group_params (num_replicas_group, num_thresholds, num_nodes/pairs)
    # individual_params (num_replicas_individual, num_subjects, num_nodes/pairs)
    def save_corrs_feature_and_param(features:torch.Tensor, group_params:torch.Tensor, individual_params:torch.Tensor, num_perms_group:int, num_perms_individual:int, num_perms_train_test:int, alpha_group:float, alpha_individual:float, feature_name:str, param_name:str, node_or_pair_wise:str, group_identifier:str, individual_identifier:str):
        group_coeffs = save_group_corrs( features=features.mean(dim=0), params=group_params, num_perms=num_perms_group, alpha=alpha_group, feature_name=f'mean_{feature_name}', param_name=f'group_{param_name}', wise=node_or_pair_wise, identifier=individual_identifier )
        individidual_lstsq_corrs = save_individual_corrs(features=features, params=individual_params, group_coeffs=group_coeffs, num_perms_p_value=num_perms_individual, num_perms_train_test=num_perms_train_test, alpha=alpha_individual, feature_name=feature_name, param_name=param_name, wise=node_or_pair_wise, identifier=group_identifier)
        save_std_mean_correlations_p_and_crit(features=features, params=individual_params, individual_lstsq_corrs=individidual_lstsq_corrs, num_perms=num_perms_group, alpha=alpha_group, feature_name=feature_name, param_name=param_name, lstsq_feature_name=feature_name, wise=node_or_pair_wise, identifier=individual_identifier)
        return 0
    
    def save_corrs_node_and_pair(node_features:torch.Tensor, sc:torch.Tensor, group_node_params:torch.Tensor, group_pair_params:torch.Tensor, individual_node_params:torch.Tensor, individual_pair_params:torch.Tensor, node_param_name:str, pair_param_name:str, group_identifier:str, individual_identifier:str):
        num_group_replicas = group_node_params.size(dim=0)
        corrected_alpha_group = base_alpha_group/num_group_replicas
        num_individual_replicas = individual_node_params.size(dim=0)
        num_nodes = individual_node_params.size(dim=-1)
        num_pairs = individual_pair_params.size(dim=-1)
        corrected_alpha_individual_node = base_alpha_individual/(num_individual_replicas*num_nodes)
        corrected_alpha_individual_pair = base_alpha_individual/(num_individual_replicas*num_pairs)
        save_corrs_feature_and_param(features=node_features, group_params=group_node_params, individual_params=individual_node_params, num_perms_group=num_perms_group_node, num_perms_individual=num_perms_individual_node, num_perms_train_test=num_perms_train_test_node, alpha_group=corrected_alpha_group, alpha_individual=corrected_alpha_individual_node, feature_name='node_features', param_name=node_param_name, node_or_pair_wise='node', group_identifier=group_identifier, individual_identifier=individual_identifier)
        save_corrs_feature_and_param(features=sc, group_params=group_pair_params, individual_params=individual_pair_params, num_perms_group=num_perms_group_pair, num_perms_individual=num_perms_individual_pair, num_perms_train_test=num_perms_train_test_pair, alpha_group=corrected_alpha_group, alpha_individual=corrected_alpha_individual_pair, feature_name='SC', param_name=pair_param_name, node_or_pair_wise='pair', group_identifier=group_identifier, individual_identifier=individual_identifier)
        return 0
    
    def save_corrs_h_and_J(node_features:torch.Tensor, sc:torch.Tensor):
        group_h, group_J = get_group_parameters()
        individual_h, individual_J = get_individual_parameters()
        save_corrs_node_and_pair(node_features=node_features, sc=sc, group_node_params=group_h, group_pair_params=group_J, individual_node_params=individual_h, individual_pair_params=individual_J, node_param_name='h', pair_param_name='J', group_identifier=group_model_short_identifier, individual_identifier=individual_model_short_identifier)
        return 0
    
    def save_corrs_mean_state_and_fc(node_features:torch.Tensor, sc:torch.Tensor):
        group_mean_state, group_fc = get_group_mean_state_and_fc()
        individual_mean_state, individual_fc = get_individual_mean_state_and_fc()
        save_corrs_node_and_pair(node_features=node_features, sc=sc, group_node_params=group_mean_state, group_pair_params=group_fc, individual_node_params=individual_mean_state, individual_pair_params=individual_fc, node_param_name='mean_state', pair_param_name='FC', group_identifier=group_fmri_file_name_part, individual_identifier=individual_fmri_file_name_part)
        return 0
    
    def save_all():
        node_features = get_node_features()
        sc = get_sc()
        save_corrs_h_and_J(node_features=node_features, sc=sc)
        save_corrs_mean_state_and_fc(node_features=node_features, sc=sc)
        return 0
    
with torch.no_grad():
    save_all()

print('done')