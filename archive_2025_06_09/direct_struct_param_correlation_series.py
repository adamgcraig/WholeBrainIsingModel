import os
import torch
from scipy import stats
import time
import argparse
import math
import isingmodellight
from isingmodellight import IsingModelLight

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')
# Set epsilon to a small non-0 number to prevent NaNs in correlations.
# The corelations may still be nonsense values.
epsilon = 0.0

parser = argparse.ArgumentParser(description="Find correlations between Ising model parameters and structural features using least-squares regression.")
parser.add_argument("-a", "--data_directory", type=str, default='D:\\ising_model_results_daai', help="directory where we can find the target mean state product files")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
parser.add_argument("-c", "--region_feature_file_part", type=str, default='node_features_all_as_is', help='region feature file name except for the .pt file extension')
parser.add_argument("-d", "--sc_file_part", type=str, default='edge_features_all_as_is', help='SC file name except for the .pt file extension')
parser.add_argument("-j", "--group_model_file_part", type=str, default='ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates', help='the part of the Ising model file name before .pt.')
parser.add_argument("-k", "--individual_model_file_part", type=str, default='group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_to_thresh_1_reps_5_subj_837_individual_updates', help='the part of the individual Ising model file name before .pt.')
parser.add_argument("-l", "--num_training_regions", type=int, default=288, help="uses the first this many regions for the train-test splits for group correlations.")
parser.add_argument("-m", "--num_training_subjects", type=int, default=670, help="uses the first this many subjects for the train-test splits for individual correlations.")
parser.add_argument("-s", "--num_perms_train_test_node", type=int, default=1000, help="number of train-test splits to use for individual subject-wise cross-validation tests when we have one correlation for each node")
parser.add_argument("-t", "--num_perms_train_test_pair", type=int, default=1000, help="number of train-test to use for individual subject-wise cross-validation tests when we have one correlation for each node pair")
parser.add_argument("-w", "--group_model_short_identifier", type=str, default='group_thresholds_31_min_0_max_3', help='abbreviated name for the group model')
parser.add_argument("-x", "--individual_model_short_identifier", type=str, default='individual_from_group_glasser_1', help='abbreviated name for the individual model')
parser.add_argument("-y", "--update_increment_group", type=int, default=1000, help="number of updates between models to test")
parser.add_argument("-z", "--min_updates_group", type=int, default=0, help="first number of updates to test")
parser.add_argument("-0", "--max_updates_group", type=int, default=3000, help="last number of updates to test")
parser.add_argument("-1", "--update_increment_individual", type=int, default=1000, help="number of updates between models to test")
parser.add_argument("-2", "--min_updates_individual", type=int, default=1000, help="first number of updates to test")
parser.add_argument("-3", "--max_updates_individual", type=int, default=33000, help="last number of updates to test")
parser.add_argument("-4", "--sim_length", type=int, default=120000, help="length of sim tests used for FC correlations, used to construct the file names")
parser.add_argument("-5", "--num_thresholds", type=int, default=31, help="number of thresholds used for group models")
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
num_training_regions= args.num_training_regions
print(f'num_training_regions={num_training_regions}')
num_training_subjects = args.num_training_subjects
print(f'num_training_subjects={num_training_subjects}')
num_perms_train_test_node = args.num_perms_train_test_node
print(f'num_perms_train_test_node={num_perms_train_test_node}')
num_perms_train_test_pair = args.num_perms_train_test_pair
print(f'num_perms_train_test_pair={num_perms_train_test_pair}')
group_model_short_identifier = args.group_model_short_identifier
print(f'group_model_short_identifier={group_model_short_identifier}')
individual_model_short_identifier = args.individual_model_short_identifier
print(f'individual_model_short_identifier={individual_model_short_identifier}')
update_increment_group = args.update_increment_group
print(f'update_increment_group={update_increment_group}')
min_updates_group = args.min_updates_group
print(f'min_updates_group={min_updates_group}')
max_updates_group = args.max_updates_group
print(f'max_updates_group={max_updates_group}')
update_increment_individual = args.update_increment_individual
print(f'update_increment_individual={update_increment_individual}')
min_updates_individual = args.min_updates_individual
print(f'min_updates_individual={min_updates_individual}')
max_updates_individual = args.max_updates_individual
print(f'max_updates_individual={max_updates_individual}')
sim_length = args.sim_length
print(f'sim_length={sim_length}')
num_thresholds = args.num_thresholds
print(f'num_thresholds={num_thresholds}')

def get_fraction_non_nan(m:torch.Tensor):
        return (   torch.count_nonzero(  torch.logical_not( torch.isnan(m) )  )/m.numel()   ).item()
    
# corrs: the Tensor of correlations, any dimensions
# feature_name: the name of the feature(s) (thickness, myelination, curvature, sulcus_depth, all, SC)
# param_name: the name of the parameter (h, mean_state, J, FC)
# method: 'direct' or 'lstsq'
# wise: 'region', 'pair'
# identifier: any additional string to include last, usually the model file name part for h or J, fMRI file name part for mean_state or FC
def save_corrs(corrs:torch.Tensor, feature_name:str, param_name:str, method:str, wise:str, identifier:str):
        corrs_file_name = os.path.join(output_directory, f'{method}_{wise}wise_{param_name}_{feature_name}_{identifier}.pt')
        torch.save(obj=corrs, f=corrs_file_name)
        fraction_non_nan = get_fraction_non_nan(corrs)
        print( f'time {time.time()-code_start_time:.3f}, saved {corrs_file_name}, min {corrs.min().item():.3f}, mean {corrs.mean().item():.3f}, max {corrs.max().item():.3f}, fraction non-NaN {fraction_non_nan:.3g}' )
        return 0

def append_ones(m:torch.Tensor):
    num_thresholds, num_parts, _ = m.size()
    return torch.cat(   (  m, torch.ones( size=(num_thresholds, num_parts, 1), dtype=m.dtype, device=m.device )  ), dim=-1   )

def z_score_after_split(training:torch.Tensor, testing:torch.Tensor):
     training_std, training_mean = torch.std_mean(training, dim=1, keepdim=True)
     training_z = (training - training_mean)/training_std
     testing_z = (testing - training_mean)/training_std
     return training_z, testing_z

def z_score_after_split_keep_1s(training:torch.Tensor, testing:torch.Tensor):
     training_std, training_mean = torch.std_mean(training, dim=1, keepdim=True)
     training_z = (training - training_mean)/training_std
     training_z[:,:,-1] = 1.0
     testing_z = (testing - training_mean)/training_std
     testing_z[:,:,-1] = 1.0
     return training_z, testing_z

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

def get_model_parameters(model_file_name_part:str, goodness_file_name_part:str):
        # Select the best replica for each threshold/subject.
        # Take the elements of J above the diagonal.
        # h_best has size (num_thresholds/subjects, num_nodes).
        # J_best has size (num_thresholds/subjects, num_pairs).
        goodness_file_name = os.path.join(data_directory, f'{goodness_file_name_part}.pt')
        goodness = torch.load(f=goodness_file_name, weights_only=False)
        goodness[torch.isnan(goodness)] = -1.0*torch.inf# Avoid selecting ones with NaN goodness.
        print( f'time {time.time()-code_start_time:.3f}, loaded {goodness_file_name} size', goodness.size(), f'min {torch.min(goodness):.3g}, mean {torch.mean(goodness):.3g}, max {torch.max(goodness):.3g}' )
        max_goodness, max_goodness_index = torch.max(input=goodness, dim=0)
        print( f'time {time.time()-code_start_time:.3f}, max goodness over replicas min {torch.min(max_goodness):.3g}, mean {torch.mean(max_goodness):.3g}, max {torch.max(max_goodness):.3g}' )
        model_file_name = os.path.join(data_directory, f'{model_file_name_part}.pt')
        model = torch.load(f=model_file_name, weights_only=False)
        h_best = torch.zeros_like(model.h[0,:,:])
        J_best = torch.zeros_like(model.J[0,:,:,:])
        num_subjects = max_goodness_index.numel()
        for subject_index in range(num_subjects):
            best_index_for_subject = max_goodness_index[subject_index]
            h_best[subject_index,:] = model.h[best_index_for_subject,subject_index,:]
            J_best[subject_index,:,:] = model.J[best_index_for_subject,subject_index,:,:]
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=J_best.size(dim=-1), device=J_best.device )
        h_best = h_best.unsqueeze(dim=-1)
        J_best = torch.clone( input=J_best[:,triu_rows,triu_cols].unsqueeze(dim=-1) )
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name} h size', h_best.size(), ' J size', J_best.size() )
        return h_best, J_best

def get_group_parameters(num_updates:int):
        # ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000
        group_model_file_part_with_updates = f'{group_model_file_part}_{num_updates}'
        # fc_corr_ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_test_length_120000
        goodness_file_name_part = os.path.join(data_directory, f'fc_corr_{group_model_file_part_with_updates}_test_length_{sim_length}')
        h_group, J_group = get_model_parameters(model_file_name_part=group_model_file_part_with_updates, goodness_file_name_part=goodness_file_name_part)
        return h_group, J_group
    
def get_individual_parameters(num_updates:int):
        # ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_to_thresh_1_reps_5_subj_837_individual_updates_40000
        individual_model_file_part_with_updates = f'ising_model_light_{individual_model_file_part}_{num_updates}'
        # fc_corr_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_to_thresh_1_reps_5_subj_837_individual_updates_40000_test_length_120000
        goodness_file_name_part = os.path.join(data_directory, f'fc_corr_{individual_model_file_part}_{num_updates}_test_length_{sim_length}')
        h_individual, J_individual = get_model_parameters(model_file_name_part=individual_model_file_part_with_updates, goodness_file_name_part=goodness_file_name_part)
        return h_individual, J_individual

def get_lstsq_group_correlations_train_test(features:torch.Tensor, params:torch.Tensor, num_perms:int):
        num_thresholds, num_parts, _ = params.size()
        corrs_train = torch.zeros( size=(num_thresholds, num_perms), dtype=features.dtype, device=features.device )
        corrs_test = torch.zeros( size=(num_thresholds, num_perms), dtype=features.dtype, device=features.device )
        rmses_train = torch.zeros( size=(num_thresholds, num_perms), dtype=features.dtype, device=features.device )
        rmses_test = torch.zeros( size=(num_thresholds, num_perms), dtype=features.dtype, device=features.device )
        for perm_index in range(num_perms):
            permutation = torch.randperm(n=num_parts, dtype=int_type, device=features.device)
            indices_train = permutation[:num_training_regions]
            indices_test = permutation[num_training_regions:]
            features_train = features[:,indices_train,:]
            params_train = params[:,indices_train,:]
            features_test = features[:,indices_test,:]
            params_test = params[:,indices_test,:]
            features_train, features_test = z_score_after_split_keep_1s(training=features_train, testing=features_test)
            params_train, params_test = z_score_after_split(training=params_train, testing=params_test)
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
            rmses_train[:,perm_index] = isingmodellight.get_pairwise_rmse(mat1=params_train, mat2=predictions_train, dim=1).squeeze(dim=-1)
            rmses_test[:,perm_index] = isingmodellight.get_pairwise_rmse(mat1=params_test, mat2=predictions_test, dim=1).squeeze(dim=-1)
        return corrs_train, corrs_test, rmses_train, rmses_test

def get_lstsq_group_correlations(features:torch.Tensor, params:torch.Tensor, num_perms:int):
        num_thresholds, num_parts, _ = params.size()
        corrs = torch.zeros( size=(num_thresholds, num_perms), dtype=features.dtype, device=features.device )
        for perm_index in range(num_perms):
            permutation = torch.randperm(n=num_parts, dtype=int_type, device=features.device)
            indices_train = permutation[:num_training_regions]
            indices_test = permutation[num_training_regions:]
            features_train = features[:,indices_train,:]
            params_train = params[:,indices_train,:]
            features_test = features[:,indices_test,:]
            params_test = params[:,indices_test,:]
            features_train, features_test = z_score_after_split_keep_1s(training=features_train, testing=features_test)
            params_train, params_test = z_score_after_split(training=params_train, testing=params_test)
            coeffs = torch.linalg.lstsq(features_train, params_train).solution
            predictions_train = torch.matmul(features_train, coeffs)
            corrs[:,perm_index] = isingmodellight.get_pairwise_correlation(mat1=params_train, mat2=predictions_train, epsilon=epsilon, dim=1).squeeze(dim=-1)
        return corrs
    
def save_direct_group_correlations_train_test(features:torch.Tensor, params:torch.Tensor, feature_name:str, param_name:str, wise:str, identifer:str):
    corrs = isingmodellight.get_pairwise_correlation(mat1=params, mat2=features, epsilon=epsilon, dim=1)
    save_corrs(corrs=corrs, feature_name=feature_name, param_name=param_name, method='direct_corr', wise=wise, identifier=identifer)
    return corrs
    
def save_lstsq_group_correlations_train_test(features:torch.Tensor, params:torch.Tensor, feature_name:str, param_name:str, wise:str, identifer:str):
    num_targets, num_samples, _ = features.size()
    one_col = torch.ones( size=(num_targets, num_samples, 1), dtype=features.dtype, device=features.device )
    features_and_1 = torch.cat( tensors=(features, one_col), dim=-1 )
    coeffs = torch.linalg.lstsq(features_and_1, params).solution
    params_pred = torch.matmul(features_and_1, coeffs)
    corrs = isingmodellight.get_pairwise_correlation(mat1=params, mat2=params_pred, epsilon=epsilon, dim=1)
    print( 'features-and-1 size', features_and_1.size(), 'params size', params.size(), 'coeffs size', coeffs.size(), 'params_pred size', params_pred.size(), 'corrs size', corrs.size() )
    save_corrs(corrs=corrs, feature_name=feature_name, param_name=param_name, method='lstsq_corr', wise=wise, identifier=identifer)
    return corrs

def save_group_corrs_h_and_J(node_features:torch.Tensor, sc:torch.Tensor, num_updates:int):
     h, J = get_group_parameters(num_updates=num_updates)
     corrs_lstsq_node = save_lstsq_group_correlations_train_test(features=node_features, params=h, feature_name='all_mean', param_name='h_group', wise='node', identifer=group_model_short_identifier)
     return corrs_lstsq_node
     # corrs_node = save_direct_group_correlations_train_test(features=node_features, params=h, feature_name='all_mean', param_name='h_group', wise='node', identifer=group_model_short_identifier)
     # corrs_pair = save_direct_group_correlations_train_test(features=sc, params=J, feature_name='sc_mean', param_name='J_group', wise='pair', identifer=group_model_short_identifier)
     # return corrs_node, corrs_pair

def save_group_corrs_h_and_J_series(node_features:torch.Tensor, sc:torch.Tensor):
    group_updates = torch.arange(start=min_updates_group, end=max_updates_group+1, step=update_increment_group, dtype=int_type, device=device)
    num_num_updates = group_updates.numel()
    # Take the mean over subjects, but keep the singleton dimension to align with parameter thresholds.
    node_features = node_features.mean(dim=0, keepdim=True)
    num_node_features = node_features.size(dim=-1)
    sc = sc.mean(dim=0, keepdim=True)
    num_pair_features = sc.size(dim=-1)
    lstsq_corrs_node = torch.zeros( size=(num_num_updates, num_thresholds, 1), dtype=node_features.dtype, device=node_features.device )
    # corrs_node = torch.zeros( size=(num_num_updates, num_thresholds, num_node_features), dtype=node_features.dtype, device=node_features.device )
    # corrs_pair = torch.zeros( size=(num_num_updates, num_thresholds, num_pair_features), dtype=node_features.dtype, device=node_features.device )
    for update_index in range(num_num_updates):
        num_updates = group_updates[update_index]
        # corrs_node[update_index,:,:], corrs_pair[update_index,:,:] = save_group_corrs_h_and_J(node_features=node_features, sc=sc, num_updates=num_updates)
        lstsq_corrs_node[update_index,:,:] = save_group_corrs_h_and_J(node_features=node_features, sc=sc, num_updates=num_updates)
    method = 'direct_corr'
    node_feature_name = 'all_mean'
    pair_feature_name = 'sc_mean'
    node_param_name = 'h_group'
    pair_param_name = 'J_group'
    node_wise = 'node'
    pair_wise = 'pair'
    identifier_series = f'{group_model_short_identifier}_update_min_{min_updates_group}_max_{max_updates_group}_inc_{update_increment_group}'
    save_corrs(corrs=lstsq_corrs_node, feature_name=node_feature_name, param_name=node_param_name, method='lstsq_corr', wise=node_wise, identifier=identifier_series)
    # save_corrs(corrs=corrs_node, feature_name=node_feature_name, param_name=node_param_name, method=method, wise=node_wise, identifier=identifier_series)
    # save_corrs(corrs=corrs_pair, feature_name=pair_feature_name, param_name=pair_param_name, method=method, wise=pair_wise, identifier=identifier_series)
    return 0
    
# features and params should have aligned dimensions.
# features (num_nodes/pairs, num_subjects, num_features+1)
# params (num_nodes/pairs, num_subjects, 1)
# The correlations are along the subjects dimension.
# squeeze() out the singleton feature dimension of correlations.
# Keep the singleton dimension of coeffs, which makes it a stack of num_features+1 x 1 matrices.
# We need this dimension in order to do more matmul() calls with it.
# corrs (num_nodes/pairs)
# coeffs (num_nodes/pairs, num_features+1, 1)
def get_lstsq_individual_correlations(features:torch.Tensor, params:torch.Tensor):
    coeffs = torch.linalg.lstsq(features, params).solution
    predictions = torch.matmul(features, coeffs)
    corrs = isingmodellight.get_pairwise_correlation(mat1=predictions, mat2=params, epsilon=epsilon, dim=1).squeeze(dim=-1)
    rmses = isingmodellight.get_pairwise_rmse(mat1=predictions, mat2=params, dim=1).squeeze(dim=-1)
    return corrs, rmses, coeffs
    
def get_lstsq_individual_correlations_train_and_test(features:torch.Tensor, params:torch.Tensor, num_perms:int, num_training_subjects:int):
    params = params.transpose(dim0=0, dim1=1)# transpose to get (nodes/pairs, subjects, 1)
    num_parts, num_subjects, _ = params.size()
    train_corrs = torch.zeros( size=(num_parts, num_perms), dtype=features.dtype, device=features.device )
    test_corrs = torch.zeros( size=(num_parts, num_perms), dtype=features.dtype, device=features.device )
    train_rmses = torch.zeros( size=(num_parts, num_perms), dtype=features.dtype, device=features.device )
    test_rmses = torch.zeros( size=(num_parts, num_perms), dtype=features.dtype, device=features.device )
    for perm_index in range(num_perms):
        perm = torch.randperm(n=num_subjects, dtype=int_type, device=params.device)
        features_perm = features[:,perm,:]
        features_train = features_perm[:,:num_training_subjects,:]
        features_test = features_perm[:,num_training_subjects:,:]
        features_train, features_test = z_score_after_split_keep_1s(training=features_train, testing=features_test)
        params_perm = params[:,perm,:]
        params_train = params_perm[:,:num_training_subjects,:]
        params_test = params_perm[:,num_training_subjects:,:]
        params_train, params_test = z_score_after_split(training=params_train, testing=params_test)
        train_corrs[:,perm_index], train_rmses[:,perm_index], coeffs = get_lstsq_individual_correlations(features=features_train, params=params_train)
        predictions = torch.matmul(features_test, coeffs)
        test_corrs[:,perm_index] = isingmodellight.get_pairwise_correlation(mat1=predictions, mat2=params_test, epsilon=epsilon, dim=1).squeeze(dim=-1)
        test_rmses[:,perm_index] = isingmodellight.get_pairwise_rmse(mat1=predictions, mat2=params_test, dim=1).squeeze(dim=-1)
    return train_corrs, test_corrs, train_rmses, test_rmses

def save_lstsq_individual_correlations_train_and_test(features:torch.Tensor, params:torch.Tensor, num_perms:int, feature_name:str, param_name:str, identifer:str):
    train_corrs, test_corrs, train_rmses, test_rmses = get_lstsq_individual_correlations_train_and_test(features=features, params=params, num_perms=num_perms, num_training_subjects=num_training_subjects)
    wise = 'subject'
    num_testing_subjects = params.size(dim=0) - num_training_subjects
    method = 'lstsq_corr'
    save_corrs(corrs=train_corrs, feature_name=feature_name, param_name=param_name, method=method, wise=wise, identifier=f'train_subj_{num_training_subjects}_perms_{num_perms}_{identifer}')
    save_corrs(corrs=test_corrs, feature_name=feature_name, param_name=param_name, method=method, wise=wise, identifier=f'test_subj_{num_testing_subjects}_perms_{num_perms}_{identifer}')
    method = 'lstsq_rmse'
    save_corrs(corrs=train_rmses, feature_name=feature_name, param_name=param_name, method=method, wise=wise, identifier=f'train_subj_{num_training_subjects}_perms_{num_perms}_{identifer}')
    save_corrs(corrs=test_rmses, feature_name=feature_name, param_name=param_name, method=method, wise=wise, identifier=f'test_subj_{num_testing_subjects}_perms_{num_perms}_{identifer}')
    return train_rmses, test_rmses

def save_individual_corrs_h_and_J(node_features:torch.Tensor, sc:torch.Tensor, num_updates:int):
    h, J = get_individual_parameters(num_updates=num_updates)
    train_rmses_node, test_rmses_node = save_lstsq_individual_correlations_train_and_test(features=node_features, params=h, num_perms=num_perms_train_test_node, feature_name='all', param_name='h', identifer=individual_model_short_identifier)
    train_rmses_pair, test_rmses_pair = save_lstsq_individual_correlations_train_and_test(features=sc, params=J, num_perms=num_perms_train_test_pair, feature_name='sc', param_name='J', identifer=individual_model_short_identifier)
    return train_rmses_node, test_rmses_node, train_rmses_pair, test_rmses_pair

def save_individual_corrs_h_and_J_series(node_features:torch.Tensor, sc:torch.Tensor):
    individual_updates = torch.arange(start=min_updates_individual, end=max_updates_individual+1, step=update_increment_individual, dtype=int_type, device=device)
    num_num_updates = individual_updates.numel()
    node_features = append_ones( node_features.transpose(dim0=0, dim1=1) )# transpose to (nodes, subjects, features)
    num_nodes = node_features.size(dim=0)
    sc = append_ones( sc.transpose(dim0=0, dim1=1) )# transpose to (pairs, subjects, features)
    num_pairs = sc.size(dim=0)
    rmses_train_node = torch.zeros( size=(num_num_updates, num_nodes, num_perms_train_test_node), dtype=node_features.dtype, device=node_features.device )
    rmses_test_node = torch.zeros_like(input=rmses_train_node)
    rmses_train_pair = torch.zeros( size=(num_num_updates, num_pairs, num_perms_train_test_node), dtype=node_features.dtype, device=node_features.device )
    rmses_test_pair = torch.zeros_like(input=rmses_train_pair)
    for update_index in range(num_num_updates):
        num_updates = individual_updates[update_index]
        rmses_train_node[update_index,:,:], rmses_test_node[update_index,:,:], rmses_train_pair[update_index,:,:], rmses_test_pair[update_index,:,:] = save_individual_corrs_h_and_J(node_features=node_features, sc=sc, num_updates=num_updates)
    method = 'lstsq_rmse'
    node_feature_name = 'all'
    pair_feature_name = 'sc'
    node_param_name = 'h'
    pair_param_name = 'J'
    node_wise = 'subject'
    pair_wise = 'subject'
    identifier_series = f'{individual_model_short_identifier}_update_min_{min_updates_individual}_max_{max_updates_individual}_inc_{update_increment_individual}'
    train_identifier = f'train_{identifier_series}'
    test_identifier = f'test_{identifier_series}'
    save_corrs(corrs=rmses_train_node, feature_name=node_feature_name, param_name=node_param_name, method=method, wise=node_wise, identifier=train_identifier)
    save_corrs(corrs=rmses_test_node, feature_name=node_feature_name, param_name=node_param_name, method=method, wise=node_wise, identifier=test_identifier)
    save_corrs(corrs=rmses_train_pair, feature_name=pair_feature_name, param_name=pair_param_name, method=method, wise=pair_wise, identifier=train_identifier)
    save_corrs(corrs=rmses_test_pair, feature_name=pair_feature_name, param_name=pair_param_name, method=method, wise=pair_wise, identifier=test_identifier)
    return 0

def save_all():
    node_features = get_node_features()
    sc = get_sc()
    save_group_corrs_h_and_J_series(node_features=node_features, sc=sc)
    # save_individual_corrs_h_and_J_series(node_features=node_features, sc=sc)
    return 0

with torch.no_grad():
    save_all()

print('done')