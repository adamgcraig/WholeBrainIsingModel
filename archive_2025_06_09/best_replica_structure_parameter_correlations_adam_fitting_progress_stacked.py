import os
import torch
from scipy import stats
import time
import argparse
import math
import isingmodellight
from isingmodellight import IsingModelLight
from collections import OrderedDict

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
# device = torch.device('cpu')
# Set epsilon to a small non-0 number to prevent NaNs in correlations.
# The corelations may still be nonsense values.
epsilon = 0.0

parser = argparse.ArgumentParser(description="Find correlations between Ising model parameters and structural features using Adam optimized linear regression.")
parser.add_argument("--data_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
parser.add_argument("--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
parser.add_argument("--region_feature_file_part", type=str, default='node_features_all_as_is', help='region feature file name except for the .pt file extension')
parser.add_argument("--sc_file_part", type=str, default='edge_features_all_as_is', help='SC file name except for the .pt file extension')
parser.add_argument("--group_fmri_file_name_part", type=str, default='thresholds_31_min_0_max_3', help="the multi-threshold group data mean state and state product file name after mean_state_ or mean_state_product_ and before .pt.")
parser.add_argument("--individual_fmri_file_name_part", type=str, default='all_mean_std_1', help="the single-threshold individual data mean state and state product file name after mean_state_ or mean_state_product_ and before .pt.")
parser.add_argument("--group_model_goodness_file_part", type=str, default='fc_corr_ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_63000_test_length_120000', help="the file name of values to use to select the best group model replica (highest value) before .pt.")
parser.add_argument("--individual_model_goodness_file_part", type=str, default='fc_corr_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_to_thresh_1_reps_5_subj_837_individual_updates_50000_test_length_120000', help="the file name of values to use to select the best individual model replica (highest value) before .pt.")
parser.add_argument("--group_model_file_part", type=str, default='ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_63000', help='the part of the Ising model file name before .pt.')
parser.add_argument("--individual_model_file_part", type=str, default='ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_to_thresh_1_reps_5_subj_837_individual_updates_50000', help='the part of the individual Ising model file name before .pt.')
parser.add_argument("--group_model_short_identifier", type=str, default='group_thresholds_31_min_0_max_3_progress', help='abbreviated name for the group model')
parser.add_argument("--individual_model_short_identifier", type=str, default='individual_from_group_glasser_1', help='abbreviated name for the individual model')
parser.add_argument("--num_training_regions", type=int, default=288, help="uses the first this many regions for the train-test splits for group correlations.")
parser.add_argument("--num_training_pairs", type=int, default=51696, help="uses the first this many regions for the train-test splits for group correlations.")
parser.add_argument("--num_training_subjects", type=int, default=670, help="uses the first this many subjects for the train-test splits for individual correlations.")
parser.add_argument("--num_perms_group_node", type=int, default=10, help="number of permutations to use for group node-wise permutation tests")
parser.add_argument("--num_perms_group_pair", type=int, default=10, help="number of permutations to use for group node-pair-wise permutation tests")
parser.add_argument("--num_perms_individual_node", type=int, default=10, help="number of permutations to use for individual subject-wise permutation tests when we have one for each node")
parser.add_argument("--num_perms_individual_pair", type=int, default=10, help="number of permutations to use for individual subject-wise permutation tests when we have one for each node pair")
parser.add_argument("--num_hidden_group_node", type=int, default=1, help='number of hidden layers in MLP')
parser.add_argument("--num_hidden_group_pair", type=int, default=1, help='number of hidden layers in MLP')
parser.add_argument("--num_hidden_individual_node", type=int, default=1, help='number of hidden layers in MLP')
parser.add_argument("--num_hidden_individual_pair", type=int, default=1, help='number of hidden layers in MLP')
parser.add_argument("--hidden_width_group_node", type=int, default=4, help='width of hidden layers in MLP')
parser.add_argument("--hidden_width_group_pair", type=int, default=2, help='width of hidden layers in MLP')
parser.add_argument("--hidden_width_individual_node", type=int, default=4, help='width of hidden layers in MLP')
parser.add_argument("--hidden_width_individual_pair", type=int, default=2, help='width of hidden layers in MLP')
parser.add_argument("--num_epochs_group_node", type=int, default=10000, help='number of epochs for which to run Adam optimizer')
parser.add_argument("--num_epochs_group_pair", type=int, default=10000, help='number of epochs for which to run Adam optimizer')
parser.add_argument("--num_epochs_individual_node", type=int, default=10000, help='number of epochs for which to run Adam optimizer')
parser.add_argument("--num_epochs_individual_pair", type=int, default=10000, help='number of epochs for which to run Adam optimizer')
parser.add_argument("--batch_size_group_node", type=int, default=-1, help='number samples per batch in Adam optimization, default -1 means a single batch with all training data')
parser.add_argument("--batch_size_group_pair", type=int, default=-1, help='number samples per batch in Adam optimization, default -1 means a single batch with all training data')
parser.add_argument("--batch_size_individual_node", type=int, default=-1, help='number samples per batch in Adam optimization, default -1 means a single batch with all training data')
parser.add_argument("--batch_size_individual_pair", type=int, default=-1, help='number samples per batch in Adam optimization, default -1 means a single batch with all training data')
parser.add_argument("--learning_rate_group_node", type=float, default=0.0001, help='learning rate of Adam optimizer')
parser.add_argument("--learning_rate_group_pair", type=float, default=0.0001, help='learning rate of Adam optimizer')
parser.add_argument("--learning_rate_individual_node", type=float, default=0.0001, help='learning rate of Adam optimizer')
parser.add_argument("--learning_rate_individual_pair", type=float, default=0.0001, help='learning rate of Adam optimizer')
parser.add_argument("--max_num_snapshots", type=int, default=1000, help='maximum number of RMSEs and correlations to save')
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
group_model_goodness_file_part = args.group_model_goodness_file_part
print(f'group_model_goodness_file_part={group_model_goodness_file_part}')
individual_model_goodness_file_part = args.individual_model_goodness_file_part
print(f'individual_model_goodness_file_part={individual_model_goodness_file_part}')
group_model_file_part = args.group_model_file_part
print(f'group_model_file_part={group_model_file_part}')
individual_model_file_part = args.individual_model_file_part
print(f'individual_model_file_part={individual_model_file_part}')
group_model_short_identifier = args.group_model_short_identifier
print(f'group_model_short_identifier={group_model_short_identifier}')
individual_model_short_identifier = args.individual_model_short_identifier
print(f'individual_model_short_identifier={individual_model_short_identifier}')
num_training_regions= args.num_training_regions
print(f'num_training_regions={num_training_regions}')
num_training_pairs= args.num_training_pairs
print(f'num_training_pairs={num_training_pairs}')
num_training_subjects = args.num_training_subjects
print(f'num_training_subjects={num_training_subjects}')
num_perms_group_node = args.num_perms_group_node
print(f'num_perms_group_node={num_perms_group_node}')
num_perms_group_pair = args.num_perms_group_pair
print(f'num_perms_group_pair={num_perms_group_pair}')
num_perms_individual_node = args.num_perms_individual_node
print(f'num_perms_individual_node={num_perms_individual_node}')
num_perms_individual_pair = args.num_perms_individual_pair
print(f'num_perms_individual_pair={num_perms_individual_pair}')
num_hidden_group_node = args.num_hidden_group_node
print(f'num_hidden_group_node={num_hidden_group_node}')
num_hidden_group_pair = args.num_hidden_group_pair
print(f'num_hidden_group_pair={num_hidden_group_pair}')
num_hidden_individual_node = args.num_hidden_individual_node
print(f'num_hidden_individual_node={num_hidden_individual_node}')
num_hidden_individual_pair = args.num_hidden_individual_pair
print(f'num_hidden_individual_pair={num_hidden_individual_pair}')
hidden_width_group_node = args.hidden_width_group_node
print(f'hidden_width_group_node={hidden_width_group_node}')
hidden_width_group_pair = args.hidden_width_group_pair
print(f'hidden_width_group_pair={hidden_width_group_pair}')
hidden_width_individual_node = args.hidden_width_individual_node
print(f'hidden_width_individual_node={hidden_width_individual_node}')
hidden_width_individual_pair = args.hidden_width_individual_pair
print(f'hidden_width_individual_pair={hidden_width_individual_pair}')
num_epochs_group_node = args.num_epochs_group_node
print(f'num_epochs_group_node={num_epochs_group_node}')
num_epochs_group_pair = args.num_epochs_group_pair
print(f'num_epochs_group_pair={num_epochs_group_pair}')
num_epochs_individual_node = args.num_epochs_individual_node
print(f'num_epochs_individual_node={num_epochs_individual_node}')
num_epochs_individual_pair = args.num_epochs_individual_pair
print(f'num_epochs_individual_pair={num_epochs_individual_pair}')
batch_size_group_node = args.batch_size_group_node
print(f'batch_size_group_node={batch_size_group_node}')
batch_size_group_pair = args.batch_size_group_pair
print(f'batch_size_group_pair={batch_size_group_pair}')
batch_size_individual_node = args.batch_size_individual_node
print(f'batch_size_individual_node={batch_size_individual_node}')
batch_size_individual_pair = args.batch_size_individual_pair
print(f'batch_size_individual_pair={batch_size_individual_pair}')
learning_rate_group_node = args.learning_rate_group_node
print(f'learning_rate_group_node={learning_rate_group_node}')
learning_rate_group_pair = args.learning_rate_group_pair
print(f'learning_rate_group_pair={learning_rate_group_pair}')
learning_rate_individual_node = args.learning_rate_individual_node
print(f'learning_rate_individual_node={learning_rate_individual_node}')
learning_rate_individual_pair = args.learning_rate_individual_pair
print(f'learning_rate_individual_pair={learning_rate_individual_pair}')
max_num_snapshots = args.max_num_snapshots
print(f'max_num_snapshots={max_num_snapshots}')

class ParallelMLP(torch.nn.Module):
    # The input should have size (num_linears, num_samples, in_features).
    # The output then has size (num_linears, num_samples, out_features).
    def __init__(self, num_linears:int, in_features:int, out_features:int, num_hidden_layers:int, hidden_layer_width:int, dtype, device):
        super(ParallelMLP, self).__init__()
        make_relu = lambda index: ( f'relu{index}', torch.nn.ReLU() )
        make_linear = lambda index: ( f'linear{index}', torch.nn.Linear(in_features=hidden_layer_width, out_features=hidden_layer_width, dtype=dtype, device=device) )
        self.coeffs = torch.nn.Parameter(  torch.randn( size=(num_linears, in_features, hidden_layer_width), dtype=dtype, device=device, requires_grad=True )/math.sqrt(in_features)  )
        self.mlp = torch.nn.Sequential(  OrderedDict( [layer(index) for index in range(1, num_hidden_layers) for layer in (make_relu, make_linear)] + [( 'reluoutput', torch.nn.ReLU() ), ( 'linearoutput', torch.nn.Linear(in_features=hidden_layer_width, out_features=out_features, dtype=dtype, device=device) )] )  )
        return
    def forward(self, input:torch.Tensor):
        return self.mlp( torch.matmul(input, self.coeffs) )

def get_node_features():
        # Select out the actual structural features, omitting the region coordinates from the Atlas.
        # Clone so that we do not retain memory of the larger Tensor after exiting the function.
        # transpose() so that we compute a separate model for each pair with subjects as samples.
        # region_features has size (num_nodes, num_subjects, num_features).
        node_features_file = os.path.join(data_directory, f'{region_feature_file_part}.pt')
        node_features = torch.clone(  torch.transpose( torch.load(node_features_file, weights_only=False)[:,:,:4], dim0=0, dim1=1 )  )
        print( f'time {time.time()-code_start_time:.3f}, loaded {node_features_file}, region features size', node_features.size() )
        return node_features
    
def get_sc():
        # Select out only the SC.
        # Clone so that we do not retain memory of the larger Tensor after exiting the function.
        # Unsqueeze to get a dimension that aligns with the features dimension of region_features.
        # transpose() so that we compute a separate model for each pair with subjects as samples.
        # sc has size (num_pairs, num_subjects, 1).
        sc_file = os.path.join(data_directory, f'{sc_file_part}.pt')
        sc = torch.clone( torch.load(sc_file, weights_only=False)[:,:,0] ).unsqueeze(dim=-1).transpose(dim0=0, dim1=1)
        print( f'time {time.time()-code_start_time:.3f}, loaded {sc_file}, SC size', sc.size() )
        return sc

def get_model_parameters(model_file_name_part:str, goodness_file_name_part:str):
        # Select the best replica for each threshold/subject.
        # Take the elements of J above the diagonal.
        # Unsqueeze to get a singleton feature dimension.
        # h_best has size (num_thresholds/subjects, num_nodes, 1).
        # J_best has size (num_thresholds/subjects, num_pairs, 1).
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

def get_group_parameters():
        h_group, J_group = get_model_parameters(model_file_name_part=group_model_file_part, goodness_file_name_part=group_model_goodness_file_part)
        return h_group, J_group
    
def get_individual_parameters():
        h_individual, J_individual = get_model_parameters(model_file_name_part=individual_model_file_part, goodness_file_name_part=individual_model_goodness_file_part)
        # Transpose so that the stacked model dimension (0) is nodes/pairs while the sample dimension (1) is subjects.
        return h_individual.transpose(dim0=0, dim1=1), J_individual.transpose(dim0=0, dim1=1)
    
def get_mean_state_and_fc(fmri_file_name_part:str, mean_over_scans:bool=False):
        # Compute FC from mean state and mean state product.
        # Take the part of FC above the diagonal.
        # Clone so that we do not retain memory of the larger Tensor.
        # unsqueeze() to get a singleton feature dimension.
        # mean_state has size (num_thresholds/subjects, num_nodes, 1).
        # fc has size (num_thresholds/subjects, num_pairs, 1).
        mean_state_file = os.path.join(data_directory, f'mean_state_{fmri_file_name_part}.pt')
        mean_state = torch.clone( torch.load(f=mean_state_file, weights_only=False) )
        mean_state_product_file = os.path.join(data_directory, f'mean_state_product_{fmri_file_name_part}.pt')
        mean_state_product = torch.clone( torch.load(f=mean_state_product_file, weights_only=False) )
        if mean_over_scans:
            mean_state = torch.mean(input=mean_state, dim=0)
            mean_state_product = torch.mean(input=mean_state_product, dim=0)
        fc = isingmodellight.get_fc(state_mean=mean_state, state_product_mean=mean_state_product, epsilon=epsilon)
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=fc.size(dim=-1), device=fc.device )
        mean_state = mean_state.unsqueeze(dim=-1)
        fc = torch.clone( fc[:,triu_rows,triu_cols].unsqueeze(dim=-1) )
        print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_file} and {mean_state_product_file}, mean state size', mean_state.size(), 'FC size', fc.size() )
        return mean_state, fc
    
def save_tensor(corrs:torch.Tensor, identifier:str):
        corrs_file_name = os.path.join(output_directory, f'{identifier}.pt')
        torch.save(obj=corrs, f=corrs_file_name)
        is_nan = torch.isnan(corrs)
        num_nan = torch.count_nonzero(is_nan)
        is_non_nan = torch.logical_not(is_nan)
        num_non_nan = torch.count_nonzero(is_non_nan)
        corrs_non_nan = corrs[is_non_nan]
        print( f'time {time.time()-code_start_time:.3f}, saved {corrs_file_name}, min {corrs_non_nan.min().item():.3f}, mean {corrs_non_nan.mean().item():.3f}, max {corrs_non_nan.max().item():.3f}, NaN {num_nan}, non-NaN {num_non_nan}' )
        return 0
    
def save_model(model:torch.nn.Module, identifier:str):
        model_file_name = os.path.join(output_directory, f'model_{identifier}.pt')
        torch.save(obj=model, f=model_file_name)
        print(f'time {time.time()-code_start_time:.3f}, saved {model_file_name}')
        return 0

def append_ones(m:torch.Tensor):
    num_thresholds, num_parts, _ = m.size()
    return torch.cat(   (  m, torch.ones( size=(num_thresholds, num_parts, 1), dtype=m.dtype, device=m.device )  ), dim=-1   )

def z_score_return_std_mean(training:torch.Tensor):
     training_std, training_mean = torch.std_mean(training, dim=1, keepdim=True)
     training_z = (training - training_mean)/training_std
     return training_z, training_std, training_mean

def z_score(training:torch.Tensor):
     training_std, training_mean = torch.std_mean(training, dim=1, keepdim=True)
     training_z = (training - training_mean)/training_std
     return training_z

def z_score_after_split(training:torch.Tensor, testing:torch.Tensor):
     training_std, training_mean = torch.std_mean(training, dim=1, keepdim=True)
     training_z = (training - training_mean)/training_std
     testing_z = (testing - training_mean)/training_std
     return training_z, testing_z

def z_score_keep_1s_return_std_mean(training:torch.Tensor):
     training_std, training_mean = torch.std_mean(training, dim=1, keepdim=True)
     training_z = (training - training_mean)/training_std
     training_z[:,:,-1] = 1.0
     return training_z, training_std, training_mean

def z_score_keep_1s(training:torch.Tensor):
     training_std, training_mean = torch.std_mean(training, dim=1, keepdim=True)
     training_z = (training - training_mean)/training_std
     training_z[:,:,-1] = 1.0
     return training_z

def z_score_after_split_keep_1s(training:torch.Tensor, testing:torch.Tensor):
     training_std, training_mean = torch.std_mean(training, dim=1, keepdim=True)
     training_z = (training - training_mean)/training_std
     training_z[:,:,-1] = 1.0
     testing_z = (testing - training_mean)/training_std
     testing_z[:,:,-1] = 1.0
     return training_z, testing_z

def fit_mlp_model(features_train:torch.Tensor, params_train:torch.Tensor, features_test:torch.Tensor, params_test:torch.Tensor, num_snapshots:int, epochs_per_snapshot:int, num_hidden_layers:int, hidden_layer_width:int, num_epochs:int, learning_rate:float, batch_size:int=-1):
    in_features = features_train.size(dim=-1)
    num_outputs, num_samples_train, out_features = params_train.size()
    rmse_train = torch.zeros( size=(num_snapshots, num_outputs), dtype=features_train.dtype, device=features_train.device, requires_grad=False )
    rmse_test = torch.zeros( size=(num_snapshots, num_outputs), dtype=features_test.dtype, device=features_test.device, requires_grad=False )
    corr_train = torch.zeros( size=(num_snapshots, num_outputs), dtype=features_train.dtype, device=features_train.device, requires_grad=False )
    corr_test = torch.zeros( size=(num_snapshots, num_outputs), dtype=features_test.dtype, device=features_test.device, requires_grad=False )
    local_batch_size = batch_size
    if local_batch_size == -1:
        local_batch_size = num_samples_train
    num_batches = num_samples_train//local_batch_size
    model = ParallelMLP(num_linears=num_outputs, in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers, hidden_layer_width=hidden_layer_width, dtype=features_train.dtype, device=features_train.device)
    # print(  f'model starts with {torch.count_nonzero( torch.isnan(model.coeffs) )} NaN and {torch.count_nonzero( torch.isinf(model.coeffs) )}  of {model.coeffs.numel()} coefficients.'  )
    # pred_params = model(features)
    # print(  f'prediction starts with {torch.count_nonzero( torch.isnan(pred_params) )} NaN and {torch.count_nonzero( torch.isinf(pred_params) )}  of {pred_params.numel()} parameters.'  )
    opt = torch.optim.Adam( params=model.parameters(), lr=learning_rate )
    mse_fn = torch.nn.MSELoss()
    snapshot_index = 0
    print( 'starting training, train features size', features_train.size(), 'test features size', features_test.size(), 'train params size', params_train.size(), 'test params size', params_test.size() )
    for epoch_index in range(num_epochs):
        order = torch.randperm(n=num_samples_train, dtype=int_type, device=features_train.device)
        shuffled_features = features_train[:,order,:]
        shuffled_params = params_train[:,order,:]
        batch_start = 0
        batch_end = local_batch_size
        for batch_index in range(num_batches):
              opt.zero_grad()
              mse = mse_fn( model(shuffled_features[:,batch_start:batch_end,:]), shuffled_params[:,batch_start:batch_end,:] )
              mse.backward()
              # print( 'epoch', epoch_index, 'MSE', mse.item(), 'coeffs gradient min', model.coeffs.grad.min(), 'max', model.coeffs.grad.max() )
              opt.step()
              batch_start += local_batch_size
              batch_end += local_batch_size
        if (epoch_index % epochs_per_snapshot) == 0:
            with torch.no_grad():
                predictions_train = model(features_train)
                predictions_test = model(features_test)
                rmse_train[snapshot_index,:] = isingmodellight.get_pairwise_rmse(mat1=predictions_train, mat2=params_train, dim=1).squeeze(dim=-1)
                corr_train[snapshot_index,:] = isingmodellight.get_pairwise_correlation(mat1=predictions_train, mat2=params_train, epsilon=epsilon, dim=1).squeeze(dim=-1)
                rmse_test[snapshot_index,:] = isingmodellight.get_pairwise_rmse(mat1=predictions_test, mat2=params_test, dim=1).squeeze(dim=-1)
                corr_test[snapshot_index,:] = isingmodellight.get_pairwise_correlation(mat1=predictions_test, mat2=params_test, epsilon=epsilon, dim=1).squeeze(dim=-1)
                snapshot_index += 1
    return model, rmse_train, rmse_test, corr_train, corr_test

# features and params should have aligned dimensions.
# features (1, num_nodes/pairs, num_features+1) (append a dimension of 1s)
# params (num_thresholds, num_nodes/pairs, 1)
# The correlations are along the nodes/pairs dimension.
# We squeeze() coeffs at the end to get rid of the singleton feature dimension.
# coeffs also has a singleton dimension, but it serves to make the stacked matrices of size num_features x 1.
# We keep it so that we can use coeffs for future matrix multiplications.
# coeffs (num_thresholds, num_features+1, 1)
# corrs (num_thresholds)
def get_adam_correlations_train_test(features:torch.Tensor, params:torch.Tensor, num_perms:int, num_train:int, num_hidden_layers:int, hidden_layer_width:int, num_epochs:int, learning_rate:float, batch_size:int=-1):
    if num_epochs > max_num_snapshots:
        num_snapshots = max_num_snapshots
        epochs_per_snapshot = num_epochs//max_num_snapshots
    else:
        num_snapshots = num_epochs
        epochs_per_snapshot = 1
    features = append_ones(features)
    num_thresholds, num_parts, _ = params.size()
    stat_size = (num_perms, num_snapshots, num_thresholds)
    corrs_train = torch.zeros(size=stat_size, dtype=features.dtype, device=features.device, requires_grad=False )
    corrs_test = torch.zeros_like(corrs_train)
    rmses_train = torch.zeros_like(corrs_train)
    rmses_test = torch.zeros_like(corrs_train)
    best_model = None
    best_rmse_test = torch.inf
    for perm_index in range(num_perms):
        permutation = torch.randperm(n=num_parts, dtype=int_type, device=features.device)
        indices_train = permutation[:num_train]
        indices_test = permutation[num_train:]
        features_train = features[:,indices_train,:]
        params_train = params[:,indices_train,:]
        features_test = features[:,indices_test,:]
        params_test = params[:,indices_test,:]
        features_train, features_test = z_score_after_split_keep_1s(training=features_train, testing=features_test)
        params_train, params_test = z_score_after_split(training=params_train, testing=params_test)
        model, rmses_train[perm_index,:,:], current_rmse_test, corrs_train[perm_index,:,:], corrs_test[perm_index,:,:] = fit_mlp_model(features_train=features_train, params_train=params_train, features_test=features_test, params_test=params_test, num_snapshots=num_snapshots, epochs_per_snapshot=epochs_per_snapshot, num_hidden_layers=num_hidden_layers, hidden_layer_width=hidden_layer_width, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size)
        rmses_test[perm_index,:,:] = current_rmse_test
        total_rmse_test = torch.sqrt(  torch.mean( torch.square(current_rmse_test) )  )# square() to get back MSE, then mean over all to get one value, then sqrt() to get RMSE.
        if total_rmse_test < best_rmse_test:
            best_rmse_test = total_rmse_test
            best_model = model
    return best_model, corrs_train, corrs_test, rmses_train, rmses_test

def save_adam_correlations_train_test(feature_name:str, param_name:str, ising_model_identifier:str, features:torch.Tensor, params:torch.Tensor, num_perms:int, num_train:int, num_hidden_layers:int, hidden_layer_width:int, num_epochs:int, learning_rate:float, batch_size:int=-1):
    model, corrs_train, corrs_test, rmses_train, rmses_test = get_adam_correlations_train_test(features=features, params=params, num_perms=num_perms, num_train=num_train, num_hidden_layers=num_hidden_layers, hidden_layer_width=hidden_layer_width, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size)
    num_test = params.size(dim=1) - num_train
    model_identifier = f'{feature_name}_to_{param_name}_{ising_model_identifier}_perms_{num_perms}_train_{num_train}_test_{num_test}_hidden_depth_{num_hidden_layers}_width_{hidden_layer_width}_epochs_{num_epochs}_batchsz_{batch_size}_lr_{learning_rate:.3g}'
    save_model(model=model, identifier=f'mlp_{model_identifier}')
    save_tensor(corrs=corrs_train, identifier=f'corr_train_{model_identifier}')
    save_tensor(corrs=corrs_test, identifier=f'corr_test_{model_identifier}')
    save_tensor(corrs=rmses_train, identifier=f'rmse_train_{model_identifier}')
    save_tensor(corrs=rmses_test, identifier=f'rmse_test_{model_identifier}')
    return 0
# get_adam_correlations_train_test(features:torch.Tensor, params:torch.Tensor, num_perms:int, num_hidden_layers:int, hidden_layer_width:int, num_epochs:int, learning_rate:float, batch_size:int=-1)

def save_corrs_group_h_and_J(node_features:torch.Tensor, sc:torch.Tensor):
     group_h, group_J = get_group_parameters()
     mean_node_features = node_features.mean(dim=1,keepdim=False).unsqueeze(dim=0)
     mean_sc = sc.mean(dim=1,keepdim=False).unsqueeze(dim=0)
     save_adam_correlations_train_test(feature_name=f'mean_all', param_name=f'group_h', ising_model_identifier=group_model_short_identifier, features=mean_node_features, params=group_h, num_perms=num_perms_group_node, num_train=num_training_regions, num_hidden_layers=num_epochs_group_node, hidden_layer_width=hidden_width_group_node, num_epochs=num_epochs_group_node, learning_rate=learning_rate_group_node, batch_size=batch_size_group_node)
     save_adam_correlations_train_test(feature_name=f'mean_sc', param_name=f'group_J', ising_model_identifier=group_model_short_identifier, features=mean_sc, params=group_J, num_perms=num_perms_group_pair, num_train=num_training_pairs, num_hidden_layers=num_hidden_group_pair, hidden_layer_width=hidden_width_group_pair, num_epochs=num_epochs_group_pair, learning_rate=learning_rate_group_pair, batch_size=batch_size_group_pair)
     return 0

def save_corrs_individual_h_and_J(node_features:torch.Tensor, sc:torch.Tensor):
     individual_h, individual_J = get_individual_parameters()
     save_adam_correlations_train_test(feature_name=f'all', param_name=f'h', ising_model_identifier=individual_model_short_identifier, features=node_features, params=individual_h, num_perms=num_perms_individual_node, num_train=num_training_subjects, num_hidden_layers=num_hidden_individual_node, hidden_layer_width=hidden_width_individual_node, num_epochs=num_epochs_individual_node, learning_rate=learning_rate_individual_node, batch_size=batch_size_individual_node)
     save_adam_correlations_train_test(feature_name=f'sc', param_name=f'J', ising_model_identifier=individual_model_short_identifier, features=sc, params=individual_J, num_perms=num_perms_individual_pair, num_train=num_training_subjects, num_hidden_layers=num_hidden_individual_pair, hidden_layer_width=hidden_width_individual_pair, num_epochs=num_epochs_individual_pair, learning_rate=learning_rate_individual_pair, batch_size=batch_size_individual_pair)
     return 0
    
def save_all():
    node_features = get_node_features()
    sc = get_sc()
    save_corrs_group_h_and_J(node_features=node_features, sc=sc)
    save_corrs_individual_h_and_J(node_features=node_features, sc=sc)
    return 0
    
save_all()

print('done')