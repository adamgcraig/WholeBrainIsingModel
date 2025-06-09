import os
import torch
import time
import argparse
import hcpdatautils as hcp
import isingutils as ising
from collections import OrderedDict

start_time = time.time()

parser = argparse.ArgumentParser(description="Predict individual offsets relative to the group Ising model from structural MRI features, reconstruct the individual Ising models, and simulate to find the FC RMSE vs original fMRI data FC.")

# directories
parser.add_argument("-d", "--data_dir", type=str, default='E:\\HCP_data', help="directory containing the structural MRI features data file")
parser.add_argument("-m", "--model_dir", type=str, default="E:\\Ising_model_results_daai", help="directory containing the struct-to-offsets models")
parser.add_argument("-s", "--stats_dir", type=str, default="E:\\Ising_model_results_daai", help="directory to which to write the output Fisher information matrices and other results")
parser.add_argument("-f", "--fim_param_string", type=str, default='nodes_21_window_50_lr_0.000_threshold_0.100_beta_0.500_reps_1000_epoch_4', help="the part of the Ising model FIM file name between 'fim_ising_' and '.pt'")
# parser.add_argument("-o", "--struct_to_offset_param_string", type=str, default='reps_1000_epoch_4_depth_0_width_2_batch_1000_lr_0.0001', help="the part of the struct-to-offsets model file name after the Ising model param string and before '_dim_*.pt'")
parser.add_argument("-o", "--struct_to_offset_param_string", type=str, default='depth_10_width_1000_batch_10000_lr_0.01', help="the part of the struct-to-offsets model file name after the FIM param string and before '_dim_*.pt'")
parser.add_argument("-n", "--num_nodes", type=int, default=21, help="number of nodes in each Ising model")
parser.add_argument("-l", "--threshold", type=float, default=0.1, help="threshold at which to binarize the fMRI data, in standard deviations above the mean")
parser.add_argument("-b", "--beta", type=float, default=0.5, help="beta constant to use when simulating the Ising model")
parser.add_argument("-t", "--num_steps", type=int, default=48000, help="number of steps to use when running Ising model simulations to compare FC")
parser.add_argument("-z", "--z_score", type=bool, default=True, help="set to True to z-score the data before training, using the training sample mean and std. dev. for both training and validation data")

# We are not counting the first or last linear layer as hidden,
# so every network has at least two layers.

args = parser.parse_args()

data_dir = args.data_dir
print(f'data_dir {data_dir}')
model_dir = args.model_dir
print(f'model_dir {model_dir}')
stats_dir = args.stats_dir
print(f'stats_dir {stats_dir}')
fim_param_string = args.fim_param_string
print(f'fim_param_string {fim_param_string}')
struct_to_offset_param_string = args.struct_to_offset_param_string
print(f'struct_to_offset_param_string {struct_to_offset_param_string}')
num_nodes = args.num_nodes
print(f'num_nodes {num_nodes}')
num_steps = args.num_steps
print(f'num_steps {num_steps}')
threshold = args.threshold
print(f'threshold={threshold:.3g}')
beta = args.beta
print(f'beta {beta}')
z_score = args.z_score
print(f'z_score {z_score}')
num_offsets = num_nodes + ( num_nodes*(num_nodes-1) )//2
print(f'num_offsets {num_offsets}')

float_type = torch.float
device = torch.device('cuda')

class Struct2Param(torch.nn.Module):

    # helper function for initialization => Do not call this elsewhere.
    def get_hidden_layer(self, n:int):
        index = n//2
        if n % 2 == 0:
            return ( f'hidden_linear{index}', torch.nn.Linear(in_features=self.hidden_layer_width, out_features=self.hidden_layer_width, device=device, dtype=float_type) )
        else:
            return ( f'hidden_relu{index}', torch.nn.ReLU() )

    # previously worked well with 21-node model:
    # def __init__(self, num_features:int, rep_dims:int=15, hidden_layer_width_1:int=15, hidden_layer_width_2:int=15, dtype=float_type, device=device)
    def __init__(self, num_features:int, num_hidden_layer:int=1, hidden_layer_width:int=90, dtype=float_type, device=device):
        super(Struct2Param, self).__init__()
        self.num_features = num_features
        self.num_hidden_layer = num_hidden_layer
        self.hidden_layer_width = hidden_layer_width
        layer_list = [
            ( 'input_linear', torch.nn.Linear(in_features=self.num_features, out_features=self.hidden_layer_width, dtype=dtype, device=device) ),
            ( 'input_relu', torch.nn.ReLU() )
            ] + [
            self.get_hidden_layer(n) for n in range(2*self.num_hidden_layer)
            ] + [
            ( 'output_linear', torch.nn.Linear(in_features=self.hidden_layer_width, out_features=1, dtype=dtype, device=device) )
            ]
        layer_dict = OrderedDict(layer_list)
        self.ff_layers = torch.nn.Sequential(layer_dict)
    
    def forward(self, features):
        return self.ff_layers(features).squeeze()

# creates a num_rows*num_cols 1-D Tensor of booleans where each value is True if and only if it is part of the upper triangle of a flattened num_rows x num_cols matrix.
# If we want the upper triangular part of a Tensor with one or more batch dimensions, we can flatten the last two dimensions together, and then use this.
def get_triu_logical_index(num_rows:int, num_cols:int):
    return ( torch.arange(start=0, end=num_rows, dtype=torch.int, device=device)[:,None] < torch.arange(start=0, end=num_cols, dtype=torch.int, device=device)[None,:] ).flatten()

def prepare_structural_data(subset:str, z_score:bool=True, struct_std:torch.Tensor=None, struct_mean:torch.Tensor=None, num_nodes:int=num_nodes, structural_data_dir:str=data_dir):
    subjects = hcp.load_subject_subset(directory_path=structural_data_dir, subject_subset=subset, require_sc=True)
    num_subjects = len(subjects)
    # Pre-allocate space for the data.
    node_features = torch.zeros( (num_subjects, num_nodes, hcp.features_per_area), dtype=float_type, device=device )
    sc = torch.zeros( (num_subjects, num_nodes, num_nodes), dtype=float_type, device=device )
    # Load all the data from the individual files.
    for subject_index in range(num_subjects):
        subject_id = subjects[subject_index]
        features_file = hcp.get_area_features_file_path(directory_path=structural_data_dir, subject_id=subject_id)
        node_features[subject_index,:,:] = hcp.load_matrix_from_binary(file_path=features_file, dtype=float_type, device=device).transpose(dim0=0, dim1=1)[:num_nodes,:]
        sc_file = hcp.get_structural_connectivity_file_path(directory_path=structural_data_dir, subject_id=subject_id)
        sc[subject_index,:,:] = hcp.load_matrix_from_binary(file_path=sc_file, dtype=float_type, device=device)[:num_nodes,:num_nodes]
    # node_features is num_subjects x num_nodes x features_per_area.
    # sc is num_subjects x num_nodes x num_nodes.
    # Flatten node_features to num_subjects x num_nodes*features_per_area.
    # Flatten sc to num_subjects x num_nodes*num_nodes, and use logical indexing to take only the SC values that correspond to upper triangular elements.
    # Then concatenate them into one num_subjects x ( num_nodes*features_per_area + num_nodes*(num_nodes-1)/2 ) Tensor.
    ut_logical = get_triu_logical_index(num_rows=num_nodes, num_cols=num_nodes)
    structural_features = torch.cat(  ( node_features.flatten(start_dim=-2, end_dim=-1), sc.flatten(start_dim=-2, end_dim=-1)[:,ut_logical] ), dim=-1  )
    no_std = type(struct_std) == type(None)
    no_mean = type(struct_mean) == type(None)
    if no_std and no_mean:
        struct_std, struct_mean = torch.std_mean(structural_features, dim=0, keepdim=True)
    elif no_std:
        struct_std = torch.std(structural_features, dim=0, keepdim=True)
    elif no_mean:
        struct_mean = torch.mean(structural_features, dim=0, keepdim=True)
    if z_score:
        structural_features = (structural_features - struct_mean)/struct_std
    return structural_features, struct_std, struct_mean

def load_struct_to_offset_models(fim_param_string:str, struct_to_offset_param_string:str, num_offsets:int):
    return [torch.load( f=os.path.join(model_dir, f'struct_to_offset_{fim_param_string}_{struct_to_offset_param_string}_dim_{offset_index}.pt') ) for offset_index in range(num_offsets)]

def get_h_and_J(params:torch.Tensor, num_nodes:int=num_nodes):
    num_subjects, num_reps, num_params = params.size()
    h = torch.index_select( input=params, dim=-1, index=torch.arange(end=num_nodes, device=params.device) )
    J_flat_ut = torch.index_select( input=params, dim=-1, index=torch.arange(start=num_nodes, end=num_params, device=params.device) )
    ut_indices = get_triu_logical_index(num_rows=num_nodes, num_cols=num_nodes).nonzero().flatten()
    J_flat = torch.zeros( (num_subjects, num_reps, num_nodes * num_nodes), dtype=params.dtype, device=params.device )
    J_flat.index_copy_(dim=-1, index=ut_indices, source=J_flat_ut)
    J = J_flat.unflatten( dim=-1, sizes=(num_nodes, num_nodes) )
    J_sym = J + J.transpose(dim0=-2, dim1=-1)
    return h, J_sym

def load_data_ts_for_fc(data_directory:str, subject_ids:list, num_nodes:int=num_nodes, threshold:torch.float=threshold):
    num_subjects = len(subject_ids)
    data_fc = torch.zeros( (num_subjects, num_nodes, num_nodes), dtype=float_type, device=device )
    for subject_index in range(num_subjects):
        subject_id = subject_ids[subject_index]
        print(f'loaded fMRI data of subject {subject_index} of {num_subjects}, {subject_id}')
        data_fc[subject_index,:,:] = hcp.get_fc( ising.standardize_and_binarize_ts_data( ts=hcp.load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject_id, dtype=float_type, device=device), threshold=threshold, time_dim=1 ).flatten(start_dim=0, end_dim=-2)[:,:num_nodes] )
    return data_fc

def reconstruct_and_test_ising(offsets_indi:torch.Tensor, projections_group_mean:torch.Tensor, V_group:torch.Tensor, data_fc:torch.Tensor, beta:torch.float=beta, num_steps:int=num_steps, num_nodes:int=num_nodes):
    print( 'offsets_indi size', offsets_indi.size() )
    print( 'projections_group_mean size', projections_group_mean.size() )
    print( 'V_group size', V_group.size() )
    projections_indi = projections_group_mean + offsets_indi
    print( 'projections_indi size', projections_indi.size() )
    params_indi = torch.matmul( projections_indi.type(V_group.dtype), torch.linalg.inv(V_group) ).real
    print( 'params_indi size', params_indi.size() )
    h_indi, J_indi = get_h_and_J(params=params_indi, num_nodes=num_nodes)
    print( 'h_indi size', h_indi.size() )
    print( 'J_indi size', J_indi.size() )
    s = ising.get_random_state_like(h_indi)
    sim_fc_indi, s = ising.run_batched_balanced_metropolis_sim_for_fc(J=J_indi, h=h_indi, s=s, num_steps=num_steps, beta=beta)
    print( 'sim_fc_indi size', sim_fc_indi.size() )
    fc_rmse = hcp.get_triu_rmse_batch(sim_fc_indi, data_fc)
    print( 'fc_rmse size', fc_rmse.size() )
    fc_corr = hcp.get_triu_corr_batch(sim_fc_indi, data_fc)
    print( 'fc_corr size', fc_corr.size() )
    return fc_rmse, fc_corr

with torch.no_grad():
    code_start_time = time.time()

    # Load group model eigenvectors.
    V_file_name = os.path.join(stats_dir, f'V_ising_{fim_param_string}.pt')
    V_group = torch.load(V_file_name)
    print(f'loaded {V_file_name}, time {time.time() - code_start_time:.3f}')
    # Load projections of the group model parameters onto the group model eigenvectors.
    projections_group_file_name = os.path.join(stats_dir, f'projections_group_ising_{fim_param_string}.pt')
    projections_group = torch.load(projections_group_file_name)
    projections_group_mean = projections_group.mean(dim=0, keepdim=True).unsqueeze(dim=0)
    print(f'loaded {projections_group_file_name}, time {time.time() - code_start_time:.3f}')

    model_list = load_struct_to_offset_models(fim_param_string=fim_param_string, struct_to_offset_param_string=struct_to_offset_param_string, num_offsets=num_offsets)

    # calculations with training set individual models
    training_features, training_std, training_mean = prepare_structural_data(subset='training', z_score=z_score, struct_std=None, struct_mean=None, num_nodes=num_nodes, structural_data_dir=data_dir)
    num_training_subjects = training_features.size(dim=0)
    print(f'loaded training structural data, time {time.time() - code_start_time:.3f}')
    validation_features, _, _ = prepare_structural_data(subset='validation', z_score=z_score, struct_std=training_std, struct_mean=training_mean, num_nodes=num_nodes, structural_data_dir=data_dir)
    num_validation_subjects = validation_features.size(dim=0)
    print(f'loaded validation structural data, time {time.time() - code_start_time:.3f}')
    training_offsets = torch.zeros( (num_training_subjects, num_offsets), dtype=float_type, device=device )
    validation_offsets = torch.zeros( (num_validation_subjects, num_offsets), dtype=float_type, device=device )
    for offset_index in range(num_offsets):
        model_file = os.path.join(model_dir, f'struct_to_offset_{fim_param_string}_{struct_to_offset_param_string}_dim_{offset_index}.pt')
        model = torch.load(model_file)
        training_offsets[:,offset_index] = model(training_features)
        validation_offsets[:,offset_index] = model(validation_features)
    for subset_name, predicted_offsets in zip(['training', 'validation'], [training_offsets, validation_offsets]):
        num_subjects, num_offsets = predicted_offsets.size()
        print(f'predicted Ising model offsets from structural data, time {time.time() - code_start_time:.3f}')
        # Load and compute FCs from the subject fMRI data.
        subject_ids = hcp.load_subject_subset(directory_path=data_dir, subject_subset=subset_name, require_sc=True)
        data_fc = load_data_ts_for_fc(data_directory=data_dir, subject_ids=subject_ids)
        print(f'loaded fMRI data time series and computed FCs, time {time.time() - code_start_time:.3f}')
        # Use them to reconstruct the individual Ising models.
        # Simulate each Ising model.
        # Compare its FC to the FC of the original fMRI data of the subject.
        fc_rmse = torch.zeros( (num_subjects, num_offsets+1), dtype=float_type, device=device )
        fc_corr = torch.zeros_like(fc_rmse)
        fc_rmse_current, fc_corr_current = reconstruct_and_test_ising(offsets_indi=predicted_offsets, projections_group_mean=projections_group_mean, V_group=V_group, data_fc=data_fc, beta=beta, num_steps=num_steps, num_nodes=num_nodes)
        fc_rmse[:,-1] = fc_rmse_current
        fc_corr[:,-1] = fc_corr_current
        print(f'computed data-vs-sim FC of {subset_name} subjects for full Ising models, median RMSE {fc_rmse_current.median():.3g}, median correlation {fc_corr_current.median():.3g}, time {time.time() - code_start_time:.3f}')
        # Repeat this process but with one offset zeroed out.
        for offset_index in range(num_offsets):
            offsets_indi_minus_1 = predicted_offsets.clone()
            offsets_indi_minus_1[:,offset_index] = 0.0
            fc_rmse_current, fc_corr_current = reconstruct_and_test_ising(offsets_indi=offsets_indi_minus_1, projections_group_mean=projections_group_mean, V_group=V_group, data_fc=data_fc, beta=beta, num_steps=num_steps, num_nodes=num_nodes)
            fc_rmse[:,offset_index] = fc_rmse_current
            fc_corr[:,offset_index] = fc_corr_current
            print(f'computed data-vs-sim FC of {subset_name} subjects with offset {offset_index} zeroed, median RMSE {fc_rmse_current.median():.3g}, median correlation {fc_corr_current.median():.3g}, time {time.time() - code_start_time:.3f}')
        # Save the results.
        error_param_string = f'struct_to_offset_{fim_param_string}_{struct_to_offset_param_string}_steps_{num_steps}_{subset_name}'
        fc_rmse_file = os.path.join(stats_dir, f'fc_rmse_{error_param_string}.pt')
        torch.save(obj=fc_rmse, f=fc_rmse_file)
        print(f'saved {fc_rmse_file}, time {time.time() - code_start_time:.3f}')
        fc_corr_file = os.path.join(stats_dir, f'fc_corr_{error_param_string}.pt')
        torch.save(obj=fc_corr, f=fc_corr_file)
        print(f'saved {fc_corr_file}, time {time.time() - code_start_time:.3f}')
    print('done')