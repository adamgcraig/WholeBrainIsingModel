import os
import torch
import time
import argparse
import hcpdatautils as hcp
import isingutils as ising

start_time = time.time()

parser = argparse.ArgumentParser(description="Load the group Ising model and simulate to find the FC RMSE vs original fMRI data FCs of individual subjects.")

# directories
parser.add_argument("-d", "--data_dir", type=str, default='E:\\HCP_data', help="directory containing the structural MRI features data file")
parser.add_argument("-s", "--stats_dir", type=str, default="E:\\Ising_model_results_daai", help="directory to which to write the output Fisher information matrices and other results")
parser.add_argument("-p", "--fim_param_string", type=str, default='nodes_21_window_50_lr_0.000_threshold_0.100_beta_0.500_reps_1000_epoch_4', help="the part of the group FIM file name and before between 'fim_ising_' and '.pt'")
parser.add_argument("-n", "--num_nodes", type=int, default=21, help="number of nodes in each Ising model")
parser.add_argument("-l", "--threshold", type=float, default=0.1, help="threshold at which to binarize the fMRI data, in standard deviations above the mean")
parser.add_argument("-b", "--beta", type=float, default=0.5, help="beta constant to use when simulating the Ising model")
parser.add_argument("-t", "--num_steps", type=int, default=48000, help="number of steps to use when running Ising model simulations to compare FC")

# We are not counting the first or last linear layer as hidden,
# so every network has at least two layers.

args = parser.parse_args()

data_dir = args.data_dir
print(f'data_dir {data_dir}')
stats_dir = args.stats_dir
print(f'stats_dir {stats_dir}')
fim_param_string = args.fim_param_string
print(f'fim_param_string {fim_param_string}')
num_nodes = args.num_nodes
print(f'num_nodes {num_nodes}')
num_steps = args.num_steps
print(f'num_steps {num_steps}')
threshold = args.threshold
print(f'threshold={threshold:.3g}')
beta = args.beta
print(f'beta {beta}')

float_type = torch.float
device = torch.device('cuda')

# creates a num_rows*num_cols 1-D Tensor of booleans where each value is True if and only if it is part of the upper triangle of a flattened num_rows x num_cols matrix.
# If we want the upper triangular part of a Tensor with one or more batch dimensions, we can flatten the last two dimensions together, and then use this.
def get_triu_logical_index(num_rows:int, num_cols:int):
    return ( torch.arange(start=0, end=num_rows, dtype=torch.int, device=device)[:,None] < torch.arange(start=0, end=num_cols, dtype=torch.int, device=device)[None,:] ).flatten()

def get_h_and_J(params:torch.Tensor, num_nodes:int=num_nodes):
    num_params = params.size(dim=-1)
    h = torch.index_select( input=params, dim=-1, index=torch.arange(end=num_nodes, device=params.device) )
    J_flat_ut = torch.index_select( input=params, dim=-1, index=torch.arange(start=num_nodes, end=num_params, device=params.device) )
    ut_indices = get_triu_logical_index(num_rows=num_nodes, num_cols=num_nodes).nonzero().flatten()
    J_flat = torch.zeros( (num_nodes * num_nodes), dtype=params.dtype, device=params.device )
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

with torch.no_grad():
    code_start_time = time.time()

    # Load group model eigenvectors.
    V_file_name = os.path.join(stats_dir, f'V_ising_{fim_param_string}.pt')
    V_group = torch.load(V_file_name)
    print(f'loaded {V_file_name}, time {time.time() - code_start_time:.3f}')
    # Load projections of the group model parameters onto the group model eigenvectors.
    projections_group_file_name = os.path.join(stats_dir, f'projections_group_ising_{fim_param_string}.pt')
    projections_group = torch.load(projections_group_file_name)
    projections_group_mean = projections_group.mean(dim=0)
    print(f'loaded {projections_group_file_name}, time {time.time() - code_start_time:.3f}')
    params_group_mean = torch.matmul( projections_group_mean.type(V_group.dtype), torch.linalg.inv(V_group) ).real
    h_group_mean, J_group_mean = get_h_and_J(params=params_group_mean, num_nodes=num_nodes)
    print(f'reconstructed group mean h and J, time {time.time() - code_start_time:.3f}')
    s = ising.get_random_state_like(h_group_mean)
    sim_fc_group_mean, s = ising.run_batched_balanced_metropolis_sim_for_fc( J=J_group_mean.unsqueeze(dim=0), h=h_group_mean.unsqueeze(dim=0), s=s.unsqueeze(dim=0), num_steps=num_steps, beta=beta )
    print(f'simulated mean group Ising model and found FC, time {time.time() - code_start_time:.3f}')

    # calculations with training set individual models
    for data_subset in ['training', 'validation', 'testing']:
        print(f'testing effect of zeroing out individual offsets with {data_subset}..., time {time.time() - code_start_time:.3f}')
        # Load the offsets along group model eigenvector directions.
        offsets_indi_file_name = os.path.join(stats_dir, f'offsets_indi_{data_subset}_ising_{fim_param_string}.pt')
        offsets_indi = torch.load(offsets_indi_file_name)
        num_subjects, num_reps, num_offsets = offsets_indi.size()
        print(f'loaded {offsets_indi_file_name}, time {time.time() - code_start_time:.3f}')
        # Load and compute FCs from the subject fMRI data.
        subject_ids = hcp.load_subject_subset(directory_path=data_dir, subject_subset=data_subset, require_sc=True)
        print(f'loaded fMRI data time series, time {time.time() - code_start_time:.3f}')
        data_fc = load_data_ts_for_fc(data_directory=data_dir, subject_ids=subject_ids)
        print(f'computed FCs of {data_subset} subjects, time {time.time() - code_start_time:.3f}')
        # Use them to reconstruct the individual Ising models.
        # Simulate each Ising model.
        # Compare its FC to the FC of the original fMRI data of the subject.
        sim_fc_group_mean_rep = sim_fc_group_mean.repeat( (num_subjects, 1, 1) )
        fc_rmse = hcp.get_triu_rmse_batch(sim_fc_group_mean_rep, data_fc)
        fc_corr = hcp.get_triu_corr_batch(sim_fc_group_mean_rep, data_fc)
        print(f'computed data-vs-sim FC of {data_subset} subjects for mean goup Ising model, median RMSE {fc_rmse.median():.3g}, median correlation {fc_corr.median():.3g}, time {time.time() - code_start_time:.3f}')
        # Save the results.
        error_param_string = f'group_ising_vs_individual_fmri_{fim_param_string}_steps_{num_steps}_{data_subset}'
        fc_rmse_file = os.path.join(stats_dir, f'fc_rmse_{error_param_string}.pt')
        torch.save(obj=fc_rmse, f=fc_rmse_file)
        print(f'saved {fc_rmse_file}, time {time.time() - code_start_time:.3f}')
        fc_corr_file = os.path.join(stats_dir, f'fc_corr_{error_param_string}.pt')
        torch.save(obj=fc_corr, f=fc_corr_file)
        print(f'saved {fc_corr_file}, time {time.time() - code_start_time:.3f}')
    print('done')