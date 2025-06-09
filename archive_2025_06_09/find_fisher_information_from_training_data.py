import os
import torch
import hcpdatautils as hcp
import isingutils as ising
import time
import argparse

start_time = time.time()

int_type = torch.int
float_type = torch.float
device = torch.device('cuda')

parser = argparse.ArgumentParser(description="Estimate the Fisher information for Ising model parameters directly from the full set of training data.")
parser.add_argument("-i", "--data_dir", type=str, default='E:\\HCP_data', help="directory containing the fMRI data directory")
parser.add_argument("-o", "--output_dir", type=str, default='E:\\Ising_model_results_daai', help="directory to which to write the output")
args = parser.parse_args()
data_dir = args.data_dir
output_dir = args.output_dir

# https://pytorch.org/docs/stable/generated/torch.cov.html
# "Estimates the covariance matrix of the variables given by the input matrix, where rows are the variables and columns are the observations."

# Take a batch of time series, and output one Fisher information matrix
# fast version that uses more GPU memory.
def get_fisher_info_matrix(ts:torch.Tensor):
    print( 'ts batch size:', ts.size() )
    num_nodes = ts.size(dim=-1)
    ut_indices = torch.triu_indices(row=num_nodes, col=num_nodes, offset=1, device=device)
    ut_rows = ut_indices[0]
    ut_cols = ut_indices[1]
    tsbyts = ts[ut_rows,:] * ts[ut_cols,:]
    ts_params = torch.cat( (ts, tsbyts), dim=0 )
    return torch.cov(ts_params)

# Take a batch of time series, and output one Fisher information matrix
# slow version that uses less GPU memory.
def get_fisher_info_matrix_less_memory(ts:torch.Tensor):
    print( 'ts batch size:', ts.size() )
    num_time_points = ts.size(dim=1)
    ts_augmented = torch.cat(   (  torch.ones( (1,num_time_points), dtype=ts.dtype, device=ts.device ), ts  ), dim=0   )
    num_nodes = ts_augmented.size(dim=0)
    # We do not want to fill in the diagonal here, because we do not want the product of a node with itself.
    ut_indices = torch.triu_indices(row=num_nodes, col=num_nodes, offset=1, device=ts.device)
    ut_rows = ut_indices[0]
    ut_cols = ut_indices[1]
    num_pairs = ut_rows.numel()
    fim = torch.zeros( (num_pairs, num_pairs), dtype=ts.dtype, device=ts.device )
    # We do want to fill in the diagonal here, because we do want the variance of a product of nodes (or node*1).
    utut_indices = torch.triu_indices(row=num_pairs, col=num_pairs, offset=0, device=ts.device)
    utut_rows = utut_indices[0]
    utut_cols = utut_indices[1]
    num_pairs_of_pairs = utut_rows.numel()
    for pair_of_pairs_index in range(num_pairs_of_pairs):
        pair_index_1 = utut_rows[pair_of_pairs_index]
        pair_index_2 = utut_cols[pair_of_pairs_index]
        node_index_11 = ut_rows[pair_index_1]
        node_index_12 = ut_cols[pair_index_1]
        product_1 = ts_augmented[node_index_11,:] * ts_augmented[node_index_12,:]
        node_index_21 = ut_rows[pair_index_2]
        node_index_22 = ut_cols[pair_index_2]
        product_2 = ts_augmented[node_index_21,:] * ts_augmented[node_index_22,:]
        fi = torch.mean(product_1 * product_2) - torch.mean(product_1) * torch.mean(product_2)
        fim[pair_index_1, pair_index_2] = fi
        fim[pair_index_2, pair_index_1] = fi
    return fim

# Take a batch of time series, and output one Fisher information matrix
# slow version that uses less GPU memory.
def get_fisher_info_matrix_even_less_memory(ts:torch.Tensor):
    print( 'ts batch size:', ts.size() )
    num_time_points = ts.size(dim=1)
    ts_augmented = torch.cat(   (  torch.ones( (1,num_time_points), dtype=ts.dtype, device=ts.device ), ts  ), dim=0   )
    num_nodes = ts_augmented.size(dim=0)
    # We do not want to fill in the diagonal here, because we do not want the product of a node with itself.
    ut_indices = torch.triu_indices(row=num_nodes, col=num_nodes, offset=1, device=ts.device)
    ut_rows = ut_indices[0]
    ut_cols = ut_indices[1]
    num_pairs = ut_rows.numel()
    fim = torch.zeros( (num_pairs, num_pairs), dtype=ts.dtype, device=ts.device )
    # We do want to fill in the diagonal here, because we do want the variance of a product of nodes (or node*1).
    for pair_index_1 in range(num_pairs):
        node_index_11 = ut_rows[pair_index_1]
        node_index_12 = ut_cols[pair_index_1]
        product_1 = ts_augmented[node_index_11,:] * ts_augmented[node_index_12,:]
        for pair_index_2 in range(pair_index_1, num_pairs):
            node_index_21 = ut_rows[pair_index_2]
            node_index_22 = ut_cols[pair_index_2]
            product_2 = ts_augmented[node_index_21,:] * ts_augmented[node_index_22,:]
            fi = torch.mean(product_1 * product_2) - torch.mean(product_1) * torch.mean(product_2)
            fim[pair_index_1, pair_index_2] = fi
            fim[pair_index_2, pair_index_1] = fi
    return fim

print('loading training data...')
last_time = time.time()
subject_ids = hcp.load_training_subjects(data_dir)
data_ts = ising.standardize_and_binarize_ts_data( hcp.load_all_time_series_for_subjects(directory_path=data_dir, subject_ids=subject_ids, dtype=float_type, device=device) ).flatten(start_dim=0, end_dim=-2).transpose(dim0=0, dim1=1)
time_to_load = time.time() - last_time
print(f'time to load data {time_to_load:.3f}')
print( 'data ts size', data_ts.size() )

print('computing Fisher information...')
last_time = time.time()
fim = get_fisher_info_matrix_even_less_memory(data_ts)
time_to_fim = time.time() - last_time
print(f'time to compute FIM {time_to_fim:.3f}')
print( 'FIM size', fim.size() )

print('saving results...')
last_time = time.time()
fim_file = os.path.join(output_dir, 'training_data_fim.pt')
torch.save(fim, fim_file)
current_time = time.time()
time_to_save = current_time - last_time
total_time = current_time - start_time
print(f'done, time to save {time_to_save:.3f}, total time {total_time:.3f}')