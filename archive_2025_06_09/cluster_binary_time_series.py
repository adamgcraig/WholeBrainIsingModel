import os
import torch
import time
import argparse
import pickle
import hcpdatautils as hcp
import isingutils as ising

parser = argparse.ArgumentParser(description="Fit multiple Ising models to each subject's fMRI data, binarizing at the median.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_batch', help="directory where we can find the time series files")
parser.add_argument("-c", "--data_subset", type=str, default='training', help="which list of subjects to use, either training, validation, testing, or all")
parser.add_argument("-n", "--num_nodes", type=int, default=360, help="number of nodes to model")
parser.add_argument("-d", "--num_time_points", type=int, default=4800, help="number of time points per simulation")
parser.add_argument("-t", "--target_num_nodes", type=int, default=21, help="number of nodes at which to save the coarse-grained time series")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
data_subset = args.data_subset
print(f'data_subset={data_subset}')
num_time_points = args.num_time_points
print(f'num_time_points={num_time_points}')
num_nodes = args.num_nodes
print(f'num_nodes={num_nodes}')
target_num_nodes = args.target_num_nodes
print(f'target_num_nodes={target_num_nodes}')

# Get the rwo and column of the first occurrence of the maximum similarity in a 2D similarity matrix.
def get_max_2d(values_2d:torch.Tensor):
    num_cols = values_2d.size(dim=-1)
    # Remove the diagonal, since self is trivially the most similar.
    reverse_eye = 1 - torch.eye(n=num_cols, dtype=values_2d.dtype, device=values_2d.device)
    index = torch.argmax( values_2d * reverse_eye )
    row = index//num_cols
    col = index % num_cols
    return row, col

with torch.no_grad():
    code_start_time = time.time()
    int_type = torch.int
    float_type = torch.float
    device = torch.device('cuda')
    # device = torch.device('cpu')
    subject_ids = hcp.load_subject_subset(directory_path=data_directory, subject_subset=data_subset, require_sc=True)
    num_subjects = len(subject_ids)
    print(f'Data subset {data_subset} has {num_subjects} subjects. Loading time series...')
    num_total_time_points = num_subjects * num_time_points
    data_ts = torch.zeros( (num_total_time_points, num_nodes), dtype=int_type, device=device )
    for subject_index in range(num_subjects):
        subject_id = subject_ids[subject_index]
        subject_start = subject_index * num_time_points
        subject_end = subject_start  + num_time_points
        single_data_ts = hcp.load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject_id, dtype=float_type, device=device).flatten(start_dim=0, end_dim=1)
        data_ts[subject_start:subject_end,:] = single_data_ts > single_data_ts.median(dim=-2, keepdim=True).values
    pair_l2s = torch.zeros( (num_nodes, num_nodes), dtype=int_type, device=device )
    time_elapsed = time.time() - code_start_time
    print(f'time {time_elapsed:.3f}')
    print(f'Computing initial l2 matrix...')
    for node_index in range(num_nodes):
        pair_l2s[node_index, :] = torch.sum(  torch.square( data_ts[:,node_index].unsqueeze(dim=1) + data_ts ), dim=0  )
    time_elapsed = time.time() - code_start_time
    print(f'time {time_elapsed:.3f}')
    new_num_nodes = num_nodes
    # The nodes tree is just nested lists of node indices for the original time series.
    # The leaf nodes are the integer indices.
    nodes = list( range(num_nodes) )
    # The node_l2s tree is for keeping track of how l2 changes as we go further up the tree.
    # Each node is a pair of an l2 value followed by a list containing either 2 (for branch nodes) or 0 (for leaf nodes) child nodes.
    node_l2s = [  [ num_total_time_points//2, [] ]  ]*num_nodes
    print('Clustering...')
    for iteration in range(num_nodes-1):
        source, target = get_max_2d(pair_l2s)
        old_l2 = pair_l2s[source, target].item()
        print(f'step {iteration} of {num_nodes}, {new_num_nodes} nodes remaining, combining nodes {source} and {target}, l2 {old_l2}')
        keep = torch.ones( (new_num_nodes,), dtype=torch.bool, device=device )
        keep[source] = False
        keep[target] = False
        # Replace the old nodes with the combined branch node.
        new_node = [ nodes[source], nodes[target] ]
        nodes = [node for (keep_node, node) in zip(keep, nodes) if keep_node] + [new_node]
        # Replace the two old [l2, [child l2s if any]] pairs with a new one with the selected l2 as the l2 and the two old l2s as its children.
        new_node_l2 = [  old_l2, [ node_l2s[source], node_l2s[target] ]  ]
        node_l2s = [node_l2 for (keep_l2, node_l2) in zip(keep, node_l2s) if keep_l2] + [new_node_l2]
        # Replace the time series for the old nodes with the combined time series.
        new_ts = data_ts[:,source] + data_ts[:,target]
        data_ts[:,:-2] = data_ts[:,keep]
        data_ts[:,-2] = new_ts
        data_ts = data_ts[:,:-1]
        # Replace the rows and columns of l2s for the old nodes with the combined l2s for the new time series.
        new_l2 = torch.sum(  torch.square( new_ts.unsqueeze(dim=1) + data_ts ), dim=0  )
        pair_l2s[:-2,:-2] = pair_l2s[keep,:][:,keep]
        pair_l2s[:-1,-2] = new_l2
        pair_l2s[-2,:-1] = new_l2
        pair_l2s = pair_l2s[:-1,:-1]
        # The network now has 1 node fewer.
        new_num_nodes -= 1
        # If this is the number of nodes we want, save the time series.
        if new_num_nodes == target_num_nodes:
            ts_file_name = os.path.join( output_directory, f'ts_binary_sums_{data_subset}_nodes_{target_num_nodes}.pt' )
            torch.save( data_ts.clone(), ts_file_name )
            print(f'saved {target_num_nodes}-node time series to {ts_file_name}')
    # When done, save the node tree and the l2 tree.
    print('saving results...')
    node_file_name = os.path.join(output_directory, f'node_tree_{data_subset}.pkl')
    with open(file=node_file_name, mode='wb') as node_file:
        pickle.dump(obj=nodes, file=node_file)
    print(f'saved node tree to {node_file_name}')
    node_l2_file_name = os.path.join(output_directory, f'node_l2_tree_{data_subset}.pkl')
    with open(file=node_l2_file_name, mode='wb') as node_l2_file:
        pickle.dump(obj=node_l2s, file=node_l2_file)
    print(f'saved l2 tree to {node_l2_file_name}')
    time_elapsed = time.time() - code_start_time
    print(f'done, time {time_elapsed:.3f}')