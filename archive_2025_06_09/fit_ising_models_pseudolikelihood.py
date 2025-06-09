# Based on the definitions of pseudolikelihood for Ising model parameters in
# H. Chau Nguyen, Riccardo Zecchina & Johannes Berg (2017)
# Inverse statistical problems: from the inverse Ising problem to data science,
# Advances in Physics, 66:3, 197-261, DOI: 10.1080/00018732.2017.1341604
# and
# Aurell, E., & Ekeberg, M. (2012).
# Inverse Ising inference using all the data.
# Physical review letters, 108(9), 090201.

import os
import torch
import time
import argparse

code_start_time = time.time()
int_type = torch.int
float_type = torch.float
device = torch.device('cuda')

parser = argparse.ArgumentParser(description="Train a bunch of Ising models on fMRI time series data using pseudolikelihood maximization.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_batch', help="directory where we can find the time series files")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate by which we multiply updates to weights and biases at each step")
parser.add_argument("-a", "--beta", type=float, default=0.5, help="factor we can use to adjust the chance of accepting a flip that increases the energy")
parser.add_argument("-t", "--threshold", type=float, default=0.1, help="threshold at which to binarize the fMRI data, in standard deviations above the mean")
parser.add_argument("-p", "--prob_update", type=float, default=0.02, help="probability of updating the model parameters on any given step")
parser.add_argument("-n", "--num_nodes", type=int, default=360, help="number of nodes to model")
parser.add_argument("-e", "--num_epochs", type=int, default=50, help="number of times to repeat the training time series")
parser.add_argument("-r", "--num_reps", type=int, default=1, help="number of models to train for each subject")
parser.add_argument("-d", "--data_subset", type=str, default='training', help="which data subset to use, either training, validation or testing")
parser.add_argument("-o", "--subjects_start", type=int, default=0, help="index of first subject in slice on which to train")
parser.add_argument("-x", "--subjects_end", type=int, default=5, help="index one past last subject in slice on which to train")
parser.add_argument("-v", "--print_every_seconds", type=int, default=10, help="minimum number of seconds between printouts of training status")
parser.add_argument("-c", "--batch_size", type=int, default=600, help="number of time points per batch on which to train in parallel")
args = parser.parse_args()
data_directory = args.data_directory
output_directory = args.output_directory
learning_rate = args.learning_rate
beta = args.beta
threshold = args.threshold
prob_update = args.prob_update
num_nodes = args.num_nodes
num_epochs = args.num_epochs
num_reps = args.num_reps
data_subset = args.data_subset
subjects_start = args.subjects_start
subjects_end = args.subjects_end
print_every_seconds = args.print_every_seconds
batch_size = args.batch_size

def get_fc(ts:torch.Tensor):
    # We want to take the correlations between pairs of areas over time.
    # torch.corrcoef() assumes observations are in columns and variables in rows.
    # As such, we have to take the transpose.
    return torch.corrcoef( ts.transpose(dim0=0, dim1=1) )

def get_fc_batch(ts_batch:torch.Tensor):
    # Take the FC of each individual time series.
    batch_size = ts_batch.size(dim=0)
    num_rois = ts_batch.size(dim=-1)
    fc = torch.zeros( (batch_size, num_rois, num_rois), dtype=ts_batch.dtype, device=ts_batch.device )
    for ts_index in range(batch_size):
        fc[ts_index,:,:] = get_fc(ts_batch[ts_index,:,:])
    return fc

def get_fcd(ts:torch.Tensor, window_length:int=90, window_step:int=1):
    # Calculate functional connectivity dynamics.
    # Based on Melozzi, Francesca, et al. "Individual structural features constrain the mouse functional connectome." Proceedings of the National Academy of Sciences 116.52 (2019): 26961-26969.
    dtype = ts.dtype
    device = ts.device
    ts_size = ts.size()
    T = ts_size[0]
    R = ts_size[1]
    triu_index = torch.triu_indices(row=R, col=R, offset=1, dtype=torch.int, device=device)
    triu_row = triu_index[0]
    triu_col = triu_index[1]
    num_indices = triu_index.size(dim=1)
    left_window_margin = window_length//2 + 1
    right_window_margin = window_length - left_window_margin
    window_centers = torch.arange(start=left_window_margin, end=T-left_window_margin+1, step=window_step, dtype=torch.int, device=device)
    window_offsets = torch.arange(start=-right_window_margin, end=left_window_margin, dtype=torch.int, device=device)
    num_windows = window_centers.size(dim=0)
    window_fc = torch.zeros( (num_windows, num_indices), dtype=dtype, device=device )
    for c in range(num_windows):
        fc = get_fc(ts[window_offsets+window_centers[c],:])
        window_fc[c,:] = fc[triu_row, triu_col]
    return torch.corrcoef(window_fc)

# only assumes the tensors are the same shape
def get_rmse(tensor1:torch.Tensor, tensor2:torch.Tensor):
    return torch.sqrt(  torch.mean( torch.square(tensor2 - tensor1) )  )

# takes the mean over all but the first dimension
# For our 3D batch-first tensors, this gives us a 1D tensor of RMSE values, one for each batch.
def get_rmse_batch(tensor1:torch.Tensor, tensor2:torch.Tensor):
    dim_indices = tuple(   range(  len( tensor1.size() )  )   )
    return torch.sqrt(  torch.mean( torch.square(tensor2 - tensor1), dim=dim_indices[1:] )  )

# In several cases, we have symmetric square matrices with fixed values on the diagonal.
# In particular, this is true of the functional connectivity and structural connectivity matrices.
# For such matrices, it is more meaningful to only calculate the RMSE of the elements above the diagonal.
def get_triu_rmse(tensor1:torch.Tensor, tensor2:torch.Tensor):
    indices = torch.triu_indices( row=tensor1.size(0), col=tensor1.size(1), offset=1, device=tensor1.device )
    indices_r = indices[0]
    indices_c = indices[1]
    return get_rmse( tensor1[indices_r,indices_c], tensor2[indices_r,indices_c] )

def get_triu_rmse_batch(tensor1:torch.Tensor, tensor2:torch.Tensor):
    indices = torch.triu_indices( tensor1.size(-2), tensor1.size(-1), 1 )
    indices_r = indices[0]
    indices_c = indices[1]
    batch_size = tensor1.size(dim=0)
    num_triu_elements = indices.size(dim=1)
    dtype = tensor1.dtype
    device = tensor1.device
    triu1 = torch.zeros( (batch_size, num_triu_elements), dtype=dtype, device=device )
    triu2 = torch.zeros_like(triu1)
    for b in range(batch_size):
        triu1[b,:] = tensor1[b,indices_r,indices_c]
        triu2[b,:] = tensor2[b,indices_r,indices_c]
    return get_rmse_batch(triu1, triu2)

def get_triu_corr(tensor1:torch.Tensor, tensor2:torch.Tensor):
    indices = torch.triu_indices( row=tensor1.size(0), col=tensor1.size(1), offset=1, device=tensor1.device )
    indices_r = indices[0]
    indices_c = indices[1]
    tensor1_triu = tensor1[indices_r,indices_c].unsqueeze(dim=0)
    # print( tensor1_triu.size() )
    tensor2_triu = tensor2[indices_r,indices_c].unsqueeze(dim=0)
    # print( tensor2_triu.size() )
    tensor_pair_triu = torch.cat( (tensor1_triu,tensor2_triu), dim=0 )
    # print( tensor_pair_triu.size() )
    corr = torch.corrcoef(tensor_pair_triu)
    # print(corr)
    return corr[0,1]

def get_triu_corr_batch(tensor1:torch.Tensor, tensor2:torch.Tensor):
    indices = torch.triu_indices( tensor1.size(-2), tensor1.size(-1), 1 )
    indices_r = indices[0]
    indices_c = indices[1]
    batch_size = tensor1.size(dim=0)
    dtype = tensor1.dtype
    device = tensor1.device
    corr_batch = torch.zeros( (batch_size,), dtype=dtype, device=device )
    for b in range(batch_size):
        triu_pair = torch.stack( (tensor1[b,indices_r,indices_c], tensor2[b,indices_r,indices_c]) )
        pair_corr = torch.corrcoef(triu_pair)
        corr_batch[b] = pair_corr[0,1]
    return corr_batch
    
def get_data_time_series_from_single_file(data_directory:str, data_subset:str, subjects_start:int, subjects_end:int, num_nodes:int, data_threshold:float):
    ts_file = os.path.join( data_directory, f'sc_fmri_ts_{data_subset}.pt' )
    data_ts = torch.load(ts_file)[subjects_start:subjects_end,:,:,:num_nodes]
    data_std, data_mean = torch.std_mean(data_ts, dim=-2, keepdim=True)
    data_ts -= data_mean
    data_ts /= data_std
    data_ts = data_ts.flatten(start_dim=1, end_dim=-2)
    data_ts = torch.sign(data_ts - data_threshold)
    return data_ts
    
def get_batched_ising_models(batch_size:int, num_nodes:int):
    J = torch.randn( (batch_size, num_nodes, num_nodes), dtype=float_type, device=device )
    J = 0.5*( J + J.transpose(dim0=-2, dim1=-1) )
    J = J - torch.diag_embed( torch.diagonal(J,dim1=-2,dim2=-1) )
    h = torch.zeros( (batch_size, num_nodes), dtype=float_type, device=device )
    s = 2.0*torch.randint( 2, (batch_size, num_nodes), dtype=float_type, device=device ) - 1.0
    return J, h, s
    
def print_if_nan(mat:torch.Tensor):
    mat_is_nan = torch.isnan(mat)
    num_nan = torch.count_nonzero(mat_is_nan)
    if num_nan > 0:
        print( mat.size(), f'matrix contains {num_nan} NaNs.' )

def get_negative_log_pseudolikelihood_loss(h:torch.Tensor, J:torch.Tensor, data_ts:torch.Tensor, node_index:int, beta:float):
    not_node_indices = torch.arange( data_ts.size(dim=-1), dtype=int_type, device=device ) != node_index
    nlpll = torch.mean(  torch.log( 1+torch.exp(-2*beta*data_ts[:,:,node_index]*(h[:,node_index] + torch.sum( data_ts[:,:,not_node_indices] * J[:,not_node_indices,node_index]))) )  )
    return nlpll

# Use this model to train the bias and input weights to a single node of the Ising model.
# node_ts should have size (num_subjects, num_time_points, 1).
# not_node_ts should have size (num_subjects, num_time_points, num_nodes-1).
# beta is a fixed, scalar hyperparameter of the Ising model.
# The return value is a scalar, the mean negative log pseudolikelihood
# of getting the observed node_ts given the observed not_node_ts and h_node and J_not_node_vs_node.
class IsingModelNegativeLogPseudoLikelihoodNode(torch.nn.Module):
    def __init__(self, num_subjects:int, num_other_nodes:int, dtype=float_type, device=device, beta:float=beta):
        super(IsingModelNegativeLogPseudoLikelihoodNode, self).__init__()
        self.beta = beta
        self.h_node = torch.nn.Parameter( torch.randn( (num_subjects, 1, 1), dtype=dtype, device=device ) )
        self.J_not_node_vs_node = torch.nn.Parameter(  torch.randn( (num_subjects, num_other_nodes, 1), dtype=dtype, device=device )  )
    def forward(self, node_ts:torch.Tensor, not_node_ts):
        deltaH = 2.0 * node_ts * ( self.h_node + torch.matmul(not_node_ts, self.J_not_node_vs_node) )
        prob_accept = torch.clamp( torch.exp(-self.beta*deltaH), min=0.0, max=1.0 )
        return torch.mean( torch.log(1+prob_accept) )

def run_ising_model_sim(h:torch.Tensor, J:torch.Tensor, num_time_points:int, beta:float=beta):
    print( f'NaNs in h: {h.isnan().count_nonzero()}' )
    print( f'NaNs in J: {J.isnan().count_nonzero()}' )
    num_subjects, num_nodes, _ = h.size()
    print(num_subjects, num_nodes)
    sim_ts = torch.zeros( (num_subjects, num_time_points, num_nodes), dtype=float_type, device=device )
    num_flips = 0
    state = 2.0*torch.randint( low=0, high=1, size=(num_subjects, 1, num_nodes), device=device ) - 1.0
    # print( state.size() )
    all_indices = torch.arange( data_ts.size(dim=-1), dtype=int_type, device=device )
    for t in range(num_time_points):
        node_order = torch.randperm(n=num_nodes, dtype=int_type, device=device)
        for i in range(num_nodes):
            node_index = node_order[i]
            not_node_indices = all_indices != node_index
            node_state = state[:,:,node_index].reshape( (num_subjects, 1, 1) )
            not_node_state = state[:,:,not_node_indices].reshape( (num_subjects, 1, num_nodes-1) )
            h_node = h[:,node_index].reshape( (num_subjects, 1, 1) )
            J_not_node_vs_node = J[:,not_node_indices,node_index].reshape( (num_subjects, num_nodes-1, 1) )
            deltaH = 2 * node_state * ( h_node + torch.matmul(not_node_state, J_not_node_vs_node) )
            prob_accept = torch.clamp( torch.exp(-beta*deltaH), min=0.0, max=1.0 )
            if prob_accept.isnan().count_nonzero() > 0:
                print( deltaH.squeeze() )
                print( prob_accept.squeeze() )
                prob_accept[prob_accept.isnan().flatten(),:,:] = 0.0
            flip = torch.bernoulli(prob_accept)
            state[:,0,node_index] *= ( 1.0 - 2.0 * flip[:,0,0].float() )
            num_flips += torch.count_nonzero(flip)
        sim_ts[:,t,:] = state.squeeze()
        for node_index in range(num_nodes):
            if state[0,0,node_index] > 0:
                print('+', end="")
            else:
                print('-', end="")
        print(f'|{t}')
    return sim_ts, num_flips

print('loading fMRI data...')
data_ts = get_data_time_series_from_single_file(data_directory=data_directory, data_subset=data_subset, subjects_start=subjects_start, subjects_end=subjects_end, num_nodes=num_nodes, data_threshold=threshold)
num_subjects, num_time_points, num_nodes = data_ts.size()
num_batches = num_time_points//batch_size
full_h = torch.zeros( (num_subjects, num_nodes, 1), dtype=float_type, device=device )
full_J = torch.zeros( (num_subjects, num_nodes, num_nodes), dtype=float_type, device=device )
for rep in range(num_reps):
    last_time = time.time()
    for node_index in range(num_nodes):
        not_node_indices = torch.arange( data_ts.size(dim=-1), dtype=int_type, device=device ) != node_index
        node_data_ts = data_ts[:,:,node_index].reshape( (num_subjects, num_time_points, 1) )
        not_node_data_ts = data_ts[:,:,not_node_indices].reshape( (num_subjects, num_time_points, num_nodes-1) )
        imnlpln_fn = IsingModelNegativeLogPseudoLikelihoodNode(num_subjects=num_subjects, num_other_nodes=num_nodes-1, dtype=float_type, device=device)
        optimizer = torch.optim.Adam( imnlpln_fn.parameters(), lr=learning_rate )
        for epoch in range(num_epochs):
            time_point_order = torch.randperm(num_time_points, dtype=int_type, device=device)
            node_data_ts_shuffled = node_data_ts[:,time_point_order,:]
            not_node_data_ts_shuffled = not_node_data_ts[:,time_point_order,:]
            for batch in range(num_batches):
                batch_start = batch*batch_size
                batch_end = batch_start+batch_size
                node_data_ts_batch = node_data_ts_shuffled[:,batch_start:batch_end,:]
                not_node_data_ts_batch = not_node_data_ts_shuffled[:,batch_start:batch_end,:]
                optimizer.zero_grad()
                loss = imnlpln_fn(node_data_ts_batch, not_node_data_ts_batch)
                loss.backward()
                optimizer.step()
                current_time = time.time()
            if current_time - last_time >= print_every_seconds:
                loss = imnlpln_fn(node_data_ts, not_node_data_ts)
                print( f'{node_index}, {epoch}, {loss.item()}, {current_time-code_start_time}' )
                last_time = current_time
        full_h[:,node_index,0] = imnlpln_fn.h_node[:,0,0]
        full_J[:,not_node_indices,node_index] = imnlpln_fn.J_not_node_vs_node[:,:,0]
        loss = imnlpln_fn(node_data_ts, not_node_data_ts)
        print( f'{node_index}, {epoch}, {loss.item()}, {current_time-code_start_time}' )
    # Save the trained model for the subject as two files, one for J and one for h.
    file_suffix = f'pl_data_{data_subset}_nodes_{num_nodes}_rep_{rep}_epochs_{num_epochs}_lr_{learning_rate}_threshold_{threshold}_beta_{beta}_start_{subjects_start}_end_{subjects_end}'
    print('saving Ising model J and h...')
    J_file_name = os.path.join(output_directory, f'J_{file_suffix}.pt')
    torch.save(full_J, J_file_name)
    h_file_name = os.path.join(output_directory, f'h_{file_suffix}.pt')
    torch.save(full_h, h_file_name)
    print('simulating...')
    with torch.no_grad():
        sim_ts, num_flips = run_ising_model_sim(h=full_h, J=full_J, num_time_points=num_time_points, beta=beta)
        data_fc = get_fc_batch(data_ts)
        sim_fc = get_fc_batch(sim_ts)
        fc_corr = get_triu_corr_batch(sim_fc, data_fc)
        fc_rmse = get_triu_rmse_batch(sim_fc, data_fc)
        current_time = time.time()
        print(f'{rep:.3g}\t{num_flips}\t{fc_corr.min():.3g}\t{fc_rmse.max():.3g}\t{current_time - code_start_time:.3g}')
current_time = time.time()
print(f'done, time {current_time - code_start_time}')