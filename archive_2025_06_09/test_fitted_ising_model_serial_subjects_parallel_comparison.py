import os
import torch
import hcpdatautils as hcp
import isingutils as ising
import time

def run_batched_parallel_metropolis_sim(sim_ts:torch.Tensor, J:torch.Tensor, h:torch.Tensor, s:torch.Tensor, num_steps:int, beta:float=0.5):
    # batch_size, num_nodes = s.size()
    # if not torch.is_tensor(sim_ts):
    #     sim_ts = torch.zeros( (batch_size, num_steps, num_nodes), dtype=s.dtype, device=s.device )
    for t in range(num_steps):
        deltaH = 2.0 * (   torch.matmul( s[:,None,:], J ).squeeze() + h   ) * s
        prob_accept = torch.clamp( torch.exp(-beta*deltaH), min=0.0, max=1.0 )
        flip = torch.bernoulli(prob_accept)
        s *= ( 1.0 - 2.0*flip )
        sim_ts[:,t,:] = s
    return sim_ts, s

start_time = time.time()

int_type = torch.int
float_type = torch.float
device = torch.device('cuda')

subject_list = ['516742']
# param_string = 'window_length_test_nodes_360_epochs_10000_max_window_2400_lr_0.001_threshold_0.100'
param_string = 'nodes_90_reps_100_epochs_24000_window_240_lr_0.001_threshold_0.100'
batch_size = 100
data_root_dir = 'E:'
model_dir = 'Ising_model_results_daai'
data_dir = 'HCP_data'
for subject in subject_list:
    print('loading data...')
    data_ts = ising.standardize_and_binarize_ts_data(  ts=hcp.load_all_time_series_for_subject( directory_path=os.path.join(data_root_dir, data_dir), subject_id=subject, dtype=float_type, device=device )  ).flatten(start_dim=0, end_dim=1)
    num_time_points, num_nodes_data = data_ts.size()
    print(f'data ts size {num_time_points} x {num_nodes_data}')
    data_fc = hcp.get_fc(data_ts).unsqueeze(dim=0).repeat( (batch_size, 1, 1) )
    print('loading models...')
    h = torch.load( os.path.join(data_root_dir, model_dir, f'h_{param_string}_subject_{subject}.pt') ).squeeze()
    print( 'h size', h.size() )
    J = torch.load( os.path.join(data_root_dir, model_dir, f'J_{param_string}_subject_{subject}.pt') )
    print( 'J size', J.size() )
    num_models, num_nodes = h.size()
    num_batches = num_models//batch_size + int(num_models % batch_size > 0)
    print('allocating time series Tensor...')
    sim_ts = torch.zeros( (batch_size, num_time_points, num_nodes), dtype=float_type, device=device )
    sim_ts_p = torch.zeros( (batch_size, num_time_points, num_nodes), dtype=float_type, device=device )
    fc_rmse_set = torch.zeros( (num_models,), dtype=float_type, device=device )
    fc_corr_set = torch.zeros( (num_models,), dtype=float_type, device=device )
    fc_rmse_set_p = torch.zeros( (num_models,), dtype=float_type, device=device )
    fc_corr_set_p = torch.zeros( (num_models,), dtype=float_type, device=device )
    print( 'sim_ts size ', sim_ts.size() )
    for batch in range(num_batches):
        batch_start = batch*batch_size
        batch_end = min(batch_start + batch_size, num_models)
        current_batch_size = batch_end - batch_start
        print(f'batch {batch} of {num_batches}, {batch_start} to {batch_end}')
        h_batch = h[batch_start:batch_end,:]
        J_batch = J[batch_start:batch_end,:,:]
        print('initializing state...')
        s = ising.get_random_state(batch_size=current_batch_size, num_nodes=num_nodes, dtype=float_type, device=device)
        s_p = ising.get_random_state(batch_size=current_batch_size, num_nodes=num_nodes, dtype=float_type, device=device)
        print('running sim...')
        sim_ts, s = ising.run_batched_balanced_metropolis_sim(sim_ts=sim_ts, J=J_batch, h=h_batch, s=s, num_steps=num_time_points)
        sim_ts_p, s_p = run_batched_parallel_metropolis_sim(sim_ts=sim_ts_p, J=J_batch, h=h_batch, s=s, num_steps=num_time_points)
        sim_fc = hcp.get_fc_batch(sim_ts)
        sim_fc_p = hcp.get_fc_batch(sim_ts_p)
        fc_rmse_set[batch_start:batch_end] = hcp.get_triu_rmse_batch(sim_fc, data_fc)
        fc_corr_set[batch_start:batch_end] = hcp.get_triu_corr_batch(sim_fc, data_fc)
        fc_rmse_set_p[batch_start:batch_end] = hcp.get_triu_rmse_batch(sim_fc_p, data_fc)
        fc_corr_set_p[batch_start:batch_end] = hcp.get_triu_corr_batch(sim_fc_p, data_fc)
        elapsed_time = time.time() - start_time
        print(f'time {elapsed_time:.3f}')
    num_parallel_worse_rmse = torch.count_nonzero(fc_rmse_set_p > fc_rmse_set)
    num_parallel_worse_corr = torch.count_nonzero(fc_corr_set_p < fc_corr_set)
    print('RMSEs serial vs parallel')
    print(  torch.stack( (fc_rmse_set, fc_rmse_set_p), dim=0 )  )
    print('correlations serial vs parallel')
    print(  torch.stack( (fc_corr_set, fc_corr_set_p), dim=0 )  )
    print(f'{num_parallel_worse_corr} had worse correlations and {num_parallel_worse_rmse} out of {num_models} had worse RMSEs when using the parallel Ising model simulation.')
    print('saving results...')
    rmse_file = os.path.join('E:', 'Ising_model_results_batch', 'fc_rmse_'+param_string+'.pt')
    torch.save(fc_rmse_set, rmse_file)
    corr_file = os.path.join('E:', 'Ising_model_results_batch', 'fc_corr_'+param_string+'.pt')
    torch.save(fc_corr_set, corr_file)
    elapsed_time = time.time() - start_time
    print(f'done, time {elapsed_time:.3f}')
