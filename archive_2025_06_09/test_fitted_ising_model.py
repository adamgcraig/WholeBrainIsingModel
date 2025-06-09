import os
import torch
# import hcpdatautils as hcp
import isingutils as ising
import time

start_time = time.time()

int_type = torch.int
float_type = torch.float
device = torch.device('cuda')

print('loading models...')
model_dir = 'E:\\Ising_model_results_daai'
param_string = 'pl_data_validation_nodes_360_rep_19_epochs_100000_lr_1e-05_threshold_0.1_beta_0.5_start_0_end_83'
h = torch.load( os.path.join(model_dir, f'h_{param_string}.pt') ).squeeze()
print( 'h size', h.size() )
J = torch.load( os.path.join(model_dir, f'J_{param_string}.pt') )
print( 'J size', J.size() )

num_subjects, num_nodes = h.size()
num_time_points = 4800
print('allocating time series Tensor...')
sim_ts = torch.zeros( (num_subjects, num_time_points, num_nodes), dtype=float_type, device=device )
print( 'sim_ts size ', sim_ts.size() )
print('initializing state...')
_, _, s = ising.get_batched_ising_models(batch_size=num_subjects, num_nodes=num_nodes, dtype=float_type, device=device)
print('running sim...')
sim_ts, s = ising.run_batched_balanced_metropolis_sim(sim_ts=sim_ts, J=J, h=h, s=s, num_steps=num_time_points)
print('saving results...')
torch.save( sim_ts, os.path.join('E:', 'Ising_model_results_batch', 'sim_ts_'+param_string+'.pt') )
elapsed_time = time.time() - start_time
print(f'done, time {elapsed_time:.3f}')
