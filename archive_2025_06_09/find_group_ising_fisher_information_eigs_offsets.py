import os
import torch
import time
import argparse
import hcpdatautils as hcp
import isingutils as ising

start_time = time.time()

parser = argparse.ArgumentParser(description="Compute Fisher information matrix of group-fMRI-trained Ising model. Then find its eigenvalues and eigenvectors and the offsets of individual Ising models from it along eigenvectors.")

# directories
parser.add_argument("-d", "--data_dir", type=str, default='E:\\HCP_data', help="directory containing the structural MRI features data file")
parser.add_argument("-m", "--model_dir", type=str, default='E:\\Ising_model_results_daai', help="directory containing the fitted Ising model h and J .pt files")
parser.add_argument("-s", "--stats_dir", type=str, default="E:\\Ising_model_results_daai", help="directory to which to write the output Fisher information matrices and other results")
parser.add_argument("-n", "--num_nodes", type=int, default=21, help="number of nodes in each Ising model")
parser.add_argument("-p", "--model_param_string", type=str, default='nodes_21_window_50_lr_0.000_threshold_0.100_beta_0.500', help="the part of the h or J file name after 'h_' or 'J_' and before '_subject_*_rep_*_epoch_*.pt'")
parser.add_argument("-i", "--num_reps_indi", type=int, default=100, help="number of Ising models per subject")
parser.add_argument("-e", "--epoch_indi", type=int, default=1999, help="epoch at which we ended individual Ising model fitting, used for constructing file names")
parser.add_argument("-r", "--num_reps_group", type=int, default=1000, help="number of group Ising models")
parser.add_argument("-g", "--epoch_group", type=int, default=4, help="epoch at which we ended group Ising model fitting, used for constructing file names")
parser.add_argument("-b", "--beta", type=float, default=0.5, help="beta constant to use when simulating the Ising model")
parser.add_argument("-t", "--num_steps", type=int, default=48000, help="number of steps to use when running Ising model simulations to compare FC")

# We are not counting the first or last linear layer as hidden,
# so every network has at least two layers.

args = parser.parse_args()

data_dir = args.data_dir
print(f'data_dir {data_dir}')
model_dir = args.model_dir
print(f'model_dir {model_dir}')
stats_dir = args.stats_dir
print(f'stats_dir {stats_dir}')
num_nodes = args.num_nodes
print(f'num_nodes {num_nodes}')
model_param_string = args.model_param_string
print(f'model_param_string {model_param_string}')
num_reps_group = args.num_reps_group
print(f'num_reps_group {num_reps_group}')
num_reps_indi = args.num_reps_indi
print(f'num_reps_indi {num_reps_indi}')
epoch_indi = args.epoch_indi
print(f'epoch_indi {epoch_indi}')
epoch_group = args.epoch_group
print(f'epoch_group {epoch_group}')
num_steps = args.num_steps
print(f'num_steps {num_steps}')
beta = args.beta
print(f'beta {beta}')

float_type = torch.float
device = torch.device('cuda')

def load_group_ising_models(model_param_string:str, num_reps:int, epoch:int, num_nodes:int):
    h = torch.zeros( (num_reps, num_nodes), dtype=float_type, device=device )
    J = torch.zeros( (num_reps, num_nodes, num_nodes), dtype=float_type, device=device )
    for rep in range(num_reps):
        # The group Ising model should always be based on the training data so that we are not snooping the other subsets.
        file_suffix = f'{model_param_string}_group_training_rep_{rep}_epoch_{epoch}.pt'
        h_file = os.path.join(model_dir, f'h_{file_suffix}')
        J_file = os.path.join(model_dir, f'J_{file_suffix}')
        h[rep, :] = torch.load(f=h_file)
        J[rep, :, :] = torch.load(f=J_file)
    return h, J

def load_indi_ising_models(model_param_string:str, subject_ids:list, num_reps:int, epoch:int, num_nodes:int):
    num_subjects = len(subject_ids)
    h = torch.zeros( (num_subjects, num_reps, num_nodes), dtype=float_type, device=device )
    J = torch.zeros( (num_subjects, num_reps, num_nodes, num_nodes), dtype=float_type, device=device )
    for subject_index in range(num_subjects):
        subject_id = subject_ids[subject_index]
        for rep in range(num_reps):
            file_suffix = f'{model_param_string}_subject_{subject_id}_rep_{rep}_epoch_{epoch}.pt'
            h_file = os.path.join(model_dir, f'h_{file_suffix}')
            J_file = os.path.join(model_dir, f'J_{file_suffix}')
            h[subject_index, rep, :] = torch.load(f=h_file)
            J[subject_index, rep, :, :] = torch.load(f=J_file)
    return h, J

# creates a num_rows*num_cols 1-D Tensor of booleans where each value is True if and only if it is part of the upper triangle of a flattened num_rows x num_cols matrix.
# If we want the upper triangular part of a Tensor with one or more batch dimensions, we can flatten the last two dimensions together, and then use this.
def get_triu_logical_index(num_rows:int, num_cols:int):
    return ( torch.arange(start=0, end=num_rows, dtype=torch.int, device=device)[:,None] < torch.arange(start=0, end=num_cols, dtype=torch.int, device=device)[None,:] ).flatten()

# Run the Ising model simulation, computing the Fisher information matrix as we go.
# The method is based on
# Panas, D., Amin, H., Maccione, A., Muthmann, O., van Rossum, M., Berdondini, L., & Hennig, M. H. (2015).
# Sloppiness in spontaneously active neuronal networks. Journal of Neuroscience, 35(22), 8480-8492.
# https://www.jneurosci.org/content/35/22/8480
# For each parameter, we have a variable x.
# For h_i, x_i is the state of node i.
# For J_ij, x_{num_nodes + i*num_nodes + j} is the product of the states of nodes i and j.
# The Fisher information matrix then has elements F[i,j] = cov(x_i,x_j) taken over the course of the simulation.
# To save memory, instead of recording the entire time series and then calculating the FIM,
# we use the formula cov(x_i,x_j) = mean(x_i * x_j) - mean(x_i) * mean(x_j)
# and compute each relevant mean by adding the relevant value to a running total over the course of the simulation
# and dividing and subtracting as appropriate at the end.
def get_ising_model_fisher_information_matrix(h:torch.Tensor, J:torch.Tensor, num_steps:int=4800, beta:torch.float=0.5):
    num_h_dims = len( h.size() )
    if num_h_dims < 2:
        h=h.unsqueeze(dim=0)
    elif num_h_dims > 2:
        h=h.flatten(start_dim=0, end_dim=1)
    num_J_dims = len( J.size() )
    if num_J_dims < 3:
        J=J.unsqueeze(dim=0)
    elif num_J_dims > 3:
        J=J.flatten(start_dim=0, end_dim=-3)
    batch_size, num_nodes = h.size()
    ut_logical = get_triu_logical_index(num_rows=num_nodes, num_cols=num_nodes)
    s = ising.get_random_state(batch_size=batch_size, num_nodes=num_nodes, dtype=h.dtype, device=device)
    s_sum = torch.zeros_like(s)
    s_product_sum = s_sum[:,:,None] * s_sum[:,None,:]
    params = torch.cat(  ( s_sum, s_product_sum.flatten(start_dim=-2, end_dim=-1)[:,ut_logical] ), dim=-1  )
    param_product_sum = params[:,:,None] * params[:,None,:]
    for _ in range(num_steps):
        s = ising.run_batched_balanced_metropolis_sim_step(J=J, h=h, s=s, beta=beta)
        s_sum += s
        s_product = (s[:,:,None] * s[:,None,:])
        s_product_sum += s_product
        params = torch.cat(  ( s, s_product.flatten(start_dim=-2, end_dim=-1)[:,ut_logical] ), dim=-1  )
        param_product_sum += params[:,:,None] * params[:,None,:]
    # In the end, we only want one num_params x num_params matrix where num_params = num_nodes + num_nodes*(num_nodes-1)/2.
    total_num_steps = batch_size * num_steps
    param_mean = torch.cat(  ( s_sum, s_product_sum.flatten(start_dim=-2, end_dim=-1)[:,ut_logical] ), dim=-1  ).sum(dim=0)/total_num_steps
    param_cov = param_product_sum.sum(dim=0)/total_num_steps - (param_mean[:,None] * param_mean[None,:])
    return param_cov

def get_param_vectors(h:torch.Tensor, J:torch.Tensor):
    num_nodes = J.size(dim=-1)
    J_flat = J.flatten(start_dim=-2, end_dim=-1)
    ut_indices = get_triu_logical_index(num_rows=num_nodes, num_cols=num_nodes).nonzero().flatten()
    J_flat_ut = torch.index_select(input=J_flat, dim=-1, index=ut_indices)
    return torch.cat( (h, J_flat_ut), dim=-1 )

with torch.no_grad():
    code_start_time = time.time()

    # group model-only calculations
    print(f'loading group Ising model..., time {time.time() - code_start_time:.3f}')
    h_group, J_group = load_group_ising_models(model_param_string=model_param_string, num_reps=num_reps_group, epoch=epoch_group, num_nodes=num_nodes)
    print(f'computing FIMs of group models..., time {time.time() - code_start_time:.3f}')
    fim_group = get_ising_model_fisher_information_matrix(h=h_group, J=J_group, num_steps=num_steps, beta=beta)
    fim_param_string = f'{model_param_string}_reps_{num_reps_group}_epoch_{epoch_group}'
    fim_file_name = os.path.join(stats_dir, f'fim_ising_{fim_param_string}.pt')
    torch.save(obj=fim_group, f=fim_file_name)
    print(f'saved {fim_file_name}, time {time.time() - code_start_time:.3f}')
    print(f'finding eigenvalues and eigenvectors of FIM..., time {time.time() - code_start_time:.3f}')
    L_group, V_group = torch.linalg.eig(fim_group)
    L_file_name = os.path.join(stats_dir, f'L_ising_{fim_param_string}.pt')
    torch.save(obj=L_group, f=L_file_name)
    print(f'saved {L_file_name}, time {time.time() - code_start_time:.3f}')
    V_file_name = os.path.join(stats_dir, f'V_ising_{fim_param_string}.pt')
    torch.save(obj=V_group, f=V_file_name)
    print(f'saved {V_file_name}, time {time.time() - code_start_time:.3f}')
    log_L_group = torch.log2(L_group.real)
    print(f'computing projections of group Ising models onto FIM eigenvectors..., time {time.time() - code_start_time:.3f}')
    params_group = get_param_vectors(h=h_group, J=J_group)
    # params_group should now be num_reps x num_params, and V_group should now be num_params x num_params.
    projections_group = torch.matmul( params_group, V_group )
    projections_group_file_name = os.path.join(stats_dir, f'projections_group_ising_{fim_param_string}.pt')
    torch.save(obj=projections_group, f=projections_group_file_name)
    print(f'saved {projections_group_file_name}, time {time.time() - code_start_time:.3f}')
    proj_var_group = projections_group.real.var(dim=0)
    log_proj_var_group = torch.log2(proj_var_group)
    L_var_corr_group = torch.corrcoef(  torch.stack( (log_L_group, log_proj_var_group), dim=0 )  )[0,1].item()
    print(f'correlation between variance of separately fitted group model parameters along eigenvector and eigenvalue in log-log scale {L_var_corr_group:.3g}')

    # calculations with training set individual models
    for data_subset in ['training', 'validation', 'testing']:
        print(f'computing projections of individual {data_subset} Ising models onto FIM eigenvectors..., time {time.time() - code_start_time:.3f}')
        indi_param_string = f'indi_{data_subset}_ising_{fim_param_string}'
        subject_ids = hcp.load_subject_subset(directory_path=data_dir, subject_subset=data_subset, require_sc=True)
        h_indi, J_indi = load_indi_ising_models(model_param_string=model_param_string, subject_ids=subject_ids, num_reps=num_reps_indi, epoch=epoch_indi, num_nodes=num_nodes)
        print(f'computing projections of individual {data_subset} Ising models onto FIM eigenvectors..., time {time.time() - code_start_time:.3f}')
        params_indi = get_param_vectors(h=h_indi, J=J_indi)
        # params_indi should now be num_subjects x num_reps x num_params, while V_group is still num_params x num_params.
        projections_indi = torch.matmul( params_indi, V_group.unsqueeze(dim=0) )
        projections_indi_file_name = os.path.join(stats_dir, f'projections_{indi_param_string}.pt')
        torch.save(obj=projections_indi, f=projections_indi_file_name)
        print(f'saved {projections_indi_file_name}, time {time.time() - code_start_time:.3f}')
        proj_var_indi = projections_indi.real.var( dim=(0,1) )
        log_proj_var_indi = torch.log(proj_var_indi)
        L_var_corr_indi = torch.corrcoef(  torch.stack( (log_L_group, log_proj_var_indi), dim=0 )  )[0,1].item()
        print(f'correlation between variance of individual model parameters along eigenvector and eigenvalue in log-log scale {L_var_corr_indi:.3g}')
        offsets_indi = projections_indi - projections_group.mean(dim=0, keepdim=True)
        offsets_indi_file_name = os.path.join(stats_dir, f'offsets_{indi_param_string}.pt')
        torch.save(obj=offsets_indi, f=offsets_indi_file_name)
        print(f'saved {offsets_indi_file_name}, time {time.time() - code_start_time:.3f}')

    print('done')