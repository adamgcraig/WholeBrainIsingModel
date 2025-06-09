import os
import torch
import time
import argparse
import copy
import hcpdatautils as hcp
import isingutils as ising
from collections import OrderedDict

code_start_time = time.time()

parser = argparse.ArgumentParser(description="Compute Fisher information matrix of group-fMRI-trained Ising model. Then find its eigenvalues and eigenvectors and the offsets of individual Ising models from it along eigenvectors.")

# directories
parser.add_argument("-d", "--data_dir", type=str, default='E:\\HCP_data', help="directory containing the structural MRI features data file")
parser.add_argument("-m", "--model_dir", type=str, default='E:\\Ising_model_results_daai', help="directory containing the fitted Ising model h and J .pt files")
parser.add_argument("-s", "--stats_dir", type=str, default="E:\\Ising_model_results_daai", help="directory to which to write the output Fisher information matrices and other results")
parser.add_argument("-n", "--num_nodes", type=int, default=21, help="number of nodes in each Ising model")
parser.add_argument("-i", "--fim_param_string", type=str, default='nodes_21_window_50_lr_0.000_threshold_0.100_beta_0.500_reps_1000_epoch_4', help="the part of the h or J file name after 'h_' or 'J_' and before '_subject_*_rep_*_epoch_*.pt'")
# parser.add_argument("-o", "--struct_to_offset_param_string", type=str, default='reps_1000_epoch_4_depth_0_width_2_batch_1000_lr_0.0001_reps_1000_epoch_4', help="the part of the struct-to-offsets model file name after the Ising model param string and before '_dim_*.pt'")
parser.add_argument("-o", "--struct_to_offset_param_string", type=str, default='depth_10_width_1000_batch_10000_lr_0.01', help="the part of the struct-to-offsets model file name after the Ising model param string and before '_dim_*.pt'")
parser.add_argument("-r", "--num_reps_group", type=int, default=1000, help="number of group Ising models")
parser.add_argument("-g", "--epoch_group", type=int, default=4, help="epoch at which we ended group Ising model fitting, used for constructing file names")
parser.add_argument("-b", "--beta", type=float, default=0.5, help="beta constant to use when simulating the Ising model")
parser.add_argument("-t", "--num_steps", type=int, default=48000, help="number of steps to use when running Ising model simulations to compare FC")
parser.add_argument("-z", "--z_score", type=bool, default=True, help="set to True to z-score the data before training, using the training sample mean and std. dev. for both training and validation data")
parser.add_argument("-y", "--num_optimizer_steps", type=int, default=100000, help="number of steps to use when optimizing the group structural features to produce the group Ising model")
parser.add_argument("-l", "--optimizer_learning_rate", type=float, default=0.000001, help="learning rate to use when optimizing the group structural features to produce the group Ising model")
parser.add_argument("-p", "--optimizer_print_every_steps", type=int, default=1000, help="number of steps between printouts when optimizing the group structural features to produce the group Ising model")

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
fim_param_string = args.fim_param_string
print(f'fim_param_string {fim_param_string}')
struct_to_offset_param_string = args.struct_to_offset_param_string
print(f'struct_to_offset_param_string {struct_to_offset_param_string}')
num_reps_group = args.num_reps_group
print(f'num_reps_group {num_reps_group}')
epoch_group = args.epoch_group
print(f'epoch_group {epoch_group}')
num_steps = args.num_steps
print(f'num_steps {num_steps}')
beta = args.beta
print(f'beta {beta}')
z_score = args.z_score
print(f'z_score {z_score}')
num_optimizer_steps = args.num_optimizer_steps
print(f'num_optimizer_steps {num_optimizer_steps}')
optimizer_learning_rate = args.optimizer_learning_rate
print(f'optimizer_learning_rate {optimizer_learning_rate}')
optimizer_print_every_steps = args.optimizer_print_every_steps
print(f'optimizer_print_every_steps {optimizer_print_every_steps}')

float_type = torch.float
device = torch.device('cuda')

combined_param_string = f'struct_to_offset_{fim_param_string}_{struct_to_offset_param_string}'

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

# We want to get the gradient of the output of a model with respect to its inputs.
# To do this, we need to register the inputs themselves as a Parameter set.
class GradientTracker(torch.nn.Module):

    def __init__(self, model:torch.nn.Module, input_values:torch.Tensor):
        super(GradientTracker, self).__init__()
        self.model = model
        self.stored_input = torch.nn.Parameter( input_values.clone() )
    
    # Use this when we just want to do a forward pass of the data through the underlying model.
    def forward(self, input:torch.Tensor=None):
        return self.model(input)
    
    # Use this when we want the output of the underlying model
    # and also the gradient of just the output of this underlying model
    # with respect to the input Tensor
    # at the input values provided or, if none, the stored values.
    def get_output_and_gradients(self, input:torch.Tensor=None):
        # print('input to get_output_and_gradients', input)
        if type(input) != type(None):
            with torch.no_grad():
                self.stored_input.copy_(input)
        if type(self.stored_input.grad) != type(None):
            self.stored_input.grad.zero_()
        output = self(self.stored_input)
        output.backward( torch.ones_like(output) )
        return output, self.stored_input.grad

# We want to keep the gradient trackers for all the offsets together.
# All offset models take the same structural data input vector.
class GradientTrackerList(torch.nn.Module):

    def __init__(self, offset_models:list, input_values:torch.Tensor) -> None:
        super(GradientTrackerList, self).__init__()
        self.shared_input = torch.nn.Parameter(input_values)
        self.gradient_trackers = torch.nn.ModuleList( [GradientTracker(model=offset_model, input_values=input_values) for offset_model in offset_models] )
    
    def forward(self):
        return torch.stack( [gradient_tracker(self.shared_input) for gradient_tracker in self.gradient_trackers], dim=0 )
    
    def get_outputs_and_jacobian(self, input:torch.Tensor=None):
        # print('input to get_outputs_and_jacobian', input)
        if type(input) == type(None):
            input = self.shared_input
        pairs = [gradient_tracker.get_output_and_gradients(input) for gradient_tracker in self.gradient_trackers]
        # for pair in pairs:
        #     print( pair[0], pair[1].size() )
        output = torch.stack([pair[0] for pair in pairs], dim=0)
        jacobian = torch.stack([pair[1] for pair in pairs], dim=0)
        # print( 'output size', output.size() )
        # print( 'jacobian size', jacobian.size() )
        return output, jacobian

def optimize_shared_input(gradient_tracker_list:GradientTrackerList, target_output:torch.Tensor, weights:torch.Tensor=None, learning_rate:float=optimizer_learning_rate, num_steps:int=num_optimizer_steps, print_every_steps:int=optimizer_print_every_steps):
    if type(weights) == type(None):
        weights = torch.ones_like(target_output)
    weighted_target_output = target_output * weights
    # Set requires_grad so that we are only optimizing the shared input.
    for param in gradient_tracker_list.parameters():
        param.requires_grad = False
    gradient_tracker_list.shared_input.requires_grad = True
    # Train.
    optimizer = torch.optim.Adam( gradient_tracker_list.parameters(), lr=learning_rate )
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        input_mag = gradient_tracker_list.shared_input.square().sum().sqrt()
        output = gradient_tracker_list().square().sum().sqrt()
        loss = loss_fn(output * weights, weighted_target_output).sqrt()
        print(f'initial input magnitude {input_mag:.3g}, RMSE loss {loss:.3g}')
    for step in range(num_steps):
        optimizer.zero_grad()
        loss_fn( gradient_tracker_list() * weights, weighted_target_output ).backward()
        optimizer.step()
        if (step % print_every_steps) == (print_every_steps-1):
            with torch.no_grad():
                input_mag = gradient_tracker_list.shared_input.square().sum().sqrt()
                output = gradient_tracker_list().square().sum().sqrt()
                loss = loss_fn(output  * weights, weighted_target_output).sqrt()
                print(f'step {step+1} input magnitude {input_mag:.3g}, RMSE loss {loss:.3g}, time {time.time() - code_start_time:.3f}')
    with torch.no_grad():
        input_mag = gradient_tracker_list.shared_input.square().sum().sqrt()
        output = gradient_tracker_list().square().sum().sqrt()
        loss = loss_fn(output * weights, weighted_target_output).sqrt()
        print(f'final input magnitude {input_mag:.3g}, RMSE loss {loss:.3g}')
    # Set requires_grad back to True for everything so that we can compute gradients for the Jacobian.
    for param in gradient_tracker_list.parameters():
        param.requires_grad = True
    return gradient_tracker_list.shared_input

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

def load_struct_to_offset_models(fim_param_string:str, struct_to_offset_param_string:str, num_offsets:int, structural_features:torch.Tensor=None):
    print(f'loading struct-to-offset models from files..., time {time.time() - code_start_time:.3f}')
    model_list = [copy.deepcopy(  torch.load( f=os.path.join(model_dir, f'struct_to_offset_{fim_param_string}_{struct_to_offset_param_string}_dim_{offset_index}.pt') )  ) for offset_index in range(num_offsets)]
    print(f'adding all models to GradientTrackerList..., time {time.time() - code_start_time:.3f}')
    return GradientTrackerList(offset_models=model_list, input_values=structural_features)

def get_h_and_J(params:torch.Tensor, num_nodes:int=num_nodes):
    h = torch.index_select( input=params, dim=-1, index=torch.arange(end=num_nodes, device=params.device) )
    J_flat_ut = torch.index_select(  input=params, dim=-1, index=torch.arange( start=num_nodes, end=params.size(dim=-1), device=params.device )  )
    ut_indices = get_triu_logical_index(num_rows=num_nodes, num_cols=num_nodes).nonzero().flatten()
    J_flat_dims = list( J_flat_ut.size() )
    J_flat_dims[-1] = num_nodes * num_nodes
    J_flat = torch.zeros(J_flat_dims, dtype=params.dtype, device=params.device)
    J_flat.index_copy_(dim=-1, index=ut_indices, source=J_flat_ut)
    J = J_flat.unflatten( dim=-1, sizes=(num_nodes, num_nodes) )
    J_sym = J + J.transpose(dim0=-2, dim1=-1)
    # print( 'h size', h.size(), 'J_sym size', J_sym.size() )
    return h, J_sym

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

# Note that V_inv_group_ising can be of a complex number data type.
# Since it is the set of eigenvectors of the FIM of the group Ising model, which is itself real-valued and symmetric, the imaginary components should be 0.
# However, they may only be approximately 0.
# In general, it is safer to cast to the type where you do not lose any information, so we mostly follow that practice here.
# However, we do only use the real parts of the reconstructed Ising model parameters when actually running the Ising model.
# Only real numbers are meaningful for Ising model couplings and external fields, and letting them stay complex-valued introduces some errors.
def compute_and_save_struct_fim_and_offsets(gradient_tracker_list:GradientTrackerList, V_inv_group_ising:torch.Tensor, projections_group_ising:torch.Tensor, struct_features_centroid:torch.Tensor, structural_data_dict:dict, centroid_name:str, num_nodes:int=num_nodes, num_steps:int=num_steps, beta:torch.float=beta, data_dir:str=data_dir, stats_dir:str=stats_dir):
    # print('struct_features_centroid', struct_features_centroid)
    offset_ising, offset_jacobian = gradient_tracker_list.get_outputs_and_jacobian(input=struct_features_centroid)

    with torch.no_grad():
        offset_jacobian_file = os.path.join(stats_dir, f'offset_jacobian_{centroid_name}.pt')
        torch.save(obj=offset_jacobian, f=offset_jacobian_file)
        
        complex_type = V_inv_group_ising.dtype
        params_mean_struct = torch.matmul( (projections_group_ising + offset_ising).type(complex_type), V_inv_group_ising )
        # We do not need complex weights when running the Ising model. We do need to preserve the sign.
        h_mean_struct, J_mean_struct = get_h_and_J(params=params_mean_struct.real, num_nodes=num_nodes)
        print(f'constructed Ising model predicted at mean structural feature values, ready to simulate, time {time.time() - code_start_time:.3f}')
        fim_ising_mean_struct = get_ising_model_fisher_information_matrix(h=h_mean_struct, J=J_mean_struct, num_steps=num_steps, beta=beta).type(complex_type)
        print(f'done simulating and computing Ising model FIM, time {time.time() - code_start_time:.3f}')

        param_jacobian = torch.matmul( torch.transpose(V_inv_group_ising, dim0=0, dim1=1), offset_jacobian.type(complex_type) )
        fim_struct = torch.matmul(  torch.matmul( param_jacobian.transpose(dim0=0, dim1=1), fim_ising_mean_struct ), param_jacobian  )
        fim_struct_file = os.path.join(stats_dir, f'fim_{centroid_name}.pt')
        torch.save(obj=fim_struct, f=fim_struct_file)
        print(f'computed FIM with respect to structural features, saved to {fim_struct_file}, time {time.time() - code_start_time:.3f}')

        L_struct, V_struct = torch.linalg.eig(fim_struct)
        L_struct_file = os.path.join(stats_dir, f'L_{centroid_name}.pt')
        torch.save(obj=L_struct, f=L_struct_file)
        print(f'saved eigenvalues of structural features FIM to {L_struct_file}, time {time.time() - code_start_time:.3f}')
        V_struct_file = os.path.join(stats_dir, f'V_{centroid_name}.pt')
        torch.save(obj=V_struct, f=V_struct_file)
        print(f'saved eigenvectors of structural features FIM to {V_struct_file}, time {time.time() - code_start_time:.3f}')

        # calculations with training set individual models
        log_L = torch.log2(L_struct)
        # Note that we need to do the training subjects first to get the mean and std. dev. values.
        print( 'struct_features_centroid size', struct_features_centroid.size() )
        print( 'V_struct size', V_struct.size() )
        for data_subset_name, structural_features in structural_data_dict.items():
            print( data_subset_name, 'features size', structural_features.size() )
            offsets = torch.matmul(  ( structural_features - struct_features_centroid.unsqueeze(dim=0) ).type(complex_type), V_struct  )
            # print( 'offsets size', offsets.size() )
            offsets_file = os.path.join(stats_dir, f'projections_{centroid_name}_{data_subset_name}.pt')
            torch.save(obj=offsets, f=offsets_file)
            print(f'computed projections of individual {data_subset_name} structural features onto FIM eigenvectors, saved to {offsets_file}, time {time.time() - code_start_time:.3f}')
            offsets_var = offsets.real.var(dim=0)
            # print( 'offsets_var size', offsets_var.size() )
            log_offsets_var = torch.log2(offsets_var)
            L_var_corr = torch.corrcoef(  torch.stack( (log_L, log_offsets_var), dim=0 )  )[0,1].item()
            print(f'correlation between variance of individual structural features along eigenvector and eigenvalue in log-log scale {L_var_corr:.3g}')
    return offsets

# Load the models into the gradient tracker list.
# We z-score the structural features within each feature over using the mean and std. dev. of the training data subjects, so the "average" structural vector should be all zeros.
num_structural_features = hcp.features_per_area * num_nodes + ( num_nodes*(num_nodes-1) )//2
mean_structural_features = torch.zeros( (num_structural_features,), dtype=float_type, device=device )
num_offsets = num_nodes + ( num_nodes*(num_nodes-1) )//2
gradient_tracker_list = load_struct_to_offset_models(fim_param_string=fim_param_string, struct_to_offset_param_string=struct_to_offset_param_string, num_offsets=num_offsets, structural_features=mean_structural_features)
print(f'loaded struct-to-offset models from files, time {time.time() - code_start_time:.3f}')

L_file_name = os.path.join(stats_dir, f'L_ising_{fim_param_string}.pt')
L_group = torch.load(L_file_name)
print(f'loaded {L_file_name}, time {time.time() - code_start_time:.3f}')

V_file_name = os.path.join(stats_dir, f'V_ising_{fim_param_string}.pt')
V_group = torch.load(V_file_name)
V_group_inv = torch.linalg.inv(V_group)
print(f'loaded {V_file_name}, time {time.time() - code_start_time:.3f}')

projections_group_file_name = os.path.join(stats_dir, f'projections_group_ising_{fim_param_string}.pt')
projections_mean_group = torch.load(projections_group_file_name).mean(dim=0)# We only care about the mean group model, not individual replications.
print(f'loaded {projections_group_file_name}, time {time.time() - code_start_time:.3f}')

with torch.no_grad():
    training_std = None
    training_mean = None
    data_subset_names = ['training', 'validation', 'testing']
    data_subset_features = [[], [], []]
    for data_subset_index in range( len(data_subset_names) ):
        data_subset_name = data_subset_names[data_subset_index]
        data_subset_features[data_subset_index], current_std, current_mean = prepare_structural_data(subset=data_subset_name, z_score=z_score, struct_std=training_std, struct_mean=training_mean, num_nodes=num_nodes, structural_data_dir=data_dir)
        if data_subset_name == 'training':
            training_std = current_std
            training_mean = current_mean
        print(f'loaded {data_subset_name} structural data, time {time.time() - code_start_time:.3f}')
    structural_data_dict = {name: features for (name, features) in zip(data_subset_names, data_subset_features)}
    # print('structural_data_dict', structural_data_dict)

# Compute the FIM with respect to the structural features and the projections of individual structural features onto the eigenvectors of the FIM.
# print('mean_structural_features', mean_structural_features)
struct_to_offset_param_string
mean_struct_string = f'mean_{combined_param_string}'
offsets_mean_struct = compute_and_save_struct_fim_and_offsets(gradient_tracker_list=gradient_tracker_list, V_inv_group_ising=V_group_inv, projections_group_ising=projections_mean_group, struct_features_centroid=mean_structural_features, structural_data_dict=structural_data_dict, centroid_name=mean_struct_string)

# Optimize the structural features centroid so that its predicted offsets are as close to 0 as possible.
# This structure is the one that should give us an Ising model as close as possible to the mean group Ising model.
target_offsets = torch.zeros_like(projections_mean_group)
# old_model = copy.deepcopy(gradient_tracker_list)
optimized_structural_features = optimize_shared_input(gradient_tracker_list=gradient_tracker_list, target_output=target_offsets, weights=None, learning_rate=optimizer_learning_rate, num_steps=num_optimizer_steps, print_every_steps=optimizer_print_every_steps)
# Check to make sure optimization did not alter anything besides shared_input.
# param_comparison = torch.tensor(  [torch.equal(old_param, new_param) for old_param, new_param in zip( old_model.parameters(), gradient_tracker_list.parameters() )]  )
# print(f'{param_comparison.count_nonzero()} of {param_comparison.numel()} parameters are the same as before optimization.')
# Then evaluate the structural features FIM at this new central "group structural features" point.
optim_struct_string = f'optim_{combined_param_string}_steps_{num_optimizer_steps}_lr_{optimizer_learning_rate}'
optim_struct_file = os.path.join(stats_dir, f'group_features_{optim_struct_string}.pt')
torch.save(obj=optimized_structural_features, f=optim_struct_file)
print(f'saved optimized group structural features to {optim_struct_file}, time {time.time() - code_start_time:.3f}')
offsets_optim_struct = compute_and_save_struct_fim_and_offsets(gradient_tracker_list=gradient_tracker_list, V_inv_group_ising=V_group_inv, projections_group_ising=projections_mean_group, struct_features_centroid=optimized_structural_features, structural_data_dict=structural_data_dict, centroid_name=optim_struct_string)

# Repeat this process but weighting the offset dimensions by the magnitudes of their eigenvalues.
# Start from zero, the same initial condition as the previous optimization.
target_offsets = torch.zeros_like(projections_mean_group)
with torch.no_grad():
    gradient_tracker_list.shared_input.zero_()
# For the weights, we only want the total magnitude of the eigenvalue, not the sign or phase.
weighted_structural_features = optimize_shared_input( gradient_tracker_list=gradient_tracker_list, target_output=target_offsets, weights=L_group.abs(), learning_rate=optimizer_learning_rate, num_steps=num_optimizer_steps, print_every_steps=optimizer_print_every_steps )
weighted_struct_string = f'weighted_{combined_param_string}_steps_{num_optimizer_steps}_lr_{optimizer_learning_rate}'
weighted_struct_file = os.path.join(stats_dir, f'group_features_{weighted_struct_string}.pt')
torch.save(obj=weighted_structural_features, f=weighted_struct_file)
print(f'saved weighted optimized group structural features to {weighted_struct_file}, time {time.time() - code_start_time:.3f}')
offsets_weighted_struct = compute_and_save_struct_fim_and_offsets(gradient_tracker_list=gradient_tracker_list, V_inv_group_ising=V_group_inv, projections_group_ising=projections_mean_group, struct_features_centroid=weighted_structural_features, structural_data_dict=structural_data_dict, centroid_name=weighted_struct_string)

print(f'done, time {time.time() - code_start_time:.3f}')