import os
import torch
import time
import argparse
import pandas
import hcpdatautils as hcp
import isingutils as ising
from collections import OrderedDict

start_time = time.time()

parser = argparse.ArgumentParser(description="Predict Ising model parameters from structural MRI and DT-MRI structural connectivity data mean for a population.")

# directories
parser.add_argument("-d", "--data_dir", type=str, default='E:\\HCP_data', help="directory containing the structural MRI features data file")
parser.add_argument("-m", "--model_dir", type=str, default='E:\\Ising_model_results_daai', help="directory containing the trained struct-to-Ising node model and edge model .pt files")
parser.add_argument("-s", "--stats_dir", type=str, default="E:\\Ising_model_results_daai", help="directory to which to write the output Fisher information matrices")
parser.add_argument("-n", "--num_nodes", type=int, default=21, help="number of nodes in Ising model")
parser.add_argument("-b", "--beta", type=float, default=0.5, help="beta constant used in Ising model")
parser.add_argument("-t", "--num_steps", type=int, default=48000, help="number of steps to use when running Ising model simulations to compare FC")
parser.add_argument("-p", "--data_subset", type=str, default='training', help="which list of subjects to use, either 'training', 'validation', or 'testing'")
parser.add_argument("-z", "--z_score", type=bool, default=True, help="set to True to z-score the data before training, using the training sample mean and std. dev. for both training and validation data")
parser.add_argument("-y", "--model_param_string", type=str, default='struct2ising_epochs_2000_val_batch_100_steps_4800_lr_0.0001_batches_1000_node_hl_3_node_w_7_edge_hl_3_edge_w_15_ising_nodes_21_reps_100_epochs_1000_window_50_lr_0.001_threshold_0.100', help="the part of the node model or edge model file name after 'node_model_' or 'edge_model_'")
# parser.add_argument("-y", "--model_param_string", type=str, default='struct2ising_epochs_500_val_batch_100_steps_4800_lr_0.0001_batches_1000_node_hl_2_node_w_21_edge_hl_2_edge_w_441_ising_nodes_21_reps_100_epochs_1000_window_50_lr_0.001_threshold_0.100', help="the part of the node model or edge model file name after 'node_model_' or 'edge_model_'")

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
beta = args.beta
print(f'beta {beta}')
num_steps = args.num_steps
print(f'num_steps {num_steps}')
data_subset = args.data_subset
print(f'data_subset {data_subset}')
model_param_string = args.model_param_string
print(f'model_param_string {model_param_string}')
z_score = args.z_score
print(f'z_score {z_score}')

float_type = torch.float
device = torch.device('cuda')

def load_roi_info(directory_path:str, dtype=torch.float, device='cpu'):
    roi_info = pandas.read_csv( os.path.join(directory_path, 'roi_info.csv') )
    names = roi_info['name'].values
    coords = torch.tensor( data=roi_info[['x','y','z']].values, dtype=dtype, device=device )
    return names, coords

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
        self.stored_input = torch.nn.Parameter(input_values)
    
    def forward(self):
        output_values = self.model(self.stored_input)
        return output_values
    
    def get_output_and_gradients(self, input:torch.Tensor=None):
        if type(input) != type(None):
            with torch.no_grad():
                self.stored_input.copy_(input)
        if type(self.stored_input.grad) != type(None):
            self.stored_input.grad.zero_()
        output = self()
        output.backward( torch.ones_like(output) )
        return output, self.stored_input.grad

def get_structural_data_std_mean(subset:str, coords:torch.Tensor):

    last_time = time.time()
    if subset == 'training':
        subjects = hcp.load_training_subjects(directory_path=data_dir)
    else:
        subjects = hcp.load_validation_subjects(directory_path=data_dir)
    subjects = hcp.get_has_sc_subject_list(directory_path=data_dir, subject_list=subjects)
    num_subjects = len(subjects)

    # Pre-allocate space for the data.
    node_features = torch.zeros( (num_subjects, num_nodes, hcp.features_per_area), dtype=float_type, device=device )
    sc = torch.zeros( (num_subjects, num_nodes, num_nodes), dtype=float_type, device=device )

    # Load all the data from the individual files.
    for subject_index in range(num_subjects):
        subject_id = subjects[subject_index]
        features_file = hcp.get_area_features_file_path(directory_path=data_dir, subject_id=subject_id)
        node_features[subject_index,:,:] = hcp.load_matrix_from_binary(file_path=features_file, dtype=float_type, device=device).transpose(dim0=0, dim1=1)[:num_nodes,:]
        sc_file = hcp.get_structural_connectivity_file_path(directory_path=data_dir, subject_id=subject_id)
        sc[subject_index,:,:] = hcp.load_matrix_from_binary(file_path=sc_file, dtype=float_type, device=device)[:num_nodes,:num_nodes]
    current_time = time.time()
    print('loaded data from files, time', current_time-last_time )
    last_time = current_time
    # coords = coords[:num_nodes,:]
    coords_std, coords_mean = torch.std_mean(coords, dim=0, keepdim=False)# mean over regions
    # coords = (coords - coords_mean)/coords_std
    node_std, node_mean = torch.std_mean( node_features, dim=(0,1), keepdim=False )# mean over subjects and regions
    # node_features = (node_features - node_mean)/node_std
    sc_std, sc_mean = torch.std_mean(sc, keepdim=False)# mean over subjects and region pairs
    # sc = (sc - sc_mean)/sc_std

    return coords_std, coords_mean, node_std, node_mean, sc_std, sc_mean

# For the following functions, we assume we are working with one subject at a time.
# As such, the Tensors should have the following dimensions:
# coords: num_brain_regions x 3
# region_features: num_brain_regions x 4
# node_features: num_nodes x 7 (It is just region_features and coords truncated to num_nodes and concatenated.)
# structural_connectivity: num_nodes x num_nodes
# edge_features: num_nodes x num_nodes x 15 (concatenation of source node_features, target node_features, and structural_connectivity)

def get_node_features(coords:torch.Tensor, region_features:torch.Tensor, num_nodes:int):
    return torch.cat( (region_features[:num_nodes,:], coords[:num_nodes,:]), dim=-1 )

def get_edge_features(node_features:torch.Tensor, structural_connectivity:torch.Tensor, num_nodes:int):
    return torch.cat(   (  node_features[None,:num_nodes,:].repeat( (num_nodes,1,1) ), node_features[:num_nodes,None,:].repeat( (1,num_nodes,1) ), structural_connectivity[:num_nodes,:num_nodes,None]  ), dim=-1   )

# This function takes the gradients we get from PyTorch Autograd and rearranges them into a Jacobian matrix in the right shape for our Fisher information matrix calculation.
# h_gradient is num_nodes x num_node_features, the same size as node_features.
# J_gradient is num_nodes x num_nodes x num_edge_features, the same size as edge_features.
# For the Jacobian, we want one row for each of the (num_nodes + num_nodes*num_nodes) Ising model parameters
# and one column for each of the ( (3+4)*num_nodes + num_nodes*(num_nodes-1)/2 ) features.
# The SC matrix is symmetric with diagonal fixed to 0, so we only want columns for the elements above the diagonal.
# On the other hand, we do store separate gradients for J(i,j) and J(j,i), since we cannot guarantee that J is symmetric.
# We do, however, force the diagonal of J to be 0, so its gradients should always be 0.
def get_jacobian(h_gradient:torch.Tensor, J_gradient:torch.Tensor):
    num_nodes, num_node_features = h_gradient.size()
    num_ising_params = num_nodes*(1 + num_nodes)
    sc_indices = torch.triu_indices(num_nodes, num_nodes, 1, device=device)
    source_indices = sc_indices[0]
    target_indices = sc_indices[1]
    num_sc_values = source_indices.numel()
    sc_columns_start = num_nodes * num_node_features
    num_structural_features = sc_columns_start + num_sc_values
    jacobian = torch.zeros( (num_ising_params, num_structural_features), dtype=h_gradient.dtype, device=device )
    for node_index in range(num_nodes):
        node_features_start = node_index * num_node_features
        node_features_end = node_features_start + num_node_features
        jacobian[node_index,node_features_start:node_features_end] = h_gradient[node_index,:]
    for pair_index in range(num_sc_values):
        source_index = source_indices[pair_index]
        target_index = target_indices[pair_index]
        pair_row_1 = num_nodes + num_nodes*source_index + target_index
        pair_row_2 = num_nodes + num_nodes*target_index + source_index
        source_col_start = source_index * num_node_features
        source_col_end = source_col_start + num_node_features
        target_col_start = target_index * num_node_features
        target_col_end = target_col_start + num_node_features
        sc_col = sc_columns_start + pair_index
        jacobian[pair_row_1, source_col_start:source_col_end] = J_gradient[source_index, target_index, :num_node_features]
        jacobian[pair_row_1, target_col_start:target_col_end] = J_gradient[source_index, target_index, num_node_features:(2*num_node_features)]
        jacobian[pair_row_1, sc_col] = J_gradient[source_index, target_index, -1]
        jacobian[pair_row_2, target_col_start:target_col_end] = J_gradient[target_index, source_index, :num_node_features]
        jacobian[pair_row_2, source_col_start:source_col_end] = J_gradient[target_index, source_index, num_node_features:(2*num_node_features)]
        jacobian[pair_row_2, sc_col] = J_gradient[target_index, source_index, -1]
    return jacobian

def get_ising_model_and_jacobian(node_model:torch.nn.Module, edge_model:torch.nn.Module, coords:torch.Tensor, region_features:torch.Tensor, structural_connectivity:torch.Tensor, num_nodes:int):
    node_features = get_node_features(coords=coords, region_features=region_features, num_nodes=num_nodes)
    print( 'node features extended with coords:', node_features.size() )
    edge_features = get_edge_features(node_features=node_features, structural_connectivity=structural_connectivity, num_nodes=num_nodes)
    print( 'node features paired between source and target and SC appended to create edge features:', edge_features.size() )
    node_gradient_tracker = GradientTracker(model=node_model, input_values=node_features)
    h, h_gradient = node_gradient_tracker.get_output_and_gradients()
    edge_gradient_tracker = GradientTracker(model=edge_model, input_values=edge_features)
    J, J_gradient = edge_gradient_tracker.get_output_and_gradients()
    jacobian = get_jacobian(h_gradient=h_gradient, J_gradient=J_gradient)
    return h, J, jacobian

# The important points:
# Region features come before SC.
# Feature iterates fastest, while region name increments once we get through all the features of the current region.
# SC indices consist of pairs of nodes such that source < target ordered such that we iterate over all targets before incrementing source.
def get_jacobian_column_labels(node_labels:list):
    feature_names = hcp.feature_names + ['x', 'y', 'z']
    num_nodes = len(node_labels)
    sc_indices = torch.triu_indices(num_nodes, num_nodes, 1, device=device)
    source_indices = sc_indices[0]
    target_indices = sc_indices[1]
    num_sc_indices = source_indices.numel()
    node_feature_labels = [f'{feature} of {node}' for node in node_labels for feature in feature_names]
    sc_labels = [f'SC from {node_labels[ source_indices[sc_index] ]} to {node_labels[ target_indices[sc_index] ]}' for sc_index in range(num_sc_indices)]
    return node_feature_labels + sc_labels

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
    s = ising.get_random_state(batch_size=batch_size, num_nodes=num_nodes, dtype=h.dtype, device=device)
    s_sum = torch.zeros_like(s)
    s_product_sum = s_sum[:,:,None] * s_sum[:,None,:]
    params = torch.cat(  ( s_sum, s_product_sum.flatten(start_dim=-2, end_dim=-1) ), dim=-1  )
    param_product_sum = params[:,:,None] * params[:,None,:]
    for _ in range(num_steps):
        s = ising.run_batched_balanced_metropolis_sim_step(J=J, h=h, s=s, beta=beta)
        s_sum += s
        s_product = (s[:,:,None] * s[:,None,:])
        s_product_sum += s_product
        params = torch.cat(  ( s, s_product.flatten(start_dim=-2, end_dim=-1) ), dim=-1  )
        param_product_sum += params[:,:,None] * params[:,None,:]
    param_mean = torch.cat(  ( s_sum, s_product_sum.flatten(start_dim=-2, end_dim=-1) ), dim=-1  )/num_steps
    param_cov = param_product_sum/num_steps - (param_mean[:,:,None] * param_mean[:,None,:])
    return param_cov

node_model_file = os.path.join(model_dir, f'node_model_{model_param_string}')
node_model = torch.load(node_model_file)
print('loaded node model:')
print(node_model)
print( node_model.ff_layers[0].weight.device )
edge_model_file = os.path.join(model_dir, f'edge_model_{model_param_string}')
edge_model = torch.load(edge_model_file)
print('loaded edge model:')
print(edge_model)
print( 'time', time.time() - start_time )

node_names, node_coords = load_roi_info(data_dir, dtype=float_type, device=device)
num_nodes_coords, num_coord_dimensions = node_coords.size()
print( 'loaded region names and coordinates with size ', node_coords.size() )
print(node_coords.device)
print( 'time', time.time() - start_time )

if data_subset == 'training':
    subjects = hcp.load_training_subjects(directory_path=data_dir)
elif data_subset == 'validation':
    subjects = hcp.load_validation_subjects(directory_path=data_dir)
elif data_subset == 'testing':
    subjects = hcp.load_testing_subjects(directory_path=data_dir)
else:
    print(f'unrecognized data subset name {data_subset}')
    exit(1)
subjects = hcp.get_has_sc_subject_list(directory_path=data_dir, subject_list=subjects)
num_subjects = len(subjects)
print(f'The {data_subset} data set includes {num_subjects} subjects.')
print( 'time', time.time() - start_time )

struct_param_labels = get_jacobian_column_labels(node_labels=node_names[:num_nodes])
param_label_file_name = os.path.join(stats_dir, f'param_labels_{model_param_string}_steps_{num_steps}.txt')
with open(param_label_file_name, 'w') as param_label_file:
    for label in struct_param_labels:
        param_label_file.write(f'{label}\n')
num_struct_params = len(struct_param_labels)
print(f'We have {num_struct_params} structural parameters in all.')
print( 'time', time.time() - start_time )

coords_std, coords_mean, node_std, node_mean, sc_std, sc_mean = get_structural_data_std_mean(subset=data_subset, coords=node_coords)
node_coords = (node_coords - coords_mean)/coords_std
print(f'computed mean and std. dev. for each structural feature over {data_subset} data')
print( 'time', time.time() - start_time )

node_features_sum = torch.zeros( (num_nodes, hcp.features_per_area), dtype=float_type, device=device )
sc_sum = torch.zeros( (num_nodes, num_nodes), dtype=float_type, device=device )
for subject_index in range(num_subjects):
    subject_id = subjects[subject_index]
    # print(f'starting on subject {subject_id} ({subject_index+1} of {num_subjects})')
    # print( 'time', time.time() - start_time )
    features_file = hcp.get_area_features_file_path(directory_path=data_dir, subject_id=subject_id)
    node_features_sum += hcp.load_matrix_from_binary(file_path=features_file, dtype=float_type, device=device).transpose(dim0=0, dim1=1)[:num_nodes,:]
    sc_file = hcp.get_structural_connectivity_file_path(directory_path=data_dir, subject_id=subject_id)
    sc_sum += hcp.load_matrix_from_binary(file_path=sc_file, dtype=float_type, device=device)[:num_nodes,:num_nodes]
node_features = (node_features_sum/num_subjects - node_mean)/node_std
print( 'found mean node features with size', node_features.size() )
print( 'time', time.time() - start_time )
sc = (sc_sum/num_subjects - sc_mean)/sc_std
print( 'found mean structural connectivity with size', sc.size() )
print( 'time', time.time() - start_time )

fim_struct_sum = torch.zeros( (num_struct_params, num_struct_params), dtype=float_type, device=device )
h, J, jacobian = get_ising_model_and_jacobian(node_model=node_model, edge_model=edge_model, coords=node_coords, region_features=node_features, structural_connectivity=sc, num_nodes=num_nodes)
print( 'found Ising model and struct-vs-Ising Jacobian' )
print( 'h size', h.size() )
# print(h.device)
print( 'J size', J.size() )
# print(J.device)
print( 'Jacobian size', jacobian.size() )
file_suffix = f'{model_param_string}_mean_{data_subset}_steps_{num_steps}'
jacobian_file = os.path.join(stats_dir, f'jacobian_{file_suffix}.pt')
torch.save(jacobian, jacobian_file)
print( 'time', time.time() - start_time )
print('starting Ising model simulation to calculate FIM...')
with torch.no_grad():
    fim_ising = get_ising_model_fisher_information_matrix(h=h, J=J, num_steps=num_steps, beta=beta).squeeze()
    print( 'found Ising model FIM with size', fim_ising.size() )
# print(fim_ising.device)
fim_ising_file = os.path.join(stats_dir, f'fim_ising_{file_suffix}.pt')
torch.save(fim_ising, fim_ising_file)
print( 'time', time.time() - start_time )
fim_struct = torch.matmul( jacobian.transpose(dim0=0, dim1=1), torch.matmul(fim_ising, jacobian) )
print( 'found struct-to-Ising FIM with size', fim_struct.size() )
fim_struct_file = os.path.join(stats_dir, f'fim_struct2ising_{file_suffix}.pt')
torch.save(fim_struct, fim_struct_file)
print( 'time', time.time() - start_time )
print( 'done' )