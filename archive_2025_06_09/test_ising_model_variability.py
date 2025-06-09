import os
import torch
import hcpdatautils as hcp
import isingutils as ising
import time
import argparse
from collections import OrderedDict

start_time = time.time()
int_type = torch.int
float_type = torch.float
device = torch.device('cuda')

parser = argparse.ArgumentParser(description="Test how distinct the FCs are for the original fMRI data, fMRI-fitted Ising models, and structure-predicted Ising models.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-m", "--model_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the model")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_batch', help="directory to which to write the FC RMSE and correlation files")
parser.add_argument("-d", "--data_subset", type=str, default='training', help="if we want multiple subjects, set it to training or validation")
parser.add_argument("-s", "--subject_id", type=int, default=None, help="if we want one subject, set it to the ID of the subject of the model")
parser.add_argument("-p", "--ising_param_string", type=str, default='_parallel_nodes_21_epochs_1000_reps_100_window_50_lr_0.001_threshold_0.100_beta_0.500', help="parameter string of the Ising model file, the characters between h_/J_ and _[subject ID]")
parser.add_argument("-y", "--model_param_string", type=str, default='struct2ising_epochs_2000_val_batch_100_steps_4800_lr_0.0001_batches_1000_node_hl_3_node_w_7_edge_hl_3_edge_w_15_ising_nodes_21_reps_100_epochs_1000_window_50_lr_0.001_threshold_0.100', help="the part of the node model or edge model file name after 'node_model_' or 'edge_model_'")
parser.add_argument("-n", "--num_nodes", type=int, default=21, help="number of nodes to model")
parser.add_argument("-o", "--num_models", type=int, default=100, help="number of models per subject in the saved model files")
parser.add_argument("-l", "--sim_length", type=int, default=48000, help="number of steps for which to run the simulation, defaults to the same number as the number of timepoints in the data")
parser.add_argument("-t", "--threshold", type=float, default=0.1, help="threshold at which to binarize the fMRI data, in standard deviations above the mean")
parser.add_argument("-a", "--beta", type=float, default=0.5, help="factor we can use to adjust the chance of accepting a flip that increases the energy")

args = parser.parse_args()

print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
model_directory = args.model_directory
print(f'model_directory={model_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
data_subset = args.data_subset
print(f'data_subset={data_subset}')
subject_id = args.subject_id
print(f'subject_id={subject_id}')
ising_param_string = args.ising_param_string
print(f'ising_param_string={ising_param_string}')
model_param_string = args.model_param_string
print(f'model_param_string {model_param_string}')
num_nodes = args.num_nodes
print(f'num_nodes={num_nodes}')
num_models = args.num_models
print(f'num_models={num_models}')
sim_length = args.sim_length
print(f'sim_length={sim_length}')
threshold = args.threshold
print(f'threshold={threshold:.3g}')
beta = args.beta
print(f'beta={beta:.3g}')

def get_structural_data_std_mean(subject_list:list, num_nodes:int, coords:torch.Tensor, dtype=float_type, device=device):
    num_subjects = len(subject_list)
    # Pre-allocate space for the data.
    node_features = torch.zeros( (num_subjects, num_nodes, hcp.features_per_area), dtype=dtype, device=device )
    sc = torch.zeros( (num_subjects, num_nodes, num_nodes), dtype=dtype, device=device )
    # Load all the data from the individual files.
    for subject_index in range(num_subjects):
        subject_id = subject_list[subject_index]
        features_file = hcp.get_area_features_file_path(directory_path=data_directory, subject_id=subject_id)
        node_features[subject_index,:,:] = hcp.load_matrix_from_binary(file_path=features_file, dtype=dtype, device=device).transpose(dim0=0, dim1=1)[:num_nodes,:]
        sc_file = hcp.get_structural_connectivity_file_path(directory_path=data_directory, subject_id=subject_id)
        sc[subject_index,:,:] = hcp.load_matrix_from_binary(file_path=sc_file, dtype=dtype, device=device)[:num_nodes,:num_nodes]
    print('loaded data from files, time', time.time() - start_time )
    coords_std, coords_mean = torch.std_mean(coords, dim=0, keepdim=False)# mean over regions
    coords = (coords - coords_mean)/coords_std
    node_std, node_mean = torch.std_mean( node_features, dim=(0,1), keepdim=False )# mean over subjects and regions
    node_features = (node_features - node_mean)/node_std
    sc_std, sc_mean = torch.std_mean(sc, keepdim=False)# mean over subjects and region pairs
    sc = (sc - sc_mean)/sc_std
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

def get_data_fc(data_directory:str, subject_list:list, num_nodes:int, threshold:float, dtype=float_type, device=device):
    num_subjects = len(subject_list)
    data_fc = torch.zeros( (num_subjects, num_nodes, num_nodes), dtype=dtype, device=device )
    for subject_index in range(num_subjects):
        subject = subject_list[subject_index]
        data_ts = ising.standardize_and_binarize_ts_data(  ts=hcp.load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject, dtype=float_type, device=device), threshold=threshold  ).flatten(start_dim=0, end_dim=1)[:,:num_nodes]
        data_fc[subject_index,:,:] = hcp.get_fc(data_ts)
    return data_fc

def get_fitted_fc(model_directory:str, subject_list:list, ising_param_string:str, num_nodes:int, sim_length:int, beta:float, dtype=float_type, device=device):
    num_subjects = len(subject_list)
    fitted_fc = torch.zeros( (num_subjects, num_nodes, num_nodes), dtype=dtype, device=device )
    for subject_index in range(num_subjects):
        subject = subject_list[subject_index]
        print(f'subject {subject} ({subject_index} of {num_subjects})')
        print('loading models...')
        h = torch.load( os.path.join(model_directory, f'h_{ising_param_string}_subject_{subject}.pt') ).squeeze()
        print( 'h size', h.size() )
        J = torch.load( os.path.join(model_directory, f'J_{ising_param_string}_subject_{subject}.pt') )
        print( 'J size', J.size() )
        num_models = h.size(dim=0)
        s = ising.get_random_state(batch_size=num_models, num_nodes=num_nodes, dtype=dtype, device=device)
        current_sim_fc, _ = ising.run_batched_balanced_metropolis_sim_for_fc(J=J, h=h, s=s, num_steps=sim_length, beta=beta)
        fitted_fc[subject_index,:,:] = current_sim_fc.mean(dim=0)
    return fitted_fc

def get_fitted_fc_all_at_once(model_directory:str, subject_list:list, ising_param_string:str, num_nodes:int, num_models:int, sim_length:int, beta:float, dtype=float_type, device=device):
    num_subjects = len(subject_list)
    h = torch.zeros( (num_subjects, num_models, num_nodes), dtype=dtype, device=device )
    J = torch.zeros( (num_subjects, num_models, num_nodes, num_nodes), dtype=dtype, device=device )
    print('loading models...')
    for subject_index in range(num_subjects):
        subject = subject_list[subject_index]
        # print(f'subject {subject} ({subject_index} of {num_subjects})')
        h[subject_index,:,:] = torch.load( os.path.join(model_directory, f'h_{ising_param_string}_subject_{subject}.pt') ).squeeze()
        J[subject_index,:,:,:] = torch.load( os.path.join(model_directory, f'J_{ising_param_string}_subject_{subject}.pt') )
    print( 'h size', h.size() )
    print( 'J size', J.size() )
    h = h.flatten(start_dim=0, end_dim=1)
    J = J.flatten(start_dim=0, end_dim=1)
    print('simulating...')
    s = ising.get_random_state(batch_size=num_subjects*num_models, num_nodes=num_nodes, dtype=dtype, device=device)
    current_sim_fc, _ = ising.run_batched_balanced_metropolis_sim_for_fc(J=J, h=h, s=s, num_steps=sim_length, beta=beta)
    fitted_fc = current_sim_fc.unflatten( dim=0, sizes=(num_subjects, num_models) )
    return fitted_fc

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

def get_predicted_fc(data_directory:str, model_directory:str, subject_list:list, model_param_string:str, num_nodes:int, sim_length:int, beta:float, dtype=float_type, device=device):
    num_subjects = len(subject_list)
    predicted_fc = torch.zeros( (num_subjects, num_nodes, num_nodes), dtype=dtype, device=device )
    node_model_file = os.path.join(model_directory, f'node_model_{model_param_string}.pt')
    node_model = torch.load(node_model_file)
    print('loaded node model:')
    print(node_model)
    print( node_model.ff_layers[0].weight.device )
    edge_model_file = os.path.join(model_directory, f'edge_model_{model_param_string}.pt')
    edge_model = torch.load(edge_model_file)
    print('loaded edge model:')
    print(edge_model)
    print( 'time', time.time() - start_time )
    _, node_coords = hcp.load_roi_info(directory_path=data_directory, dtype=dtype, device=device)
    print( 'loaded region names and coordinates with size ', node_coords.size() )
    print( 'time', time.time() - start_time )
    coords_std, coords_mean, node_std, node_mean, sc_std, sc_mean = get_structural_data_std_mean(subject_list=subject_list, num_nodes=num_nodes, coords=node_coords)
    node_coords = (node_coords - coords_mean)/coords_std
    h = torch.zeros( (num_subjects, num_nodes), dtype=dtype, device=device )
    J = torch.zeros( (num_subjects, num_nodes, num_nodes), dtype=dtype, device=device )
    for subject_index in range(num_subjects):
        subject_id = subject_list[subject_index]
        print(f'subject {subject_id} ({subject_index} of {num_subjects})')
        print('loading structural MRI data...')
        features_file = hcp.get_area_features_file_path(directory_path=data_directory, subject_id=subject_id)
        node_features = ( hcp.load_matrix_from_binary(file_path=features_file, dtype=dtype, device=device).transpose(dim0=0, dim1=1) - node_mean )/node_std
        print( 'loaded node features with size', node_features.size() )
        print( 'time', time.time() - start_time )
        sc_file = hcp.get_structural_connectivity_file_path(directory_path=data_directory, subject_id=subject_id)
        sc = ( hcp.load_matrix_from_binary(file_path=sc_file, dtype=dtype, device=device) - sc_mean )/sc_std
        node_features = get_node_features(coords=node_coords, region_features=node_features, num_nodes=num_nodes)
        print( 'node features extended with coords:', node_features.size() )
        edge_features = get_edge_features(node_features=node_features, structural_connectivity=sc, num_nodes=num_nodes)
        print( 'node features paired between source and target and SC appended to create edge features:', edge_features.size() )
        print('predicting models...')
        h[subject_index,:] = node_model(node_features)
        J[subject_index,:,:] = edge_model(edge_features)
    print( 'h size', h.size() )
    print( 'J size', J.size() )
    print('simulating...')
    s = ising.get_random_state(batch_size=num_subjects, num_nodes=num_nodes, dtype=dtype, device=device)
    predicted_fc, _ = ising.run_batched_balanced_metropolis_sim_for_fc(J=J, h=h, s=s, num_steps=sim_length, beta=beta)
    return predicted_fc

def print_stats(name:str, values:torch.Tensor):
    values = values.flatten()
    quantile_cutoffs = torch.tensor([0.025, 0.5, 0.975], dtype=float_type, device=device)
    quantiles = torch.quantile(values, quantile_cutoffs)
    min_val = torch.min(values)
    max_val = torch.max(values)
    print(f'The distribution of {name} values has median {quantiles[1].item():.3g} with 95% CI [{quantiles[0].item():.3g}, {quantiles[2].item():.3g}] and range [{min_val.item():.3g}, {max_val.item():.3g}].')

subject_string = data_subset
if data_subset == 'training':
    subject_list = hcp.load_training_subjects(data_directory)
elif data_subset == 'validation':
    subject_list = hcp.load_validation_subjects(data_directory)
elif data_subset == 'testing':
    subject_list = hcp.load_testing_subjects(data_directory)
elif data_subset == 'all':
    subject_list = hcp.load_training_subjects(data_directory) + hcp.load_validation_subjects(data_directory) + hcp.load_testing_subjects(data_directory)
else:
    subject_list = [subject_id]
subject_list = hcp.get_has_sc_subject_list(directory_path=data_directory, subject_list=subject_list)

with torch.no_grad():

    data_fc = get_data_fc(data_directory=data_directory, subject_list=subject_list, num_nodes=num_nodes, threshold=threshold)
    data_fc_file = os.path.join(output_directory, f'data_fc_{data_subset}_nodes_{num_nodes}_threshold_{threshold}.pt')
    torch.save(data_fc, data_fc_file)
    print('found FC mean and std. dev. for original fMRI data')
    print(f'time {time.time() - start_time:.3f}')

    # fitted_fc = get_fitted_fc(model_directory=model_directory, subject_list=subject_list, ising_param_string=ising_param_string, num_nodes=num_nodes, sim_length=sim_length, beta=beta)
    fitted_fc = get_fitted_fc_all_at_once(model_directory=model_directory, subject_list=subject_list, ising_param_string=ising_param_string, num_nodes=num_nodes, num_models=num_models, sim_length=sim_length, beta=beta)
    fitted_fc_file = os.path.join(output_directory, f'fitted_fc_{data_subset}_nodes_{num_nodes}_steps_{sim_length}_beta_{beta}_ising_{ising_param_string}.pt')
    torch.save(fitted_fc, fitted_fc_file)
    print('found FC mean and std. dev. for Ising models fitted to fMRI data')
    print(f'time {time.time() - start_time:.3f}')

    predicted_fc = get_predicted_fc(data_directory=data_directory, model_directory=model_directory, subject_list=subject_list, model_param_string=model_param_string, num_nodes=num_nodes, sim_length=sim_length, beta=beta)
    predicted_fc_file = os.path.join(output_directory, f'fitted_fc_{data_subset}_steps_{sim_length}_{model_param_string}.pt')
    torch.save(predicted_fc, predicted_fc_file)
    print('found FC mean and std. dev. for Ising models predicted from structural MRI data')
    print(f'time {time.time() - start_time:.3f}')

    data_std, data_mean = torch.std_mean(data_fc, dim=0)
    fitted_fc_subj_mean = fitted_fc.mean(dim=1)
    fitted_std, fitted_mean = torch.std_mean( fitted_fc_subj_mean, dim=0 )# Take the mean over reps for the same subject, then the mean and std over subjects.
    predicted_std, predicted_mean = torch.std_mean(predicted_fc, dim=0)
    print_stats(name=f'{data_subset} subject FC values for region pairs from fMRI data', values=data_fc)
    print_stats(name=f'{data_subset} subject mean FC values for region pairs from fMRI data', values=data_mean)
    print_stats(name=f'{data_subset} subject std. dev. FC values for region pairs from fMRI data', values=data_std)
    print_stats(name=f'{data_subset} subject FC values for region pairs from Ising models fitted to fMRI data', values=fitted_fc_subj_mean)
    print_stats(name=f'{data_subset} subject mean FC values for region pairs from Ising models fitted to fMRI data', values=fitted_mean)
    print_stats(name=f'{data_subset} subject std. dev. FC values for region pairs from Ising models fitted to fMRI data', values=fitted_std)
    print_stats(name=f'{data_subset} subject FC values for region pairs from Ising models predicted from structural MRI data', values=predicted_fc)
    print_stats(name=f'{data_subset} subject mean FC values for region pairs from Ising models predicted from structural MRI data', values=predicted_mean)
    print_stats(name=f'{data_subset} subject std. dev. FC values for region pairs from Ising models predicted from structural MRI data', values=predicted_std)
