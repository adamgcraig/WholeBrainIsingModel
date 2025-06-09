import os
import torch
import time
import argparse
import pandas
import hcpdatautils as hcp
import isingutils as ising
from collections import OrderedDict

start_time = time.time()
last_time = start_time

parser = argparse.ArgumentParser(description="Predict Ising model parameters from structural MRI and DT-MRI structural connectivity data.")

# directories
parser.add_argument("-a", "--structural_data_dir", type=str, default='E:\\HCP_data', help="directory containing the structural MRI features data file")
parser.add_argument("-b", "--ising_model_dir", type=str, default='E:\\Ising_model_results_daai', help="directory containing the fitted Ising model J parameter file")
parser.add_argument("-c", "--stats_dir", type=str, default="E:\\Ising_model_results_batch", help="directory to which to write the output files from training")

# hyperparameters of the Ising model, used for looking up which h files to load
parser.add_argument("-d", "--num_nodes", type=int, default=21, help="number of nodes in Ising model")
parser.add_argument("-e", "--num_reps", type=int, default=1000, help="number of Ising models trained for each subject")
parser.add_argument("-f", "--num_epochs_ising", type=int, default=1000, help="number of epochs for which we trained the Ising model")
parser.add_argument("-g", "--window_length", type=int, default=50, help="window length used when training Ising model")
parser.add_argument("-i", "--learning_rate_ising", type=str, default='0.001', help="learning rate used when training Ising model")
parser.add_argument("-j", "--threshold", type=str, default='0.100', help="threshold at which to binarize the fMRI data, in standard deviations above the mean, in the Ising model")
parser.add_argument("-k", "--beta", type=str, default='0.500', help="beta constant used in Ising model")
# To avoid number formatting inconsistencies, we use strings for the floating point numbers that we only use in file names.

# hyperparameters of the model training
parser.add_argument("-l", "--num_epochs", type=int, default=1, help="number of epochs for which to train")
parser.add_argument("-m", "--num_subepochs", type=int, default=1000, help="number of epochs for which to train between evaluations of the full model")
parser.add_argument("-n", "--num_steps", type=int, default=4800, help="number of steps to use when running Ising model simulations to compare FC")
parser.add_argument("-o", "--validation_batch_size", type=int, default=1, help="size of the batches we use in the validation run where we actually run the Ising model, relevant for memory footprint vs speed purposes, not the gradient descent process itself")
parser.add_argument("-p", "--learning_rate", type=float, default=0.0001, help="learning rate to use for Adam optimizer")
parser.add_argument("-q", "--num_batches", type=int, default=1000, help="number of batches into which to divide the training data")
parser.add_argument("-r", "--node_num_hidden_layers", type=int, default=2, help="number of hidden layers in the node model")
parser.add_argument("-s", "--node_hidden_layer_width", type=int, default=21, help="width of the hidden layers in the node model")
parser.add_argument("-t", "--edge_num_hidden_layers", type=int, default=2, help="number of hidden layers in the edge model")
parser.add_argument("-u", "--edge_hidden_layer_width", type=int, default=441, help="width of the hidden layers in the edge model")
parser.add_argument("-z", "--z_score", type=bool, default=True, help="set to True to z-score the data before training, using the training sample mean and std. dev. for both training and validation data")
# We are not counting the first or last linear layer as hidden,
# so every network has at least two layers.

args = parser.parse_args()

structural_data_dir = args.structural_data_dir
print(f'structural_data_dir {structural_data_dir}')
ising_model_dir = args.ising_model_dir
print(f'ising_model_dir {ising_model_dir}')
stats_dir = args.stats_dir
print(f'stats_dir {stats_dir}')
num_nodes = args.num_nodes
print(f'num_nodes {num_nodes}')
num_reps = args.num_reps
print(f'num_reps {num_reps}')
num_epochs_ising = args.num_epochs_ising
print(f'num_epochs_ising {num_epochs_ising}')
window_length = args.window_length
print(f'window_length {window_length}')
learning_rate_ising = args.learning_rate_ising
print(f'learning_rate_ising {learning_rate_ising}')
threshold = args.threshold
print(f'threshold {threshold}')
beta_str = args.beta
print(f'beta_str {beta_str}')
beta_float = float(beta_str)
num_epochs = args.num_epochs
print(f'num_epochs {num_epochs}')
num_subepochs = args.num_subepochs
print(f'num_subepochs {num_subepochs}')
num_steps = args.num_steps
print(f'num_steps {num_steps}')
validation_batch_size = args.validation_batch_size
print(f'validation_batch_size {validation_batch_size}')
learning_rate = args.learning_rate
print(f'learning_rate {learning_rate}')
num_batches = args.num_batches
print(f'num_batches {num_batches}')
node_num_hidden_layers = args.node_num_hidden_layers
print(f'node_num_hidden_layers {node_num_hidden_layers}')
node_hidden_layer_width = args.node_hidden_layer_width
print(f'node_hidden_layer_width {node_hidden_layer_width}')
edge_num_hidden_layers = args.edge_num_hidden_layers
print(f'edge_num_hidden_layers {edge_num_hidden_layers}')
edge_hidden_layer_width = args.edge_hidden_layer_width
print(f'edge_hidden_layer_width {edge_hidden_layer_width}')
z_score = args.z_score
print(f'z_score {z_score}')

float_type = torch.float
device = torch.device('cuda')

def load_roi_info(directory_path:str, dtype=torch.float, device='cpu'):
    roi_info = pandas.read_csv( os.path.join(directory_path, 'roi_info.csv') )
    names = roi_info['name'].values
    coords = torch.tensor( data=roi_info[['x','y','z']].values, dtype=dtype, device=device )
    return names, coords

names, coords = load_roi_info(structural_data_dir, dtype=float_type, device=device)
num_nodes_coords, num_coords = coords.size()
current_time = time.time()
print('coords:',num_nodes_coords,'x',num_coords, 'time', current_time-last_time, 'seconds')
last_time = current_time

training_subjects = hcp.get_has_sc_subject_list( directory_path=structural_data_dir, subject_list=hcp.load_training_subjects(directory_path=structural_data_dir) )
validation_subjects = hcp.get_has_sc_subject_list( directory_path=structural_data_dir, subject_list=hcp.load_validation_subjects(directory_path=structural_data_dir) )

def prepare_data(subset:str, coords:torch.Tensor, num_reps:int, z_score:bool=True, coords_std:torch.Tensor=None, coords_mean:torch.Tensor=None, node_std:torch.Tensor=None, node_mean:torch.Tensor=None, sc_std:torch.Tensor=None, sc_mean:torch.Tensor=None):

    last_time = time.time()
    if subset == 'training':
        subjects = hcp.load_training_subjects(directory_path=structural_data_dir)
    else:
        subjects = hcp.load_validation_subjects(directory_path=structural_data_dir)
    subjects = hcp.get_has_sc_subject_list(directory_path=structural_data_dir, subject_list=subjects)
    num_subjects = len(subjects)

    # Pre-allocate space for the data.
    node_features = torch.zeros( (num_subjects, num_nodes, hcp.features_per_area), dtype=float_type, device=device )
    sc = torch.zeros( (num_subjects, num_nodes, num_nodes), dtype=float_type, device=device )
    h = torch.zeros( (num_subjects, num_reps, num_nodes), dtype=float_type, device=device )
    J = torch.zeros( (num_subjects, num_reps, num_nodes, num_nodes), dtype=float_type, device=device )

    # Load all the data from the individual files.
    # param_string = f'nodes_{num_nodes}_reps_{num_reps}_epochs_{num_epochs_ising}_window_{window_length}_lr_{learning_rate_ising}_threshold_{threshold}'
    # param_string = f'window_length_test_nodes_{num_nodes}_epochs_{num_epochs_ising}_max_window_{window_length}_lr_{learning_rate_ising}_threshold_{threshold}'
    param_string = f'_parallel_nodes_{num_nodes}_epochs_{num_epochs_ising}_reps_{num_reps}_window_{window_length}_lr_{learning_rate_ising}_threshold_{threshold}_beta_{beta_str}'
    for subject_index in range(num_subjects):
        subject_id = subjects[subject_index]
        features_file = hcp.get_area_features_file_path(directory_path=structural_data_dir, subject_id=subject_id)
        node_features[subject_index,:,:] = hcp.load_matrix_from_binary(file_path=features_file, dtype=float_type, device=device).transpose(dim0=0, dim1=1)[:num_nodes,:]
        sc_file = hcp.get_structural_connectivity_file_path(directory_path=structural_data_dir, subject_id=subject_id)
        sc[subject_index,:,:] = hcp.load_matrix_from_binary(file_path=sc_file, dtype=float_type, device=device)[:num_nodes,:num_nodes]
        ising_model_string = f'{param_string}_subject_{subject_id}'
        ising_r_file = os.path.join(ising_model_dir, f'h_{ising_model_string}.pt')
        h[subject_index,:,:] = torch.load(ising_r_file)
        ising_J_file = os.path.join(ising_model_dir, f'J_{ising_model_string}.pt')
        J[subject_index,:,:,:] = torch.load(ising_J_file)
    current_time = time.time()
    print('loaded data from files, time', current_time-last_time )
    last_time = current_time

    coords = coords[:num_nodes,:]# torch.tensor(coords[:num_nodes,:], dtype=float_type, device=device)
    
    if z_score:
        if ( type(coords_std) == type(None) ) or ( type(coords_mean) == type(None) ):
            coords_std, coords_mean = torch.std_mean(coords, dim=0, keepdim=True)# mean over regions
        coords = (coords - coords_mean)/coords_std
        if ( type(node_std) == type(None) ) or ( type(node_mean) == type(None) ):
            node_std, node_mean = torch.std_mean( node_features, dim=(0,1), keepdim=True )# mean over subjects and regions
        node_features = (node_features - node_mean)/node_std
        if ( type(sc_std) == type(None) ) or ( type(sc_mean) == type(None) ):
            sc_std, sc_mean = torch.std_mean(sc, keepdim=True)# mean over subjects and region pairs
        sc = (sc - sc_mean)/sc_std

    # Concatenate features together and duplicate them as needed.
    node_features = torch.cat(   (  node_features, coords[None,:,:].repeat( (num_subjects,1,1) )  ), dim=-1   )
    print( 'node features extended with coords:', node_features.size() )
    edge_features = torch.cat(   (  node_features[:,None,:,:].repeat( (1,num_nodes,1,1) ), node_features[:,:,None,:].repeat( (1,1,num_nodes,1) ), sc[:,:,:,None]  ), dim=-1   )
    print( 'node features paired between source and target and SC appended to create edge features:', node_features.size() )
    # node_features = node_features[:,None,:,:].repeat( (1,num_reps,1,1) ).flatten(start_dim=0, end_dim=1)
    # print( 'repeated features once for each rep of Ising model trainig:', node_features.size() )
    # edge_features = edge_features[:,None,:,:,:].repeat( (1,num_reps,1,1,1) ).flatten(start_dim=0, end_dim=1)
    # print( 'repeated features once for each rep of Ising model trainig:', edge_features.size() )
    # h = h.flatten(start_dim=0, end_dim=1)
    # J = J.flatten(start_dim=0, end_dim=1)

    return node_features, h, edge_features, J, coords_std, coords_mean, node_std, node_mean, sc_std, sc_mean

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

class PredictionLoss(torch.nn.Module):
    def __init__(self):
        super(PredictionLoss, self).__init__()
        self.pred_loss = torch.nn.MSELoss()
    def forward(self, param_pred:torch.Tensor, param_actual:torch.Tensor):
        return self.pred_loss(param_pred, param_actual)

def do_ising_model_sim(h:torch.Tensor, J:torch.Tensor, num_steps:int, beta:float_type=0.5):
    batch_size, num_nodes = h.size()
    ts = torch.zeros( (batch_size, num_steps, num_nodes), dtype=h.dtype, device=h.device )
    s = ising.get_random_state(batch_size=batch_size, num_nodes=num_nodes, dtype=h.dtype, device=h.device)
    ts, _ = ising.run_batched_balanced_metropolis_sim(sim_ts=ts, J=J, h=h, s=s, num_steps=num_steps, beta=beta)
    fc = hcp.get_fc_batch(ts)
    return fc

def compare_ising_models_by_fc(h1:torch.Tensor, J1:torch.Tensor, h2:torch.Tensor, J2:torch.Tensor, num_steps:int, beta:float_type=0.5):
    fc1 = do_ising_model_sim(h=h1, J=J1, num_steps=num_steps, beta=beta)
    fc2 = do_ising_model_sim(h=h2, J=J2, num_steps=num_steps, beta=beta)
    fc_rmse = hcp.get_triu_rmse_batch(fc1, fc2)
    fc_corr = hcp.get_triu_corr_batch(fc1, fc2)
    return fc_rmse, fc_corr

def predict_and_compare(node_model:torch.nn.Module, edge_model:torch.nn.Module, node_features:torch.Tensor, edge_features:torch.Tensor, h:torch.Tensor, J:torch.Tensor, num_steps:int, beta:float_type=0.5):
    h_pred = node_model(node_features)
    J_pred = edge_model(edge_features)
    h_loss = h - h_pred[:,None,:]
    J_loss = J - J_pred[:,None,:,:]
    num_subjects = h.size(dim=0)
    num_reps = h.size(dim=1)
    fc = do_ising_model_sim( h=h.flatten(start_dim=0, end_dim=1), J=J.flatten(start_dim=0, end_dim=1), num_steps=num_steps, beta=beta )
    fc_pred = do_ising_model_sim(h=h_pred, J=J_pred, num_steps=num_steps, beta=beta)[:,None,:,:].repeat( (1,num_reps,1,1) ).flatten(start_dim=0, end_dim=1)
    fc_rmse = hcp.get_triu_rmse_batch(fc, fc_pred).unflatten( dim=0, sizes=(num_subjects, num_reps) )
    fc_corr = hcp.get_triu_corr_batch(fc, fc_pred).unflatten( dim=0, sizes=(num_subjects, num_reps) )
    return h_loss, J_loss, fc_rmse, fc_corr

# During training, we select batches of node or edge feature vectors as randomly as possible,
# intermixing different reps, subjects, and regions/region pairs.
node_features, h, edge_features, J, coords_std, coords_mean, node_std, node_mean, sc_std, sc_mean = prepare_data('training', coords=coords, num_reps=num_reps, z_score=z_score)
num_subjects, num_nodes, num_node_features = node_features.size()
num_node_samples = num_subjects * num_reps * num_nodes
node_sample_subject_indices = torch.arange(end=num_subjects, device=device)[:,None,None].repeat( (1, num_reps, num_nodes) ).flatten()
node_sample_rep_indices = torch.arange(end=num_reps, device=device)[None,:,None].repeat( (num_subjects, 1, num_nodes) ).flatten()
node_sample_node_indices = torch.arange(end=num_nodes, device=device)[None,None,:].repeat( (num_subjects, num_reps, 1) ).flatten()
num_edge_samples = num_subjects * num_reps * num_nodes * num_nodes
edge_sample_subject_indices = torch.arange(end=num_subjects, device=device)[:,None,None,None].repeat( (1, num_reps, num_nodes, num_nodes) ).flatten()
edge_sample_rep_indices = torch.arange(end=num_reps, device=device)[None,:,None,None].repeat( (num_subjects, 1, num_nodes, num_nodes) ).flatten()
edge_sample_source_indices = torch.arange(end=num_nodes, device=device)[None,None,:,None].repeat( (num_subjects, num_reps, 1, num_nodes) ).flatten()
edge_sample_target_indices = torch.arange(end=num_nodes, device=device)[None,None,None,:].repeat( (num_subjects, num_reps, num_nodes, 1) ).flatten()
# During validation, we test an entire model, so we need to keep all region data for the same subject together.
# If the batch size does not divide the number of models evenly, include a short batch at the end.
num_validation_batches = num_subjects//validation_batch_size + int(num_subjects % validation_batch_size > 0)
num_edge_features = edge_features.size(dim=-1)
node_features_val, h_val, edge_features_val, J_val, coords_std, coords_mean, node_std, node_mean, sc_std, sc_mean = prepare_data('validation', coords=coords, num_reps=num_reps, z_score=z_score, coords_std=coords_std, coords_mean=coords_mean, node_std=node_std, node_mean=node_mean, sc_std=sc_std, sc_mean=sc_mean)
num_subjects_val = node_features_val.size(dim=0)
num_validation_batches_val = num_subjects_val//validation_batch_size + int(num_subjects_val % validation_batch_size > 0)

# Allocate some storage for loss values.
loss_record_shape = (num_subjects, num_reps)
h_loss = torch.zeros_like(h)
J_loss = torch.zeros_like(J)
fc_rmse = torch.zeros(loss_record_shape, dtype=float_type, device=device)
fc_corr = torch.zeros(loss_record_shape, dtype=float_type, device=device)
loss_record_shape_val = (num_subjects_val, num_reps)
h_loss_val = torch.zeros_like(h_val)
J_loss_val = torch.zeros_like(J_val)
fc_rmse_val = torch.zeros(loss_record_shape_val, dtype=float_type, device=device)
fc_corr_val = torch.zeros(loss_record_shape_val, dtype=float_type, device=device)

loss_fn = PredictionLoss()
node_batch_size = num_node_samples//num_batches
edge_batch_size = num_edge_samples//num_batches
# Define our ML stuff.
node_model = Struct2Param(num_features=num_node_features, hidden_layer_width=node_hidden_layer_width, num_hidden_layer=node_num_hidden_layers, dtype=float_type, device=device)
edge_model = Struct2Param(num_features=num_edge_features, hidden_layer_width=edge_hidden_layer_width, num_hidden_layer=edge_num_hidden_layers, dtype=float_type, device=device)
node_optimizer = torch.optim.Adam( node_model.parameters(), lr=learning_rate )
edge_optimizer = torch.optim.Adam( edge_model.parameters(), lr=learning_rate )
ising_model_string = f'nodes_{num_nodes}_reps_{num_reps}_epochs_{num_epochs_ising}_window_{window_length}_lr_{learning_rate_ising}_threshold_{threshold}'
print('starting training...')
last_time = time.time()
for epoch in range(num_epochs):
    for subepoch in range(num_subepochs):
        node_sample_order = torch.randperm(num_node_samples, dtype=torch.int, device=device)
        node_shuffled_subject_indices = node_sample_subject_indices[node_sample_order]
        node_shuffled_rep_indices = node_sample_rep_indices[node_sample_order]
        node_shuffled_node_indices = node_sample_node_indices[node_sample_order]
        node_features_shuffled = node_features[node_shuffled_subject_indices, node_shuffled_node_indices, :]
        h_shuffled = h[node_shuffled_subject_indices, node_shuffled_rep_indices, node_shuffled_node_indices]
        edge_sample_order = torch.randperm(num_edge_samples, dtype=torch.int, device=device)
        edge_shuffled_subject_indices = edge_sample_subject_indices[edge_sample_order]
        edge_shuffled_rep_indices = edge_sample_rep_indices[edge_sample_order]
        edge_shuffled_source_indices = edge_sample_source_indices[edge_sample_order]
        edge_shuffled_target_indices = edge_sample_target_indices[edge_sample_order]
        edge_features_shuffled = edge_features[edge_shuffled_subject_indices, edge_shuffled_source_indices, edge_shuffled_target_indices, :]
        J_shuffled = J[edge_shuffled_subject_indices, edge_shuffled_rep_indices, edge_shuffled_source_indices, edge_shuffled_target_indices]
        for batch in range(num_batches):
            # Do the training step for the node model.
            node_optimizer.zero_grad()
            batch_start = node_batch_size*batch
            batch_end = batch_start + node_batch_size
            batch_node_features = node_features_shuffled[batch_start:batch_end,:]
            batch_h = h_shuffled[batch_start:batch_end]
            h_pred = node_model(batch_node_features)
            node_loss = loss_fn(h_pred, batch_h)
            node_loss.backward()
            node_optimizer.step()
            # Do the training step for the node model.
            edge_optimizer.zero_grad()
            batch_start = edge_batch_size*batch
            batch_end = batch_start + edge_batch_size
            batch_edge_features = edge_features_shuffled[batch_start:batch_end,:]
            batch_J = J_shuffled[batch_start:batch_end]
            J_pred = edge_model(batch_edge_features)
            edge_loss = loss_fn(J_pred, batch_J)
            edge_loss.backward()
            edge_optimizer.step()
            # print(f'epoch {epoch}, subepoch {subepoch}, batch {batch}, node loss {node_loss.item():.3g}, edge_loss {edge_loss.item():.3g}')
    param_string = f'struct2ising_epochs_{(epoch+1)*num_subepochs}_val_batch_{validation_batch_size}_steps_{num_steps}_lr_{learning_rate}_batches_{num_batches}_node_hl_{node_num_hidden_layers}_node_w_{node_hidden_layer_width}_edge_hl_{edge_num_hidden_layers}_edge_w_{edge_hidden_layer_width}_ising_{ising_model_string}'
    node_model_file = os.path.join(stats_dir, f'node_model_{param_string}')
    torch.save(node_model, node_model_file)
    edge_model_file = os.path.join(stats_dir, f'edge_model_{param_string}')
    torch.save(edge_model, edge_model_file)
    with torch.no_grad():
        print('validating...')
        for batch in range(num_validation_batches):
            batch_start = validation_batch_size*batch
            batch_end = min(batch_start + validation_batch_size, num_subjects)
            # Do the training step for the node model.
            batch_node_features = node_features[batch_start:batch_end,:,:]
            batch_h = h[batch_start:batch_end,:,:]
            batch_edge_features = edge_features[batch_start:batch_end,:,:,:]
            batch_J = J[batch_start:batch_end,:,:,:]
            # Run the model, and compare the FC to that of the data time series.
            hl, jl, fcr, fcc = predict_and_compare(node_model, edge_model, batch_node_features, batch_edge_features, batch_h, batch_J, num_steps)
            h_loss[batch_start:batch_end,:,:] = hl
            J_loss[batch_start:batch_end,:,:,:] = jl
            fc_rmse[batch_start:batch_end,:] = fcr
            fc_corr[batch_start:batch_end,:] = fcc
            print(f'training batch {batch}')
        for batch in range(num_validation_batches_val):
            batch_start = validation_batch_size*batch
            batch_end = min(batch_start + validation_batch_size, num_subjects_val)
            # Do the training step for the node model.
            batch_node_features = node_features_val[batch_start:batch_end,:,:]
            batch_h = h_val[batch_start:batch_end,:,:]
            batch_edge_features = edge_features_val[batch_start:batch_end,:,:,:]
            batch_J = J_val[batch_start:batch_end,:,:,:]
            # Run the model, and compare the FC to that of the data time series.
            hl, jl, fcr, fcc = predict_and_compare(node_model, edge_model, batch_node_features, batch_edge_features, batch_h, batch_J, num_steps)
            h_loss_val[batch_start:batch_end,:,:] = hl
            J_loss_val[batch_start:batch_end,:,:,:] = jl
            fc_rmse_val[batch_start:batch_end,:] = fcr
            fc_corr_val[batch_start:batch_end,:] = fcc
            print(f'validation batch {batch}')
        # Print out the mean values.
        mean_h_loss = h_loss.abs().mean()
        mean_h_loss_val = h_loss_val.abs().mean()
        mean_J_loss = J_loss.abs().mean()
        mean_J_loss_val = J_loss_val.abs().mean()
        mean_fc_rmse = fc_rmse.mean()
        mean_fc_rmse_val = fc_rmse_val.mean()
        mean_fc_corr = fc_corr.mean()
        mean_fc_corr_val = fc_corr_val.mean()
        current_time = time.time() - last_time
        print(f'epoch {epoch},\th t {mean_h_loss:.3g},\th v {mean_h_loss_val:.3g},\tJ t {mean_J_loss:.3g},\tJ v {mean_J_loss_val:.3g},\tFC RMSE t {mean_fc_rmse:.3g},\tFC RMSE v {mean_fc_rmse_val:.3g},\tFC corr t {mean_fc_corr:.3g},\tFC corr v {mean_fc_corr_val:.3g},\ttime {current_time:.3g}')
print('saving...')
last_time = time.time()
param_string = f'struct2ising_epochs_{num_epochs*num_subepochs}_val_batch_{validation_batch_size}_steps_{num_steps}_lr_{learning_rate}_batches_{num_batches}_node_hl_{node_num_hidden_layers}_node_w_{node_hidden_layer_width}_edge_hl_{edge_num_hidden_layers}_edge_w_{edge_hidden_layer_width}_ising_{ising_model_string}'
node_model_file = os.path.join(stats_dir, f'node_model_{param_string}.pt')
torch.save(node_model, node_model_file)
edge_model_file = os.path.join(stats_dir, f'edge_model_{param_string}.pt')
torch.save(edge_model, edge_model_file)
h_loss_file = os.path.join(stats_dir, f'h_loss_{param_string}.pt')
torch.save(h_loss, h_loss_file)
J_loss_file = os.path.join(stats_dir, f'J_loss_{param_string}.pt')
torch.save(J_loss, J_loss_file)
fc_rmse_file = os.path.join(stats_dir, f'fc_rmse_{param_string}.pt')
torch.save(fc_rmse, fc_rmse_file)
fc_corr_file = os.path.join(stats_dir, f'fc_corr_{param_string}.pt')
torch.save(fc_corr, fc_corr_file)
h_loss_val_file = os.path.join(stats_dir, f'h_loss_val_{param_string}.pt')
torch.save(h_loss_val, h_loss_val_file)
J_loss_val_file = os.path.join(stats_dir, f'J_loss_val_{param_string}.pt')
torch.save(J_loss_val, J_loss_val_file)
fc_rmse_val_file = os.path.join(stats_dir, f'fc_rmse_val_{param_string}.pt')
torch.save(fc_rmse_val, fc_rmse_val_file)
fc_corr_val_file = os.path.join(stats_dir, f'fc_corr_val_{param_string}.pt')
torch.save(fc_corr_val, fc_corr_val_file)
current_time = time.time()
print('done,', current_time-last_time, 'seconds to save,', current_time-start_time, 'seconds total')