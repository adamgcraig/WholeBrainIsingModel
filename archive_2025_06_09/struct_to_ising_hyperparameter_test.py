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
parser.add_argument("-s", "--structural_data_dir", type=str, default='E:\\HCP_data', help="directory containing the structural MRI features data file")
parser.add_argument("-i", "--ising_model_dir", type=str, default='E:\\Ising_model_results_daai', help="directory containing the fitted Ising model J parameter file")
parser.add_argument("-v", "--stats_dir", type=str, default="E:\\Ising_model_results_batch", help="directory to which to write the output files from VAE training")

# hyperparameters of the Ising model, used for looking up which h files to load
parser.add_argument("-n", "--num_nodes", type=int, default=21, help="number of nodes in Ising model")
parser.add_argument("-m", "--num_reps", type=int, default=100, help="number of Ising models trained for each subject")
parser.add_argument("-o", "--num_epochs_ising", type=int, default=1000, help="number of epochs for which we trained the Ising model")
parser.add_argument("-p", "--window_length", type=int, default=50, help="window length used when training Ising model")
parser.add_argument("-j", "--learning_rate_ising", type=str, default='0.001', help="learning rate used when training Ising model")
parser.add_argument("-t", "--threshold", type=str, default='0.100', help="threshold at which to binarize the fMRI data, in standard deviations above the mean, in the Ising model")
parser.add_argument("-b", "--beta", type=str, default='0.500', help="beta constant used in Ising model")
# To avoid number formatting inconsistencies, we use strings for the floating point numbers that we only use in file names.

# hyperparameters of the model training
parser.add_argument("-e", "--num_epochs", type=int, default=1, help="number of epochs for which to train")
parser.add_argument("-y", "--num_subepochs", type=int, default=1000, help="number of epochs for which to train between evaluations of the full model")
parser.add_argument("-z", "--num_steps", type=int, default=1200, help="number of steps to use when running Ising model simulations to compare FC")
parser.add_argument("-a", "--validation_batch_size", type=int, default=100, help="size of the batches we use in the validation run where we actually run the Ising model, relevant for memory footprint vs speed purposes, not the gradient descent process itself")

args = parser.parse_args()

structural_data_dir = args.structural_data_dir
ising_model_dir = args.ising_model_dir
stats_dir = args.stats_dir

num_nodes = args.num_nodes
num_reps = args.num_reps
num_epochs_ising = args.num_epochs_ising
window_length = args.window_length
learning_rate_ising = args.learning_rate_ising
threshold = args.threshold
beta_str = args.beta
beta_float = float(beta_str)
num_epochs = args.num_epochs
num_subepochs = args.num_subepochs
num_steps = args.num_steps
validation_batch_size = args.validation_batch_size

learning_rates = [0.01, 0.001, 0.0001]
num_learning_rates = len(learning_rates)
num_batches_choices = [100, 1000, 10000]
num_batch_sizes = len(num_batches_choices)
node_hidden_layer_widths = [14, 7, 3]
node_num_hidden_layer_widths = len(node_hidden_layer_widths)
node_num_hidden_layers = [10, 3, 0]
node_num_num_hidden_layers = len(node_num_hidden_layers)
edge_hidden_layer_widths = [30, 15, 7]
edge_num_hidden_layer_widths = len(edge_hidden_layer_widths)
edge_num_hidden_layers = [10, 3, 0]
edge_num_num_hidden_layers = len(edge_num_hidden_layers)
# For the purposes of this list, we are not counting the first or last linear layer as hidden,
# so every network has at least two layers.

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

def prepare_data(subset:str, coords:list, num_reps:int):

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

    # Concatenate features together and duplicate them as needed.
    coords = coords[:num_nodes,:]# torch.tensor(coords[:num_nodes,:], dtype=float_type, device=device)
    node_features = torch.cat(   (  node_features, coords[None,:,:].repeat( (num_subjects,1,1) )  ), dim=-1   )
    print( 'node features extended with coords:', node_features.size() )
    edge_features = torch.cat(   (  node_features[:,None,:,:].repeat( (1,num_nodes,1,1) ), node_features[:,:,None,:].repeat( (1,1,num_nodes,1) ), sc[:,:,:,None]  ), dim=-1   )
    print( 'node features paired between source and target and SC appended to create edge features:', node_features.size() )
    node_features = node_features[:,None,:,:].repeat( (1,num_reps,1,1) ).flatten(start_dim=0, end_dim=1)
    print( 'repeated features once for each rep of Ising model trainig:', node_features.size() )
    edge_features = edge_features[:,None,:,:,:].repeat( (1,num_reps,1,1,1) ).flatten(start_dim=0, end_dim=1)
    print( 'repeated features once for each rep of Ising model trainig:', edge_features.size() )
    h = h.flatten(start_dim=0, end_dim=1)
    J = J.flatten(start_dim=0, end_dim=1)

    return node_features, h, edge_features, J

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

def get_loss_components(model:torch.nn.Module, features:torch.Tensor, params:torch.Tensor):
    param_pred = model(features)
    param_loss = torch.abs(param_pred - params).squeeze()
    return param_loss

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
    h_loss = torch.nn.functional.mse_loss(h, h_pred).sqrt()
    J_loss = torch.nn.functional.mse_loss(J, J_pred).sqrt()
    fc_rmse, fc_corr = compare_ising_models_by_fc(h_pred, J_pred, h, J, num_steps, beta)
    return h_loss, J_loss, fc_rmse, fc_corr

node_features, h, edge_features, J = prepare_data('training', coords=coords, num_reps=num_reps)
num_models, num_nodes, num_node_features = node_features.size()
num_node_samples = num_models * num_nodes
node_sample_rows = torch.repeat_interleave( input=torch.arange(end=num_models, device=device), repeats=num_nodes )
node_sample_cols = torch.arange(end=num_nodes, device=device).repeat(num_models)
num_edge_samples = num_models * num_nodes * num_nodes
edge_sample_pages = torch.arange(end=num_models, device=device)[:,None,None].repeat( (1, num_nodes, num_nodes) ).flatten()
edge_sample_rows = torch.arange(end=num_nodes, device=device)[None,:,None].repeat( (num_models, 1, num_nodes) ).flatten()
edge_sample_cols = torch.arange(end=num_nodes, device=device)[None,None,:].repeat( (num_models, num_nodes, 1) ).flatten()
# If the batch size does not divide the number of models evenly, include a short batch at the end.
num_validation_batches = num_models//validation_batch_size + int(num_models % validation_batch_size > 0)
num_edge_features = edge_features.size(dim=-1)
node_features_val, h_val, edge_features_val, J_val = prepare_data('validation', coords=coords, num_reps=num_reps)
num_models_val = node_features_val.size(dim=0)
num_validation_batches_val = num_models_val//validation_batch_size + int(num_models_val % validation_batch_size > 0)

# Allocate some storage for loss values.
loss_record_shape = (num_learning_rates, num_batch_sizes, node_num_hidden_layer_widths, node_num_num_hidden_layers, edge_num_hidden_layer_widths, edge_num_num_hidden_layers, num_models)
h_loss = torch.zeros(loss_record_shape, dtype=float_type, device=device)
J_loss = torch.zeros(loss_record_shape, dtype=float_type, device=device)
fc_rmse = torch.zeros(loss_record_shape, dtype=float_type, device=device)
fc_corr = torch.zeros(loss_record_shape, dtype=float_type, device=device)
loss_record_shape_val = (num_learning_rates, num_batch_sizes, node_num_hidden_layer_widths, node_num_num_hidden_layers, edge_num_hidden_layer_widths, edge_num_num_hidden_layers, num_models_val)
h_loss_val = torch.zeros(loss_record_shape_val, dtype=float_type, device=device)
J_loss_val = torch.zeros(loss_record_shape_val, dtype=float_type, device=device)
fc_rmse_val = torch.zeros(loss_record_shape_val, dtype=float_type, device=device)
fc_corr_val = torch.zeros(loss_record_shape_val, dtype=float_type, device=device)

loss_fn = PredictionLoss()

for lr_index in range(num_learning_rates):
    learning_rate = learning_rates[lr_index]
    for bs_index in range(num_batch_sizes):
        num_batches = num_batches_choices[bs_index]
        node_batch_size = num_node_samples//num_batches
        edge_batch_size = num_edge_samples//num_batches
        for nhlw_index in range(node_num_hidden_layer_widths):
            node_hidden_layer_width = node_hidden_layer_widths[nhlw_index]
            for nnhl_index in range(node_num_num_hidden_layers):
                node_num_hidden_layer = node_num_hidden_layers[nnhl_index]# This is not grammatically correct, but it fits the naming convention.
                for ehlw_index in range(edge_num_hidden_layer_widths):
                    edge_hidden_layer_width = edge_hidden_layer_widths[ehlw_index]
                    for enhl_index in range(edge_num_num_hidden_layers):
                        edge_num_hidden_layer = edge_num_hidden_layers[enhl_index]
                        # Define our ML stuff.
                        node_model = Struct2Param(num_features=num_node_features, hidden_layer_width=node_hidden_layer_width, num_hidden_layer=node_num_hidden_layer, dtype=float_type, device=device)
                        edge_model = Struct2Param(num_features=num_edge_features, hidden_layer_width=edge_hidden_layer_width, num_hidden_layer=edge_num_hidden_layer, dtype=float_type, device=device)
                        node_optimizer = torch.optim.Adam( node_model.parameters(), lr=learning_rate )
                        edge_optimizer = torch.optim.Adam( edge_model.parameters(), lr=learning_rate )
                        print('starting training...')
                        last_time = time.time()
                        for epoch in range(num_epochs):
                            for subepoch in range(num_subepochs):
                                node_sample_order = torch.randperm(num_node_samples, dtype=torch.int, device=device)
                                node_shuffled_rows = node_sample_rows[node_sample_order]
                                node_shuffled_cols = node_sample_cols[node_sample_order]
                                node_features_shuffled = node_features[node_shuffled_rows, node_shuffled_cols, :]
                                h_shuffled = h[node_shuffled_rows, node_shuffled_cols]
                                edge_sample_order = torch.randperm(num_edge_samples, dtype=torch.int, device=device)
                                edge_shuffled_pages = edge_sample_pages[edge_sample_order]
                                edge_shuffled_rows = edge_sample_rows[edge_sample_order]
                                edge_shuffled_cols = edge_sample_cols[edge_sample_order]
                                edge_features_shuffled = edge_features[edge_shuffled_pages, edge_shuffled_rows, edge_shuffled_cols, :]
                                J_shuffled = J[edge_shuffled_pages, edge_shuffled_rows, edge_shuffled_cols]
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
                            with torch.no_grad():
                                for batch in range(num_validation_batches):
                                    batch_start = validation_batch_size*batch
                                    batch_end = min(batch_start + validation_batch_size, num_models)
                                    # Do the training step for the node model.
                                    batch_node_features = node_features[batch_start:batch_end,:,:]
                                    batch_h = h[batch_start:batch_end,:]
                                    batch_edge_features = edge_features[batch_start:batch_end,:,:,:]
                                    batch_J = J[batch_start:batch_end,:,:]
                                    # Run the model, and compare the FC to that of the data time series.
                                    hl, jl, fcr, fcc = predict_and_compare(node_model, edge_model, batch_node_features, batch_edge_features, batch_h, batch_J, num_steps)
                                    h_loss[lr_index, bs_index, nhlw_index, nnhl_index, ehlw_index, enhl_index, batch_start:batch_end] = hl
                                    J_loss[lr_index, bs_index, nhlw_index, nnhl_index, ehlw_index, enhl_index, batch_start:batch_end] = jl
                                    fc_rmse[lr_index, bs_index, nhlw_index, nnhl_index, ehlw_index, enhl_index, batch_start:batch_end] = fcr
                                    fc_corr[lr_index, bs_index, nhlw_index, nnhl_index, ehlw_index, enhl_index, batch_start:batch_end] = fcc
                                for batch in range(num_validation_batches_val):
                                    batch_start = validation_batch_size*batch
                                    batch_end = min(batch_start + validation_batch_size, num_models_val)
                                    # Do the training step for the node model.
                                    batch_node_features = node_features_val[batch_start:batch_end,:,:]
                                    batch_h = h_val[batch_start:batch_end,:]
                                    batch_edge_features = edge_features_val[batch_start:batch_end,:,:,:]
                                    batch_J = J_val[batch_start:batch_end,:,:]
                                    # Run the model, and compare the FC to that of the data time series.
                                    hl, jl, fcr, fcc = predict_and_compare(node_model, edge_model, batch_node_features, batch_edge_features, batch_h, batch_J, num_steps)
                                    h_loss_val[lr_index, bs_index, nhlw_index, nnhl_index, ehlw_index, enhl_index, batch_start:batch_end] = hl
                                    J_loss_val[lr_index, bs_index, nhlw_index, nnhl_index, ehlw_index, enhl_index, batch_start:batch_end] = jl
                                    fc_rmse_val[lr_index, bs_index, nhlw_index, nnhl_index, ehlw_index, enhl_index, batch_start:batch_end] = fcr
                                    fc_corr_val[lr_index, bs_index, nhlw_index, nnhl_index, ehlw_index, enhl_index, batch_start:batch_end] = fcc
                                # Print out the mean values.
                                mean_h_loss = h_loss[lr_index, bs_index, nhlw_index, nnhl_index, ehlw_index, enhl_index, :].mean()
                                mean_h_loss_val = h_loss_val[lr_index, bs_index, nhlw_index, nnhl_index, ehlw_index, enhl_index, :].mean()
                                mean_J_loss = J_loss[lr_index, bs_index, nhlw_index, nnhl_index, ehlw_index, enhl_index, :].mean()
                                mean_J_loss_val = J_loss_val[lr_index, bs_index, nhlw_index, nnhl_index, ehlw_index, enhl_index, :].mean()
                                mean_fc_rmse = fc_rmse[lr_index, bs_index, nhlw_index, nnhl_index, ehlw_index, enhl_index, :].mean()
                                mean_fc_rmse_val = fc_rmse_val[lr_index, bs_index, nhlw_index, nnhl_index, ehlw_index, enhl_index, :].mean()
                                mean_fc_corr = fc_corr[lr_index, bs_index, nhlw_index, nnhl_index, ehlw_index, enhl_index, :].mean()
                                mean_fc_corr_val = fc_corr_val[lr_index, bs_index, nhlw_index, nnhl_index, ehlw_index, enhl_index, :].mean()
                                current_time = time.time() - last_time
                                print(f'epoch {epoch},\th t {mean_h_loss:.3g},\th v {mean_h_loss_val:.3g},\tJ t {mean_J_loss:.3g},\tJ v {mean_J_loss_val:.3g},\tFC RMSE t {mean_fc_rmse:.3g},\tFC RMSE v {mean_fc_rmse_val:.3g},\tFC corr t {mean_fc_corr:.3g},\tFC corr v {mean_fc_corr_val:.3g},\ttime {current_time:.3g}')

print('saving...')
last_time = time.time()
ising_model_string = f'nodes_{num_nodes}_reps_{num_reps}_epochs_{num_epochs_ising}_window_{window_length}_lr_{learning_rate_ising}_threshold_{threshold}'
param_string = f'struct2ising_hyperparam_test_epochs_{num_epochs}_subepochs_{num_subepochs}_val_batch_{validation_batch_size}_steps_{num_steps}_ising_{ising_model_string}'
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