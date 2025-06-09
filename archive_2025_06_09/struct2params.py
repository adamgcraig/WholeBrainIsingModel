import os
import torch
import time
import argparse
import pandas
import hcpdatautils as hcp

start_time = time.time()
last_time = start_time

parser = argparse.ArgumentParser(description="Predict Ising model parameters from structural MRI and DT-MRI structural connectivity data.")

# directories
parser.add_argument("-s", "--structural_data_dir", type=str, default='E:\\HCP_data', help="directory containing the structural MRI features data file")
parser.add_argument("-i", "--ising_model_dir", type=str, default='E:\\Ising_model_results_daai', help="directory containing the fitted Ising model J parameter file")
parser.add_argument("-v", "--vae_dir", type=str, default="E:\\Ising_VAE_results", help="directory to which to write the output files from VAE training")

# hyperparameters of the Ising model, used for looking up which h files to load
# parser.add_argument("-n", "--num_nodes", type=int, default=21, help="number of nodes in Ising model")
# parser.add_argument("-m", "--num_reps", type=int, default=10, help="number of Ising models trained for each subject")
# parser.add_argument("-o", "--num_epochs_ising", type=int, default=200, help="number of epochs for which we trained the Ising model")
# parser.add_argument("-p", "--prob_update", type=str, default='0.019999999552965164', help="probability of updating the model parameters on any given step used when training Ising model")
# parser.add_argument("-j", "--learning_rate_ising", type=str, default='0.001', help="learning rate used when training Ising model")
# parser.add_argument("-t", "--threshold", type=float, default=0.1, help="threshold at which to binarize the fMRI data, in standard deviations above the mean, in the Ising model")

parser.add_argument("-n", "--num_nodes", type=int, default=90, help="number of nodes in Ising model")
parser.add_argument("-m", "--num_reps", type=int, default=5, help="number of Ising models trained for each subject")
parser.add_argument("-o", "--num_epochs_ising", type=int, default=1000, help="number of epochs for which we trained the Ising model")
parser.add_argument("-p", "--prob_update", type=str, default='0.05000000074505806', help="probability of updating the model parameters on any given step used when training Ising model")
parser.add_argument("-j", "--learning_rate_ising", type=str, default='0.001', help="learning rate used when training Ising model")
parser.add_argument("-t", "--threshold", type=float, default=0.1, help="threshold at which to binarize the fMRI data, in standard deviations above the mean, in the Ising model")

# hyperparameters of the VAE
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate by which we multiply updates to weights and biases at each step")
parser.add_argument("-r", "--recon_weight", type=float, default=0.0, help="weight to give to reconstruction loss of structural feature vector")
parser.add_argument("-k", "--kl_weight", type=float, default=0.0, help="weight to give to KL-divergence of representation distribution from standard normal distribution")
parser.add_argument("-w", "--J_weight", type=float, default=1.0, help="weight to give to prediction loss of J from trained Ising model")
parser.add_argument("-b", "--batch_size", type=int, default=60, help="size of batch of structural feature vectors over which to calculate the gradient at each learning step")
parser.add_argument("-e", "--num_epochs", type=int, default=10000, help="number of epochs for which to train")

args = parser.parse_args()

structural_data_dir = args.structural_data_dir
ising_model_dir = args.ising_model_dir
vae_dir = args.vae_dir

num_nodes = args.num_nodes
num_reps = args.num_reps
num_epochs_ising = args.num_epochs_ising
prob_update = args.prob_update
learning_rate_ising = args.learning_rate_ising
threshold = args.threshold

J_weight = args.J_weight
kl_weight = args.kl_weight
recon_weight = args.recon_weight
learning_rate = args.learning_rate
batch_size = args.batch_size
num_epochs = args.num_epochs
num_subepochs = 1

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

ising_model_string = f'nodes_{num_nodes}_reps_{num_reps}_epochs_{num_epochs_ising}_p_{prob_update}_lr_{learning_rate_ising}_threshold_{threshold}'

def prepare_data(subset:str, coords:list, num_reps:int):

    last_time = time.time()
    if subset == 'training':
        num_subjects = 669
        num_subjects_fn = 699
    else:
        num_subjects = 83
        num_subjects_fn = 83
    
    h = torch.zeros( (num_reps, num_subjects, num_nodes), dtype=float_type, device=device )
    J = torch.zeros( (num_reps, num_subjects, num_nodes, num_nodes), dtype=float_type, device=device )

    for rep in range(num_reps):
        ising_model_string = f'nodes_{num_nodes}_rep_{rep}_epochs_{num_epochs_ising}_p_{prob_update}_lr_{learning_rate_ising}_threshold_{threshold}'
        ising_r_file = os.path.join(ising_model_dir, f'h_data_{subset}_{ising_model_string}_start_0_end_{num_subjects_fn}.pt')
        h[rep,:,:] = torch.load(ising_r_file)[:num_subjects,:]
        # num_reps, num_subjects_h, num_nodes_h = h.size()
        # current_time = time.time()
        # print('h:',num_subjects_h,'x',num_nodes_h, 'time', current_time-last_time, 'seconds')
        # last_time = current_time

        ising_J_file = os.path.join(ising_model_dir, f'J_data_{subset}_{ising_model_string}_start_0_end_{num_subjects_fn}.pt')
        J[rep,:,:,:] = torch.load(ising_J_file)[:num_subjects,:,:]
        # num_reps, num_subjects_J, num_nodes_J, num_nodes_J2 = J.size()
        # current_time = time.time()
        # print('J:', num_subjects_J, 'x', num_nodes_J, 'x', num_nodes_J2, ', time', current_time-last_time, 'seconds')
        # last_time = current_time
    current_time = time.time()
    print('loaded h and J, time', current_time-last_time )
    last_time = current_time

    features_file_name = f'E:\\HCP_data\\sc_mri_structural_features_{subset}.pt'
    # In the file, the order of dimensions is (subject, feature, node).
    # We want (subject, node, feature).
    node_features = torch.transpose( torch.load(features_file_name), dim0=-2, dim1=-1 )
    num_subjects_features, num_nodes_features, num_features = node_features.size()
    current_time = time.time()
    print('features:', num_subjects_features, 'x', num_nodes_features, 'x', num_features, 'time', current_time-last_time, 'seconds')
    last_time = current_time

    sc_file_name = f'E:\\HCP_data\\dtmri_sc_{subset}.pt'
    sc = torch.transpose( torch.load(sc_file_name), dim0=-2, dim1=-1 )
    num_subjects_sc, num_nodes_sc, num_nodes_sc2 = sc.size()
    current_time = time.time()
    print('sc:',num_subjects_sc,'x',num_nodes_sc,'x',num_nodes_sc2, 'time', current_time-last_time, 'seconds')
    last_time = current_time

    # num_subjects = min( min(num_subjects_features, num_subjects_sc), num_subjects_J )
    # print('using', num_subjects, 'subjects')
    # num_nodes = min( min(num_nodes_coords, num_nodes_features), min(num_nodes_J, num_nodes_sc) )
    # print('using', num_nodes, 'nodes')

    coords = coords[:num_nodes,:]
    node_features = node_features[:num_subjects,:num_nodes,:]
    sc = sc[:num_subjects,:num_nodes,:num_nodes]
    h = h[:,:num_subjects,:num_nodes]
    J = J[:,:num_subjects,:num_nodes,:num_nodes]

    node_features = torch.cat(   (  node_features, coords[None,:,:].repeat( (num_subjects,1,1) )  ), dim=-1   )
    print( 'node features extended with coords:', node_features.size() )
    edge_features = torch.cat(   (  node_features[:,None,:,:].repeat( (1,num_nodes,1,1) ), node_features[:,:,None,:].repeat( (1,1,num_nodes,1) ), sc[:,:,:,None]  ), dim=-1   )
    print( 'node features paired between source and target and SC appended to create edge features:', node_features.size() )
    node_features = node_features[None,:,:,:].repeat( (num_reps,1,1,1) )
    print( 'repeated features once for each rep of Ising model trainig:', node_features.size() )
    edge_features = edge_features[None,:,:,:,:].repeat( (num_reps,1,1,1,1) )
    print( 'repeated features once for each rep of Ising model trainig:', edge_features.size() )

    return node_features, h, edge_features, J

class Struct2ParamVAE(torch.nn.Module):
    # previously worked well with 21-node model:
    # def __init__(self, num_features:int, rep_dims:int=15, hidden_layer_width_1:int=15, hidden_layer_width_2:int=15, dtype=float_type, device=device)
    def __init__(self, num_features:int, rep_dims:int=90, hidden_layer_width_1:int=90, hidden_layer_width_2:int=90, dtype=float_type, device=device):
        super(Struct2ParamVAE, self).__init__()
        self.rep_dims = rep_dims
        self.struct_enc = torch.nn.Sequential(
            torch.nn.Linear(in_features=num_features, out_features=hidden_layer_width_1, dtype=dtype, device=device),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_layer_width_1, out_features=hidden_layer_width_2, dtype=dtype, device=device),
            torch.nn.ReLU()
        )
        self.mu_layer = torch.nn.Linear(in_features=hidden_layer_width_2, out_features=rep_dims, dtype=dtype, device=device)
        # self.sigma_layer = torch.nn.Linear(in_features=hidden_layer_width_2, out_features=rep_dims, dtype=dtype, device=device)
        # self.struct_dec = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=rep_dims, out_features=hidden_layer_width_2, dtype=dtype, device=device),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(in_features=hidden_layer_width_2, out_features=hidden_layer_width_1, dtype=dtype, device=device),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(in_features=hidden_layer_width_1, out_features=num_features, dtype=dtype, device=device)
        # )
        self.param_dec = torch.nn.Sequential(
            torch.nn.Linear(in_features=rep_dims, out_features=hidden_layer_width_2, dtype=dtype, device=device),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_layer_width_2, out_features=hidden_layer_width_1, dtype=dtype, device=device),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_layer_width_1, out_features=1, dtype=dtype, device=device)
        )
    
    def forward(self, features):
        rep = self.struct_enc(features)
        mu = self.mu_layer(rep)
        # log_sigma = self.sigma_layer(rep)
        # z = mu + torch.exp(log_sigma)*torch.randn_like(mu)
        param_pred = self.param_dec(mu).squeeze()
        return param_pred

def kl_div_norm_vs_std_norm(mu:torch.Tensor, log_sigma:torch.Tensor):
    return 0.5 * torch.mean( torch.pow(mu,2.0) + torch.exp(log_sigma) - 1 - log_sigma  )

class RAndNegELBO(torch.nn.Module):

    def __init__(self):
        super(RAndNegELBO, self).__init__()
        self.recon_loss = torch.nn.MSELoss()
        self.J_loss = torch.nn.MSELoss()

    def forward(self, J_pred:torch.Tensor, J:torch.Tensor):
        J_loss = self.J_loss(J_pred, J)
        return J_loss

def get_loss_components(model:torch.nn.Module, features:torch.Tensor, params:torch.Tensor):
    param_pred = model(features)
    param_loss = torch.abs(param_pred - params).squeeze()
    return param_loss

def do_ising_model_sim(h:torch.Tensor, J:torch.Tensor, num_steps:int, beta:float_type=0.5):
    num_models, num_nodes = h.size()
    dtype = h.dtype
    device = h.device
    sim_ts = torch.zeros( (num_models, num_steps, num_nodes), dtype=dtype, device=device )
    state = 2.0*torch.randint_like(h, low=0, high=1) - 1.0
    for t in range(num_steps):
        node_order = torch.randperm(num_nodes, dtype=torch.int, device=device)
        for n in range(num_nodes):
            node_index = node_order[n]
            deltaH = 2*(  torch.sum( J[:,:,node_index] * state, dim=-1 ) + h[:,node_index]  )*state[:,node_index]
            prob_accept = torch.clamp( torch.exp(-beta*deltaH), min=0.0, max=1.0 )
            state[:,node_index] *= ( 1.0 - 2.0*torch.bernoulli(prob_accept) )
        sim_ts[:,t,:] = state
    return sim_ts

def compare_ising_models_by_fc(h1:torch.Tensor, J1:torch.Tensor, h2:torch.Tensor, J2:torch.Tensor, num_steps:int, beta:float_type=0.5):
    ts1 = do_ising_model_sim(h1, J1, num_steps, beta)
    ts2 = do_ising_model_sim(h2, J2, num_steps, beta)
    fc1 = hcp.get_fc_batch(ts1)
    fc2 = hcp.get_fc_batch(ts2)
    fc_rmse = hcp.get_triu_rmse_batch(fc1, fc2)
    fc_corr = hcp.get_triu_corr_batch(fc1, fc2)
    return fc_rmse, fc_corr

def predict_and_compare(node_model:torch.nn.Module, edge_model:torch.nn.Module, node_features:torch.Tensor, edge_features:torch.Tensor, h:torch.Tensor, J:torch.Tensor, num_steps:int, beta:float_type=0.5):
    h_pred = node_model(node_features)
    J_pred = edge_model(edge_features)
    fc_rmse, fc_corr = compare_ising_models_by_fc(h_pred, J_pred, h, J, num_steps, beta)
    return fc_rmse, fc_corr

node_features, h, edge_features, J = prepare_data('training', coords=coords, num_reps=num_reps)
num_reps, num_subjects, num_nodes, num_node_features = node_features.size()
num_edge_features = edge_features.size(dim=-1)
node_features_val, h_val, edge_features_val, J_val = prepare_data('validation', coords=coords, num_reps=num_reps)
num_reps_val, num_subjects_val, _, _ = node_features_val.size()
# Flatten the rep and subject dimensions together so that we only need to deal with one batch dimension when training.
node_features = node_features.flatten(start_dim=0, end_dim=1)
edge_features = edge_features.flatten(start_dim=0, end_dim=1)
h = h.flatten(start_dim=0, end_dim=1)
J = J.flatten(start_dim=0, end_dim=1)
node_features_val = node_features_val.flatten(start_dim=0, end_dim=1)
edge_features_val = edge_features_val.flatten(start_dim=0, end_dim=1)
h_val = h_val.flatten(start_dim=0, end_dim=1)
J_val = J_val.flatten(start_dim=0, end_dim=1)
num_models = node_features.size(dim=0)
num_models_val = node_features_val.size(dim=0)

# Define our ML stuff.
node_model = Struct2ParamVAE(num_features=num_node_features, dtype=float_type, device=device)
edge_model = Struct2ParamVAE(num_features=num_edge_features, dtype=float_type, device=device)
loss_fn = RAndNegELBO()
node_optimizer = torch.optim.Adam( node_model.parameters(), lr=learning_rate )
edge_optimizer = torch.optim.Adam( edge_model.parameters(), lr=learning_rate )
num_batches = num_models//batch_size

# Make some Tensors into which we can record our results.
# node_recon_loss_history = torch.zeros( (num_epochs, num_models, num_nodes), dtype=float_type, device=device )
# node_kl_loss_history = torch.zeros( (num_epochs, num_models, num_nodes), dtype=float_type, device=device )
# h_loss_history = torch.zeros( (num_epochs, num_models, num_nodes), dtype=float_type, device=device )
# node_recon_loss_history_val = torch.zeros( (num_epochs, num_models_val, num_nodes), dtype=float_type, device=device )
# node_kl_loss_history_val = torch.zeros( (num_epochs, num_models_val, num_nodes), dtype=float_type, device=device )
# h_loss_history_val = torch.zeros( (num_epochs, num_models_val, num_nodes), dtype=float_type, device=device )
# edge_recon_loss_history = torch.zeros( (num_epochs, num_models, num_nodes, num_nodes), dtype=float_type, device=device )
# edge_kl_loss_history = torch.zeros( (num_epochs, num_models, num_nodes, num_nodes), dtype=float_type, device=device )
# fc_rmse_history = torch.zeros( (num_epochs, num_models), dtype=float_type, device=device )
# fc_corr_history = torch.zeros( (num_epochs, num_models), dtype=float_type, device=device )
# J_loss_history = torch.zeros( (num_epochs, num_models, num_nodes, num_nodes), dtype=float_type, device=device )
# edge_recon_loss_history_val = torch.zeros( (num_epochs, num_models_val, num_nodes, num_nodes), dtype=float_type, device=device )
# edge_kl_loss_history_val = torch.zeros( (num_epochs, num_models_val, num_nodes, num_nodes), dtype=float_type, device=device )
# J_loss_history_val = torch.zeros( (num_epochs, num_models_val, num_nodes, num_nodes), dtype=float_type, device=device )
# fc_rmse_history_val = torch.zeros( (num_epochs, num_models_val), dtype=float_type, device=device )
# fc_corr_history_val = torch.zeros( (num_epochs, num_models_val), dtype=float_type, device=device )

print('starting training...')
last_time = time.time()
for epoch in range(num_epochs):
    sample_order = torch.randperm(num_models, dtype=torch.int, device=device)
    node_features_shuffled = node_features[sample_order,:,:]
    h_shuffled = h[sample_order,:]
    edge_features_shuffled = edge_features[sample_order,:,:,:]
    J_shuffled = J[sample_order,:,:]
    for subepoch in range(num_subepochs):
        for batch in range(num_batches):
            batch_start = batch_size*batch
            batch_end = batch_start + batch_size
            # Do the training step for the node model.
            node_optimizer.zero_grad()
            batch_node_features = node_features_shuffled[batch_start:batch_end,:,:]
            batch_h = h_shuffled[batch_start:batch_end,:]
            h_pred = node_model(batch_node_features)
            node_loss = loss_fn(h_pred, batch_h)
            node_loss.backward()
            node_optimizer.step()
            # Do the training step for the node model.
            edge_optimizer.zero_grad()
            batch_edge_features = edge_features_shuffled[batch_start:batch_end,:,:,:]
            batch_J = J_shuffled[batch_start:batch_end,:,:]
            J_pred = edge_model(batch_edge_features)
            edge_loss = loss_fn(J_pred, batch_J)
            edge_loss.backward()
            edge_optimizer.step()
    with torch.no_grad():
        # Get the individual kinds of losses for individual data points.
        num_training_subjects_to_use = batch_size# edge_features_val.size(dim=0)
        num_validation_subjects_to_use = batch_size
        h_loss = get_loss_components(node_model, node_features[:num_training_subjects_to_use], h[:num_training_subjects_to_use])
        h_loss_val = get_loss_components(node_model, node_features_val[:num_validation_subjects_to_use], h_val[:num_validation_subjects_to_use])
        J_loss = get_loss_components(edge_model, edge_features[:num_training_subjects_to_use], J[:num_training_subjects_to_use])
        J_loss_val = get_loss_components(edge_model, edge_features_val[:num_validation_subjects_to_use], J_val[:num_validation_subjects_to_use])
        fc_rmse, fc_corr = predict_and_compare(node_model, edge_model, node_features[:num_training_subjects_to_use], edge_features[:num_training_subjects_to_use], h[:num_training_subjects_to_use], J[:num_training_subjects_to_use], hcp.num_time_points)
        fc_rmse_val, fc_corr_val = predict_and_compare(node_model, edge_model, node_features_val[:num_validation_subjects_to_use], edge_features_val[:num_validation_subjects_to_use], h_val[:num_validation_subjects_to_use], J_val[:num_validation_subjects_to_use], hcp.num_time_points)
        # Store them in the history Tensors.
        # node_recon_loss_history[epoch,:,:] = node_recon_loss
        # node_kl_loss_history[epoch,:,:] = node_kl_loss
        # h_loss_history[epoch,:,:] = h_loss
        # node_recon_loss_history_val[epoch,:,:] = node_recon_loss_val
        # node_kl_loss_history_val[epoch,:,:] = node_kl_loss_val
        # h_loss_history_val[epoch,:,:] = h_loss_val
        # edge_recon_loss_history[epoch,:,:,:] = edge_recon_loss
        # edge_kl_loss_history[epoch,:,:,:] = edge_kl_loss
        # J_loss_history[epoch,:,:,:] = J_loss
        # edge_recon_loss_history_val[epoch,:,:,:] = edge_recon_loss_val
        # edge_kl_loss_history_val[epoch,:,:,:] = edge_kl_loss_val
        # J_loss_history_val[epoch,:,:,:] = J_loss_val
        # fc_rmse_history[epoch,:] = fc_rmse
        # fc_corr_history[epoch,:] = fc_corr
        # fc_rmse_history_val[epoch,:] = fc_rmse_val
        # fc_corr_history_val[epoch,:] = fc_corr_val
        # Print out the mean values.
        mean_h_loss = h_loss.mean()
        mean_h_loss_val = h_loss_val.mean()
        mean_J_loss = J_loss.mean()
        mean_J_loss_val = J_loss_val.mean()
        mean_fc_rmse = fc_rmse.mean()
        mean_fc_rmse_val = fc_rmse_val.mean()
        mean_fc_corr = fc_corr.mean()
        mean_fc_corr_val = fc_corr_val.mean()
        current_time = time.time()
        print(f'epoch {epoch},\th t {mean_h_loss:.3g},\th v {mean_h_loss_val:.3g},\tJ t {mean_J_loss:.3g},\tJ v {mean_J_loss_val:.3g},\tFC RMSE t {mean_fc_rmse:.3g},\tFC RMSE v {mean_fc_rmse_val:.3g},\tFC corr t {mean_fc_corr:.3g},\tFC corr v {mean_fc_corr_val:.3g},\ttime {current_time-last_time:.3g}')
last_time = current_time


param_string = f'epochs_{num_epochs}_batch_{batch_size}_lr_{learning_rate:.3g}_recon_{recon_weight:.3g}_kl_{kl_weight:.3g}_J_{J_weight:.3g}_ising_{ising_model_string}'
path_prefix = os.path.join(vae_dir, f'ising_model_vae_nosubep_{param_string}')

print('saving...')
torch.save(node_model, f'{path_prefix}_node_model.pt')
torch.save(edge_model, f'{path_prefix}_edge_model.pt')
# torch.save(node_recon_loss_history, f'{path_prefix}_node_recon_loss_history_train.pt')
# torch.save(node_kl_loss_history, f'{path_prefix}_node_kl_loss_history_train.pt')
# torch.save(h_loss_history, f'{path_prefix}_h_loss_history_train.pt')
# torch.save(edge_recon_loss_history, f'{path_prefix}_edge_recon_loss_history_train.pt')
# torch.save(edge_kl_loss_history, f'{path_prefix}_edge_kl_loss_history_train.pt')
# torch.save(J_loss_history, f'{path_prefix}_J_loss_history_train.pt')
# torch.save(fc_rmse_history, f'{path_prefix}_fc_rmse_loss_history_train.pt')
# torch.save(fc_corr_history, f'{path_prefix}_fc_corr_loss_history_train.pt')
# torch.save(node_recon_loss_history_val, f'{path_prefix}_node_recon_loss_history_validate.pt')
# torch.save(node_kl_loss_history_val, f'{path_prefix}_node_kl_loss_history_validate.pt')
# torch.save(h_loss_history_val, f'{path_prefix}_h_loss_history_validate.pt')
# torch.save(edge_recon_loss_history_val, f'{path_prefix}_edge_recon_loss_history_validate.pt')
# torch.save(edge_kl_loss_history_val, f'{path_prefix}_edge_kl_loss_history_validate.pt')
# torch.save(J_loss_history_val, f'{path_prefix}_J_loss_history_validate.pt')
# torch.save(fc_rmse_history_val, f'{path_prefix}_fc_rmse_loss_history_validate.pt')
# torch.save(fc_corr_history_val, f'{path_prefix}_fc_corr_loss_history_validate.pt')
current_time = time.time()
print('done,', current_time-last_time, 'seconds to save,', current_time-start_time, 'seconds total')