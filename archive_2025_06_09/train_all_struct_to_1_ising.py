import os
import torch
import time
import argparse
import copy
import hcpdatautils as hcp
from collections import OrderedDict

start_time = time.time()
last_time = start_time

parser = argparse.ArgumentParser(description="Predict Ising model parameters from structural MRI and DT-MRI structural connectivity data.")

# directories
parser.add_argument("-d", "--structural_data_dir", type=str, default='E:\\HCP_data', help="directory containing the structural MRI features data file")
parser.add_argument("-m", "--ising_model_dir", type=str, default='E:\\Ising_model_results_daai', help="directory containing the fitted Ising model J parameter file")
parser.add_argument("-s", "--stats_dir", type=str, default="E:\\Ising_model_results_batch", help="directory to which to write the output files from training")

# hyperparameters of the Ising model, used for looking up which h files to load
parser.add_argument("-f", "--fim_param_string", type=str, default="nodes_21_window_50_lr_0.000_threshold_0.100_beta_0.500_reps_1000_epoch_4", help="portion of offsets file name after offsets_indi_[data_subset]_ising_")
parser.add_argument("-n", "--num_nodes", type=int, default=21, help="number of nodes in Ising model")

# hyperparameters of the model training
parser.add_argument("-l", "--num_epochs", type=int, default=1000, help="number of epochs for which to train")
parser.add_argument("-t", "--training_batch_size", type=int, default=1000, help="number of training example (structural feature vector, Ising model offset) pairs we use in each optimizer step")
parser.add_argument("-v", "--validation_batch_size", type=int, default=1000, help="size of subjects we process at one time when computing individual training and validation error values at the end of each epoch, relevant for memory footprint vs speed purposes, not the gradient descent process itself")
parser.add_argument("-p", "--learning_rate", type=float, default=0.0001, help="learning rate to use for Adam optimizer")
parser.add_argument("-r", "--num_hidden_layers", type=int, default=2, help="number of hidden layers in the node model")
parser.add_argument("-w", "--hidden_layer_width", type=int, default=21, help="width of the hidden layers in the node model")
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
fim_param_string = args.fim_param_string
print(f'fim_param_string {fim_param_string}')
num_nodes = args.num_nodes
print(f'num_nodes {num_nodes}')
num_epochs = args.num_epochs
print(f'num_epochs {num_epochs}')
training_batch_size = args.training_batch_size
print(f'training_batch_size {training_batch_size}')
validation_batch_size = args.validation_batch_size
print(f'validation_batch_size {validation_batch_size}')
learning_rate = args.learning_rate
print(f'learning_rate {learning_rate}')
num_hidden_layers = args.num_hidden_layers
print(f'num_hidden_layers {num_hidden_layers}')
hidden_layer_width = args.hidden_layer_width
print(f'hidden_layer_width {hidden_layer_width}')
z_score = args.z_score
print(f'z_score {z_score}')

float_type = torch.float
device = torch.device('cuda')

# creates a num_rows*num_cols 1-D Tensor of booleans where each value is True if and only if it is part of the upper triangle of a flattened num_rows x num_cols matrix.
# If we want the upper triangular part of a Tensor with one or more batch dimensions, we can flatten the last two dimensions together, and then use this.
def get_triu_logical_index(num_rows:int, num_cols:int):
    return ( torch.arange(start=0, end=num_rows, dtype=torch.int, device=device)[:,None] < torch.arange(start=0, end=num_cols, dtype=torch.int, device=device)[None,:] ).flatten()

def prepare_structural_data(subset:str, z_score:bool=True, struct_std:torch.Tensor=None, struct_mean:torch.Tensor=None, num_nodes:int=num_nodes):
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
    def forward(self, offset_pred:torch.Tensor, offset_actual:torch.Tensor):
        return self.pred_loss(offset_pred, offset_actual)

code_start_time = time.time()

struct_to_offset_param_string = f'struct_to_offset_{fim_param_string}_depth_{num_hidden_layers}_width_{hidden_layer_width}_batch_{training_batch_size}_lr_{learning_rate}'

training_offsets_file = os.path.join(ising_model_dir, f'offsets_indi_training_ising_{fim_param_string}.pt')
training_offsets = torch.load(training_offsets_file)
print( 'training Ising model offsets size', training_offsets.size() )
num_training_subjects, num_training_reps, num_offsets = training_offsets.size()

validation_offsets_file = os.path.join(ising_model_dir, f'offsets_indi_validation_ising_{fim_param_string}.pt')
validation_offsets = torch.load(validation_offsets_file)
print( 'validation Ising model offsets size', validation_offsets.size() )
num_validation_subjects, num_validation_reps, _ = validation_offsets.size()

training_features, training_std, training_mean = prepare_structural_data(subset='training', z_score=z_score, num_nodes=num_nodes)
print( 'training structural feature data size ', training_features.size() )
num_features = training_features.size(dim=-1)

validation_features, _, _ = prepare_structural_data(subset='validation', z_score=z_score, struct_std=training_std, struct_mean=training_mean, num_nodes=num_nodes)
print( 'validation structural feature data size ', validation_features.size() )

print(f'loaded data from files, time {time.time() - code_start_time:.3f}')

# Flatten the subjects x reps dimensions together to make the training data easier to shuffle.
training_features_flat = training_features[:,None,:].repeat( (1,num_training_reps,1) ).flatten(start_dim=0, end_dim=1)
training_offsets_flat = training_offsets.flatten(start_dim=0, end_dim=1)
num_training_samples = num_training_subjects * num_training_reps
# validation_features = validation_features.flatten(start_dim=0, end_dim=1)
# validation_offsets = validation_offsets.flatten(start_dim=0, end_dim=1)

partial_training_flat_batch_size = num_training_samples % training_batch_size
has_partial_training_flat_batch = partial_training_flat_batch_size > 0
num_training_flat_batches = num_training_samples//training_batch_size + int(has_partial_training_flat_batch)

# Use the validation batch size, since checking the individual errors for the unflattened training data is part of the validation process.
partial_training_batch_size = num_training_subjects % validation_batch_size
has_partial_training_batch = partial_training_batch_size > 0
num_training_batches = num_training_subjects//training_batch_size + int(has_partial_training_batch)

partial_validation_batch_size = num_validation_subjects % validation_batch_size
has_partial_validation_batch = partial_validation_batch_size > 0
num_validation_batches = num_validation_subjects//validation_batch_size + int(has_partial_validation_batch)

training_errors = torch.zeros( (num_training_subjects, num_training_reps, num_offsets), dtype=float_type, device=device )
validation_errors = torch.zeros( (num_validation_subjects, num_validation_reps, num_offsets), dtype=float_type, device=device )
best_training_errors = torch.zeros_like(training_errors)
best_validation_errors = torch.zeros_like(validation_errors)
median_abs_training_error = torch.zeros( (num_epochs+1, num_offsets), dtype=float_type, device=device )
median_abs_validation_error = torch.zeros( (num_epochs+1, num_offsets), dtype=float_type, device=device )
best_epoch = torch.zeros( (num_offsets,), dtype=torch.int, device=device )
for offset_index in range(num_offsets):
    best_model = None
    best_median_abs_val_error = torch.finfo().max
    offset_model = Struct2Param(num_features=num_features, hidden_layer_width=hidden_layer_width, num_hidden_layer=num_hidden_layers, dtype=float_type, device=device)
    loss_fn = PredictionLoss()
    offset_optimizer = torch.optim.Adam( offset_model.parameters(), lr=learning_rate )
    training_offset_flat = training_offsets_flat[:,offset_index]
    training_offset = training_offsets[:,:,offset_index]
    validation_offset = validation_offsets[:,:,offset_index]
    for epoch in range(num_epochs):
        # print(f'epoch {epoch}')
        with torch.no_grad():
            for batch in range(num_training_batches):
                if has_partial_training_batch and (batch == num_training_batches-1):
                    current_batch_size = partial_training_batch_size
                else:
                    current_batch_size = validation_batch_size
                batch_start = batch*training_batch_size
                batch_end = batch_start + current_batch_size
                current_features = training_features[batch_start:batch_end,:]
                current_offset = training_offset[batch_start:batch_end,:]
                predicted_offset = offset_model(current_features)
                training_errors[batch_start:batch_end,:,offset_index] = predicted_offset[:,None] - current_offset
            abs_train_errors = training_errors[:,:,offset_index].abs()
            median_ate = abs_train_errors.median()
            median_abs_training_error[epoch,offset_index] = median_ate
            if epoch % 100 == 0:
                print(f'offset {offset_index}, epoch {epoch}, min abs train error {abs_train_errors.min():.3g}, median abs train error {median_ate:.3g}, max abs train error {abs_train_errors.max():.3g}, time {time.time() - code_start_time:.3f}')
            for batch in range(num_validation_batches):
                if has_partial_validation_batch and (batch == num_validation_batches-1):
                    current_batch_size = partial_validation_batch_size
                else:
                    current_batch_size = validation_batch_size
                batch_start = batch*validation_batch_size
                batch_end = batch_start + current_batch_size
                current_features = validation_features[batch_start:batch_end,:]
                current_offset = validation_offset[batch_start:batch_end,:]
                predicted_offset = offset_model(current_features)
                validation_errors[batch_start:batch_end,:,offset_index] = predicted_offset[:,None] - current_offset
            abs_val_errors = validation_errors[:,:,offset_index].abs()
            median_ave = abs_val_errors.median()
            median_abs_validation_error[epoch,offset_index] = median_ave
            if median_ave < best_median_abs_val_error:
                best_median_abs_val_error = median_ave
                best_model = copy.deepcopy(offset_model)
                best_epoch[offset_index] = epoch
                best_training_errors[:,:,offset_index] = training_errors[:,:,offset_index]
                best_validation_errors[:,:,offset_index] = validation_errors[:,:,offset_index]
            if epoch % 100 == 0:
                print(f'offset {offset_index}, epoch {epoch}, min abs val error {abs_val_errors.min():.3g}, median abs val error {median_ave:.3g}, max abs val error {abs_val_errors.max():.3g}, time {time.time() - code_start_time:.3f}')
        # Note that we are now outside the torch.no_grad() block, since we are going to train the model.
        # We randomize the order of the training samples but not of the validation samples.
        rand_indices = torch.randperm(n=num_training_samples, dtype=torch.int, device=device)
        for batch in range(num_training_flat_batches):
            # print(f'batch {batch}')
            if has_partial_training_flat_batch and (batch == num_training_flat_batches-1):
                current_batch_size = partial_training_flat_batch_size
            else:
                current_batch_size = training_batch_size
            batch_start = batch*training_batch_size
            batch_end = batch_start + current_batch_size
            batch_indices = rand_indices[batch_start:batch_end]
            current_features = training_features_flat[batch_indices,:]
            current_offset = training_offset_flat[batch_indices]
            offset_optimizer.zero_grad()
            predicted_offset = offset_model(current_features)
            offset_loss = loss_fn(current_offset, predicted_offset)
            offset_loss.backward()
            offset_optimizer.step()
    # Do one last round of testing at the end.
    epoch = num_epochs
    with torch.no_grad():
        for batch in range(num_training_batches):
            if has_partial_training_batch and (batch == num_training_batches-1):
                current_batch_size = partial_training_batch_size
            else:
                current_batch_size = validation_batch_size
            batch_start = batch*training_batch_size
            batch_end = batch_start + current_batch_size
            current_features = training_features[batch_start:batch_end,:]
            current_offset = training_offset[batch_start:batch_end,:]
            predicted_offset = offset_model(current_features)
            training_errors[batch_start:batch_end,:,offset_index] = predicted_offset[:,None] - current_offset
        abs_train_errors = training_errors[:,:,offset_index].abs()
        median_ate = abs_train_errors.median()
        median_abs_training_error[epoch,offset_index] = median_ate
        print(f'offset {offset_index}, epoch {epoch}, min abs train error {abs_train_errors.min():.3g}, median abs train error {median_ate:.3g}, max abs train error {abs_train_errors.max():.3g}, time {time.time() - code_start_time:.3f}')
        for batch in range(num_validation_batches):
            if has_partial_validation_batch and (batch == num_validation_batches-1):
                current_batch_size = partial_validation_batch_size
            else:
                current_batch_size = validation_batch_size
            batch_start = batch*validation_batch_size
            batch_end = batch_start + current_batch_size
            current_features = validation_features[batch_start:batch_end,:]
            current_offset = validation_offset[batch_start:batch_end,:]
            predicted_offset = offset_model(current_features)
            validation_errors[batch_start:batch_end,:,offset_index] = predicted_offset[:,None] - current_offset
        abs_val_errors = validation_errors[:,:,offset_index].abs()
        median_ave = abs_val_errors.median()
        median_abs_validation_error[epoch,offset_index] = median_ave
        if median_ave < best_median_abs_val_error:
            best_median_abs_val_error = median_ave
            best_model = copy.deepcopy(offset_model)
            best_epoch[offset_index] = epoch
            best_training_errors[:,:,offset_index] = training_errors[:,:,offset_index]
            best_validation_errors[:,:,offset_index] = validation_errors[:,:,offset_index]
        print(f'offset {offset_index}, epoch {epoch}, min abs val error {abs_val_errors.min():.3g}, median abs val error {median_ave:.3g}, max abs val error {abs_val_errors.max():.3g}, time {time.time() - code_start_time:.3f}')
    offset_model_file = os.path.join(ising_model_dir, f'{struct_to_offset_param_string}_dim_{offset_index}.pt')
    torch.save(obj=best_model, f=offset_model_file)
    print(f'saved {offset_model_file}, time {time.time() - code_start_time:.3f}')

best_epoch_file = os.path.join(stats_dir, f'best_epoch_{struct_to_offset_param_string}.pt')
torch.save(obj=best_epoch, f=best_epoch_file)
print(f'saved {best_epoch_file}, time {time.time() - code_start_time:.3f}')

training_error_file = os.path.join(stats_dir, f'training_error_{struct_to_offset_param_string}.pt')
torch.save(obj=best_training_errors, f=training_error_file)
print(f'saved {training_error_file}, time {time.time() - code_start_time:.3f}')

validation_error_file = os.path.join(stats_dir, f'validation_error_{struct_to_offset_param_string}.pt')
torch.save(obj=best_validation_errors, f=validation_error_file)
print(f'saved {validation_error_file}, time {time.time() - code_start_time:.3f}')

median_abs_training_error_history_file = os.path.join(stats_dir, f'median_abs_training_error_history_{struct_to_offset_param_string}.pt')
torch.save(obj=median_abs_training_error, f=median_abs_training_error_history_file)
print(f'saved {median_abs_training_error_history_file}, time {time.time() - code_start_time:.3f}')

median_abs_validation_error_history_file = os.path.join(stats_dir, f'median_abs_validation_error_history_{struct_to_offset_param_string}.pt')
torch.save(obj=median_abs_validation_error, f=median_abs_validation_error_history_file)
print(f'saved {median_abs_validation_error_history_file}, time {time.time() - code_start_time:.3f}')

print('done')