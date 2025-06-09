import os
import torch
import time
import argparse
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from graph2graphcnn import MultiLayerPerceptron
from graph2graphcnn import UniformMultiLayerPerceptron

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')
num_nodes = 360
num_node_features = 4 + 3 + 1# thickness, myelination, curvature, sulcus depth + x, y, z + SC sum
num_edge_features = 1 + 2*num_node_features# SC + node features of first endpoint + node features of second endpoint
num_h_features = 1
num_J_features = 1

parser = argparse.ArgumentParser(description="Compare performance of the structure-to-Ising model MLP pairs vs an ensemble of the same models trained on shuffled data.")
parser.add_argument("-a", "--input_directory", type=str, default='E:\\g2gcnn_examples', help="directory from which we read the training examples")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the trained models and pandas DataFrame of loss results")
parser.add_argument("-c", "--file_name_fragment", type=str, default='rectangular_max_rmse_2', help="part of the input example files between example_[index]_ and _[validation|training]_example.pt, will also be part of the output file names")
parser.add_argument("-d", "--output_file_name_fragment", type=str, default='small', help="another string we can incorporate into the output file names to help differentiate separate runs")
parser.add_argument("-s", "--optimizer_name", type=str, default='Adam', help="either Adam or SGD")
parser.add_argument("-e", "--num_epochs", type=int, default=1000, help="number of epochs for which to train each model")
parser.add_argument("-f", "--patience", type=int, default=1000, help="Number of epochs with no noticeable improvement in either loss before we stop and move on to the next model.")
parser.add_argument("-g", "--min_improvement", type=float, default=10e-10, help="Minimal amount of improvement to count as noticeable.")
parser.add_argument("-i", "--save_models", action='store_true', default=False, help="Set this flag in order to have the script save each trained G2GCNN model.")
parser.add_argument("-j", "--num_training_examples", type=int, default=3345, help="number of training examples")
parser.add_argument("-k", "--num_validation_examples", type=int, default=420, help="number of validation examples")
parser.add_argument("-l", "--num_instances", type=int, default=100, help="number of shuffled and real pairs of models to train and validate")
parser.add_argument("-o", "--mlp_hidden_layers", type=int, default=10, help="number of hidden layers to use in each multi-layer perceptron")
parser.add_argument("-p", "--rep_dims", type=int, default=20, help="number of nodes in each MLP hidden layer")
parser.add_argument("-q", "--batch_size", type=int, default=50, help="batch size")
parser.add_argument("-r", "--learning_rate", type=float, default=0.01, help="learning rate")
parser.add_argument("-t", "--include_coords", action='store_true', default=True, help="Set this flag in order to include the x, y, and z coordinates of each brain region.")
args = parser.parse_args()
print('getting arguments...')
input_directory = args.input_directory
print(f'input_directory={input_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
file_name_fragment = args.file_name_fragment
print(f'file_name_fragment={file_name_fragment}')
output_file_name_fragment = args.output_file_name_fragment
print(f'output_file_name_fragment={output_file_name_fragment}')
num_epochs = args.num_epochs
print(f'num_epochs={num_epochs}')
patience = args.patience
print(f'patience={patience}')
min_improvement = args.min_improvement
print(f'min_improvement={min_improvement:.3g}')
save_models = args.save_models
print(f'save_models={save_models}')
num_training_examples = args.num_training_examples
print(f'num_training_examples={num_training_examples}')
num_validation_examples = args.num_validation_examples
print(f'num_validation_examples={num_validation_examples}')
num_instances = args.num_instances
print(f'num_instances={num_instances}')
mlp_hidden_layers = args.mlp_hidden_layers
print(f'mlp_hidden_layers={mlp_hidden_layers}')
rep_dims = args.rep_dims
print(f'rep_dims={rep_dims}')
optimizer_name = args.optimizer_name
print(f'optimizer_name={optimizer_name}')
batch_size = args.batch_size
print(f'batch_size={batch_size}')
learning_rate = args.learning_rate
print(f'learning_rate={learning_rate}')
include_coords = args.include_coords
print(f'include_coords={include_coords}')
if include_coords:
    coords_string = 'coords_yes'
else:
    coords_string = 'coords_no'

class StructParamsFileDataset(Dataset):
    def __init__(self, file_directory:str, file_suffix:str, num_nodes:int, num_node_features:int, num_edge_features:int, num_h_features:int, num_J_features:int, dtype, device, num_examples:int):
        super(StructParamsFileDataset,self).__init__()
        self.file_directory = file_directory
        self.file_suffix = file_suffix
        self.num_examples = num_examples
        self.num_nodes = num_nodes
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.num_h_features = num_h_features
        self.num_J_features = num_J_features
        self.dtype = dtype
        self.device = device
    def __len__(self):
        return self.num_examples
    def __getitem__(self, idx:int):
        return torch.load( f=os.path.join(self.file_directory, f'example_{idx}_{self.file_suffix}.pt') )

class ShuffledStructParamsFileDataset(Dataset):
    def __init__(self, file_directory:str, file_suffix:str, num_nodes:int, num_node_features:int, num_edge_features:int, num_h_features:int, num_J_features:int, dtype, device, num_examples:int):
        super(ShuffledStructParamsFileDataset,self).__init__()
        self.file_directory = file_directory
        self.file_suffix = file_suffix
        self.num_examples = num_examples
        self.num_nodes = num_nodes
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.num_h_features = num_h_features
        self.num_J_features = num_J_features
        self.dtype = dtype
        self.device = device
        self.shuffle_indices = torch.randperm(n=self.num_examples, dtype=int_type, device=self.device)
    def reshuffle(self):
        self.shuffle_indices = torch.randperm(n=self.num_examples, dtype=int_type, device=self.device)
    def __len__(self):
        return self.num_examples
    def __getitem__(self, idx:int):
        # Load the structural features and the Ising model parameters from different files so that they are mismatched.
        shuffled_index = self.shuffle_indices[idx]
        node_features, edge_features, _, _ = torch.load( f=os.path.join(self.file_directory, f'example_{idx}_{self.file_suffix}.pt') )
        _, _, h, J = torch.load( f=os.path.join(self.file_directory, f'example_{shuffled_index}_{self.file_suffix}.pt') )
        return node_features, edge_features, h, J

# Assumes
# node_features has size batch_size x num_nodes x num_node_features
# edge_features has size batch_size x num_nodes x num_nodes x num_edge_features
def recombine_node_and_edge_features(node_features:torch.Tensor, edge_features:torch.Tensor):
    num_nodes = node_features.size(dim=-2)
    node_features = torch.cat(  ( node_features, torch.sum(edge_features, dim=-2) ), dim=-1  )
    edge_features = torch.cat(   (  node_features[:,:,None,:].repeat( (1,1,num_nodes,1) ), edge_features, node_features[:,None,:,:].repeat( (1,num_nodes,1,1) )  ), dim=-1   )
    return node_features, edge_features

def get_loss(data_loader:DataLoader, loss_fn:torch.nn.Module, node_mlp:MultiLayerPerceptron, edge_mlp:MultiLayerPerceptron):
    h_loss_sum = 0
    h_loss_count = 0
    J_loss_sum = 0
    J_loss_count = 0
    for node_features_batch, edge_features_batch, h_batch, J_batch in data_loader:
        node_features_batch, edge_features_batch = recombine_node_and_edge_features(node_features=node_features_batch, edge_features=edge_features_batch)
        batch_size = node_features_batch.size(dim=0)
        h_pred = node_mlp(node_features_batch)
        h_loss_sum += batch_size*loss_fn(h_pred, h_batch).item()
        h_loss_count += batch_size
        J_pred = edge_mlp(edge_features_batch)
        J_loss_sum += loss_fn(J_pred, J_batch).item()
        J_loss_count += batch_size
    return h_loss_sum/h_loss_count, J_loss_sum/J_loss_count

def train_model(training_data_set:StructParamsFileDataset, validation_data_set:StructParamsFileDataset, node_mlp_file:str, edge_mlp_file:str, num_epochs:int=1000, optimizer_name:str='Adam', rep_dims:int=7, mlp_hidden_layers:int=3, batch_size:int=10, learning_rate:torch.float=0.001, save_model:bool=False, patience:int=10, improvement_threshold:torch.float=0.0001):
    dtype = training_data_set.dtype
    device = training_data_set.device
    num_node_features = training_data_set.num_node_features
    num_edge_features = training_data_set.num_edge_features
    num_h_features = training_data_set.num_h_features
    num_J_features = training_data_set.num_J_features
    training_data_loader = DataLoader(dataset=training_data_set, shuffle=True, batch_size=batch_size)
    validation_data_loader = DataLoader(dataset=validation_data_set, shuffle=False, batch_size=batch_size)
    node_mlp = UniformMultiLayerPerceptron(num_in_features=num_node_features, num_out_features=num_h_features, hidden_layer_width=rep_dims, num_hidden_layers=mlp_hidden_layers, dtype=dtype, device=device)
    edge_mlp = UniformMultiLayerPerceptron(num_in_features=num_edge_features, num_out_features=num_J_features, hidden_layer_width=rep_dims, num_hidden_layers=mlp_hidden_layers, dtype=dtype, device=device)
    loss_fn = torch.nn.MSELoss()
    if optimizer_name == 'SGD':
        node_optimizer = torch.optim.SGD( params=node_mlp.parameters(), lr=learning_rate )
        edge_optimizer = torch.optim.SGD( params=edge_mlp.parameters(), lr=learning_rate )
    else:
        node_optimizer = torch.optim.Adam( params=node_mlp.parameters(), lr=learning_rate )
        edge_optimizer = torch.optim.Adam( params=edge_mlp.parameters(), lr=learning_rate )
    num_no_improvement_epochs = 0
    last_training_h_loss, last_training_J_loss = get_loss(data_loader=training_data_loader, loss_fn=loss_fn, node_mlp=node_mlp, edge_mlp=edge_mlp)
    last_validation_h_loss, last_validation_J_loss = get_loss(data_loader=validation_data_loader, loss_fn=loss_fn, node_mlp=node_mlp, edge_mlp=edge_mlp)
    print(f'time {time.time() - code_start_time:.3f}, starting training MLPs with {num_node_features} input node features, {num_edge_features} input edge features, {rep_dims} latent representaion dimensions, {mlp_hidden_layers} hidden layers per MLP, batch size {batch_size}, learning rate {learning_rate:.3g}, optimizer {optimizer_name}.')
    for epoch in range(num_epochs):
        for node_features_batch, edge_features_batch, h_batch, J_batch in training_data_loader:
            node_features_batch, edge_features_batch = recombine_node_and_edge_features(node_features=node_features_batch, edge_features=edge_features_batch)
            # Do one step for the node model.
            node_optimizer.zero_grad()
            h_pred = node_mlp(node_features_batch)
            node_loss = loss_fn(h_pred, h_batch)
            node_loss.backward()
            node_optimizer.step()
            # Do one step for the edge model.
            edge_optimizer.zero_grad()
            J_pred = edge_mlp(edge_features_batch)
            edge_loss = loss_fn(J_pred, J_batch)
            edge_loss.backward()
            edge_optimizer.step()
        with torch.no_grad():
            training_h_loss, training_J_loss = get_loss(data_loader=training_data_loader, loss_fn=loss_fn, node_mlp=node_mlp, edge_mlp=edge_mlp)
            validation_h_loss, validation_J_loss = get_loss(data_loader=validation_data_loader, loss_fn=loss_fn, node_mlp=node_mlp, edge_mlp=edge_mlp)
            training_h_loss_diff = last_training_h_loss - training_h_loss
            training_J_loss_diff = last_training_J_loss - training_J_loss
            validation_h_loss_diff = last_validation_h_loss - validation_h_loss
            validation_J_loss_diff = last_validation_J_loss - validation_J_loss
            num_no_improvement_epochs += int( (training_h_loss_diff < improvement_threshold) and (training_J_loss_diff < improvement_threshold) and (validation_h_loss_diff < improvement_threshold) and (validation_J_loss_diff < improvement_threshold) )
            last_training_h_loss = training_h_loss
            last_training_J_loss = training_J_loss
            last_validation_h_loss = validation_h_loss
            last_validation_J_loss = validation_J_loss
            print(f'time {time.time() - code_start_time:.3f}, epoch {epoch+1}, h training loss {training_h_loss:.3g}, change {training_h_loss_diff:.3g}, validation loss {validation_h_loss:.3g}, change {validation_h_loss_diff:.3g}, J training loss {training_J_loss:.3g}, change {training_J_loss_diff:.3g}, validation loss {validation_J_loss:.3g}, change {validation_J_loss_diff:.3g}, num no-improvement epochs {num_no_improvement_epochs}')
            if num_no_improvement_epochs >= patience:
                print(f'patience exceeded, moving on...')
                break
            if math.isnan(training_h_loss) and math.isnan(training_J_loss) and math.isnan(validation_h_loss) and math.isnan(validation_J_loss):
                print('encountered all-NaNs, moving on...')
                break
    if save_model:
        torch.save(obj=node_mlp, f=node_mlp_file)
        torch.save(obj=edge_mlp, f=edge_mlp_file)
        print(f'time {time.time() - code_start_time:.3f}, saved {node_mlp_file} and {edge_mlp_file}')
    return training_h_loss, training_J_loss, validation_h_loss, validation_J_loss

# Load the training and validation data.
training_data_set = StructParamsFileDataset(file_directory=input_directory, file_suffix=f'{file_name_fragment}_training_example', num_nodes=num_nodes, num_node_features=num_node_features, num_edge_features=num_edge_features, num_h_features=num_h_features, num_J_features=num_J_features, dtype=float_type, device=device, num_examples=num_training_examples)
shuffled_training_data_set = ShuffledStructParamsFileDataset(file_directory=input_directory, file_suffix=f'{file_name_fragment}_training_example', num_nodes=num_nodes, num_node_features=num_node_features, num_edge_features=num_edge_features, num_h_features=num_h_features, num_J_features=num_J_features, dtype=float_type, device=device, num_examples=num_training_examples)
validation_data_set = StructParamsFileDataset(file_directory=input_directory, file_suffix=f'{file_name_fragment}_validation_example', num_nodes=num_nodes, num_node_features=num_node_features, num_edge_features=num_edge_features, num_h_features=num_h_features, num_J_features=num_J_features, dtype=float_type, device=device, num_examples=num_validation_examples)
# Check whether we have already created the results pickle file in a previous run.
more_complete_output_file_name_fragment = f'{output_file_name_fragment}_{file_name_fragment}_epochs_{num_epochs}_patience_{patience}_min_improvement_{min_improvement:.3g}_opt_{optimizer_name}_batch_{batch_size}_lr_{learning_rate:.3g}_rep_dims_{rep_dims}_mlp_depth_{mlp_hidden_layers}'
real_h_training_rmse = torch.zeros( size=(num_instances,), dtype=float_type, device=device )
real_J_training_rmse = torch.zeros( size=(num_instances,), dtype=float_type, device=device )
real_h_validation_rmse = torch.zeros( size=(num_instances,), dtype=float_type, device=device )
real_J_validation_rmse = torch.zeros( size=(num_instances,), dtype=float_type, device=device )
shuffled_h_training_rmse = torch.zeros( size=(num_instances,), dtype=float_type, device=device )
shuffled_J_training_rmse = torch.zeros( size=(num_instances,), dtype=float_type, device=device )
shuffled_h_validation_rmse = torch.zeros( size=(num_instances,), dtype=float_type, device=device )
shuffled_J_validation_rmse = torch.zeros( size=(num_instances,), dtype=float_type, device=device )
for instance_index in range(num_instances):

    print(f'{time.time()-code_start_time:.3f}, training real model instance {instance_index+1} of {num_instances}...')
    file_name_fragment_with_hyperparams = f'{more_complete_output_file_name_fragment}_opt_{optimizer_name}_mlp_{mlp_hidden_layers}_rep_{rep_dims}_batch_sz_{batch_size}_lr_{learning_rate:.3g}_run_{instance_index}.pt'
    node_mlp_file = os.path.join(output_directory, f'node_mlp_{file_name_fragment_with_hyperparams}')
    edge_mlp_file = os.path.join(output_directory, f'edge_mlp_{file_name_fragment_with_hyperparams}')
    training_h_loss, training_J_loss, validation_h_loss, validation_J_loss = train_model(training_data_set=training_data_set, validation_data_set=validation_data_set, node_mlp_file=node_mlp_file, edge_mlp_file=edge_mlp_file, num_epochs=num_epochs, optimizer_name=optimizer_name, rep_dims=rep_dims, mlp_hidden_layers=mlp_hidden_layers, batch_size=batch_size, learning_rate=learning_rate, save_model=save_models, patience=patience, improvement_threshold=min_improvement)
    real_h_training_rmse[instance_index] = math.sqrt(training_h_loss)
    real_J_training_rmse[instance_index] = math.sqrt(training_J_loss)
    real_h_validation_rmse[instance_index] = math.sqrt(validation_h_loss)
    real_J_validation_rmse[instance_index] = math.sqrt(validation_J_loss)
    results_file = os.path.join(output_directory, f'mlp_pair_real_{instance_index+1}_vs_shuffled_0_{more_complete_output_file_name_fragment}.df')
    torch.save( obj=(real_h_training_rmse, shuffled_h_training_rmse, real_h_validation_rmse, shuffled_h_validation_rmse, real_J_training_rmse, shuffled_J_training_rmse, real_J_validation_rmse, shuffled_J_validation_rmse), f=results_file )

    print(f'{time.time()-code_start_time:.3f}, training shuffled model instance {instance_index+1} of {num_instances}...')
    file_name_fragment_with_hyperparams = f'{more_complete_output_file_name_fragment}_opt_{optimizer_name}_mlp_{mlp_hidden_layers}_rep_{rep_dims}_batch_sz_{batch_size}_lr_{learning_rate:.3g}_run_{instance_index}.pt'
    node_mlp_file = os.path.join(output_directory, f'shuffled_node_mlp_{file_name_fragment_with_hyperparams}')
    edge_mlp_file = os.path.join(output_directory, f'shuffled_edge_mlp_{file_name_fragment_with_hyperparams}')
    shuffled_training_data_set.reshuffle()
    training_h_loss, training_J_loss, validation_h_loss, validation_J_loss = train_model(training_data_set=shuffled_training_data_set, validation_data_set=validation_data_set, node_mlp_file=node_mlp_file, edge_mlp_file=edge_mlp_file, num_epochs=num_epochs, optimizer_name=optimizer_name, rep_dims=rep_dims, mlp_hidden_layers=mlp_hidden_layers, batch_size=batch_size, learning_rate=learning_rate, save_model=save_models, patience=patience, improvement_threshold=min_improvement)
    shuffled_h_training_rmse[instance_index] = math.sqrt(training_h_loss)
    shuffled_J_training_rmse[instance_index] = math.sqrt(training_J_loss)
    shuffled_h_validation_rmse[instance_index] = math.sqrt(validation_h_loss)
    shuffled_J_validation_rmse[instance_index] = math.sqrt(validation_J_loss)
    results_file = os.path.join(output_directory, f'mlp_pair_real_{num_instances}_vs_shuffled_{instance_index+1}_{more_complete_output_file_name_fragment}.df')
    torch.save( obj=(real_h_training_rmse, shuffled_h_training_rmse, real_h_validation_rmse, shuffled_h_validation_rmse, real_J_training_rmse, shuffled_J_training_rmse, real_J_validation_rmse, shuffled_J_validation_rmse), f=results_file )

print(f'time {time.time() - code_start_time:.3f}, done')