import os
import torch
import time
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from isingmodel import IsingModel# need to import the class for when we load the model from a file
from graph2graphcnn import UniformGraph2GraphCNN
from graph2graphcnn import GraphMSELoss

code_start_time = time.time()

parser = argparse.ArgumentParser(description="Train a GCNN model to predict Ising model parameters from structural features.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-n", "--node_features_file", type=str, default='node_features_group_training_and_individual_all.pt', help="file with node features with dimensions num_subjects x num_nodes x num_node_features")
parser.add_argument("-e", "--edge_features_file", type=str, default='edge_features_group_training_and_individual_all.pt', help="file with edge features with dimensions num_subjects x num_nodes x num_nodes x num_edge_features")
parser.add_argument("-m", "--ising_model_file", type=str, default='ising_model_group_training_and_individual_all_fold_1_parallel_20_steps_12000_beta_updates_25.pt', help="file containing the IsingModel object with h of size num_betas*num_subjects x num_nodes and J of size num_betas*num_subjects x num_nodes x num_nodes")
parser.add_argument("-a", "--training_start", type=int, default=1, help="first index of a training subject")
parser.add_argument("-b", "--training_end", type=int, default=670, help="last index of a training subject + 1")
parser.add_argument("-c", "--validation_start", type=int, default=670, help="first index of a training subject")
parser.add_argument("-d", "--validation_end", type=int, default=754, help="last index of a training subject + 1")
parser.add_argument("-v", "--num_graph_convolution_layers", type=int, default=3, help="number of pairs of a multi-layer perceptron layer followed by a graph convolution layer")
parser.add_argument("-l", "--num_mlp_hidden_layers", type=int, default=3, help="number of hidden layers to use in each multi-layer perceptron")
parser.add_argument("-k", "--rep_dims", type=int, default=7, help="number of dimensions to use in each latent-space representation")
parser.add_argument("-z", "--num_saves", type=int, default=1000000, help="number of points at which to save the model")
parser.add_argument("-p", "--num_epochs_per_save", type=int, default=1000, help="number of epochs between saves of the model")
parser.add_argument("-y", "--optimizer_name", type=str, default='Adam', help="either Adam or SGD")
parser.add_argument("-s", "--batch_size", type=int, default=669, help="number of model instances per training batch")
parser.add_argument("-r", "--learning_rate", type=float, default=0.001, help="learning rate")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
node_features_file = args.node_features_file
print(f'node_features_file={node_features_file}')
edge_features_file = args.edge_features_file
print(f'edge_features_file={edge_features_file}')
ising_model_file = args.ising_model_file
print(f'ising_model_file={ising_model_file}')
training_start = args.training_start
print(f'training_start={training_start}')
training_end = args.training_end
print(f'training_end={training_end}')
validation_start = args.validation_start
print(f'validation_start={validation_start}')
validation_end = args.validation_end
print(f'validation_end={validation_end}')
num_graph_convolution_layers = args.num_graph_convolution_layers
print(f'num_graph_convolution_layers={num_graph_convolution_layers}')
num_mlp_hidden_layers = args.num_mlp_hidden_layers
print(f'num_mlp_hidden_layers={num_mlp_hidden_layers}')
rep_dims = args.rep_dims
print(f'rep_dims={rep_dims}')
num_saves = args.num_saves
print(f'num_saves={num_saves}')
num_epochs_per_save = args.num_epochs_per_save
print(f'num_epochs_per_save={num_epochs_per_save}')
optimizer_name = args.optimizer_name
print(f'optimizer_name={optimizer_name}')
batch_size = args.batch_size
print(f'batch_size={batch_size}')
learning_rate = args.learning_rate
print(f'learning_rate={learning_rate}')

class StructParamsDataset(Dataset):
    def __init__(self, node_features:torch.Tensor, edge_features:torch.Tensor, h:torch.Tensor, J:torch.Tensor):
        super(StructParamsDataset,self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.h = h
        self.J = J
    def __len__(self):
        return self.node_features.size(dim=0)
    def __getitem__(self, idx):
        return self.node_features[idx,:,:], self.edge_features[idx,:,:,:], self.h[idx,:,:], self.J[idx,:,:,:]

def load_data(file_directory:str, ising_model_file:str, node_features_file:str, edge_features_file:str, training_start:int, training_end:int, validation_start:int, validation_end:int, batch_size:int):

    ising_model_file = os.path.join(file_directory, ising_model_file)
    ising_model = torch.load(ising_model_file)
    print(f'time {time.time() - code_start_time:.3f}, loaded {ising_model_file}, beta updates {ising_model.num_beta_updates}, param updates {ising_model.num_param_updates}')
    # Just multiply in beta instead of trying to learn it separately.
    # Unsqueeze so that we have a separate singleton "out-features" dimension.
    h = torch.unsqueeze( input=(ising_model.h * ising_model.beta), dim=-1 )
    J = torch.unsqueeze(  input=( ising_model.J * ising_model.beta.unsqueeze(dim=-1) ), dim=-1  )

    node_features_file = os.path.join(file_directory, node_features_file)
    node_features = torch.load(node_features_file)
    print(f'time {time.time() - code_start_time:.3f}, loaded {node_features_file}')
    print( 'node_features size', node_features.size() )
    node_features = torch.unsqueeze(node_features, dim=0).repeat( (ising_model.num_betas_per_target, 1, 1, 1) ).flatten(start_dim=0, end_dim=1)
    print( 'expanded to', node_features.size() )

    edge_features_file = os.path.join(file_directory, edge_features_file)
    edge_features = torch.load(edge_features_file)
    print(f'time {time.time() - code_start_time:.3f}, loaded {edge_features_file}')
    print( 'edge_features size', edge_features.size() )
    edge_features = torch.unsqueeze(edge_features, dim=0).repeat( (ising_model.num_betas_per_target, 1, 1, 1, 1) ).flatten(start_dim=0, end_dim=1)
    print( 'expanded to', edge_features.size() )

    # Clone these so that we can deallocate the rest of the underlying Tensors once we exit the function scope.
    training_node_features = node_features[training_start:training_end,:,:].clone()
    training_edge_features = edge_features[training_start:training_end,:,:,:].clone()
    training_h = h[training_start:training_end,:,:].clone()
    training_J = J[training_start:training_end,:,:,:].clone()
    validation_node_features = node_features[validation_start:validation_end,:,:].clone()
    validation_edge_features = edge_features[validation_start:validation_end,:,:,:].clone()
    validation_h = h[validation_start:validation_end,:,:].clone()
    validation_J = J[validation_start:validation_end,:,:,:].clone()

    training_data_set = StructParamsDataset(node_features=training_node_features, edge_features=training_edge_features, h=training_h, J=training_J)
    training_data_loader = DataLoader(dataset=training_data_set, shuffle=True, batch_size=batch_size)
    validation_data_set = StructParamsDataset(node_features=validation_node_features, edge_features=validation_edge_features, h=validation_h, J=validation_J)
    validation_data_loader = DataLoader(dataset=validation_data_set, shuffle=False, batch_size=batch_size)

    num_nodes = node_features.size(dim=1)
    num_node_features = node_features.size(dim=-1)
    num_edge_features = edge_features.size(dim=-1)
    dtype = node_features.dtype
    device = node_features.device

    return training_data_loader, validation_data_loader, num_nodes, num_node_features, num_edge_features, dtype, device

def get_loss(data_loader:DataLoader, g2gcnn_model:UniformGraph2GraphCNN):
    training_loss_sum = 0
    training_loss_count = 0
    for node_features_batch, edge_features_batch, h_batch, J_batch in data_loader:
        h_pred, J_pred = g2gcnn_model(node_in_features=node_features_batch, edge_in_features=edge_features_batch)
        loss = loss_fn(node_features_pred=h_pred, edge_features_pred=J_pred, node_features_target=h_batch, edge_features_target=J_batch)
        current_batch_size = node_features_batch.size(dim=0)
        training_loss_sum += ( current_batch_size*loss.item() )
        training_loss_count += current_batch_size
    return training_loss_sum/training_loss_count
    
# Counterintuitively, all our input files are in the output directory.
training_data_loader, validation_data_loader, num_nodes, num_node_features, num_edge_features, dtype, device = load_data(file_directory=output_directory, ising_model_file=ising_model_file, node_features_file=node_features_file, edge_features_file=edge_features_file, training_start=training_start, training_end=training_end, validation_start=validation_start, validation_end=validation_end, batch_size=batch_size)

g2gcnn_model = UniformGraph2GraphCNN(num_node_in_features=num_node_features, num_edge_in_features=num_edge_features, num_node_out_features=1, num_edge_out_features=1, mlp_hidden_width=rep_dims, num_node_mlp_hidden_layers=num_mlp_hidden_layers, num_edge_mlp_hidden_layers=num_mlp_hidden_layers, num_graph_message_passes=num_graph_convolution_layers, dtype=dtype, device=device)
print(f'time {time.time() - code_start_time:.3f}, initialized uniform graph-to-graph convolutional neural network with {num_node_features} input node features, {num_edge_features} input edge features, {rep_dims} latent representaion dimensions, {num_mlp_hidden_layers} hidden layers per MLP, and {num_graph_convolution_layers} graph convolutions.')
if optimizer_name == 'SGD':
    optimizer = torch.optim.SGD( params=g2gcnn_model.parameters(), lr=learning_rate )
else:
    optimizer = torch.optim.Adam( params=g2gcnn_model.parameters(), lr=learning_rate )
loss_fn = GraphMSELoss(num_nodes=num_nodes, dtype=dtype, device=device)

total_epochs = 0
for save in range(num_saves):
    for epoch in range(num_epochs_per_save):
        for node_features_batch, edge_features_batch, h_batch, J_batch in training_data_loader:
            optimizer.zero_grad()
            h_pred, J_pred = g2gcnn_model(node_in_features=node_features_batch, edge_in_features=edge_features_batch)
            loss = loss_fn(node_features_pred=h_pred, edge_features_pred=J_pred, node_features_target=h_batch, edge_features_target=J_batch)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            total_epochs += 1
            training_loss = get_loss(data_loader=training_data_loader, g2gcnn_model=g2gcnn_model)
            validation_loss = get_loss(data_loader=validation_data_loader, g2gcnn_model=g2gcnn_model)
            print(f'time {time.time() - code_start_time:.3f}, epoch {total_epochs}, training loss {training_loss:.3g}, validation loss {validation_loss:.3g}')
    g2gcnn_model_file = os.path.join(output_directory, f'g2gcnn_gconv_{num_graph_convolution_layers}_mlp_{num_mlp_hidden_layers}_rep_{rep_dims}_batch_sz_{batch_size}_lr_{learning_rate:.3g}_epoch_{total_epochs}.pt')
    torch.save(obj=g2gcnn_model, f=g2gcnn_model_file)
    print(f'time {time.time() - code_start_time:.3f}, saved {g2gcnn_model_file}')
print(f'time {time.time() - code_start_time:.3f}, done')