import os
import torch
import time
import argparse
import math
import isingmodel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from graph2graphcnn import MultiLayerPerceptron

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

parser = argparse.ArgumentParser(description="Measure correlations between structural features and parameters of fitted Ising models.")
parser.add_argument("-a", "--input_directory", type=str, default='E:\\g2gcnn_examples', help="directory from which we read the training examples")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the trained models and pandas DataFrame of loss results")
parser.add_argument("-c", "--file_name_fragment", type=str, default='rectangular_max_rmse_2', help="part of the input example files between example_[index]_ and _[validation|training]_example.pt, will also be part of the output file names")
parser.add_argument("-j", "--num_training_examples", type=int, default=3345, help="number of training examples")
parser.add_argument("-q", "--batch_size", type=int, default=50, help="batch size")
args = parser.parse_args()
print('getting arguments...')
input_directory = args.input_directory
print(f'input_directory={input_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
file_name_fragment = args.file_name_fragment
print(f'file_name_fragment={file_name_fragment}')
num_training_examples = args.num_training_examples
print(f'num_training_examples={num_training_examples}')
batch_size = args.batch_size
print(f'batch_size={batch_size}')

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

# Load the training and validation data.
training_data_set = StructParamsFileDataset(file_directory=input_directory, file_suffix=f'{file_name_fragment}_training_example', num_nodes=num_nodes, num_node_features=num_node_features, num_edge_features=num_edge_features, num_h_features=num_h_features, num_J_features=num_J_features, dtype=float_type, device=device, num_examples=num_training_examples)
training_data_loader = DataLoader(dataset=training_data_set, shuffle=True, batch_size=batch_size)
node_product_sum = torch.zeros( size=(num_node_features, num_h_features), dtype=float_type, device=device )
node_feature_sum = torch.zeros( size=(num_node_features,), dtype=float_type, device=device )
node_squared_sum = torch.zeros_like(node_feature_sum)
h_feature_sum = torch.zeros( size=(num_h_features,), dtype=float_type, device=device )
h_squared_sum = torch.zeros_like(h_feature_sum)
num_node_samples = 0
edge_product_sum = torch.zeros( size=(num_edge_features, num_J_features), dtype=float_type, device=device )
edge_feature_sum = torch.zeros( size=(num_edge_features,), dtype=float_type, device=device )
edge_squared_sum = torch.zeros_like(edge_feature_sum)
J_feature_sum = torch.zeros( size=(num_J_features,), dtype=float_type, device=device )
J_squared_sum = torch.zeros_like(J_feature_sum)
num_edge_samples = 0
triu_rows, triu_cols = isingmodel.get_triu_indices_for_products(num_nodes=num_nodes, device=device)
batch_number = 1
for node_features_batch, edge_features_batch, h_batch, J_batch in training_data_loader:
    node_features_batch, edge_features_batch = recombine_node_and_edge_features(node_features=node_features_batch, edge_features=edge_features_batch)
    node_product_sum += torch.sum( node_features_batch[:,:,:,None] * h_batch[:,:,None,:], dim=(0,1) )
    node_feature_sum += torch.sum( node_features_batch, dim=(0,1) )
    node_squared_sum += torch.sum( torch.square(node_features_batch), dim=(0,1) )
    h_feature_sum += torch.sum( h_batch, dim=(0,1) )
    h_squared_sum += torch.sum( torch.square(h_batch), dim=(0,1) )
    num_node_samples += ( node_features_batch.size(dim=0) * node_features_batch.size(dim=1) )
    edge_features_triu = edge_features_batch[:,triu_rows,triu_cols,:]
    J_triu = J_batch[:,triu_rows,triu_cols,:]
    edge_product_sum += torch.sum( edge_features_triu[:,:,:,None] * J_triu[:,:,None,:], dim=(0,1) )
    edge_feature_sum += torch.sum( edge_features_triu, dim=(0,1) )
    edge_squared_sum += torch.sum( torch.square(edge_features_triu), dim=(0,1) )
    J_feature_sum += torch.sum( J_triu, dim=(0,1) )
    J_squared_sum += torch.sum( torch.square(J_triu), dim=(0,1) )
    num_edge_samples += ( edge_features_triu.size(dim=0) * edge_features_triu.size(dim=1) )
    # print( f'batch {batch_number} sizes, node features ', node_features_batch.size(), ', h', h_batch.size(), ', edge features triu', edge_features_triu.size(), ', J triu', J_triu.size() )
node_feature_mean = node_feature_sum/num_node_samples
print('node_feature_mean', node_feature_mean)
node_feature_std = torch.sqrt(  ( node_squared_sum - num_node_samples*torch.square(node_feature_mean) )/(num_node_samples-1)  )
print('node_feature_std', node_feature_std)
h_mean = h_feature_sum/num_node_samples
print('h_mean', h_mean)
h_std = torch.sqrt(  ( h_squared_sum - num_node_samples*torch.square(h_mean) )/(num_node_samples-1)  )
print('h_std', h_std)
node_correlation = (node_product_sum - num_node_samples*node_feature_mean[:,None]*h_mean[None,:])/( (num_node_samples-1) * node_feature_std[:,None] * h_std[None,:] )
print( 'node correlations', node_correlation.transpose(dim0=0, dim1=1) )
edge_feature_mean = edge_feature_sum/num_edge_samples
print('edge_feature_mean', edge_feature_mean)
edge_feature_std = torch.sqrt(  ( edge_squared_sum - num_edge_samples*torch.square(edge_feature_mean) )/(num_edge_samples-1)  )
print('edge_feature_std', edge_feature_std)
J_mean = J_feature_sum/num_edge_samples
print('J_mean', J_mean)
print( 'J_squared_sum', J_squared_sum )
print( 'square(J_mean)', torch.square(J_mean) )
print('num_edge_samples', num_edge_samples)
J_std = torch.sqrt(  ( J_squared_sum - num_edge_samples*torch.square(J_mean) )/(num_edge_samples-1)  )
print('J_std', J_std)
edge_correlation = (edge_product_sum - num_edge_samples*edge_feature_mean[:,None]*J_mean[None,:])/( (num_edge_samples-1) * edge_feature_std[:,None] * J_std[None,:] )
print( 'edge correlations', edge_correlation.transpose(dim0=0, dim1=1) )
node_correlation_file = os.path.join(output_directory, f'node_feature_h_correlations.pt')
torch.save(obj=node_correlation, f=node_correlation_file)
edge_correlation_file = os.path.join(output_directory, f'edge_feature_J_correlations.pt')
torch.save(obj=edge_correlation, f=edge_correlation_file)
print(f'time {time.time() - code_start_time:.3f}, done')