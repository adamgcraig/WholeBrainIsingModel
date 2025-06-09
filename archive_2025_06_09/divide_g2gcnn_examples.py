import os
import torch
import time
import argparse
from torch.utils.data import Dataset
from isingmodel import IsingModel# need to import the class for when we load the model from a file

code_start_time = time.time()

parser = argparse.ArgumentParser(description="Compare performance of the structure-to-Ising model GCNN for different combinations of hyperparameters.")
parser.add_argument("-i", "--input_directory", type=str, default='E:\\Ising_model_results_daai', help="directory from which to read the Ising model file and big structural features files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\g2gcnn_examples', help="directory to which we can write the smaller individual example files")
parser.add_argument("-b", "--node_features_file", type=str, default='node_features_group_training_and_individual_all_rectangular.pt', help="file containing a Tensor of individual structural features data of size num_betas*num_subjects x num_nodes x num_node_features")
parser.add_argument("-c", "--edge_features_file", type=str, default='edge_features_group_training_and_individual_all.pt', help="file containing a Tensor of individual structural features data of size num_betas*num_subjects x num_nodes x num_nodes x num_edge_features")
parser.add_argument("-d", "--ising_model_file", type=str, default='ising_model_beta_updates_100_param_updates_3100_group_training_and_individual_all_fold_1_betas_5_steps_1200_lr_0.01.pt', help="file containing the IsingModel object with h of size num_betas*num_subjects x num_nodes and J of size num_betas*num_subjects x num_nodes x num_nodes")
parser.add_argument("-e", "--ising_model_rmse_file", type=str, default='combined_mean_state_rmse_beta_updates_100_param_updates_3100_group_training_and_individual_all_fold_1_betas_5_steps_1200_lr_0.01.pt', help="file containing a combined root mean squared error value for each Ising model representing disparity between observed state and state product means in the Ising model vs in the data")
parser.add_argument("-f", "--training_start", type=int, default=1, help="first index of a training subject")
parser.add_argument("-g", "--training_end", type=int, default=670, help="last index of a training subject + 1")
parser.add_argument("-p", "--validation_start", type=int, default=670, help="first index of a training subject")
parser.add_argument("-j", "--validation_end", type=int, default=754, help="last index of a training subject + 1")
parser.add_argument("-k", "--coordinate_type_name", type=str, default='rectangular', help="the type of coordinates used in node_features_file, just used in the output file names to make it easier to keep track")
parser.add_argument("-m", "--max_ising_model_rmse", type=float, default=2.0, help="maximum allowed combined parameter RMSE value for fitted Ising models we use for training and validation")
args = parser.parse_args()
print('getting arguments...')
input_directory = args.input_directory
print(f'input_directory={input_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
node_features_file = args.node_features_file
print(f'node_features_file={node_features_file}')
edge_features_file = args.edge_features_file
print(f'edge_features_file={edge_features_file}')
ising_model_file = args.ising_model_file
print(f'ising_model_file={ising_model_file}')
ising_model_rmse_file = args.ising_model_rmse_file
print(f'ising_model_rmse_file={ising_model_rmse_file}')
training_start = args.training_start
print(f'training_start={training_start}')
training_end = args.training_end
print(f'training_end={training_end}')
validation_start = args.validation_start
print(f'validation_start={validation_start}')
validation_end = args.validation_end
print(f'validation_end={validation_end}')
coordinate_type_name = args.coordinate_type_name
print(f'coordinate_type_name={coordinate_type_name}')
max_ising_model_rmse = args.max_ising_model_rmse
print(f'max_ising_model_rmse={max_ising_model_rmse}')

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')

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

def load_data(file_directory:str, ising_model_file:str, ising_model_rmse_file:str, node_features_file:str, edge_features_file:str, training_start:int, training_end:int, validation_start:int, validation_end:int, max_ising_model_rmse:float):
    # Counterintuitively, all our input files are in the output directory.

    ising_model_file = os.path.join(file_directory, ising_model_file)
    ising_model = torch.load(ising_model_file)
    print(f'time {time.time() - code_start_time:.3f}, loaded {ising_model_file}')
    # Just multiply in beta instead of trying to learn it separately.
    # Unsqueeze so that we have a separate singleton "out-features" dimension.
    h = torch.unsqueeze( input=(ising_model.h * ising_model.beta), dim=-1 ).unflatten( dim=0, sizes=(ising_model.num_betas_per_target, -1) )
    J = torch.unsqueeze(  input=( ising_model.J * ising_model.beta.unsqueeze(dim=-1) ), dim=-1  ).unflatten( dim=0, sizes=(ising_model.num_betas_per_target, -1) )

    ising_model_rmse_file = os.path.join(file_directory, ising_model_rmse_file)
    ising_model_rmse = torch.load(ising_model_rmse_file).unflatten( dim=0, sizes=(ising_model.num_betas_per_target, -1) )
    print(f'time {time.time() - code_start_time:.3f}, loaded {ising_model_rmse_file}')

    node_features_file = os.path.join(file_directory, node_features_file)
    node_features = torch.load(node_features_file)
    print(f'time {time.time() - code_start_time:.3f}, loaded {node_features_file}')
    print( 'node_features size', node_features.size() )
    node_features = torch.unsqueeze(node_features, dim=0).repeat( (ising_model.num_betas_per_target, 1, 1, 1) )
    print( 'expanded to', node_features.size() )

    edge_features_file = os.path.join(file_directory, edge_features_file)
    edge_features = torch.load(edge_features_file)
    print(f'time {time.time() - code_start_time:.3f}, loaded {edge_features_file}')
    print( 'edge_features size', edge_features.size() )
    edge_features = torch.unsqueeze(edge_features, dim=0).repeat( (ising_model.num_betas_per_target, 1, 1, 1, 1) )
    print( 'expanded to', edge_features.size() )

    # Separate out the training subjects and validation subjects.
    training_node_features = node_features[:,training_start:training_end,:,:].flatten(start_dim=0, end_dim=1)
    validation_node_features = node_features[:,validation_start:validation_end,:,:].flatten(start_dim=0, end_dim=1)
    training_edge_features = edge_features[:,training_start:training_end,:,:,:].flatten(start_dim=0, end_dim=1)
    validation_edge_features = edge_features[:,validation_start:validation_end,:,:,:].flatten(start_dim=0, end_dim=1)
    training_h = h[:,training_start:training_end,:,:].flatten(start_dim=0, end_dim=1)
    validation_h = h[:,validation_start:validation_end,:,:].flatten(start_dim=0, end_dim=1)
    training_J = J[:,training_start:training_end,:,:,:].flatten(start_dim=0, end_dim=1)
    validation_J = J[:,validation_start:validation_end,:,:,:].flatten(start_dim=0, end_dim=1)
    training_rmse = ising_model_rmse[:,training_start:training_end].flatten(start_dim=0, end_dim=1)
    validation_rmse = ising_model_rmse[:,validation_start:validation_end].flatten(start_dim=0, end_dim=1)

    # Filter out models with fitting RMSE that is too high.
    training_is_good_enough = training_rmse < max_ising_model_rmse
    validation_is_good_enough = validation_rmse < max_ising_model_rmse
    training_node_features = training_node_features[training_is_good_enough,:,:]
    validation_node_features = validation_node_features[validation_is_good_enough,:,:]
    training_edge_features = training_edge_features[training_is_good_enough,:,:,:]
    validation_edge_features = validation_edge_features[validation_is_good_enough,:,:,:]
    training_h = training_h[training_is_good_enough,:,:]
    validation_h = validation_h[validation_is_good_enough,:,:]
    training_J = training_J[training_is_good_enough,:,:,:]
    validation_J = validation_J[validation_is_good_enough,:,:,:]

    # Package the Tensors into a DataSet just so that we do not need to return as many individual things.
    # We cannot put them into a DataLoader yet, because we still do not know what batch size to use.
    training_data_set = StructParamsDataset(node_features=training_node_features, edge_features=training_edge_features, h=training_h, J=training_J)
    validation_data_set = StructParamsDataset(node_features=validation_node_features, edge_features=validation_edge_features, h=validation_h, J=validation_J)
    return training_data_set, validation_data_set

def split_and_save_data(data_set:StructParamsDataset, save_directory:str, file_suffix:str):
    example_file_list = []
    for example_index in range( data_set.__len__() ):
        node_features, edge_features, h, J = data_set.__getitem__(example_index)
        example_file = os.path.join(save_directory, f'example_{example_index}_{file_suffix}.pt')
        # Use .clone() to create separate Tensors instead of views.
        # If we save a view, it will save the entire underlying Tensor.
        torch.save(  obj=( node_features.clone(), edge_features.clone(), h.clone(), J.clone() ), f=example_file  )
        print(f'saved {example_file}')
        example_file_list.append(example_file)
    return example_file_list

# Load the training and validation data.
training_data_set, validation_data_set = load_data(file_directory=input_directory, ising_model_file=ising_model_file, ising_model_rmse_file=ising_model_rmse_file, node_features_file=node_features_file, edge_features_file=edge_features_file, training_start=training_start, training_end=training_end, validation_start=validation_start, validation_end=validation_end, max_ising_model_rmse=max_ising_model_rmse)
# Split each data set up into individual files to be loaded one-at-a-time later.
training_data_files = split_and_save_data(data_set=training_data_set, save_directory=output_directory, file_suffix=f'{coordinate_type_name}_max_rmse_{max_ising_model_rmse:.3g}_training_example')
validation_data_files = split_and_save_data(data_set=validation_data_set, save_directory=output_directory, file_suffix=f'{coordinate_type_name}_max_rmse_{max_ising_model_rmse:.3g}_validation_example')
print(f'time {time.time() - code_start_time:.3f}, done')