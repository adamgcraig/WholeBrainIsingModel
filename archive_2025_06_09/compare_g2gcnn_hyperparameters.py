import os
import torch
import time
import argparse
import math
import pandas
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from isingmodel import IsingModel# need to import the class for when we load the model from a file
from graph2graphcnn import UniformGraph2GraphCNN
from graph2graphcnn import GraphMSELoss

code_start_time = time.time()

parser = argparse.ArgumentParser(description="Compare performance of the structure-to-Ising model GCNN for different combinations of hyperparameters.")
parser.add_argument("-a", "--file_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can read and write files")
parser.add_argument("-b", "--node_features_file", type=str, default='node_features_group_training_and_individual_all_rectangular.pt', help="file containing a Tensor of individual structural features data of size num_betas*num_subjects x num_nodes x num_node_features")
parser.add_argument("-c", "--edge_features_file", type=str, default='edge_features_group_training_and_individual_all.pt', help="file containing a Tensor of individual structural features data of size num_betas*num_subjects x num_nodes x num_nodes x num_edge_features")
parser.add_argument("-d", "--ising_model_file", type=str, default='ising_model_beta_updates_100_param_updates_3100_group_training_and_individual_all_fold_1_betas_5_steps_1200_lr_0.01.pt', help="file containing the IsingModel object with h of size num_betas*num_subjects x num_nodes and J of size num_betas*num_subjects x num_nodes x num_nodes")
parser.add_argument("-e", "--ising_model_rmse_file", type=str, default='combined_mean_state_rmse_beta_updates_100_param_updates_3100_group_training_and_individual_all_fold_1_betas_5_steps_1200_lr_0.01.pt', help="file containing a combined root mean squared error value for each Ising model representing disparity between observed state and state product means in the Ising model vs in the data")
parser.add_argument("-f", "--training_start", type=int, default=1, help="first index of a training subject")
parser.add_argument("-g", "--training_end", type=int, default=670, help="last index of a training subject + 1")
parser.add_argument("-i", "--validation_start", type=int, default=670, help="first index of a training subject")
parser.add_argument("-j", "--validation_end", type=int, default=754, help="last index of a training subject + 1")
parser.add_argument("-k", "--coordinate_type_name", type=str, default='rectangular', help="the type of coordinates used in node_features_file, just used in the output file names to make it easier to keep track")
parser.add_argument("-l", "--num_epochs", type=int, default=1000, help="number of epochs for which to train each model")
parser.add_argument("-m", "--max_ising_model_rmse", type=float, default=0.15, help="maximum allowed combined parameter RMSE value for fitted Ising models we use for training and validation")
parser.add_argument("-n", "--min_graph_convolution_layers", type=int, default=1, help="minimum number of pairs of a multi-layer perceptron layer followed by a graph convolution layer")
parser.add_argument("-o", "--max_graph_convolution_layers", type=int, default=10, help="maximum number of pairs of a multi-layer perceptron layer followed by a graph convolution layer")
parser.add_argument("-p", "--num_graph_convolution_layers", type=int, default=10, help="numbers of MLP-graph convolution pair counts will be round( linspace(num_graph_convolution_layers points, ranging from min_graph_convolution_layers to max_graph_convolution_layers, inclusive) )")
parser.add_argument("-q", "--min_mlp_hidden_layers", type=int, default=1, help="minimum number of hidden layers to use in each multi-layer perceptron")
parser.add_argument("-r", "--max_mlp_hidden_layers", type=int, default=10, help="maximum number of hidden layers to use in each multi-layer perceptron")
parser.add_argument("-s", "--num_mlp_hidden_layers", type=int, default=10, help="numbers of MLP hidden layer counts will be round( linspace(num_mlp_hidden_layers points, ranging from min_mlp_hidden_layers to max_mlp_hidden_layers, inclusive) )")
parser.add_argument("-t", "--min_rep_dims", type=int, default=2, help="minimum number of dimensions to use in each latent-space representation")
parser.add_argument("-u", "--max_rep_dims", type=int, default=21, help="maximum number of dimensions to use in each latent-space representation")
parser.add_argument("-v", "--num_rep_dims", type=int, default=5, help="numbers of latent-space dimensions will be round( linspace(num_rep_dims points, ranging from min_rep_dims to max_rep_dims, inclusive) )")
parser.add_argument("-w", "--min_batch_size", type=int, default=10, help="the smallest batch size to try")
parser.add_argument("-x", "--max_batch_size", type=int, default=669, help="the smallest batch size to try")
parser.add_argument("-y", "--num_batch_sizes", type=int, default=5, help="number of batch sizes to try")
parser.add_argument("-z", "--min_learning_rate", type=float, default=0.0000001, help="the slowest learning rate to try")
parser.add_argument("-0", "--max_learning_rate", type=float, default=1.0, help="the fastest learning rate to try")
parser.add_argument("-1", "--num_learning_rates", type=int, default=5, help="number of learning rates to try")
parser.add_argument("-2", "--results_file_name", type=str, default='compare_g2gcnn_hyperparameters.pkl', help="We will save a record of the training and validation errors for all combinations of hyperparameters to this file as a pandas DataFrame pickle file.")
parser.add_argument("-3", "--save_models", action='store_true', default=False, help="Set this flag in order to have the script save each trained G2GCNN model.")
parser.add_argument("-4", "--patience", type=int, default=10, help="Number of epochs with no noticeable improvement in either loss before we stop and move on to the next model.")
parser.add_argument("-5", "--improvement_threshold", type=float, default=0.0001, help="Minimal amount of improvement to count as noticeable.")
args = parser.parse_args()
print('getting arguments...')
file_directory = args.file_directory
print(f'file_directory={file_directory}')
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
num_epochs = args.num_epochs
print(f'num_epochs={num_epochs}')
max_ising_model_rmse = args.max_ising_model_rmse
print(f'max_ising_model_rmse={max_ising_model_rmse}')
min_graph_convolution_layers = args.min_graph_convolution_layers
print(f'min_graph_convolution_layers={min_graph_convolution_layers}')
max_graph_convolution_layers = args.max_graph_convolution_layers
print(f'max_graph_convolution_layers={max_graph_convolution_layers}')
num_graph_convolution_layers = args.num_graph_convolution_layers
print(f'num_graph_convolution_layers={num_graph_convolution_layers}')
min_mlp_hidden_layers = args.min_mlp_hidden_layers
print(f'min_mlp_hidden_layers={min_mlp_hidden_layers}')
max_mlp_hidden_layers = args.max_mlp_hidden_layers
print(f'max_mlp_hidden_layers={max_mlp_hidden_layers}')
num_mlp_hidden_layers = args.num_mlp_hidden_layers
print(f'num_mlp_hidden_layers={num_mlp_hidden_layers}')
min_rep_dims = args.min_rep_dims
print(f'min_rep_dims={min_rep_dims}')
max_rep_dims = args.max_rep_dims
print(f'max_rep_dims={max_rep_dims}')
num_rep_dims = args.num_rep_dims
print(f'num_rep_dims={num_rep_dims}')
min_batch_size = args.min_batch_size
print(f'min_batch_size={min_batch_size}')
max_batch_size = args.max_batch_size
print(f'max_batch_size={max_batch_size}')
num_batch_sizes = args.num_batch_sizes
print(f'num_batch_sizes={num_batch_sizes}')
min_learning_rate = args.min_learning_rate
print(f'min_learning_rate={min_learning_rate:.3g}')
max_learning_rate = args.max_learning_rate
print(f'max_learning_rate={max_learning_rate:.3g}')
num_learning_rates = args.num_learning_rates
print(f'num_learning_rates={num_learning_rates}')
results_file_name = args.results_file_name
print(f'results_file_name={results_file_name}')
save_models = args.save_models
print(f'save_models={save_models}')
patience = args.patience
print(f'patience={patience}')
improvement_threshold = args.improvement_threshold
print(f'improvement_threshold={improvement_threshold}')

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

def init_losses_df(min_learning_rate:float, max_learning_rate:float, num_learning_rates:int, min_batch_size:int, max_batch_size:int, num_batch_sizes:int, min_graph_convolution_layers:int, max_graph_convolution_layers:int, num_graph_convolution_layers:int, min_mlp_hidden_layers:int, max_mlp_hidden_layers:int, num_mlp_hidden_layers:int, min_rep_dims:int, max_rep_dims:int, num_rep_dims:int) -> pandas.DataFrame:
    # Set up a DataFrame to store the final losses of the trained models.
    # We have a set list of predefined choices for the node coordinate space and the optimizer.
    optimizer_names = ['Adam', 'SGD']
    num_optimizers = len(optimizer_names)
    # Grow the batch sizes and learning rates at exponential rates.
    batch_sizes = torch.exp( torch.linspace( start=math.log(min_batch_size), end=math.log(max_batch_size), steps=num_batch_sizes, dtype=float_type, device=device ) ).int()
    learning_rates = torch.exp( torch.linspace( start=math.log(min_learning_rate), end=math.log(max_learning_rate), steps=num_learning_rates, dtype=float_type, device=device) )
    # Grow parameters that affect the depth and width of the GCNN at linear rates.
    graph_convolution_layers_choices = torch.linspace(start=min_graph_convolution_layers, end=max_graph_convolution_layers, steps=num_graph_convolution_layers, dtype=int_type, device=device)
    mlp_hidden_layers_choices = torch.linspace(start=min_mlp_hidden_layers, end=max_mlp_hidden_layers, steps=num_mlp_hidden_layers, dtype=int_type, device=device)
    rep_dims_choices = torch.linspace(start=min_rep_dims, end=max_rep_dims, steps=num_rep_dims, dtype=int_type, device=device)
    losses_df = pandas.DataFrame({'batch_size':pandas.Series(dtype='int'), 'graph_convolution_layers':pandas.Series(dtype='int'), 'mlp_hidden_layers':pandas.Series(dtype='int'), 'rep_dims':pandas.Series(dtype='int'), 'optimizer_name':pandas.Series(dtype='str'), 'learning_rate':pandas.Series(dtype='float'), 'training_rmse':pandas.Series(dtype='float'), 'validation_rmse':pandas.Series(dtype='float'), 'time':pandas.Series(dtype='float')})
    num_conditions = 0
    for graph_convolution_layers_index in range(num_graph_convolution_layers):
        graph_convolution_layers = graph_convolution_layers_choices[graph_convolution_layers_index]
        # print(f'{time.time()-code_start_time:.3f}, set graph_convolution_layers={graph_convolution_layers}')
        for mlp_hidden_layers_index in range(num_mlp_hidden_layers):
            mlp_hidden_layers = mlp_hidden_layers_choices[mlp_hidden_layers_index]
            # print(f'{time.time()-code_start_time:.3f}, set mlp_hidden_layers={mlp_hidden_layers}')
            for rep_dims_index in range(num_rep_dims):
                rep_dims = rep_dims_choices[rep_dims_index]
                # print(f'{time.time()-code_start_time:.3f}, set rep_dims={rep_dims}')
                for batch_size_index in range(num_batch_sizes):
                    batch_size = batch_sizes[batch_size_index]
                    # print(f'{time.time()-code_start_time:.3f}, set batch_size={batch_size}')
                    for learning_rate_index in range(num_learning_rates):
                        learning_rate = learning_rates[learning_rate_index]
                        # print(f'{time.time()-code_start_time:.3f}, set learning_rate={learning_rate:.3g}')
                        for optimizer_index in range(num_optimizers):
                            optimizer_name = optimizer_names[optimizer_index]
                            # print(f'{time.time()-code_start_time:.3f}, set optimizer_name={optimizer_name}')
                            dfrow = pandas.DataFrame({'batch_size':batch_size, 'graph_convolution_layers':graph_convolution_layers, 'mlp_hidden_layers':mlp_hidden_layers, 'rep_dims':rep_dims, 'optimizer_name':optimizer_name, 'learning_rate':learning_rate, 'training_rmse':-1.0, 'validation_rmse':-1.0, 'time':-1.0}, index=[num_conditions])
                            losses_df = pandas.concat([losses_df, dfrow], ignore_index=True)
                            num_conditions += 1
    return losses_df

def load_data(file_directory:str, ising_model_file:str, ising_model_rmse_file:str, node_features_file:str, edge_features_file:str, training_start:int, training_end:int, validation_start:int, validation_end:int, max_ising_model_rmse:float):
    # Counterintuitively, all our input files are in the output directory.

    ising_model_file = os.path.join(file_directory, ising_model_file)
    ising_model = torch.load(ising_model_file)
    print(f'time {time.time() - code_start_time:.3f}, loaded {ising_model_file}')
    # Just multiply in beta instead of trying to learn it separately.
    # Unsqueeze so that we have a separate singleton "out-features" dimension.
    h = torch.unsqueeze( input=(ising_model.h * ising_model.beta), dim=-1 )
    J = torch.unsqueeze(  input=( ising_model.J * ising_model.beta.unsqueeze(dim=-1) ), dim=-1  )

    ising_model_rmse_file = os.path.join(file_directory, ising_model_rmse_file)
    ising_model_rmse = torch.load(ising_model_rmse_file)
    print(f'time {time.time() - code_start_time:.3f}, loaded {ising_model_rmse_file}')

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

    is_good_ising_model = ising_model_rmse < max_ising_model_rmse
    training_is_good_ising_model = is_good_ising_model[training_start:training_end]
    training_indices = torch.arange(start=training_start, end=training_end, dtype=int_type, device=device)[training_is_good_ising_model]
    validation_is_good_ising_model = is_good_ising_model[validation_start:validation_end]
    validation_indices = torch.arange(start=validation_start, end=validation_end, dtype=int_type, device=device)[validation_is_good_ising_model]
    print(f'{torch.count_nonzero(training_is_good_ising_model)} of {training_is_good_ising_model.numel()} training models and {torch.count_nonzero(validation_is_good_ising_model)} of {validation_is_good_ising_model.numel()} validation models have RMSEs below threshold {max_ising_model_rmse:.3g}.')

    # Clone these so that we can deallocate the rest of the underlying Tensors once we exit the function scope.
    training_node_features = node_features[training_indices,:,:].clone()
    training_edge_features = edge_features[training_indices,:,:,:].clone()
    training_h = h[training_indices,:,:].clone()
    training_J = J[training_indices,:,:,:].clone()
    validation_node_features = node_features[validation_indices,:,:].clone()
    validation_edge_features = edge_features[validation_indices,:,:,:].clone()
    validation_h = h[validation_indices,:,:].clone()
    validation_J = J[validation_indices,:,:,:].clone()

    # Package the Tensors into a DataSet just so that we do not need to return as many individual things.
    # We cannot put them into a DataLoader yet, because we still do not know what batch size to use.
    training_data_set = StructParamsDataset(node_features=training_node_features, edge_features=training_edge_features, h=training_h, J=training_J)
    validation_data_set = StructParamsDataset(node_features=validation_node_features, edge_features=validation_edge_features, h=validation_h, J=validation_J)
    return training_data_set, validation_data_set

def get_loss(data_loader:DataLoader, loss_fn:torch.nn.Module, g2gcnn_model:UniformGraph2GraphCNN):
    training_loss_sum = 0
    training_loss_count = 0
    for node_features_batch, edge_features_batch, h_batch, J_batch in data_loader:
        h_pred, J_pred = g2gcnn_model(node_in_features=node_features_batch, edge_in_features=edge_features_batch)
        loss = loss_fn(node_features_pred=h_pred, edge_features_pred=J_pred, node_features_target=h_batch, edge_features_target=J_batch)
        current_batch_size = node_features_batch.size(dim=0)
        training_loss_sum += ( current_batch_size*loss.item() )
        training_loss_count += current_batch_size
    return training_loss_sum/training_loss_count

def train_model(training_data_set:StructParamsDataset, validation_data_set:StructParamsDataset, g2gcnn_model_file:str, num_epochs:int=1000, optimizer_name:str='Adam', rep_dims:int=7, mlp_hidden_layers:int=3, graph_convolution_layers:int=3, batch_size:int=10, learning_rate:torch.float=0.001, save_model:bool=False, patience:int=10, improvement_threshold:torch.float=0.0001):
    example_node_features, example_edge_features, example_h, example_J = training_data_set.__getitem__(0)
    dtype = example_node_features.dtype
    device = example_edge_features.device
    num_nodes, num_node_features = example_node_features.size()
    num_edge_features = example_edge_features.size(dim=-1)
    num_h_features = example_h.size(dim=-1)
    num_J_features = example_J.size(dim=-1)
    training_data_loader = DataLoader(dataset=training_data_set, shuffle=True, batch_size=batch_size)
    validation_data_loader = DataLoader(dataset=validation_data_set, shuffle=False, batch_size=batch_size)
    g2gcnn_model = UniformGraph2GraphCNN(num_node_in_features=num_node_features, num_edge_in_features=num_edge_features, num_node_out_features=num_h_features, num_edge_out_features=num_J_features, mlp_hidden_width=rep_dims, num_node_mlp_hidden_layers=mlp_hidden_layers, num_edge_mlp_hidden_layers=mlp_hidden_layers, num_graph_message_passes=graph_convolution_layers, dtype=dtype, device=device)
    loss_fn = GraphMSELoss(num_nodes=num_nodes, dtype=dtype, device=device)
    print(f'time {time.time() - code_start_time:.3f}, initialized uniform graph-to-graph convolutional neural network with {num_node_features} input node features, {num_edge_features} input edge features, {rep_dims} latent representaion dimensions, {mlp_hidden_layers} hidden layers per MLP, and {graph_convolution_layers} graph convolutions.')
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD( params=g2gcnn_model.parameters(), lr=learning_rate )
    else:
        optimizer = torch.optim.Adam( params=g2gcnn_model.parameters(), lr=learning_rate )
    num_no_improvement_epochs = 0
    last_training_loss = 10e10
    last_validation_loss = 10e10
    for epoch in range(num_epochs):
        for node_features_batch, edge_features_batch, h_batch, J_batch in training_data_loader:
            optimizer.zero_grad()
            h_pred, J_pred = g2gcnn_model(node_in_features=node_features_batch, edge_in_features=edge_features_batch)
            loss = loss_fn(node_features_pred=h_pred, edge_features_pred=J_pred, node_features_target=h_batch, edge_features_target=J_batch)
            # print( 'loss', loss )
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            training_loss = get_loss(data_loader=training_data_loader, loss_fn=loss_fn, g2gcnn_model=g2gcnn_model)
            validation_loss = get_loss(data_loader=validation_data_loader, loss_fn=loss_fn, g2gcnn_model=g2gcnn_model)
            is_no_improvement = int(  ( (last_training_loss - training_loss) < improvement_threshold  ) and ( (last_validation_loss - validation_loss) < improvement_threshold )  )
            num_no_improvement_epochs = (num_no_improvement_epochs + 1)*is_no_improvement
            last_training_loss = training_loss
            last_validation_loss = validation_loss
            print(f'time {time.time() - code_start_time:.3f}, epoch {epoch+1}, training loss {training_loss:.3g}, validation loss {validation_loss:.3g}, num no-improvement epochs {num_no_improvement_epochs}')
            if num_no_improvement_epochs >= patience:
                print(f'patience exceeded, moving on...')
                break
    if save_model:
        torch.save(obj=g2gcnn_model, f=g2gcnn_model_file)
        print(f'time {time.time() - code_start_time:.3f}, saved {g2gcnn_model_file}')
    # Use item() here just to make sure these get detached from the optimization, backpropagation, etc.,
    # allowing us to free up the CUDA memory of everything allocated inside the function scope.
    return training_loss, validation_loss

# Load the training and validation data.
training_data_set, validation_data_set = load_data(file_directory=file_directory, ising_model_file=ising_model_file, ising_model_rmse_file=ising_model_rmse_file, node_features_file=node_features_file, edge_features_file=edge_features_file, training_start=training_start, training_end=training_end, validation_start=validation_start, validation_end=validation_end, max_ising_model_rmse=max_ising_model_rmse)
# Check whether we have already created the results pickle file in a previous run.
results_file = os.path.join(file_directory, results_file_name)
if os.path.exists(results_file):
    print(f'loading results table from {results_file}...')
    losses_df = pandas.read_pickle(results_file)
    start_index = len( losses_df.loc[ losses_df['time'] != -1.0 ].index )
else:
    print('creating new results table...')
    losses_df = init_losses_df(min_learning_rate=min_learning_rate, max_learning_rate=max_learning_rate, num_learning_rates=num_learning_rates, min_batch_size=min_batch_size, max_batch_size=max_batch_size, num_batch_sizes=num_batch_sizes, min_graph_convolution_layers=min_graph_convolution_layers, max_graph_convolution_layers=max_graph_convolution_layers, num_graph_convolution_layers=num_graph_convolution_layers, min_mlp_hidden_layers=min_mlp_hidden_layers, max_mlp_hidden_layers=max_mlp_hidden_layers, num_mlp_hidden_layers=num_mlp_hidden_layers, min_rep_dims=min_rep_dims, max_rep_dims=max_rep_dims, num_rep_dims=num_rep_dims)
    start_index = 0
num_cases = len(losses_df.index)
print(f'starting from condition {start_index} of {num_cases}')
for condition_index in range(start_index, num_cases):
    batch_size = int(losses_df.at[condition_index,'batch_size'])
    graph_convolution_layers = int(losses_df.at[condition_index,'graph_convolution_layers'])
    mlp_hidden_layers = int(losses_df.at[condition_index,'mlp_hidden_layers'])
    rep_dims = int(losses_df.at[condition_index,'rep_dims'])
    optimizer_name = str(losses_df.at[condition_index,'optimizer_name'])
    learning_rate = float(losses_df.at[condition_index,'learning_rate'])
    training_start_time = time.time()
    print(f'{training_start_time-code_start_time:.3f}, training with hyperparameter set {condition_index+1} of {num_cases}...')
    g2gcnn_model_file = os.path.join(file_directory, f'g2gcnn_coord_{coordinate_type_name}_max_rmse_{max_ising_model_rmse:.3g}_epochs_{num_epochs}_opt_{optimizer_name}_gconv_{graph_convolution_layers}_mlp_{mlp_hidden_layers}_rep_{rep_dims}_batch_sz_{batch_size}_lr_{learning_rate:.3g}.pt')
    training_mse, validation_mse = train_model(training_data_set=training_data_set, validation_data_set=validation_data_set, g2gcnn_model_file=g2gcnn_model_file, num_epochs=num_epochs, optimizer_name=optimizer_name, rep_dims=rep_dims, mlp_hidden_layers=mlp_hidden_layers, graph_convolution_layers=graph_convolution_layers, batch_size=batch_size, learning_rate=learning_rate, save_model=save_models, patience=patience, improvement_threshold=improvement_threshold)
    losses_df.at[condition_index,'training_rmse'] = math.sqrt(training_mse)
    losses_df.at[condition_index,'validation_rmse'] = math.sqrt(validation_mse)
    losses_df.at[condition_index,'time'] = time.time() - training_start_time
    losses_df.to_pickle(path=results_file)
    print(f'{time.time()-code_start_time:.3f}, saved {results_file}')
print(f'time {time.time() - code_start_time:.3f}, done')