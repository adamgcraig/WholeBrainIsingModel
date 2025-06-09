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
parser.add_argument("-a", "--input_directory", type=str, default='E:\\g2gcnn_examples', help="directory from which we read the training examples")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the trained models and pandas DataFrame of loss results")
parser.add_argument("-c", "--file_name_fragment", type=str, default='rectangular_max_rmse_2', help="part of the input example files between example_[index]_ and _[validation|training]_example.pt, will also be part of the output file names")
parser.add_argument("-z", "--output_file_name_fragment", type=str, default='big_batch', help="another string we can incorporate into the output file names to help differentiate separate runs")
parser.add_argument("-d", "--num_epochs", type=int, default=1000, help="number of epochs for which to train each model")
parser.add_argument("-e", "--patience", type=int, default=10, help="Number of epochs with no noticeable improvement in either loss before we stop and move on to the next model.")
parser.add_argument("-f", "--min_improvement", type=float, default=10e-10, help="Minimal amount of improvement to count as noticeable.")
parser.add_argument("-g", "--save_models", action='store_true', default=False, help="Set this flag in order to have the script save each trained G2GCNN model.")
parser.add_argument("-i", "--num_training_examples", type=int, default=3345, help="number of training examples")
parser.add_argument("-j", "--num_validation_examples", type=int, default=420, help="number of validation examples")
parser.add_argument("-k", "--min_graph_convolution_layers", type=int, default=1, help="minimum number of pairs of a multi-layer perceptron layer followed by a graph convolution layer")
parser.add_argument("-l", "--max_graph_convolution_layers", type=int, default=10, help="maximum number of pairs of a multi-layer perceptron layer followed by a graph convolution layer")
parser.add_argument("-m", "--num_graph_convolution_layers", type=int, default=10, help="numbers of MLP-graph convolution pair counts will be round( linspace(num_graph_convolution_layers points, ranging from min_graph_convolution_layers to max_graph_convolution_layers, inclusive) )")
parser.add_argument("-n", "--min_mlp_hidden_layers", type=int, default=1, help="minimum number of hidden layers to use in each multi-layer perceptron")
parser.add_argument("-o", "--max_mlp_hidden_layers", type=int, default=10, help="maximum number of hidden layers to use in each multi-layer perceptron")
parser.add_argument("-p", "--num_mlp_hidden_layers", type=int, default=10, help="numbers of MLP hidden layer counts will be round( linspace(num_mlp_hidden_layers points, ranging from min_mlp_hidden_layers to max_mlp_hidden_layers, inclusive) )")
parser.add_argument("-q", "--min_rep_dims", type=int, default=2, help="minimum number of dimensions to use in each latent-space representation")
parser.add_argument("-r", "--max_rep_dims", type=int, default=21, help="maximum number of dimensions to use in each latent-space representation")
parser.add_argument("-s", "--num_rep_dims", type=int, default=5, help="numbers of latent-space dimensions will be round( linspace(num_rep_dims points, ranging from min_rep_dims to max_rep_dims, inclusive) )")
parser.add_argument("-t", "--min_batch_size", type=int, default=10, help="the smallest batch size to try")
parser.add_argument("-u", "--max_batch_size", type=int, default=420, help="the smallest batch size to try")
parser.add_argument("-v", "--num_batch_sizes", type=int, default=5, help="number of batch sizes to try")
parser.add_argument("-w", "--min_learning_rate", type=float, default=0.0000001, help="the slowest learning rate to try")
parser.add_argument("-x", "--max_learning_rate", type=float, default=1.0, help="the fastest learning rate to try")
parser.add_argument("-y", "--num_learning_rates", type=int, default=5, help="number of learning rates to try")
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

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')
num_nodes = 360
num_node_features = 7
num_edge_features = 1
num_h_features = 1
num_J_features = 1

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
        self.dtype =dtype
        self.device = device
    def __len__(self):
        return self.num_examples
    def __getitem__(self, idx:int):
        return torch.load( f=os.path.join(self.file_directory, f'example_{idx}_{self.file_suffix}.pt') )
    # def __getitem__(self, idx:torch.Tensor):
    #     if type(idx) == torch.Tensor:
    #         idx = idx.tolist()
    #     elif type(idx) == int:
    #         idx = [idx]
    #     num_to_load = len(idx)
    #     node_features = torch.zeros( size=(num_to_load, self.num_nodes, self.num_node_features), dtype=self.dtype, device=self.device )
    #     edge_features = torch.zeros( size=(num_to_load, self.num_nodes, self.num_nodes, self.num_edge_features), dtype=self.dtype, device=self.device )
    #     h = torch.zeros( size=(num_to_load, self.num_nodes, self.num_h_features), dtype=self.dtype, device=self.device )
    #     J = torch.zeros( size=(num_to_load, self.num_nodes, self.num_nodes, self.num_J_features), dtype=self.dtype, device=self.device )
    #     for example_index in idx:
    #         example_file = os.path.join(self.file_directory, f'example_{example_index}_{self.file_suffix}.pt')
    #         node_features[example_index,:,:], edge_features[example_index,:,:,:], h[example_index,:,:], J[example_index,:,:,:] = torch.load(f=example_file)
    #     return node_features, edge_features, h, J

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

def train_model(training_data_set:StructParamsFileDataset, validation_data_set:StructParamsFileDataset, g2gcnn_model_file:str, num_epochs:int=1000, optimizer_name:str='Adam', rep_dims:int=7, mlp_hidden_layers:int=3, graph_convolution_layers:int=3, batch_size:int=10, learning_rate:torch.float=0.001, save_model:bool=False, patience:int=10, improvement_threshold:torch.float=0.0001):
    dtype = training_data_set.dtype
    device = training_data_set.device
    num_nodes = training_data_set.num_nodes
    num_node_features = training_data_set.num_node_features
    num_edge_features = training_data_set.num_edge_features
    num_h_features = training_data_set.num_h_features
    num_J_features = training_data_set.num_J_features
    training_data_loader = DataLoader(dataset=training_data_set, shuffle=True, batch_size=batch_size)
    validation_data_loader = DataLoader(dataset=validation_data_set, shuffle=False, batch_size=batch_size)
    g2gcnn_model = UniformGraph2GraphCNN(num_node_in_features=num_node_features, num_edge_in_features=num_edge_features, num_node_out_features=num_h_features, num_edge_out_features=num_J_features, mlp_hidden_width=rep_dims, num_node_mlp_hidden_layers=mlp_hidden_layers, num_edge_mlp_hidden_layers=mlp_hidden_layers, num_graph_message_passes=graph_convolution_layers, dtype=dtype, device=device)
    loss_fn = GraphMSELoss(num_nodes=num_nodes, dtype=dtype, device=device)
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD( params=g2gcnn_model.parameters(), lr=learning_rate )
    else:
        optimizer = torch.optim.Adam( params=g2gcnn_model.parameters(), lr=learning_rate )
    num_no_improvement_epochs = 0
    last_training_loss = 10e10
    last_validation_loss = 10e10
    print(f'time {time.time() - code_start_time:.3f}, starting training G2GCNN with {num_node_features} input node features, {num_edge_features} input edge features, {rep_dims} latent representaion dimensions, {mlp_hidden_layers} hidden layers per MLP, and {graph_convolution_layers} graph convolutions, batch size {batch_size}, learning rate {learning_rate:.3g}, optimizer {optimizer_name}.')
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
            training_loss_diff = last_training_loss - training_loss
            validation_loss_diff = last_validation_loss - validation_loss
            num_no_improvement_epochs += int( (training_loss_diff < improvement_threshold) and (validation_loss_diff < improvement_threshold) )
            last_training_loss = training_loss
            last_validation_loss = validation_loss
            print(f'time {time.time() - code_start_time:.3f}, epoch {epoch+1}, training loss {training_loss:.3g}, change {training_loss_diff:.3g}, validation loss {validation_loss:.3g}, change {validation_loss_diff:.3g}, num no-improvement epochs {num_no_improvement_epochs}')
            if num_no_improvement_epochs >= patience:
                print(f'patience exceeded, moving on...')
                break
            if math.isnan(training_loss) or math.isnan(validation_loss):
                print('encountered NaN, moving on...')
                break
    if save_model:
        torch.save(obj=g2gcnn_model, f=g2gcnn_model_file)
        print(f'time {time.time() - code_start_time:.3f}, saved {g2gcnn_model_file}')
    # Use item() here just to make sure these get detached from the optimization, backpropagation, etc.,
    # allowing us to free up the CUDA memory of everything allocated inside the function scope.
    return training_loss, validation_loss

# Load the training and validation data.
training_data_set = StructParamsFileDataset(file_directory=input_directory, file_suffix=f'{file_name_fragment}_training_example', num_nodes=num_nodes, num_node_features=num_node_features, num_edge_features=num_edge_features, num_h_features=num_h_features, num_J_features=num_J_features, dtype=float_type, device=device, num_examples=num_training_examples)
validation_data_set = StructParamsFileDataset(file_directory=input_directory, file_suffix=f'{file_name_fragment}_validation_example', num_nodes=num_nodes, num_node_features=num_node_features, num_edge_features=num_edge_features, num_h_features=num_h_features, num_J_features=num_J_features, dtype=float_type, device=device, num_examples=num_validation_examples)
# Check whether we have already created the results pickle file in a previous run.
more_complete_output_file_name_fragment = f'{output_file_name_fragment}_{file_name_fragment}_epochs_{num_epochs}_patience_{patience}_min_improvement_{min_improvement:.3g}'
results_file = os.path.join(output_directory, f'loss_table_{more_complete_output_file_name_fragment}.df')
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
    g2gcnn_model_file = os.path.join(output_directory, f'g2gcnn_{more_complete_output_file_name_fragment}_opt_{optimizer_name}_gconv_{graph_convolution_layers}_mlp_{mlp_hidden_layers}_rep_{rep_dims}_batch_sz_{batch_size}_lr_{learning_rate:.3g}.pt')
    training_mse, validation_mse = train_model(training_data_set=training_data_set, validation_data_set=validation_data_set, g2gcnn_model_file=g2gcnn_model_file, num_epochs=num_epochs, optimizer_name=optimizer_name, rep_dims=rep_dims, mlp_hidden_layers=mlp_hidden_layers, graph_convolution_layers=graph_convolution_layers, batch_size=batch_size, learning_rate=learning_rate, save_model=save_models, patience=patience, improvement_threshold=min_improvement)
    losses_df.at[condition_index,'training_rmse'] = math.sqrt(training_mse)
    losses_df.at[condition_index,'validation_rmse'] = math.sqrt(validation_mse)
    losses_df.at[condition_index,'time'] = time.time() - training_start_time
    losses_df.to_pickle(path=results_file)
    print(f'{time.time()-code_start_time:.3f}, saved {results_file}')
print(f'time {time.time() - code_start_time:.3f}, done')