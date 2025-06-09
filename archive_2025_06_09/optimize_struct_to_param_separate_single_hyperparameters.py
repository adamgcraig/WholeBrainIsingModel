import os
import torch
from scipy import stats
import time
import argparse
import isingmodellight
from isingmodellight import IsingModelLight
from collections import OrderedDict
import math

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')
# When we z-score data, we replace any SD < min_std with 1.0 so that the z-scores are 0 instead of Inf or NaN.
min_std = 10e-10

parser = argparse.ArgumentParser(description="Find linear regressions to predict individual differences in Ising model parameters from individual differences in structural features.")
parser.add_argument("-a", "--data_directory", type=str, default='D:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
parser.add_argument("-c", "--region_feature_file_name_part", type=str, default='node_features_all_as_is', help="part of the output file name before .pt")
parser.add_argument("-d", "--region_pair_feature_file_name_part", type=str, default='edge_features_all_as_is', help="part of the output file name before .pt")
parser.add_argument("-e", "--model_file_name_part", type=str, default='ising_model_light_all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps_40000', help="the part of the Ising model file name before .pt.")
parser.add_argument("-f", "--data_file_name_part", type=str, default='all_mean_std_1', help="the data mean state and state product file name after mean_state_ or mean_state_product_ and before .pt.")
parser.add_argument("-g", "--num_permutations", type=int, default=10, help="number of permutations to use in each cross validation")
parser.add_argument("-i", "--num_training_subjects", type=int, default=670, help="number of subjects to use when fitting the model")
parser.add_argument("-j", "--num_validation_subjects", type=int, default=84, help="number of subjects to use when choosing the hyperparameters")
parser.add_argument("-k", "--max_num_hidden_layers", type=int, default=10, help="number of hidden layers in MLP")
parser.add_argument("-l", "--max_hidden_layer_width", type=int, default=10, help="width of each hidden layer in MLP")
parser.add_argument("-m", "--max_epochs", type=int, default=10000, help="maximum number of epochs for which to train the MLP")
parser.add_argument("-n", "--min_improvement", type=float, default=-1000.0, help="minimum improvement in validation RMSE we need to see in order to keep optimizing")
parser.add_argument("-o", "--epochs_per_validation", type=int, default=100, help="number of optimizer epochs between tests for improvement in validation RMSE")
parser.add_argument("-p", "--batch_size_increment", type=int, default=67, help="Batch sizes that we try will be multiples of this.")
parser.add_argument("-q", "--min_learning_rate_power", type=float, default=2.0, help="The largest learning rate that we try will be 10^-this.")
parser.add_argument("-r", "--max_learning_rate_power", type=float, default=6.0, help="The smallest learning rate that we try will be 10^-this.")
parser.add_argument("-s", "--output_file_name_part", type=str, default='ising_model', help="a string to include in output file names")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
region_feature_file_name_part = args.region_feature_file_name_part
print(f'region_feature_file_name_part={region_feature_file_name_part}')
region_pair_feature_file_name_part = args.region_pair_feature_file_name_part
print(f'region_pair_feature_file_name_part={region_pair_feature_file_name_part}')
model_file_name_part = args.model_file_name_part
print(f'model_file_name_part={model_file_name_part}')
data_file_name_part = args.data_file_name_part
print(f'data_file_name_part={data_file_name_part}')
num_permutations = args.num_permutations
print(f'num_permutations={num_permutations}')
num_training_subjects = args.num_training_subjects
print(f'num_training_subjects={num_training_subjects}')
num_validation_subjects = args.num_validation_subjects
print(f'num_validation_subjects={num_validation_subjects}')
num_non_testing_subjects = num_training_subjects + num_validation_subjects
max_num_hidden_layers = args.max_num_hidden_layers
print(f'max_num_hidden_layers={max_num_hidden_layers}')
max_hidden_layer_width = args.max_hidden_layer_width
print(f'max_hidden_layer_width={max_hidden_layer_width}')
max_epochs = args.max_epochs
print(f'max_epochs={max_epochs}')
min_improvement = args.min_improvement
print(f'min_improvement={min_improvement}')
epochs_per_validation = args.epochs_per_validation
print(f'epochs_per_validation={epochs_per_validation}')
batch_size_increment = args.batch_size_increment
print(f'batch_size_increment={batch_size_increment}')
min_learning_rate_power = args.min_learning_rate_power
print(f'min_learning_rate_power={min_learning_rate_power}')
max_learning_rate_power = args.max_learning_rate_power
print(f'max_learning_rate_power={max_learning_rate_power}')
output_file_name_part = args.output_file_name_part
print(f'output_file_name_part={output_file_name_part}')

hidden_layer_counts = torch.tensor(data=[0, 10, 100, 1000], dtype=int_type, device=device)
num_hidden_layer_counts = hidden_layer_counts.numel()
hidden_layer_widths = torch.tensor(data=[1, 10, 100, 1000], dtype=int_type, device=device)
num_hidden_layer_widths = hidden_layer_widths.numel()
batch_sizes = torch.tensor(data=[10, 134, 670], dtype=int_type, device=device)
num_batch_sizes = batch_sizes.numel()
learning_rates = torch.tensor(data=[0.1, 0.001, 0.00001], dtype=float_type, device=device)
num_learning_rates = learning_rates.numel()

def get_rectangular_mlp(in_features:int, out_features:int, num_hidden_layers:int, hidden_layer_width:int, dtype, device):
    if num_hidden_layers == 0:
        model = torch.nn.Linear(in_features=in_features, out_features=out_features, dtype=dtype, device=device)
    else:
        make_linear = lambda index: ( f'linear{index}', torch.nn.Linear(in_features=hidden_layer_width, out_features=hidden_layer_width, dtype=dtype, device=device) )
        make_relu = lambda index: ( f'relu{index}', torch.nn.ReLU() )
        model = torch.nn.Sequential(  OrderedDict( [( 'linearinput', torch.nn.Linear(in_features=in_features, out_features=hidden_layer_width, dtype=dtype, device=device) ), ( 'reluinput', torch.nn.ReLU() )] + [layer(index) for index in range(num_hidden_layers-1) for layer in (make_linear, make_relu)] + [( 'linearoutput', torch.nn.Linear(in_features=hidden_layer_width, out_features=out_features, dtype=dtype, device=device) )] )  )
    return model

class GraphLayer(torch.nn.Module):
    # This model takes two inputs.
    # w_ij (num_subjects, num_nodes, num_nodes)
    # x_i (num_subjects, num_nodes, num_in_features)
    # It produces two outputs.
    # y_ij = f( x_i, w_ij, x_j ), z_i = g( x_i, w_i1, w_i2, ..., w_iN )
    # y_ij (num_subjects, num_nodes, num_nodes)
    # z_i (num_subjects, num_nodes, num_out_features)
    def __init__(self, num_nodes:int, in_features:int, out_features:int, num_hidden_layers_f:int, num_hidden_layers_g:int, hidden_layer_width_f:int, hidden_layer_width_g:int, dtype, device):
        super(GraphLayer, self).__init__()
        self.in_features = in_features
        self.num_nodes = num_nodes
        in_features_f = 2*in_features + 1
        out_features_f = 1
        self.f = get_rectangular_mlp( in_features=in_features_f, out_features=out_features_f, num_hidden_layers=num_hidden_layers_f, hidden_layer_width=hidden_layer_width_f, dtype=dtype, device=device )
        in_features_g = in_features + num_nodes
        out_features_g = out_features
        self.g = get_rectangular_mlp( in_features=in_features_g, out_features=out_features_g, num_hidden_layers=num_hidden_layers_g, hidden_layer_width=hidden_layer_width_g, dtype=dtype, device=device )
        return
    def forward(self, input:torch.Tensor):
        w, x = torch.split( tensor=input, split_size_or_sections=[self.num_nodes, self.in_features], dim=-1 )
        # left x_i (num_subjects, num_nodes, num_in_features) -> (num_subjects, 1, num_nodes, num_in_features) -> (num_subjects, num_nodes, num_nodes, num_in_features)
        # right x_j (num_subjects, num_nodes, num_in_features) -> (num_subjects, num_nodes, 1, num_in_features) -> (num_subjects, num_nodes, num_nodes, num_in_features)
        # w_ij (num_subjects, num_nodes, num_nodes) -> (num_subjects, num_nodes, num_nodes, 1)
        # g(.) (num_subjects, num_nodes, num_nodes, 1) -> (num_subjects, num_nodes, num_nodes)
        input_f = torch.cat(   (  x.unsqueeze(dim=-3).repeat( (1,self.num_nodes,1,1) ), x.unsqueeze(dim=-2).repeat( (1,1,self.num_nodes,1) ), w.unsqueeze(dim=-1)  ), dim=-1   )
        output_f = torch.squeeze( input=self.f(input_f), dim=-1 )
        return torch.cat(  tensors=( output_f, self.g(input) ), dim=-1  )

def get_stacked_mlp_layers(num_nodes:int, in_features:int, out_features:int, num_hidden_layers_f:int, num_hidden_layers_g:int, hidden_layer_width_f:int, hidden_layer_width_g:int, stack_height:int, hidden_features:int, dtype, device):
    make_layer = lambda top_in, top_out: GraphLayer(num_nodes=num_nodes, in_features=top_in, out_features=top_out, num_hidden_layers_f=num_hidden_layers_f, num_hidden_layers_g=num_hidden_layers_g, hidden_layer_width_f=hidden_layer_width_f, hidden_layer_width_g=hidden_layer_width_g, dtype=dtype, device=device)
    if stack_height <= 1:
        model = make_layer(top_in=in_features, top_out=out_features)
    else:
        model = torch.nn.Sequential(   OrderedDict(  [( 'inputgraph', make_layer(top_in=in_features, top_out=hidden_features) )] + [( f'hiddengraph{n}', make_layer(top_in=hidden_features, top_out=hidden_features) ) for n in range(stack_height-2)] + [( 'outputgraph', make_layer(top_in=hidden_features, top_out=out_features) )]  )   )
    return model

def save_and_print(obj:torch.Tensor, file_name_part:str):
    file_name = os.path.join(output_directory, f'{file_name_part}.pt')
    torch.save(obj=obj, f=file_name)
    num_nan = torch.count_nonzero( torch.isnan(obj) ).item()
    print( f'time {time.time()-code_start_time:.3f}, saved {file_name}, size', obj.size(), f'num NaN {num_nan}, min {obj.min().item():.3g} mean {obj.mean(dtype=torch.float).item():.3g} max {obj.max().item():.3g}' )
    return 0

def split_and_z_score_data(independent:torch.Tensor, dependent:torch.Tensor):
    permutation = torch.randperm( n=independent.size(dim=0), dtype=int_type, device=independent.device )
    train_indices = permutation[:num_training_subjects]
    validate_indices = permutation[num_training_subjects:num_non_testing_subjects]
    test_indices = permutation[num_non_testing_subjects:]
    indep_train = independent[train_indices,:,:]
    indep_std, indep_mean = torch.std_mean(input=indep_train, dim=0, keepdim=True)
    indep_std[indep_std  < min_std] = 1.0
    indep_train = (indep_train - indep_mean)/indep_std
    dep_train = dependent[train_indices,:,:]
    dep_std, dep_mean = torch.std_mean(input=dep_train, dim=0, keepdim=True)
    dep_std[dep_std  < min_std] = 1.0
    dep_train = (dep_train - dep_mean)/dep_std
    indep_validate = (independent[validate_indices,:,:] - indep_mean)/indep_std
    dep_validate = (dependent[validate_indices,:,:] - dep_mean)/dep_std
    indep_test = (independent[test_indices,:,:] - indep_mean)/indep_std
    dep_test = (dependent[test_indices,:,:] - dep_mean)/dep_std
    return indep_train, dep_train, indep_validate, dep_validate, indep_test, dep_test

def train_mlp(features_train:torch.Tensor, params_train:torch.Tensor, features_validate:torch.Tensor, params_validate:torch.Tensor, num_hidden_layers:int=10, num_hidden_features:int=10, batch_size:int=670, learning_rate:float=0.001):
    model_dtype = features_train.dtype
    model_device = features_train.device
    num_samples, _, num_features = features_train.size()
    num_pair_params = params_train.size(dim=-1)
    num_batches = num_samples//batch_size
    # model = get_stacked_mlp_layers(num_nodes=num_nodes, in_features=num_nodes_plus_num_features-num_nodes, out_features=num_nodes_plus_num_params-num_nodes, num_hidden_layers_f=num_mlp_layers_f, num_hidden_layers_g=num_mlp_layers_g, hidden_layer_width_f=num_hidden_features_f, hidden_layer_width_g=num_hidden_features_g, stack_height=num_graph_layers, hidden_features=num_graph_hidden_features, dtype=model_dtype, device=model_device)
    model = get_rectangular_mlp(in_features=num_features, out_features=num_pair_params, num_hidden_layers=num_hidden_layers, hidden_layer_width=num_hidden_features, dtype=model_dtype, device=model_device)
    opt = torch.optim.Adam( params=model.parameters(), lr=learning_rate )
    msefn = torch.nn.MSELoss()
    rmse_train = torch.sqrt(  msefn( model(features_train), params_train )  ).item()
    rmse_validate = torch.sqrt(  msefn( model(features_validate), params_validate )  ).item()
    old_rmse = rmse_validate
    improvement = 0.0
    total_epochs = 0
    num_validation_checks = max_epochs//epochs_per_validation
    for _ in range(num_validation_checks):
        for _ in range(epochs_per_validation):
            batch_order = torch.randperm(n=num_samples, dtype=int_type, device=model_device)
            batch_start = 0
            batch_end = batch_size
            for _ in range(num_batches):
                batch_indices = batch_order[batch_start:batch_end]
                batch_sc_and_features = features_train[batch_indices,:,:]
                batch_J_and_h = params_train[batch_indices,:,:]
                opt.zero_grad()
                pred_J_and_h = model(batch_sc_and_features)
                mse = msefn(pred_J_and_h, batch_J_and_h)
                mse.backward()
                opt.step()
                batch_start += batch_size
                batch_end += batch_size
        total_epochs += epochs_per_validation
        rmse_train = torch.sqrt(  msefn( model(features_train), params_train )  ).item()
        rmse_validate = torch.sqrt(  msefn( model(features_validate), params_validate )  ).item()
        improvement = old_rmse - rmse_validate
        print(f'time {time.time()-code_start_time:.3f}, num epochs {total_epochs}, RMSE training {rmse_train:.3g} validation {rmse_validate:.3g}, validation improvement {improvement:.3g}')
        if improvement < min_improvement:
            break
        old_rmse = rmse_validate
    print(f'time {time.time()-code_start_time:.3f}, num epochs {total_epochs}, over last {epochs_per_validation} epochs, improvement {improvement:.3g} was less than {min_improvement:.3g}, so we will stop.')
    # We have one RMSE for each region or region-pair model.
    # Combine these into a single RMSE for each.
    return total_epochs, rmse_train, rmse_validate, model

def do_hyperparameter_optimization(features:torch.Tensor, params:torch.Tensor):
    model_dtype = features.dtype
    model_device = features.device
    opt_size = (num_hidden_layer_counts, num_hidden_layer_widths, num_batch_sizes, num_learning_rates, num_permutations)
    total_epochs = torch.zeros(size=opt_size, dtype=int_type, device=model_device)
    rmse_train = torch.full(size=opt_size, fill_value=torch.inf, dtype=model_dtype, device=model_device)
    rmse_validate = torch.full( size=opt_size, fill_value=torch.inf, dtype=model_dtype, device=model_device )
    for hlc_index in range(num_hidden_layer_counts):
        new_num_hidden_layers = hidden_layer_counts[hlc_index].item()
        for hlw_index in range(num_hidden_layer_widths):
            new_hidden_layer_width = hidden_layer_widths[hlw_index].item()
            for bs_index in range(num_batch_sizes):
                new_batch_size = batch_sizes[bs_index].item()
                for lr_index in range(num_learning_rates):
                    new_learning_rate = learning_rates[lr_index].item()
                    for perm_index in range(num_permutations):
                        features_train, params_train, features_validate, params_validate, features_test, params_test = split_and_z_score_data(independent=features, dependent=params)
                        print(f'graph hidden layers number {new_num_hidden_layers}, width {new_hidden_layer_width}, batch size {new_batch_size}, learning rate {new_learning_rate:.3g}, permutation {perm_index+1}')
                        try:
                            new_total_epochs, new_rmse_train, new_rmse_validate, _ = train_mlp(features_train=features_train, params_train=params_train, features_validate=features_validate, params_validate=params_validate, num_hidden_layers=new_num_hidden_layers, num_hidden_features=new_hidden_layer_width, batch_size=new_batch_size, learning_rate=new_learning_rate)
                            print(f'num epochs {new_total_epochs}, RMSE training {new_rmse_train:.3g}, validation {new_rmse_validate:.3g}')
                            total_epochs[hlc_index, hlw_index, bs_index, lr_index, perm_index] = new_total_epochs
                            rmse_train[hlc_index, hlw_index, bs_index, lr_index, perm_index] = new_rmse_train
                            rmse_validate[hlc_index, hlw_index, bs_index, lr_index,perm_index] = new_rmse_validate
                        except Exception as e:
                            print('threw an exception')
                            print(e)
    rmse_validate_mean = rmse_validate.mean(dim=-1)
    min_rmse_validate_over_lr, best_lr_indices = rmse_validate_mean.min(dim=-1)
    min_rmse_validate_over_bs, best_bs_indices = min_rmse_validate_over_lr.min(dim=-1)
    min_rmse_validate_over_graph_hlw, best_graph_hlw_indices = min_rmse_validate_over_bs.min(dim=-1)
    best_hlc_index = min_rmse_validate_over_graph_hlw.argmin().item()
    best_hlw_index = best_graph_hlw_indices[best_hlc_index].item()
    best_bs_index = best_bs_indices[best_hlc_index, best_hlw_index].item()
    best_lr_index = best_lr_indices[best_hlc_index, best_hlw_index, best_bs_index].item()
    best_num_hidden_layers = hidden_layer_counts[best_hlc_index].item()
    best_hidden_layer_width = hidden_layer_widths[best_hlw_index].item()
    best_batch_size = batch_sizes[best_bs_index].item()
    best_learning_rate = learning_rates[best_lr_index].item()
    min_rmse_train = rmse_train[best_hlc_index, best_hlw_index, best_bs_index,best_lr_index,:].mean().item()
    min_rmse_validate = rmse_validate_mean[best_hlc_index, best_hlw_index, best_bs_index,best_lr_index].item()
    print(f'best hyperparameters num hidden layers {best_num_hidden_layers}, hidden layer width {best_hidden_layer_width}, batch size {best_batch_size}, learning rate {best_learning_rate:.3g}, RMSE training {min_rmse_train:.3g}, validation {min_rmse_validate:.3g}')
    features_train, params_train, features_validate, params_validate, features_test, params_test = split_and_z_score_data(independent=features, dependent=params)
    _, _, _, best_model = train_mlp(features_train=features_train, params_train=params_train, features_validate=features_validate, params_validate=params_validate, num_hidden_layers=new_num_hidden_layers, num_hidden_features=new_hidden_layer_width, batch_size=new_batch_size, learning_rate=new_learning_rate)
    # pred_train = best_model(indep_train)
    best_train_rmse = isingmodellight.get_pairwise_rmse( mat1=best_model(features_train), mat2=params_train, dim=0 )
    # pred_validate = best_model(indep_validate)
    best_validate_rmse = isingmodellight.get_pairwise_rmse( mat1=best_model(features_validate), mat2=params_validate, dim=0 )
    # pred_test = best_model(indep_test)
    best_test_rmse = isingmodellight.get_pairwise_rmse( mat1=best_model(features_test), mat2=params_test, dim=0 )
    print(f'final model RMSE training min {best_train_rmse.min():.3g}, mean {best_train_rmse.mean():.3g}, max {best_train_rmse.max():.3g}, validation min {best_validate_rmse.min():.3g}, mean {best_validate_rmse.mean():.3g}, max {best_validate_rmse.max():.3g}, testing min {best_test_rmse.min():.3g}, mean {best_test_rmse.mean():.3g}, max {best_test_rmse.max():.3g}')
    return rmse_train, rmse_validate, best_num_hidden_layers, best_hidden_layer_width, best_batch_size, best_learning_rate, best_train_rmse, best_validate_rmse, best_test_rmse, best_model

def save_mlp_results(features:torch.Tensor, feature_name:str, params:torch.Tensor, param_name:str):
    rmse_train, rmse_validate, best_num_hidden_layers, best_hidden_layer_width, best_batch_size, best_learning_rate, best_train_rmse, best_validate_rmse, best_test_rmse, best_model = do_hyperparameter_optimization(features=features, params=params)
    all_mlp_file_part = f'{feature_name}_{param_name}_mlp_pairwise_hidden_max_nhl_{max_num_hidden_layers}_max_hlw_{max_hidden_layer_width}_bs_incr_{batch_size_increment}_lr_e_min_{min_learning_rate_power:.3g}_max_{max_learning_rate_power:.3g}_perms_{num_permutations}_{output_file_name_part}'
    save_and_print(obj=rmse_train, file_name_part=f'rmse_train_all_{all_mlp_file_part}')
    save_and_print(obj=rmse_validate, file_name_part=f'rmse_validate_all_{all_mlp_file_part}')
    best_mlp_file_part = f'{feature_name}_{param_name}_mlp_pairwise_hidden_num_{best_num_hidden_layers}_width_{best_hidden_layer_width}_batch_{best_batch_size}_lr_{best_learning_rate:.3g}_perms_{num_permutations}_{output_file_name_part}'
    save_and_print(obj=best_train_rmse, file_name_part=f'rmse_train_{best_mlp_file_part}')
    save_and_print(obj=best_validate_rmse, file_name_part=f'rmse_validate_{best_mlp_file_part}')
    save_and_print(obj=best_test_rmse, file_name_part=f'rmse_test_{best_mlp_file_part}')
    torch.save( obj=best_model, f=os.path.join(output_directory, f'model_{best_mlp_file_part}.pt') )
    return 0

def get_sc_and_features(num_region_features:int=4):
    # sc size (num_subjects, num_node_pairs, 1)
    # region_pairs size (num_subjects, num_nodes, num_node_features)
    region_feature_file_name = os.path.join(data_directory, f'{region_feature_file_name_part}.pt')
    region_features = torch.clone( input=torch.load(f=region_feature_file_name, weights_only=False)[:,:,:num_region_features] )
    print( f'time {time.time()-code_start_time:.3f}, loaded {region_feature_file_name} region features size', region_features.size() )
    region_pair_feature_file_name = os.path.join(data_directory, f'{region_pair_feature_file_name_part}.pt')
    sc = torch.clone(  input=torch.unsqueeze( input=torch.load(f=region_pair_feature_file_name, weights_only=False)[:,:,0], dim=-1 )  )
    print( f'time {time.time()-code_start_time:.3f}, loaded {region_pair_feature_file_name} SC size', sc.size() )
    return sc, region_features

def get_J_and_h():
    # J size (num_subjects, num_node_pairs, 1)
    # h (num_subjects, num_nodes, 1)
    model_file_name = os.path.join(data_directory, f'{model_file_name_part}.pt')
    model = torch.load(f=model_file_name, weights_only=False)
    triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=model.J.size(dim=-1), device=model.J.device )
    J = torch.mean( input=torch.unsqueeze(input=model.J[:,:,triu_rows,triu_cols], dim=-1), dim=0 )
    h = torch.mean( input=torch.unsqueeze(input=model.h, dim=-1), dim=0 )
    return J, h
    
def get_fc_and_mean():
    # fc size (num_subjects, num_node_pairs, 1)
    # mean_state (num_subjects, num_nodes, 1)
    mean_state_file_name = os.path.join(data_directory, f'mean_state_{data_file_name_part}.pt')
    mean_state = torch.mean( input=torch.load(f=mean_state_file_name, weights_only=False), dim=0 )
    print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_file_name}, mean state size', mean_state.size() )
    mean_state_product_file_name = os.path.join(data_directory, f'mean_state_product_{data_file_name_part}.pt')
    mean_state_product = torch.mean( input=torch.load(f=mean_state_product_file_name, weights_only=False), dim=0 )
    print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_product_file_name}, mean state product size', mean_state_product.size() )
    fc = isingmodellight.get_fc(state_mean=mean_state, state_product_mean=mean_state_product, epsilon=0.0)
    print( f'time {time.time()-code_start_time:.3f}, computed FC size', fc.size() )
    return fc.unsqueeze(dim=-1), mean_state.unsqueeze(dim=-1)

def save_region_and_pair_results(region_features:torch.Tensor, sc:torch.Tensor, node_params:torch.Tensor, node_param_name:str, node_pair_params:torch.Tensor, node_pair_param_name:str):
    save_mlp_results(features=region_features, feature_name='region_features', params=node_params, param_name=node_param_name)
    save_mlp_results(features=sc, feature_name='SC', params=node_pair_params, param_name=node_pair_param_name)
    return 0

def save_h_and_J_results(region_features:torch.Tensor, sc:torch.Tensor):
    J, h = get_J_and_h()
    save_region_and_pair_results(region_features=region_features, sc=sc, node_params=h, node_param_name='h', node_pair_params=J, node_pair_param_name='J')
    return 0

def save_mean_and_fc_results(region_features:torch.Tensor, sc:torch.Tensor):
    fc, mean_state = get_J_and_h()
    save_region_and_pair_results(region_features=region_features, sc=sc, node_params=mean_state, node_param_name='mean_state', node_pair_params=fc, node_pair_param_name='FC')
    return 0

sc, region_features = get_sc_and_features()
save_h_and_J_results(region_features=region_features, sc=sc)
save_mean_and_fc_results(region_features=region_features, sc=sc)
print('done')