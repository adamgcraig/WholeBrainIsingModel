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
parser.add_argument("-g", "--num_permutations", type=int, default=100, help="number of permutations to use in each cross validation")
parser.add_argument("-i", "--num_training_subjects", type=int, default=670, help="number of subjects to use when fitting the model")
parser.add_argument("-j", "--num_hidden_layers", type=int, default=0, help="number of hidden layers in MLP")
parser.add_argument("-k", "--hidden_layer_width", type=int, default=1, help="width of each hidden layer in MLP")
parser.add_argument("-l", "--num_epochs", type=int, default=10000, help="number of epochs for which to train the MLP")
parser.add_argument("-m", "--batch_size", type=int, default=670, help="batch size")
parser.add_argument("-n", "--learning_rate", type=float, default=0.0001, help="learning rate")
parser.add_argument("-o", "--output_file_name_part", type=str, default='linear_model_compare', help="a string to include in output file names")
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
num_hidden_layers = args.num_hidden_layers
print(f'num_hidden_layers={num_hidden_layers}')
hidden_layer_width = args.hidden_layer_width
print(f'hidden_layer_width={hidden_layer_width}')
num_epochs = args.num_epochs
print(f'num_epochs={num_epochs}')
batch_size = args.batch_size
print(f'batch_size={batch_size}')
learning_rate = args.learning_rate
print(f'learning_rate={learning_rate}')
output_file_name_part = args.output_file_name_part
print(f'output_file_name_part={output_file_name_part}')

def count_nans(mat:torch.Tensor):
    return torch.count_nonzero( torch.isnan(mat) ).item()

def save_and_print(obj:torch.Tensor, file_name_part:str):
    file_name = os.path.join(output_directory, f'{file_name_part}.pt')
    torch.save(obj=obj, f=file_name)
    num_nan = torch.count_nonzero( torch.isnan(obj) ).item()
    print( f'time {time.time()-code_start_time:.3f}, saved {file_name}, size', obj.size(), f'num NaN {num_nan}, min {obj.min().item():.3g} mean {obj.mean(dtype=torch.float).item():.3g} max {obj.max().item():.3g}' )
    return 0

def split_and_z_score_data(independent:torch.Tensor, dependent:torch.Tensor):
    permutation = torch.randperm( n=independent.size(dim=0), dtype=int_type, device=independent.device )
    train_indices = permutation[:num_training_subjects]
    test_indices = permutation[num_training_subjects:]
    indep_train = independent[train_indices,:]
    indep_std, indep_mean = torch.std_mean(input=indep_train, dim=0, keepdim=True)
    indep_std[indep_std  < min_std] = 1.0
    indep_train = (indep_train - indep_mean)/indep_std
    dep_train = dependent[train_indices,:]
    dep_std, dep_mean = torch.std_mean(input=dep_train, dim=0, keepdim=True)
    dep_std[dep_std  < min_std] = 1.0
    dep_train = (dep_train - dep_mean)/dep_std
    indep_test = (independent[test_indices,:] - indep_mean)/indep_std
    dep_test = (dependent[test_indices,:] - dep_mean)/dep_std
    # print( 'indep_train size', indep_train.size(), 'mean', indep_train.mean(dim=0), 'SD', indep_train.std(dim=0) )
    # print( 'dep_train size', dep_train.size(), 'mean', dep_train.mean(dim=0), 'SD', dep_train.std(dim=0) )
    # print( 'indep_test size', indep_test.size(), 'mean', indep_test.mean(dim=0), 'SD', indep_test.std(dim=0) )
    # print( 'dep_test size', dep_test.size(), 'mean', dep_test.mean(dim=0), 'SD', dep_test.std(dim=0) )
    return indep_train, dep_train, indep_test, dep_test

def append_ones(mat:torch.Tensor):
    return torch.cat(    tensors=(   mat, torch.ones(  size=( mat.size(dim=0), 1 ), dtype=mat.dtype, device=mat.device  )   ), dim=-1    )

def do_train_test_splits_lstsq(independent:torch.Tensor, dependent:torch.Tensor):
    model_dtype = independent.dtype
    model_device = independent.device
    opt_zeros = torch.full( size=(num_permutations,), fill_value=torch.inf, dtype=model_dtype, device=model_device )
    rmse_train = opt_zeros
    rmse_test = opt_zeros.clone()
    for perm_index in range(num_permutations):
        indep_train, dep_train, indep_test, dep_test = split_and_z_score_data(independent=independent, dependent=dependent)
        indep_train = append_ones(mat=indep_train)
        indep_test = append_ones(mat=indep_test)
        coefficients = torch.linalg.lstsq(indep_train, dep_train).solution
        pred_train = torch.matmul(indep_train, coefficients)
        rtr = isingmodellight.get_pairwise_rmse( mat1=pred_train, mat2=dep_train, dim=0 ).item()
        rmse_train[perm_index] = rtr
        pred_test = torch.matmul(indep_test, coefficients)
        rte = isingmodellight.get_pairwise_rmse( mat1=pred_test, mat2=dep_test, dim=0 ).item()
        rmse_test[perm_index] = rte
        # print(f'NaNs in indep_train {count_nans(indep_train)}, dep_train {count_nans(dep_train)}, indep_test {count_nans(indep_test)}, dep_test {count_nans(dep_test)}, coefficients {count_nans(coefficients)}, pred_train {count_nans(pred_train)}, pred_test {count_nans(pred_test)}')
        print(f'time {time.time()-code_start_time:.3f}, permutation {perm_index+1}, lstsq RMSE training {rtr:.3g}, testing {rte:.3g}')
    return rmse_train, rmse_test

def train_mlp(indep_train:torch.Tensor, dep_train:torch.Tensor, indep_test:torch.Tensor, dep_test:torch.Tensor):
    model_dtype = indep_train.dtype
    model_device = indep_train.device
    num_samples, num_features = indep_train.size()
    num_params = dep_train.size(dim=-1)
    num_batches = num_samples//batch_size
    if num_hidden_layers == 0:
        model = torch.nn.Linear(in_features=num_features, out_features=num_params, dtype=model_dtype, device=model_device)
        # model = ParallelLinear(num_linears=num_models, in_features=num_features, out_features=num_params, dtype=model_dtype, device=model_device)
    else:
        make_linear = lambda index: ( f'linear{index}', torch.nn.Linear(in_features=hidden_layer_width, out_features=hidden_layer_width, dtype=model_dtype, device=model_device) )
        # make_linear = lambda index: ( f'linear{index}', ParallelLinear(num_linears=num_models, in_features=hidden_layer_width, out_features=hidden_layer_width, dtype=model_dtype, device=model_device) )
        make_relu = lambda index: ( f'relu{index}', torch.nn.ReLU() )
        model = torch.nn.Sequential(  OrderedDict( [( 'linearinput', torch.nn.Linear(in_features=num_features, out_features=hidden_layer_width, dtype=model_dtype, device=model_device) ), ( 'reluinput', torch.nn.ReLU() )] + [layer(index) for index in range(num_hidden_layers-1) for layer in (make_linear, make_relu)] + [( 'linearoutput', torch.nn.Linear(in_features=hidden_layer_width, out_features=num_params, dtype=model_dtype, device=model_device) )] )  )
        # model = torch.nn.Sequential(  OrderedDict( [( 'linearinput', ParallelLinear(num_linears=num_models, in_features=num_features, out_features=hidden_layer_width, dtype=model_dtype, device=model_device) ), ( 'reluinput', torch.nn.ReLU() )] + [layer(index) for index in range(num_hidden_layers-1) for layer in (make_linear, make_relu)] + [( 'linearoutput', ParallelLinear(num_linears=num_models, in_features=hidden_layer_width, out_features=num_params, dtype=model_dtype, device=model_device) )] )  )
    opt = torch.optim.Adam( params=model.parameters(), lr=learning_rate )
    msefn = torch.nn.MSELoss()
    opt_zeros = torch.zeros( size=(num_epochs,), dtype=model_dtype, device=model_device )
    rmse_train = opt_zeros
    rmse_test = opt_zeros.clone()
    for epoch_index in range(num_epochs):
        batch_order = torch.randperm(n=num_samples, dtype=int_type, device=model_device)
        batch_start = 0
        batch_end = batch_size
        for _ in range(num_batches):
            batch_indices = batch_order[batch_start:batch_end]
            batch_indep = indep_train[batch_indices,:]
            batch_dep = dep_train[batch_indices,:]
            opt.zero_grad()
            pred_batch = model(batch_indep)
            mse = msefn(batch_dep, pred_batch)
            mse.backward()
            opt.step()
            batch_start += batch_size
            batch_end += batch_size
        rmse_train[epoch_index] = torch.sqrt(  msefn( model(indep_train), dep_train )  ).item()
        rmse_test[epoch_index] = torch.sqrt(  msefn( model(indep_test), dep_test )  ).item()
    # We have one RMSE for each region or region-pair model.
    # Combine these into a single RMSE for each.
    return rmse_train, rmse_test

def do_train_test_splits_mlp(independent:torch.Tensor, dependent:torch.Tensor):
    opt_zeros = torch.full( size=(num_permutations, num_epochs), fill_value=torch.inf, dtype=independent.dtype, device=independent.device )
    rmse_train = opt_zeros
    rmse_test = opt_zeros.clone()
    for perm_index in range(num_permutations):
        indep_train, dep_train, indep_test, dep_test = split_and_z_score_data(independent=independent, dependent=dependent)
        rtr, rte = train_mlp(indep_train=indep_train, dep_train=dep_train, indep_test=indep_test, dep_test=dep_test)
        rmse_train[perm_index,:] = rtr
        rmse_test[perm_index,:] = rte
        print(f'time {time.time()-code_start_time:.3f}, permutation {perm_index+1}, MLP RMSE training {rtr[-1]:.3g} testing {rte[-1]:.3g}')
    return rmse_train, rmse_test
    
def get_region_features(num_region_features:int=4):
    # Transpose so that dim 0 is regions and dim 1 is subjects.
    # We will take one correlation over inter-subject differences for each region.
    # clone() so that we can deallocate the larger Tensor of which it is a view.
    # region_features already has a feature dimension, so we do not need to unsqueeze():
    # region or region pair (batch dimension) x subject (observation dimension) x feature (feature dimension)
    region_feature_file_name = os.path.join(data_directory, f'{region_feature_file_name_part}.pt')
    region_features = torch.transpose( input=torch.load(f=region_feature_file_name, weights_only=False)[:,:,:num_region_features], dim0=0, dim1=1 ).clone()
    print( f'time {time.time()-code_start_time:.3f}, loaded {region_feature_file_name} region features size', region_features.size() )
    return region_features
    
def get_mean_state():
    # Take the mean over scans.
    # Transpose so that dim 0 is regions and dim 1 is subjects.
    # We will take one correlation over inter-subject differences for each region.
    # unsqueeze() because, for least squares regression, both independent and dependent need to have the same number of dimensions:
    # region or region pair (batch dimension) x subject (observation dimension) x feature (feature dimension)
    mean_state_file_name = os.path.join(data_directory, f'mean_state_{data_file_name_part}.pt')
    mean_state = torch.mean( input=torch.load(f=mean_state_file_name, weights_only=False), dim=0 ).transpose(dim0=0, dim1=1).unsqueeze(dim=-1)
    print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_file_name} mean state size', mean_state.size() )
    return mean_state
    
def get_h():
    # Take the mean over replicas.
    # Transpose so that dim 0 is regions and dim 1 is subjects.
    # We will take one correlation over inter-subject differences for each region.
    # unsqueeze() because, for least squares regression, both independent and dependent need to have the same number of dimensions:
    # region or region pair (batch dimension) x subject (observation dimension) x feature (feature dimension)
    model_file_name = os.path.join(data_directory, f'{model_file_name_part}.pt')
    h = torch.mean( input=torch.load(f=model_file_name, weights_only=False).h, dim=0 ).transpose(dim0=0, dim1=1).unsqueeze(dim=-1)
    print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name} h size', h.size() )
    return h
    
def get_sc():
    # Transpose so that dim 0 is region pairs and dim 1 is subjects.
    # We will take one correlation over inter-subject differences for each region pair.
    # clone() so that we can de-allocate the larger Tensor of which this is a view.
    # unsqueeze() because, for least squares regression, both independent and dependent need to have the same number of dimensions:
    # region or region pair (batch dimension) x subject (observation dimension) x feature (feature dimension)
    region_pair_feature_file_name = os.path.join(data_directory, f'{region_pair_feature_file_name_part}.pt')
    sc = torch.transpose( input=torch.load(f=region_pair_feature_file_name, weights_only=False)[:,:,0], dim0=0, dim1=1 ).unsqueeze(dim=-1).clone()
    print( f'time {time.time()-code_start_time:.3f}, loaded {region_pair_feature_file_name} SC size', sc.size() )
    return sc
    
def get_J():
    # Take the part above the diagonal, and then take the mean over replicas.
    # This gives us a smaller Tensor with which to work.
    # Transpose so that dim 0 is region pairs and dim 1 is subjects.
    # We will take one correlation over inter-subject differences for each region pair.
    # unsqueeze() because, for least squares regression, both independent and dependent need to have the same number of dimensions:
    # region or region pair (batch dimension) x subject (observation dimension) x feature (feature dimension)
    model_file_name = os.path.join(data_directory, f'{model_file_name_part}.pt')
    model = torch.load(f=model_file_name, weights_only=False)
    triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=model.J.size(dim=-1), device=model.J.device )
    J = torch.mean(input=model.J[:,:,triu_rows,triu_cols], dim=0).transpose(dim0=0, dim1=1).unsqueeze(dim=-1)
    print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name} J size', J.size() )
    return J
    
def get_fc():
    # Before computing the FC, pool subject data from individual scans with mean().
    # After computing the FC, take the part above the diagonal.
    # This gives us a smaller Tensor with which to work.
    # Transpose so that dim 0 is region pairs and dim 1 is subjects.
    # We will take one correlation over inter-subject differences for each region pair.
    # clone() so that we can deallocate the larger Tensor of which it is a view.
    mean_state_file_name = os.path.join(data_directory, f'mean_state_{data_file_name_part}.pt')
    mean_state = torch.load(f=mean_state_file_name, weights_only=False)
    print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_file_name}, mean state size', mean_state.size() )
    mean_state_product_file_name = os.path.join(data_directory, f'mean_state_product_{data_file_name_part}.pt')
    mean_state_product = torch.load(f=mean_state_product_file_name, weights_only=False)
    print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_product_file_name}, mean state product size', mean_state_product.size() )
    triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=mean_state_product.size(dim=-1), device=mean_state_product.device )
    fc = torch.transpose(  input=isingmodellight.get_fc( state_mean=torch.mean(mean_state, dim=0), state_product_mean=torch.mean(mean_state_product, dim=0), epsilon=0.0 )[:,triu_rows,triu_cols], dim0=0, dim1=1  ).unsqueeze(dim=-1).clone()
    print( f'time {time.time()-code_start_time:.3f}, computed FC size', fc.size() )
    return fc

def iterate_over_parts_and_save(features:torch.Tensor, feature_name:str, params:torch.Tensor, param_name:str):
    model_dtype = features.dtype
    model_device = features.device
    num_parts = features.size(dim=0)
    lstsq_zeros = torch.zeros( size=(num_parts, num_permutations), dtype=model_dtype, device=model_device )
    lstsq_rmse_train = lstsq_zeros
    lstsq_rmse_test = lstsq_zeros.clone()
    mlp_zeros = torch.zeros( size=(num_parts, num_permutations, num_epochs), dtype=model_dtype, device=model_device )
    mlp_rmse_train = mlp_zeros
    mlp_rmse_test = mlp_zeros.clone()
    for part_index in range(num_parts):
        features_part = features[part_index,:,:]
        params_part = params[part_index,:,:]
        lstsq_rmse_train[part_index,:], lstsq_rmse_test[part_index,:] = do_train_test_splits_lstsq(independent=features_part, dependent=params_part)
        mlp_rmse_train[part_index,:,:], mlp_rmse_test[part_index,:,:] = do_train_test_splits_mlp(independent=features_part, dependent=params_part)
    lstsq_file_name_part = f'rmse_{feature_name}_{param_name}_lstsq_{output_file_name_part}'
    save_and_print(obj=lstsq_rmse_train, file_name_part=f'train_{lstsq_file_name_part}')
    save_and_print(obj=lstsq_rmse_test, file_name_part=f'test_{lstsq_file_name_part}')
    mlp_file_name_part = f'rmse_{feature_name}_{param_name}_mlp_{output_file_name_part}_nhl_{num_hidden_layers}_hlw_{hidden_layer_width}_bs_{batch_size}_lr_{learning_rate:.3g}_perms_{num_permutations}'
    save_and_print(obj=mlp_rmse_train, file_name_part=f'train_{mlp_file_name_part}')
    save_and_print(obj=mlp_rmse_test, file_name_part=f'test_{mlp_file_name_part}')
    return 0

def do_node_models():
    region_features = get_region_features()
    iterate_over_parts_and_save( features=region_features, feature_name='all', params=get_h(), param_name='h' )
    iterate_over_parts_and_save( features=region_features, feature_name='all', params=get_mean_state(), param_name='mean_state' )
    return 0
    
def do_edge_models():
    sc = get_sc()
    iterate_over_parts_and_save( features=sc, feature_name='SC', params=get_J(), param_name='J' )
    iterate_over_parts_and_save( features=sc, feature_name='SC', params=get_fc(), param_name='FC' )
    return 0
    
do_node_models()
do_edge_models()
print('done')