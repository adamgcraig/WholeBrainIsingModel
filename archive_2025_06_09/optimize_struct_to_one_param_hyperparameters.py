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
parser.add_argument("-m", "--max_epochs", type=int, default=1000, help="maximum number of epochs for which to train the MLP")
parser.add_argument("-n", "--min_improvement", type=float, default=-1000.0, help="minimum improvement in validation RMSE we need to see in order to keep optimizing")
parser.add_argument("-o", "--epochs_per_validation", type=int, default=100, help="number of optimizer epochs between tests for improvement in validation RMSE")
parser.add_argument("-p", "--batch_size_increment", type=int, default=67, help="Batch sizes that we try will be multiples of this.")
parser.add_argument("-q", "--min_learning_rate_power", type=float, default=0.0, help="The largest learning rate that we try will be 10^-this.")
parser.add_argument("-r", "--max_learning_rate_power", type=float, default=5.0, help="The smallest learning rate that we try will be 10^-this.")
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

# hidden_layer_counts = torch.arange(start=0, end=max_hidden_layer_count+1, step=1, dtype=int_type, device=model_device)
hidden_layer_counts = torch.tensor(data=[0, 10, 100], dtype=int_type, device=device)
num_hidden_layer_counts = hidden_layer_counts.numel()
# hidden_layer_widths = torch.arange(start=1, end=max_hidden_layer_width+1, step=1, dtype=int_type, device=model_device)
hidden_layer_widths = torch.tensor(data=[1, 360, 64620], dtype=int_type, device=device)
num_hidden_layer_widths = hidden_layer_widths.numel()
# batch_sizes = torch.arange(start=batch_size_increment, end=num_training_subjects+1, step=batch_size_increment, dtype=int_type, device=model_device)
batch_sizes = torch.tensor(data=[10, 134, 670], dtype=int_type, device=device)
num_batch_sizes = batch_sizes.numel()
# lr_powers = torch.arange( start=-1.0*min_learning_rate_power, end=-1.0*(max_learning_rate_power+1.0), step=-1.0, dtype=float_type, device=model_device )
# lr_tens = torch.full_like(input=lr_powers, fill_value=10.0)
# learning_rates = torch.pow(input=lr_tens, exponent=lr_powers)
learning_rates = torch.tensor(data=[1e-2, 1e-4, 1e-8], dtype=float_type, device=device)
num_learning_rates = learning_rates.numel()

class ParallelLinear(torch.nn.Module):
    # The input should have size (num_linears, num_samples, in_features).
    # The output then has size (num_linears, num_samples, out_features).
    def __init__(self, num_linears:int, in_features:int, out_features:int, dtype, device):
        super(ParallelLinear, self).__init__()
        self.weights = torch.nn.Parameter(  torch.randn( size=(num_linears, in_features, out_features), dtype=dtype, device=device )/math.sqrt(in_features)  )
        self.biases = torch.nn.Parameter(  torch.randn( size=(num_linears, 1, out_features), dtype=dtype, device=device )  )
        return
    def forward(self, input:torch.Tensor):
        return self.biases + torch.matmul(input, self.weights)

def save_and_print(obj:torch.Tensor, file_name_part:str):
    file_name = os.path.join(output_directory, f'{file_name_part}.pt')
    torch.save(obj=obj, f=file_name)
    num_nan = torch.count_nonzero( torch.isnan(obj) ).item()
    print( f'time {time.time()-code_start_time:.3f}, saved {file_name}, size', obj.size(), f'num NaN {num_nan}, min {obj.min().item():.3g} mean {obj.mean(dtype=torch.float).item():.3g} max {obj.max().item():.3g}' )
    return 0

def fit_lstsq(independent:torch.Tensor, dependent:torch.Tensor):
    num_subjects, _ = independent.size()
    num_testing_subjects = num_subjects - num_non_testing_subjects
    model_dtype = independent.dtype
    model_device = independent.device
    indep_train = independent[:num_training_subjects,:]
    indep_std, indep_mean = torch.std_mean(indep_train, dim=0, keepdim=True)
    indep_std[indep_std < min_std] = 1.0
    dep_train = dependent[:num_training_subjects,:]
    dep_std, dep_mean = torch.std_mean(dep_train, dim=0, keepdim=True)
    dep_std[dep_std < min_std] = 1.0
    indep_train = torch.cat(   tensors=(  (indep_train-indep_mean)/indep_std, torch.ones( size=(num_training_subjects, 1), dtype=model_dtype, device=model_device )  ), dim=-1   )
    dep_train = (dep_train - dep_mean)/dep_std
    indep_validate = torch.cat(   tensors=(  (independent[num_training_subjects:num_non_testing_subjects,:] - indep_mean)/indep_std, torch.ones( size=(num_validation_subjects, 1), dtype=model_dtype, device=model_device )  ), dim=-1   )
    dep_validate = (dependent[num_training_subjects:num_non_testing_subjects,:] - dep_mean)/dep_std
    indep_test = torch.cat(   tensors=(  (independent[num_non_testing_subjects:,:] - indep_mean)/indep_std, torch.ones( size=(num_testing_subjects, 1), dtype=model_dtype, device=model_device )  ), dim=-1   )
    dep_test = (dependent[num_non_testing_subjects:,:] - dep_mean)/dep_std
    coefficients = torch.linalg.lstsq(indep_train, dep_train).solution
    pred_train = torch.matmul(indep_train, coefficients)
    rmse_train = isingmodellight.get_pairwise_rmse(mat1=pred_train, mat2=dep_train, dim=0).item()
    pred_validate = torch.matmul(indep_validate, coefficients)
    rmse_validate = isingmodellight.get_pairwise_rmse(mat1=pred_validate, mat2=dep_validate, dim=0).item()
    pred_test = torch.matmul(indep_test, coefficients)
    rmse_test = isingmodellight.get_pairwise_rmse(mat1=pred_test, mat2=dep_test, dim=0).item()
    print(f'time {time.time()-code_start_time:.3f}, least squares linear model RMSE training {rmse_train:.3g}, validation {rmse_validate:.3g}, testing {rmse_test:.3g}')
    return rmse_train, rmse_test, rmse_validate, coefficients

def train_mlp(indep_train:torch.Tensor, dep_train:torch.Tensor, indep_validate:torch.Tensor, dep_validate:torch.Tensor, num_hidden_layers:int=10, hidden_layer_width:int=10, batch_size:int=670, learning_rate:float=0.001):
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
    # pred_train = model(indep_train)
    rmse_train = torch.sqrt(  msefn( model(indep_train), dep_train )  ).item()
    # pred_validate = model(indep_validate)
    rmse_validate = torch.sqrt(  msefn( model(indep_validate), dep_validate )  ).item()
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
                batch_indep = indep_train[batch_indices,:]
                batch_dep = dep_train[batch_indices,:]
                opt.zero_grad()
                pred_batch = model(batch_indep)
                mse = msefn(batch_dep, pred_batch)
                mse.backward()
                opt.step()
                batch_start += batch_size
                batch_end += batch_size
        total_epochs += epochs_per_validation
        # pred_train = model(indep_train)
        rmse_train = torch.sqrt(  msefn( model(indep_train), dep_train )  ).item()
        # pred_validate = model(indep_validate)
        rmse_validate = torch.sqrt(  msefn( model(indep_validate), dep_validate )  ).item()
        improvement = old_rmse - rmse_validate
        print(f'time {time.time()-code_start_time:.3f}, num epochs {total_epochs}, RMSE training {rmse_train:.3g} validation {rmse_validate:.3g}, validation improvement {improvement:.3g}')
        if improvement < min_improvement:
            break
        old_rmse = rmse_validate
    print(f'time {time.time()-code_start_time:.3f}, num epochs {total_epochs}, over last {epochs_per_validation} epochs, improvement {improvement:.3g} was less than {min_improvement:.3g}, so we will stop.')
    # We have one RMSE for each region or region-pair model.
    # Combine these into a single RMSE for each.
    return total_epochs, rmse_train, rmse_validate, model

def do_hyperparameter_optimization(independent:torch.Tensor, dependent:torch.Tensor, max_hidden_layer_count:int, max_hidden_layer_width:int, num_perms:int):
    model_dtype = independent.dtype
    model_device = independent.device
    opt_size = (num_hidden_layer_counts, num_hidden_layer_widths, num_batch_sizes, num_learning_rates, num_perms)
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
                    for perm_index in range(num_perms):
                        permutation = torch.randperm(n=num_non_testing_subjects, dtype=int_type, device=model_device)
                        train_indices = permutation[:num_training_subjects]
                        validate_indices = permutation[num_training_subjects:]
                        indep_train = independent[train_indices,:]
                        indep_std, indep_mean = torch.std_mean(input=indep_train, dim=0, keepdim=True)
                        indep_std[indep_std  < min_std] = 1.0
                        indep_train = (indep_train - indep_mean)/indep_std
                        dep_train = dependent[train_indices,:]
                        dep_std, dep_mean = torch.std_mean(input=dep_train, dim=0, keepdim=True)
                        dep_std[dep_std  < min_std] = 1.0
                        dep_train = (dep_train - dep_mean)/dep_std
                        indep_validate = (independent[validate_indices,:] - indep_mean)/indep_std
                        dep_validate = (dependent[validate_indices,:] - dep_mean)/dep_std
                        print(f'hidden layers {new_num_hidden_layers}, hidden layer width {new_hidden_layer_width}, batch size {new_batch_size}, learning rate {new_learning_rate:.3g}, permutation {perm_index+1}')
                        # new_total_epochs, new_rmse_train, new_rmse_validate, _ = train_mlp(indep_train=indep_train, dep_train=dep_train, indep_validate=indep_validate, dep_validate=dep_validate, num_hidden_layers=new_num_hidden_layers, hidden_layer_width=new_hidden_layer_width, batch_size=new_batch_size, learning_rate=new_learning_rate)
                        # print(f'num epochs {new_total_epochs}, RMSE training {new_rmse_train:.3g}, validation {new_rmse_validate:.3g}')
                        # total_epochs[hlc_index,hlw_index,bs_index,lr_index,perm_index] = new_total_epochs
                        # rmse_train[hlc_index,hlw_index,bs_index,lr_index,perm_index] = new_rmse_train
                        # rmse_validate[hlc_index,hlw_index,bs_index,lr_index,perm_index] = new_rmse_validate
                        try:
                            new_total_epochs, new_rmse_train, new_rmse_validate, _ = train_mlp(indep_train=indep_train, dep_train=dep_train, indep_validate=indep_validate, dep_validate=dep_validate, num_hidden_layers=new_num_hidden_layers, hidden_layer_width=new_hidden_layer_width, batch_size=new_batch_size, learning_rate=new_learning_rate)
                            print(f'num epochs {new_total_epochs}, RMSE training {new_rmse_train:.3g}, validation {new_rmse_validate:.3g}')
                            total_epochs[hlc_index,hlw_index,bs_index,lr_index,perm_index] = new_total_epochs
                            rmse_train[hlc_index,hlw_index,bs_index,lr_index,perm_index] = new_rmse_train
                            rmse_validate[hlc_index,hlw_index,bs_index,lr_index,perm_index] = new_rmse_validate
                        except Exception as e:
                            print('threw an exception')
                            print(e)
    rmse_validate_mean = rmse_validate.mean(dim=-1)
    min_rmse_validate_over_lr, best_lr_indices = rmse_validate_mean.min(dim=-1)
    min_rmse_validate_over_bs, best_bs_indices = min_rmse_validate_over_lr.min(dim=-1)
    min_rmse_validate_over_hlw, best_hlw_indices = min_rmse_validate_over_bs.min(dim=-1)
    min_rmse_validate, best_hlc_index = min_rmse_validate_over_hlw.min(dim=-1)
    best_hlc_index = best_hlc_index.item()
    best_num_hidden_layers = hidden_layer_counts[best_hlc_index].item()
    best_hlw_index = best_hlw_indices[best_hlc_index].item()
    best_hidden_layer_width = hidden_layer_widths[best_hlw_index].item()
    best_bs_index = best_bs_indices[best_hlc_index,best_hlw_index].item()
    best_batch_size = batch_sizes[best_bs_index].item()
    best_lr_index = best_lr_indices[best_hlc_index,best_hlw_index,best_bs_index].item()
    best_learning_rate = learning_rates[best_lr_index].item()
    indep_train = independent[:num_training_subjects,:]
    indep_std, indep_mean = torch.std_mean(indep_train, dim=0, keepdim=True)
    indep_std[indep_std  < min_std] = 1.0
    indep_train = (indep_train - indep_mean)/indep_std
    dep_train = dependent[:num_training_subjects,:]
    dep_std, dep_mean = torch.std_mean(dep_train, dim=0, keepdim=True)
    dep_std[dep_std  < min_std] = 1.0
    dep_train = (dep_train - dep_mean)/dep_std
    indep_validate = (independent[num_training_subjects:num_non_testing_subjects,:] - indep_mean)/indep_std
    dep_validate = (dependent[num_training_subjects:num_non_testing_subjects,:] - dep_mean)/dep_std
    indep_test = (independent[num_non_testing_subjects:,:] - indep_mean)/indep_std
    dep_test = (dependent[num_non_testing_subjects:,:] - dep_mean)/dep_std
    min_rmse_train = rmse_train[best_hlc_index,best_hlw_index,best_bs_index,best_lr_index,:].mean().item()
    print(f'best hyperparameters num hidden layers {best_num_hidden_layers}, hidden layer width {best_hidden_layer_width}, batch size {best_batch_size}, learning rate {best_learning_rate:.3g}, RMSE training {min_rmse_train:.3g}, validation {min_rmse_validate:.3g}')
    best_total_epochs, _, _, best_model = train_mlp(indep_train=indep_train, dep_train=dep_train, indep_validate=indep_validate, dep_validate=dep_validate, num_hidden_layers=best_num_hidden_layers, hidden_layer_width=best_hidden_layer_width, batch_size=best_batch_size, learning_rate=best_learning_rate)
    # pred_train = best_model(indep_train)
    best_train_rmse = isingmodellight.get_pairwise_rmse( mat1=best_model(indep_train), mat2=dep_train, dim=0 ).item()
    # pred_validate = best_model(indep_validate)
    best_validate_rmse = isingmodellight.get_pairwise_rmse( mat1=best_model(indep_validate), mat2=dep_validate, dim=0 ).item()
    # pred_test = best_model(indep_test)
    best_test_rmse = isingmodellight.get_pairwise_rmse( mat1=best_model(indep_test), mat2=dep_test, dim=0 ).item()
    print(f'final model num training epochs {best_total_epochs}, RMSE training {best_train_rmse:.3g}, validation {best_validate_rmse:.3g}, testing {best_test_rmse:.3g}')
    return total_epochs, rmse_train, rmse_validate, best_num_hidden_layers, best_hidden_layer_width, best_batch_size, best_learning_rate, best_train_rmse, best_validate_rmse, best_test_rmse, best_model
    
def save_lstsq_results(independent:torch.Tensor, independent_name:str, dependent:torch.Tensor, dependent_name:str):
    lstsq_rmse_train, lstsq_rmse_test, lstsq_rmse_validate, lstsq_coefficients = fit_lstsq(independent=independent, dependent=dependent)
    lstsq_file_part = f'{independent_name}_{dependent_name}_lstsq_{output_file_name_part}'
    save_and_print(obj=lstsq_coefficients, file_name_part=f'coeffs_{lstsq_file_part}')
    return lstsq_rmse_train, lstsq_rmse_test, lstsq_rmse_validate

def save_mlp_results(independent:torch.Tensor, independent_name:str, dependent:torch.Tensor, dependent_name:str, num_perms:int, max_num_hidden_layers:int, max_hidden_layer_width:int):
    total_epochs, rmse_train, rmse_validate, best_num_hidden_layers, best_hidden_layer_width, best_batch_size, best_learning_rate, best_train_rmse, best_validate_rmse, best_test_rmse, best_model = do_hyperparameter_optimization(independent=independent, dependent=dependent, max_hidden_layer_count=max_num_hidden_layers, max_hidden_layer_width=max_hidden_layer_width, num_perms=num_perms)
    mlp_file_part = f'{independent_name}_{dependent_name}_mlp_hidden_num_{best_num_hidden_layers}_width_{best_hidden_layer_width}_batch_{best_batch_size}_lr_{best_learning_rate:.3g}_perms_{num_perms}_{output_file_name_part}'
    save_and_print(obj=total_epochs, file_name_part=f'num_epochs_{mlp_file_part}')
    save_and_print(obj=rmse_train, file_name_part=f'rmse_train_all_{mlp_file_part}')
    save_and_print(obj=rmse_validate, file_name_part=f'rmse_validate_all_{mlp_file_part}')
    torch.save( obj=best_model, f=os.path.join(output_directory, f'model_{mlp_file_part}.pt') )
    return best_train_rmse, best_validate_rmse, best_test_rmse

def save_optimization_results(independent:torch.Tensor, independent_name:str, dependent:torch.Tensor, dependent_name:str, num_perms:int, max_num_hidden_layers:int, max_hidden_layer_width:int):
    lstsq_rmse_train, lstsq_rmse_test, lstsq_rmse_validate = save_lstsq_results(independent=independent, independent_name=independent_name, dependent=dependent, dependent_name=dependent_name)
    best_train_rmse, best_validate_rmse, best_test_rmse = save_mlp_results(independent=independent, independent_name=independent_name, dependent=dependent, dependent_name=dependent_name, num_perms=num_perms, max_num_hidden_layers=max_num_hidden_layers, max_hidden_layer_width=max_hidden_layer_width)
    return lstsq_rmse_train, lstsq_rmse_test, lstsq_rmse_validate, best_train_rmse, best_validate_rmse, best_test_rmse
    
def get_region_features(num_region_features:int=4):
    # Transpose so that dim 0 is regions and dim 1 is subjects.
    # We will take one correlation over inter-subject differences for each region.
    # clone() so that we can deallocate the larger Tensor of which it is a view.
    # region_features already has a feature dimension, so we do not need to unsqueeze():
    # region or region pair (batch dimension) x subject (observation dimension) x feature (feature dimension)
    region_feature_file_name = os.path.join(data_directory, f'{region_feature_file_name_part}.pt')
    region_features = torch.clone(  torch.flatten( torch.load(f=region_feature_file_name, weights_only=False)[:,:,:num_region_features], start_dim=1, end_dim=2 )  )
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
    # clone() so that we can de-allocate the larger Tensor of which this is a view.
    # subject (observation dimension) x feature (region pair)
    region_pair_feature_file_name = os.path.join(data_directory, f'{region_pair_feature_file_name_part}.pt')
    sc = torch.clone( torch.load(f=region_pair_feature_file_name, weights_only=False)[:,:,0] )
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
    num_parts = params.size(dim=0)
    rmse_zeros = torch.zeros( size=(num_parts,), dtype=params.dtype, device=params.device )
    lstsq_rmse_train = rmse_zeros
    lstsq_rmse_test = rmse_zeros.clone()
    lstsq_rmse_validate = rmse_zeros.clone()
    best_train_rmse = rmse_zeros.clone()
    best_validate_rmse = rmse_zeros.clone()
    best_test_rmse = rmse_zeros.clone()
    for part_index in range(num_parts):
        lstsq_rmse_train[part_index], lstsq_rmse_test[part_index], lstsq_rmse_validate[part_index], best_train_rmse[part_index], best_validate_rmse[part_index], best_test_rmse[part_index] = save_optimization_results(independent=features, independent_name=feature_name, dependent=params[part_index,:,:], dependent_name=f'{param_name}_{part_index}', num_perms=num_permutations, max_num_hidden_layers=max_num_hidden_layers, max_hidden_layer_width=max_hidden_layer_width)
    lstsq_file_name_part = f'rmse_{feature_name}_{param_name}_lstsq_{output_file_name_part}'
    save_and_print(obj=lstsq_rmse_train, file_name_part=f'train_{lstsq_file_name_part}')
    save_and_print(obj=lstsq_rmse_validate, file_name_part=f'validate_{lstsq_file_name_part}')
    save_and_print(obj=lstsq_rmse_test, file_name_part=f'test_{lstsq_file_name_part}')
    mlp_file_name_part = f'rmse_{feature_name}_{param_name}_mlp_{output_file_name_part}'
    save_and_print(obj=best_train_rmse, file_name_part=f'train_{mlp_file_name_part}')
    save_and_print(obj=best_validate_rmse, file_name_part=f'validate_{mlp_file_name_part}')
    save_and_print(obj=best_test_rmse, file_name_part=f'test_{mlp_file_name_part}')
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