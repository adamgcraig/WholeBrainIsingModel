import os
import torch
import time
import argparse
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from graph2graphcnn import UniformMultiLayerPerceptron

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')

parser = argparse.ArgumentParser(description="Compare performance of the structure-to-Ising model MLP pairs vs an ensemble of the same models trained on shuffled data.")
parser.add_argument("-a", "--input_directory", type=str, default='E:\\Ising_model_results_daai', help="directory from which we read the training examples")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the trained models and pandas DataFrame of loss results")
parser.add_argument("-c", "--file_name_fragment", type=str, default='small', help="another string we can incorporate into the output file names to help differentiate separate runs")
parser.add_argument("-d", "--optimizer_name", type=str, default='Adam', help="either Adam or SGD")
parser.add_argument("-e", "--num_epochs", type=int, default=1000, help="number of epochs for which to train each model")
parser.add_argument("-f", "--patience", type=int, default=1000, help="Number of epochs with no noticeable improvement in either loss before we stop and move on to the next model.")
parser.add_argument("-g", "--min_improvement", type=float, default=10e-10, help="Minimal amount of improvement to count as noticeable.")
parser.add_argument("-i", "--save_models", action='store_true', default=False, help="Set this flag in order to have the script save each trained G2GCNN model.")
parser.add_argument("-j", "--num_instances", type=int, default=100, help="number of MLP models to train and validate")
parser.add_argument("-k", "--mlp_hidden_layers", type=int, default=10, help="number of hidden layers to use in each multi-layer perceptron")
parser.add_argument("-l", "--rep_dims", type=int, default=20, help="number of nodes in each MLP hidden layer")
parser.add_argument("-m", "--batch_size", type=int, default=50, help="batch size")
parser.add_argument("-n", "--learning_rate", type=float, default=0.01, help="learning rate")
parser.add_argument("-o", "--models_per_subject", type=int, default=5, help="number of fitted Ising models for each subject")
parser.add_argument("-p", "--is_edge", action='store_true', default=False, help="Set this flag in order to train the edge MLP instead of the node MLP.")
parser.add_argument("-q", "--shuffle_subjects", action='store_true', default=False, help="Set this flag in order to shuffle the pairings of subject structural features and Ising model parameters. Regions or region pairs remain matched.")
parser.add_argument("-r", "--preload_data", action='store_true', default=False, help="Set this flag in order to load all the training and validation data before beginning training instead of loading dynamically.")
args = parser.parse_args()
print('getting arguments...')
input_directory = args.input_directory
print(f'input_directory={input_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
file_name_fragment = args.file_name_fragment
print(f'file_name_fragment={file_name_fragment}')
num_epochs = args.num_epochs
print(f'num_epochs={num_epochs}')
patience = args.patience
print(f'patience={patience}')
min_improvement = args.min_improvement
print(f'min_improvement={min_improvement:.3g}')
save_models = args.save_models
print(f'save_models={save_models}')
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
models_per_subject = args.models_per_subject
print(f'models_per_subject={models_per_subject}')
is_edge = args.is_edge
print(f'is_edge={is_edge}')
if is_edge:
    model_type = 'edge'
else:
    model_type = 'node'
shuffle_subjects = args.shuffle_subjects
print(f'shuffle_subjects={shuffle_subjects}')
if shuffle_subjects:
    shuffle_string = 'shuffle_subjects'
else:
    shuffle_string = 'shuffle_none'
preload_data = args.preload_data
print(f'preload_data={preload_data}')

def load_subject_ids(file_directory:str, subject_subset:str):
    subject_id_file_name = os.path.join(file_directory, f'{subject_subset}_subject_ids.txt')
    with open(file=subject_id_file_name, mode='r') as subject_id_file:
        subject_ids = [int(line) for line in subject_id_file.readlines()]
    return subject_ids

class Feature2ParamDataset(Dataset):
    def __init__(self, file_directory:str, subject_subset:str, models_per_subject:int, is_edge:bool=False):
        super(Feature2ParamDataset,self).__init__()
        self.file_directory = file_directory
        self.subject_ids = load_subject_ids(file_directory=file_directory, subject_subset=subject_subset)
        print(f'loaded {subject_subset} subject list, found {len(self.subject_ids)} subjects')
        self.models_per_subject = models_per_subject
        if is_edge:
            self.feature_type = 'edge'
            self.model_param = 'J'
        else:
            self.feature_type = 'node'
            self.model_param = 'h'
    def __len__(self):
        return len(self.subject_ids) * self.models_per_subject
    def __getitem__(self, idx:int):
        subject_index = idx // self.models_per_subject
        subject_id = self.subject_ids[subject_index]
        model_index = idx % self.models_per_subject
        feature_file = os.path.join(self.file_directory, f'{self.feature_type}_features_subject_{subject_id}.pt')
        features = torch.load(feature_file)
        param_file = os.path.join(self.file_directory, f'{self.model_param}_subject_{subject_id}_model_{model_index}.pt')
        # The Tensor stored in the node features or edge features file is num_nodes x num_node_features or num_pairs x num_edge_features.
        # The Tensor stored in the h or J file is 1D with length num_nodes or num_pairs.
        # Unsqueeze to Nx1 so that nodes or pairs becomes a batch dimension matching up to that of the features Tensor.
        param = torch.load(param_file).unsqueeze(dim=-1)
        return features, param

class ShuffledSubjectsFeature2ParamDataset(Dataset):
    def __init__(self, file_directory:str, subject_subset:str, models_per_subject:int, is_edge:bool=False, dtype=torch.float, device='cpu'):
        super(ShuffledSubjectsFeature2ParamDataset,self).__init__()
        self.file_directory = file_directory
        self.subject_ids = load_subject_ids(file_directory=file_directory, subject_subset=subject_subset)
        print(f'loaded {subject_subset} subject list, found {len(self.subject_ids)} subjects')
        self.models_per_subject = models_per_subject
        if is_edge:
            self.feature_type = 'edge'
            self.model_param = 'J'
        else:
            self.feature_type = 'node'
            self.model_param = 'h'
        self.dtype = dtype
        self.device = device
        length = len(self.subject_ids) * self.models_per_subject
        self.shuffle_indices = torch.randperm( n=length, dtype=int_type, device=self.device )
    def reshuffle(self):
        length = len(self.subject_ids) * self.models_per_subject
        self.shuffle_indices = torch.randperm( n=length, dtype=int_type, device=self.device )
    def __len__(self):
        return len(self.subject_ids) * self.models_per_subject
    def __getitem__(self, idx:int):
        # Use the original index for the node or edge features and the shuffled index for h or J.
        # Doing this mismatches each the subject of the structural features and the subject of the Ising model parameters.
        subject_index = idx // self.models_per_subject
        subject_id = self.subject_ids[subject_index]
        shuffled_idx = self.shuffle_indices[idx]
        shuffled_subject_index = shuffled_idx // self.models_per_subject
        shuffled_subject_id = self.subject_ids[shuffled_subject_index]
        shuffled_model_index = shuffled_idx % self.models_per_subject
        feature_file = os.path.join(self.file_directory, f'{self.feature_type}_features_subject_{subject_id}.pt')
        features = torch.load(feature_file)
        param_file = os.path.join(self.file_directory, f'{self.model_param}_subject_{shuffled_subject_id}_model_{shuffled_model_index}.pt')
        # The Tensor stored in the node features or edge features file is num_nodes x num_node_features or num_pairs x num_edge_features.
        # The Tensor stored in the h or J file is 1D with length num_nodes or num_pairs.
        # Unsqueeze to Nx1 so that nodes or pairs becomes a batch dimension matching up to that of the features Tensor.
        param = torch.load(param_file).unsqueeze(dim=-1)
        return features, param

class PreloadedFeature2ParamDataset(Dataset):
    def __init__(self, file_directory:str, subject_subset:str, models_per_subject:int, is_edge:bool=False):
        super(PreloadedFeature2ParamDataset,self).__init__()
        self.file_directory = file_directory
        self.subject_ids = load_subject_ids(file_directory=file_directory, subject_subset=subject_subset)
        print(f'loaded {subject_subset} subject list, found {len(self.subject_ids)} subjects')
        self.models_per_subject = models_per_subject
        if is_edge:
            self.feature_type = 'edge'
            self.model_param = 'J'
        else:
            self.feature_type = 'node'
            self.model_param = 'h'
        num_subjects = len(self.subject_ids)
        feature_file = os.path.join(self.file_directory, f'{self.feature_type}_features_subject_{self.subject_ids[0]}.pt')
        example_features = torch.load(f=feature_file)
        example_size, num_features = example_features.size()
        dtype = example_features.dtype
        device = example_features.device
        self.features = torch.zeros( size=(num_subjects, example_size, num_features), dtype=dtype, device=device )
        for subject_index in range(num_subjects):
            self.features[subject_index,:,:] = torch.load( f=os.path.join(self.file_directory, f'{self.feature_type}_features_subject_{self.subject_ids[subject_index]}.pt') )
        self.params = torch.zeros( size=(models_per_subject, num_subjects, example_size, 1), dtype=dtype, device=device )
        for subject_index in range(num_subjects):
            for model_index in range(self.models_per_subject):
                self.params[model_index,subject_index,:,0] = torch.load( f=os.path.join(self.file_directory, f'{self.model_param}_subject_{self.subject_ids[subject_index]}_model_{model_index}.pt') )
    def reshuffle(self):
        models_per_subject = self.params.size(dim=0)
        num_subjects = self.params.size(dim=1)
        length = models_per_subject * num_subjects
        shuffle_indices = torch.randperm( n=length, dtype=int_type, device=self.params.device )
        self.params = ( self.params.flatten(start_dim=0, end_dim=1) )[shuffle_indices,:,:].unflatten( dim=0, sizes=(models_per_subject, num_subjects) )
    def __len__(self):
        return len(self.subject_ids) * self.models_per_subject
    def __getitem__(self, idx:int):
        subject_index = idx // self.models_per_subject
        model_index = idx % self.models_per_subject
        return self.features[subject_index,:,:], self.params[model_index,subject_index,:,:]
    
# Get two sets of RMSE values for the data set.
# In rmse_example, we take the mean over all nodes or node pairs to get a single RMSE for each example.
# In rmse_region, we take the mean over all examples to get a single RMSE for each node or node pair.
def get_rmses(data_loader:DataLoader, model:torch.nn.Module, num_examples:int, example_size:int):
    rmse_example = torch.zeros( size=(num_examples,), dtype=float_type, device=device )
    sum_se_region = torch.zeros( size=(example_size,), dtype=float_type, device=device )
    example_index = 0
    for features_batch, param_batch in data_loader:
        predicted_param_batch = model(features_batch)
        batch_size = features_batch.size(dim=0)
        diff_square = (predicted_param_batch - param_batch).squeeze(dim=-1).square()
        rmse_example[example_index:(example_index+batch_size)] = diff_square.mean(dim=-1).sqrt()
        sum_se_region += diff_square.sum(dim=0)
        example_index += batch_size
    mse_region = sum_se_region/num_examples
    rmse_region = mse_region.sqrt()
    rmse_overall = rmse_region.mean().sqrt()
    return rmse_example, rmse_region, rmse_overall

def get_min_mean_max_string(training_values:torch.Tensor, validation_values:torch.Tensor):
    return f'min\t{training_values.min():.3g}\t/\t{validation_values.min():.3g}\tmean\t{training_values.mean():.3g}\t/\t{validation_values.mean():.3g}\tmax\t{training_values.max():.3g}\t/\t{validation_values.max():.3g}'

def train_model(training_data_set:Feature2ParamDataset, validation_data_set:Feature2ParamDataset, model_file:str, num_epochs:int=1000, optimizer_name:str='Adam', rep_dims:int=7, mlp_hidden_layers:int=3, batch_size:int=10, learning_rate:torch.float=0.001, save_model:bool=False, patience:int=10, improvement_threshold:torch.float=0.0001):
    num_training_examples = training_data_set.__len__()
    num_validation_examples = validation_data_set.__len__()
    example_features, example_param = training_data_set.__getitem__(idx=0)
    dtype = example_features.dtype
    device = example_features.device
    num_regions = example_features.size(dim=-2)
    num_features = example_features.size(dim=-1)
    num_params = example_param.size(dim=-1)
    training_data_loader = DataLoader(dataset=training_data_set, shuffle=True, batch_size=batch_size)
    validation_data_loader = DataLoader(dataset=validation_data_set, shuffle=False, batch_size=batch_size)
    model = UniformMultiLayerPerceptron(num_in_features=num_features, num_out_features=num_params, hidden_layer_width=rep_dims, num_hidden_layers=mlp_hidden_layers, dtype=dtype, device=device)
    loss_fn = torch.nn.MSELoss()
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD( params=model.parameters(), lr=learning_rate )
    else:
        optimizer = torch.optim.Adam( params=model.parameters(), lr=learning_rate )
    num_no_improvement_epochs = 0
    print(f'time {time.time() - code_start_time:.3f}, starting training MLPs with {num_features} input features, {rep_dims} latent representaion dimensions, {mlp_hidden_layers} hidden layers per MLP, batch size {batch_size}, learning rate {learning_rate:.3g}, optimizer {optimizer_name}.')
    last_training_rmse_example, last_training_rmse_region, last_training_rmse_overall = get_rmses(data_loader=training_data_loader, model=model, num_examples=num_training_examples, example_size=num_regions)
    last_validation_rmse_example, last_validation_rmse_region, last_validation_rmse_overall = get_rmses(data_loader=validation_data_loader, model=model, num_examples=num_validation_examples, example_size=num_regions)
    print(f'initial losses training/validation example {get_min_mean_max_string(last_training_rmse_example, last_validation_rmse_example)}\tregion {get_min_mean_max_string(last_training_rmse_region, last_validation_rmse_region)}\toverall\t{last_training_rmse_overall:.3g}\t/\t{last_validation_rmse_overall:.3g}')
    for epoch in range(num_epochs):
        for features_batch, param_batch in training_data_loader:
            optimizer.zero_grad()
            param_pred = model(features_batch)
            loss = loss_fn(param_batch, param_pred)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            training_rmse_example, training_rmse_region, training_rmse_overall = get_rmses(data_loader=training_data_loader, model=model, num_examples=num_training_examples, example_size=num_regions)
            validation_rmse_example, validation_rmse_region, validation_rmse_overall = get_rmses(data_loader=validation_data_loader, model=model, num_examples=num_validation_examples, example_size=num_regions)
            diff_training_rmse_example = training_rmse_example - last_training_rmse_example
            diff_validation_rmse_example = validation_rmse_example - last_validation_rmse_example
            diff_training_rmse_region = training_rmse_region - last_training_rmse_region
            diff_validation_rmse_region = validation_rmse_region - last_validation_rmse_region
            diff_training_rmse_overall = training_rmse_overall - last_training_rmse_overall
            diff_validation_rmse_overall = validation_rmse_overall - last_validation_rmse_overall
            num_no_improvement_epochs += ( (-1*diff_validation_rmse_overall) < min_improvement )
            print(f'epoch\t{epoch+1}\tlosses training/validation example {get_min_mean_max_string(training_rmse_example, validation_rmse_example)}\tdiff {get_min_mean_max_string(diff_training_rmse_example, diff_validation_rmse_example)}\tregion {get_min_mean_max_string(training_rmse_region, validation_rmse_region)}\tdiff {get_min_mean_max_string(diff_training_rmse_region, diff_validation_rmse_region)}\t\toverall\t{training_rmse_overall:.3g}\t/\t{validation_rmse_overall:.3g}\tdiff\t{diff_training_rmse_overall:.3g}\t/\t{diff_validation_rmse_overall:.3g}')
            if num_no_improvement_epochs >= patience:
                print(f'patience exceeded, moving on...')
                break
            if math.isnan(validation_rmse_overall):
                print('encountered NaN in validation RMSE, moving on...')
                break
            last_training_rmse_example = training_rmse_example
            last_validation_rmse_example = validation_rmse_example
            last_training_rmse_region = training_rmse_region
            last_validation_rmse_region = validation_rmse_region
            last_training_rmse_overall = training_rmse_overall
            last_validation_rmse_overall = validation_rmse_overall
    if save_model:
        torch.save(obj=model, f=model_file)
        print(f'time {time.time() - code_start_time:.3f}, saved {model_file}')
    return training_rmse_example, training_rmse_region, validation_rmse_example, validation_rmse_region

# Load the training and validation data.
if preload_data:
    training_data_set = PreloadedFeature2ParamDataset(file_directory=input_directory, subject_subset='training', models_per_subject=models_per_subject, is_edge=is_edge)
    validation_data_set = PreloadedFeature2ParamDataset(file_directory=input_directory, subject_subset='validation', models_per_subject=models_per_subject, is_edge=is_edge)
elif shuffle_subjects:
    training_data_set = ShuffledSubjectsFeature2ParamDataset(file_directory=input_directory, subject_subset='training', models_per_subject=models_per_subject, is_edge=is_edge, dtype=float_type, device=device)
    validation_data_set = ShuffledSubjectsFeature2ParamDataset(file_directory=input_directory, subject_subset='validation', models_per_subject=models_per_subject, is_edge=is_edge, dtype=float_type, device=device)
else:
    training_data_set = Feature2ParamDataset(file_directory=input_directory, subject_subset='training', models_per_subject=models_per_subject, is_edge=is_edge)
    validation_data_set = Feature2ParamDataset(file_directory=input_directory, subject_subset='validation', models_per_subject=models_per_subject, is_edge=is_edge)
# Check whether we have already created the results pickle file in a previous run.
output_file_name_fragment = f'{model_type}_mlp_{file_name_fragment}_models_per_subject_{models_per_subject}_epochs_{num_epochs}_patience_{patience}_min_imp_{min_improvement:.3g}_opt_{optimizer_name}_batch_{batch_size}_lr_{learning_rate:.3g}_width_{rep_dims}_depth_{mlp_hidden_layers}_{shuffle_string}'
for instance_index in range(num_instances):

    print(f'{time.time()-code_start_time:.3f}, training model instance {instance_index+1} of {num_instances}...')
    if shuffle_subjects:
        training_data_set.reshuffle()
    file_name_fragment_with_instance = f'{output_file_name_fragment}_{instance_index+1}'
    model_file = os.path.join(output_directory, f'{file_name_fragment_with_instance}.pt')
    training_rmse_example, training_rmse_region, validation_rmse_example, validation_rmse_region = train_model(training_data_set=training_data_set, validation_data_set=validation_data_set, model_file=model_file, num_epochs=num_epochs, optimizer_name=optimizer_name, rep_dims=rep_dims, mlp_hidden_layers=mlp_hidden_layers, batch_size=batch_size, learning_rate=learning_rate, save_model=save_models, patience=patience, improvement_threshold=min_improvement)
    
    training_rmse_example_file = os.path.join(output_directory, f'training_rmse_example_{file_name_fragment_with_instance}.pt')
    torch.save(obj=training_rmse_example, f=training_rmse_example_file)
    print(f'time {time.time() - code_start_time:.3f}, saved {training_rmse_example_file}')

    validation_rmse_example_file = os.path.join(output_directory, f'validation_rmse_example_{file_name_fragment_with_instance}.pt')
    torch.save(obj=validation_rmse_example, f=validation_rmse_example_file)
    print(f'time {time.time() - code_start_time:.3f}, saved {validation_rmse_example_file}')
    
    training_rmse_region_file = os.path.join(output_directory, f'training_rmse_region_{file_name_fragment_with_instance}.pt')
    torch.save(obj=training_rmse_region, f=training_rmse_region_file)
    print(f'time {time.time() - code_start_time:.3f}, saved {training_rmse_region_file}')

    validation_rmse_region_file = os.path.join(output_directory, f'validation_rmse_region_{file_name_fragment_with_instance}.pt')
    torch.save(obj=validation_rmse_region, f=validation_rmse_region_file)
    print(f'time {time.time() - code_start_time:.3f}, saved {validation_rmse_region_file}')

print(f'time {time.time() - code_start_time:.3f}, done')