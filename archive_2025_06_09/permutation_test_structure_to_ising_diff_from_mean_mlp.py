import os
import torch
import time
import argparse
import isingmodel
from isingmodel import IsingModel
from graph2graphcnn import UniformMultiLayerPerceptron

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')

parser = argparse.ArgumentParser(description="Measure correlations between offsets of parameters of fitted Ising models from their mean values with predicted offsets based on structural feature offsets from their mean values.")
parser.add_argument("-a", "--input_directory", type=str, default='E:\\Ising_model_results_daai', help="directory from which we read the training examples")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the trained models and pandas DataFrame of loss results")
parser.add_argument("-c", "--file_name_fragment", type=str, default='group_training_and_individual_all', help="part of the input example files between example_[index]_ and _[validation|training]_example.pt, will also be part of the output file names")
parser.add_argument("-d", "--training_index_start", type=int, default=1, help="first index of training subjects")
parser.add_argument("-e", "--training_index_end", type=int, default=670, help="last index of training subjects + 1")
parser.add_argument("-f", "--num_permutations", type=int, default=1000, help="number of permutations of permuted pairings to try")
parser.add_argument("-g", "--abs_params", action='store_true', default=False, help="Set this flag in order to take the absolute values of parameters.")
parser.add_argument("-j", "--multiply_beta", action='store_true', default=False, help="Set this flag in order to multiply beta into the h and J parameters before taking the correlations.")
parser.add_argument("-k", "--num_epochs", type=int, default=1000, help="number of epochs for which to train the h and J prediction models")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate with which to train the h and J prediction models")
parser.add_argument("-m", "--hidden_layer_width", type=int, default=7, help="number of nodes per layer to use in the h and J prediction models")
parser.add_argument("-n", "--num_hidden_layers", type=int, default=1, help="number of hidden layers to use in the h and J prediction models")
args = parser.parse_args()
print('getting arguments...')
input_directory = args.input_directory
print(f'input_directory={input_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
file_name_fragment = args.file_name_fragment
print(f'file_name_fragment={file_name_fragment}')
training_index_start = args.training_index_start
print(f'training_index_start={training_index_start}')
training_index_end = args.training_index_end
print(f'training_index_end={training_index_end}')
num_permutations = args.num_permutations
print(f'num_permutations={num_permutations}')
abs_params = args.abs_params
print(f'abs_params={abs_params}')
multiply_beta = args.multiply_beta
print(f'multiply_beta={multiply_beta}')
num_epochs = args.num_epochs
print(f'num_epochs={num_epochs}')
learning_rate = args.learning_rate
print(f'learning_rate={learning_rate}')
hidden_layer_width = args.hidden_layer_width
print(f'hidden_layer_width={hidden_layer_width}')
num_hidden_layers = args.num_hidden_layers
print(f'num_hidden_layers={num_hidden_layers}')

if abs_params:
    abs_params_str = 'abs_params'
else:
    abs_params_str = 'signed_params'
if multiply_beta:
    multiply_beta_str = 'times_beta'
else:
    multiply_beta_str = 'no_beta'

def load_structural_features(input_directory:str, file_name_fragment:str, training_index_start:int, training_index_end:int):

    node_features_file = os.path.join(input_directory, f'node_features_{file_name_fragment}.pt')
    node_features = torch.load(f=node_features_file)
    print( 'loaded node_features size', node_features.size() )
    # Take out the x, y, and z coordinates, since they have no individual variation.
    node_features = node_features[training_index_start:training_index_end,:,3:]

    edge_features_file = os.path.join(input_directory, f'edge_features_{file_name_fragment}.pt')
    edge_features = torch.load(f=edge_features_file)
    print( 'loaded edge_features size', edge_features.size() )
    edge_features = edge_features[training_index_start:training_index_end,:,:,:]

    # We want to use the mean SC values of edges to a given node as an additional node feature.
    # It is easier to calculate it here, before we have converted edge_features from square to upper triangular form.
    node_features = torch.cat(  ( torch.mean(edge_features, dim=2), node_features ), dim=-1  )

    # Select the part of edge features above the diagonal.
    num_nodes = edge_features.size(dim=2)
    triu_rows, triu_cols = isingmodel.get_triu_indices_for_products(num_nodes=num_nodes, device=device)
    edge_features = edge_features[:,triu_rows,triu_cols,:]

    # Append some derived features.

    # node features
    # 0: mean SC, 1: thickness, 2: myelination, 3: curvature, 4: sulcus depth
    # We start with edge feature 0: SC and append
    # 1: distance
    # 2-11: abs(diff(p)) where p is each of the node features for the endpoints.
    num_subjects, num_pairs, original_num_edge_features = edge_features.size()
    num_node_features = node_features.size(dim=-1)
    new_num_edge_features = original_num_edge_features + num_node_features
    extended_edge_features = torch.zeros( size=(num_subjects, num_pairs, new_num_edge_features), dtype=edge_features.dtype, device=edge_features.device )
    extended_edge_features[:,:,:original_num_edge_features] = edge_features
    extended_edge_features[:,:,-num_node_features:] = torch.abs(node_features[:,triu_rows,:]-node_features[:,triu_cols,:])

    node_features_diff = node_features - torch.mean(node_features, dim=0, keepdim=True)
    edge_features_diff = extended_edge_features - torch.mean(extended_edge_features, dim=0, keepdim=True)
    return node_features_diff, edge_features_diff

def load_ising_model(input_directory:str, file_name_fragment:str, training_index_start:int, training_index_end:int, abs_params:bool, multiply_beta:bool):

    ising_model_file = os.path.join(input_directory, f'ising_model_{file_name_fragment}_fold_1_betas_5_steps_1200_lr_0.01.pt')
    ising_model = torch.load(f=ising_model_file)
    print(f'ising model after beta updates {ising_model.num_beta_updates}, param updates {ising_model.num_param_updates}')
    beta = ising_model.beta
    print( 'loaded beta size', beta.size() )
    h = ising_model.h
    print( 'loaded h size', h.size() )
    J = ising_model.J
    print( 'loaded J size', J.size() )
    
    triu_rows, triu_cols = ising_model.get_triu_indices_for_products()
    J = J[:,triu_rows,triu_cols]

    num_models = h.size(dim=0)
    models_per_subject = ising_model.num_betas_per_target
    num_subjects = num_models//models_per_subject
    beta = beta.unflatten( dim=0, sizes=(models_per_subject, num_subjects) )
    h = h.unflatten( dim=0, sizes=(models_per_subject, num_subjects) )
    J = J.unflatten( dim=0, sizes=(models_per_subject, num_subjects) )

    beta = beta[:,training_index_start:training_index_end,:]
    h = h[:,training_index_start:training_index_end,:]
    J = J[:,training_index_start:training_index_end,:]

    if abs_params:
        h = torch.abs(h)
        J = torch.abs(J)
    
    if multiply_beta:
        h = beta*h
        J = beta*J
    
    h_diff = h - torch.mean( h, dim=(0,1), keepdim=True )
    J_diff = J - torch.mean( J, dim=(0,1), keepdim=True )
    
    print('After taking upper triangular parts of square matrices, unsqueezing the batch dimension, and selecting training subjects')
    print( 'beta size', beta.size() )
    print( 'h size', h.size() )
    print( 'J size', J.size() )

    # clone() so that we do not keep the larger underlying Tensors of which these are views.
    return h_diff, J_diff

def load_features(input_directory:str, file_name_fragment:str, training_index_start:int, training_index_end:int, abs_params:bool, multiply_beta:bool):
    h_diff, J_diff = load_ising_model(input_directory=input_directory, file_name_fragment=file_name_fragment, training_index_start=training_index_start, training_index_end=training_index_end, abs_params=abs_params, multiply_beta=multiply_beta)
    node_features_diff, edge_features_diff = load_structural_features(input_directory=input_directory, file_name_fragment=file_name_fragment, training_index_start=training_index_start, training_index_end=training_index_end)
    if abs_params:
        correlation_names = ['|h| vs predicted |h|', '|J| vs predicted |J|']
    else:
        correlation_names = ['h vs predicted h', 'J vs predicted J']
    return node_features_diff, edge_features_diff, h_diff, J_diff, correlation_names

def save_features_and_params(node_features_diff:torch.Tensor, edge_features_diff:torch.Tensor, h_diff:torch.Tensor, J_diff:torch.Tensor, output_directory:str, file_name_fragment:str):

    node_features_file = os.path.join(output_directory, f'node_features_diff_{file_name_fragment}.pt')
    torch.save(obj=node_features_diff, f=node_features_file)
    print(f'{time.time() - code_start_time:.3f}, saved {node_features_file}')

    edge_features_file = os.path.join(output_directory, f'edge_features_diff_{file_name_fragment}.pt')
    torch.save(obj=edge_features_diff, f=edge_features_file)
    print(f'{time.time() - code_start_time:.3f}, saved {edge_features_file}')

    h_file = os.path.join(output_directory, f'h_diff_{file_name_fragment}.pt')
    torch.save(obj=h_diff, f=h_file)
    print(f'{time.time() - code_start_time:.3f}, saved {h_file}')

    J_file = os.path.join(output_directory, f'J_diff_{file_name_fragment}.pt')
    torch.save(obj=J_diff, f=J_file)
    print(f'{time.time() - code_start_time:.3f}, saved {J_file}')

    return node_features_file, edge_features_file, h_file, J_file

def get_mlp_predictions(features:torch.Tensor, param:torch.Tensor, num_epochs:int=1000, learning_rate:float=0.001, num_hidden_layers:int=1, hidden_layer_width:int=7):
    num_batches = param.size(dim=0)
    param = param.unsqueeze(dim=-1)
    model = UniformMultiLayerPerceptron( num_in_features=features.size(dim=-1), num_out_features=1, hidden_layer_width=hidden_layer_width, num_hidden_layers=num_hidden_layers, dtype=features.dtype, device=features.device )
    optimizer = torch.optim.Adam( params=model.parameters(), lr=learning_rate )
    loss_fn = torch.nn.MSELoss()
    # param_prediction = model(features)
    # for batch in range(num_batches):
    #     loss = loss_fn(param_prediction, param[batch,:,:,:])
    #     print(f'initial batch {batch+1} loss {loss:.3g}')
    for _ in range(num_epochs):
        for batch in range(num_batches):
            optimizer.zero_grad()
            param_prediction = model(features)
            loss = loss_fn(param_prediction, param[batch,:,:,:])
            loss.backward()
            optimizer.step()
            # print(f'epoch {epoch+1}, batch {batch+1}, loss {loss:.3g}')
    return param_prediction.detach()

# model_param has size models_per_subject x num_subjects x num_nodes (or num_pairs)
# features has size num_subjects x num_nodes (or num_pairs) x num_features
# We want to replicate features across models_per_subject and replicate model_param across num_features
# and take the correlation over models per subject, subjects, and nodes (or node pairs)
# so that we end up with a correlation matrix that is 1D with num_features elements.
def model_feature_correlation(model_param:torch.Tensor, feature:torch.Tensor, epsilon:torch.float=0.0):
    std_1, mean_1 = torch.std_mean( model_param, dim=(0,1,2) )# Take std and mean over model instance, subject, and node/pair.
    std_2, mean_2 = torch.std_mean( feature, dim=(0,1) )# Take std and mean over subject and node/pair
    return ( torch.mean( model_param.unsqueeze(dim=-1) * feature.unsqueeze(dim=0), dim=(0,1,2) ) - mean_1 * mean_2 + epsilon )/(std_1 * std_2 + epsilon)

def get_correlations(node_features_diff:torch.Tensor, edge_features_diff:torch.Tensor, h_diff:torch.Tensor, J_diff:torch.Tensor, num_epochs:int=1000, learning_rate:float=0.001, num_hidden_layers:int=1, hidden_layer_width:int=7, epsilon:torch.float=0.0):
    print(f'time {time.time()-code_start_time:.3f}, training node model...')
    h_diff_prediction = get_mlp_predictions(features=node_features_diff, param=h_diff, num_epochs=num_epochs, learning_rate=learning_rate, num_hidden_layers=num_hidden_layers, hidden_layer_width=hidden_layer_width)
    print(f'time {time.time()-code_start_time:.3f}, training edge model...')
    J_diff_prediction = get_mlp_predictions(features=edge_features_diff, param=J_diff, num_epochs=num_epochs, learning_rate=learning_rate, num_hidden_layers=num_hidden_layers, hidden_layer_width=hidden_layer_width)
    with torch.no_grad():
        h_correlations = model_feature_correlation(model_param=h_diff, feature=h_diff_prediction, epsilon=epsilon)
        print( 'h correlations', h_correlations )
        # print(f'time {time.time()-code_start_time:.3f}, h prediction correlation {h_correlations:.3g}')
        J_correlations = model_feature_correlation(model_param=J_diff, feature=J_diff_prediction, epsilon=epsilon)
        print( 'J correlations', J_correlations )
        # print(f'time {time.time()-code_start_time:.3f}, J prediction correlation {J_correlations:.3g}')
    return torch.cat( (h_correlations, J_correlations), dim=0 )

def compare_correlation_to_permuted(node_features_diff:torch.Tensor, edge_features_diff:torch.Tensor, h_diff:torch.Tensor, J_diff:torch.Tensor, num_permutations:int, shuffle_subjects:bool, shuffle_nodes:bool, correlation_names:list, true_correlations:torch.Tensor, file_name:str, num_epochs:int=1000, learning_rate:float=0.001, num_hidden_layers:int=1, hidden_layer_width:int=7, epsilon:torch.float=0.0):
    num_subjects = node_features_diff.size(dim=0)
    num_nodes = node_features_diff.size(dim=1)
    num_pairs = edge_features_diff.size(dim=1)
    num_correlations = len(correlation_names)
    if os.path.exists(file_name):
        perm_correlations = torch.load(f=file_name)
        print(f'loaded correlations from {file_name}')
    else:
        perm_correlations = torch.zeros( (num_permutations, num_correlations), dtype=float_type, device=device, requires_grad=False )
        # Because of how indexing works in PyTorch, we have to permute one index at a time.
        for perm in range(num_permutations):
            # print(f'time {time.time() - code_start_time:.3f},\tpermutation {perm} of {num_permutations}...')
            node_features_diff_perm = node_features_diff
            edge_features_diff_perm = edge_features_diff
            if shuffle_subjects:
                subject_order = torch.randperm(n=num_subjects, dtype=int_type, device=device)
                node_features_diff_perm = node_features_diff_perm[subject_order,:,:]
                edge_features_diff_perm = edge_features_diff_perm[subject_order,:,:]
            if shuffle_nodes:
                node_order = torch.randperm(n=num_nodes, dtype=int_type, device=device)
                node_features_diff_perm = node_features_diff_perm[:,node_order,:]
                pair_order = torch.randperm(n=num_pairs, dtype=int_type, device=device)
                edge_features_diff_perm = edge_features_diff_perm[:,pair_order,:]
            print(f'time {time.time()-code_start_time:.3f}, permutation {perm+1}...')
            perm_correlations[perm,:] = get_correlations(node_features_diff=node_features_diff_perm, edge_features_diff=edge_features_diff_perm, h_diff=h_diff, J_diff=J_diff, num_epochs=num_epochs, learning_rate=learning_rate, num_hidden_layers=num_hidden_layers, hidden_layer_width=hidden_layer_width, epsilon=epsilon)
        torch.save(obj=perm_correlations, f=file_name)
        print(f'saved correlations to {file_name}')
    if type(true_correlations) == type(None):
        p_bigger = None
    else:
        p_bigger = torch.count_nonzero( perm_correlations.abs() >= true_correlations.abs(), dim=0 )/num_permutations
    return p_bigger, perm_correlations

def summarize_distribution(values:torch.Tensor):
    values = values.flatten()
    quantile_cutoffs = torch.tensor([0.005, 0.5, 0.995], dtype=float_type, device=device)
    quantiles = torch.quantile(values, quantile_cutoffs)
    min_val = torch.min(values)
    max_val = torch.max(values)
    return f'median\t{quantiles[1].item():.3g}\t0.5%ile\t{quantiles[0].item():.3g}\t99.5%ile\t{quantiles[2].item():.3g}\tmin\t{min_val.item():.3g}\tmax\t{max_val.item():.3g}'

def try_all_permutation_cases(node_features_diff:torch.Tensor, edge_features_diff:torch.Tensor, h_diff:torch.Tensor, J_diff:torch.Tensor, num_permutations:int, correlation_names:list, output_directory:str, file_name_fragment:str, num_epochs:int=1000, learning_rate:float=0.001, num_hidden_layers:int=1, hidden_layer_width:int=7, epsilon:torch.float=0.0):
    true_correlations_file = os.path.join(output_directory, f'true_correlations_{file_name_fragment}.pt')
    _, true_correlations = compare_correlation_to_permuted(node_features_diff=node_features_diff, edge_features_diff=edge_features_diff, h_diff=h_diff, J_diff=J_diff, num_permutations=num_permutations, shuffle_subjects=False, shuffle_nodes=False, correlation_names=correlation_names, true_correlations=None, file_name=true_correlations_file, num_epochs=num_epochs, learning_rate=learning_rate, num_hidden_layers=num_hidden_layers, hidden_layer_width=hidden_layer_width, epsilon=epsilon)
    print(f'time {time.time() - code_start_time:.3f},\tstarting subject permutations...')
    perm_correlations_subjects_file = os.path.join(output_directory, f'perm_correlations_subjects_{file_name_fragment}.pt')
    p_bigger_subjects, perm_correlations_subjects = compare_correlation_to_permuted(node_features_diff=node_features_diff, edge_features_diff=edge_features_diff, h_diff=h_diff, J_diff=J_diff, num_permutations=num_permutations, shuffle_subjects=True, shuffle_nodes=False, correlation_names=correlation_names, true_correlations=true_correlations, file_name=perm_correlations_subjects_file, num_epochs=num_epochs, learning_rate=learning_rate, num_hidden_layers=num_hidden_layers, hidden_layer_width=hidden_layer_width, epsilon=epsilon)
    print(f'time {time.time() - code_start_time:.3f},\tstarting node and node-pair permutations...')
    perm_correlations_nodes_file = os.path.join(output_directory, f'perm_correlations_nodes_{file_name_fragment}.pt')
    p_bigger_nodes, perm_correlations_nodes = compare_correlation_to_permuted(node_features_diff=node_features_diff, edge_features_diff=edge_features_diff, h_diff=h_diff, J_diff=J_diff, num_permutations=num_permutations, shuffle_subjects=False, shuffle_nodes=True, correlation_names=correlation_names, true_correlations=true_correlations, file_name=perm_correlations_nodes_file, num_epochs=num_epochs, learning_rate=learning_rate, num_hidden_layers=num_hidden_layers, hidden_layer_width=hidden_layer_width, epsilon=epsilon)
    print(f'time {time.time() - code_start_time:.3f},\tstarting (subject, node) and (subject, node-pair) permutations...')
    perm_correlations_both_file = os.path.join(output_directory, f'perm_correlations_both_{file_name_fragment}.pt')
    p_bigger_both, perm_correlations_both = compare_correlation_to_permuted(node_features_diff=node_features_diff, edge_features_diff=edge_features_diff, h_diff=h_diff, J_diff=J_diff, num_permutations=num_permutations, shuffle_subjects=True, shuffle_nodes=True, correlation_names=correlation_names, true_correlations=true_correlations, file_name=perm_correlations_both_file, num_epochs=num_epochs, learning_rate=learning_rate, num_hidden_layers=num_hidden_layers, hidden_layer_width=hidden_layer_width, epsilon=epsilon)
    num_correlations = len(correlation_names)
    for correlation_index in range(num_correlations):
        true_distribution_str = summarize_distribution(true_correlations[:,correlation_index])
        subjects_distribution_str = summarize_distribution(perm_correlations_subjects[:,correlation_index])
        nodes_distribution_str = summarize_distribution(perm_correlations_nodes[:,correlation_index])
        both_distribution_str = summarize_distribution(perm_correlations_both[:,correlation_index])
        print(f'time\t{time.time() - code_start_time:.3f}\t{correlation_names[correlation_index]}\ttrue correlation\t{true_distribution_str}\tprobability abs(correlation) with randomized subjects >= actual\t{p_bigger_subjects[correlation_index]:.3g}\t{subjects_distribution_str}\tprobability abs(correlation) with randomized nodes >= actual\t{p_bigger_nodes[correlation_index]:.3g}\t{nodes_distribution_str}\tprobability abs(correlation) with randomized subjects and nodes >= actual\t{p_bigger_both[correlation_index]:.3g}\t{both_distribution_str}')
    return true_correlations, perm_correlations_subjects, perm_correlations_nodes, perm_correlations_both

with torch.no_grad():
    node_features_diff, edge_features_diff, h_diff, J_diff, correlation_names = load_features(input_directory=input_directory, file_name_fragment=file_name_fragment, training_index_start=training_index_start, training_index_end=training_index_end, abs_params=abs_params, multiply_beta=multiply_beta)
perms_file_name_fragment = f'diff_vs_diff_{file_name_fragment}_{abs_params_str}_{multiply_beta_str}_epochs_{num_epochs}_lr_{learning_rate:.3g}_depth_{num_hidden_layers}_width_{hidden_layer_width}_permutations_{num_permutations}'
true_correlations, perm_correlations_subjects, perm_correlations_nodes, perm_correlations_both = try_all_permutation_cases(node_features_diff=node_features_diff, edge_features_diff=edge_features_diff, h_diff=h_diff, J_diff=J_diff, num_permutations=num_permutations, correlation_names=correlation_names, output_directory=output_directory, file_name_fragment=perms_file_name_fragment, num_epochs=num_epochs, learning_rate=learning_rate, num_hidden_layers=num_hidden_layers, hidden_layer_width=hidden_layer_width, epsilon=0.0)
print(f'time {time.time()-code_start_time:.3f}, done')