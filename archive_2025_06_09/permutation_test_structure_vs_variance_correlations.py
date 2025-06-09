import os
import torch
import time
import argparse
import isingmodel
from isingmodel import IsingModel

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')

parser = argparse.ArgumentParser(description="Measure correlations between structural features and parameters of fitted Ising models.")
parser.add_argument("-a", "--input_directory", type=str, default='E:\\Ising_model_results_daai', help="directory from which we read the training examples")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the trained models and pandas DataFrame of loss results")
parser.add_argument("-c", "--file_name_fragment", type=str, default='group_training_and_individual_all', help="part of the input example files between example_[index]_ and _[validation|training]_example.pt, will also be part of the output file names")
parser.add_argument("-d", "--training_index_start", type=int, default=1, help="first index of training subjects")
parser.add_argument("-e", "--training_index_end", type=int, default=670, help="last index of training subjects + 1")
parser.add_argument("-f", "--num_permutations", type=int, default=1000, help="number of permutations of permuted pairings to try")
parser.add_argument("-g", "--sim_length", type=int, default=1200, help="number of simulation steps to run to get state and state product means")
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
sim_length = args.sim_length
print(f'sim_length={sim_length}')

def get_spherical_coords(rectangular_coords:torch.Tensor):
    x = rectangular_coords[:,:,0].unsqueeze(dim=-1)
    y = rectangular_coords[:,:,1].unsqueeze(dim=-1)
    z = rectangular_coords[:,:,2].unsqueeze(dim=-1)
    x_sq_plus_y_sq = x.square() + y.square()
    # radius = sqrt(x^2 + y^2 + z^2)
    radius = torch.sqrt( x_sq_plus_y_sq + z.square() )
    # inclination = arccos(z/radius)
    inclination = torch.arccos(z/radius)
    # azimuth = sgn(y)*arccos( x/sqrt(x^2+y^2) )
    azimuth = torch.sign(y) * torch.arccos( x/torch.sqrt(x_sq_plus_y_sq) )
    return torch.cat( (radius, inclination, azimuth), dim=-1 )

def load_structural_features(input_directory:str, file_name_fragment:str, training_index_start:int, training_index_end:int):

    node_features_file = os.path.join(input_directory, f'node_features_{file_name_fragment}.pt')
    node_features = torch.load(f=node_features_file)
    print( 'loaded node_features size', node_features.size() )
    node_features = node_features[training_index_start:training_index_end,:,:]

    edge_features_file = os.path.join(input_directory, f'edge_features_{file_name_fragment}.pt')
    edge_features = torch.load(f=edge_features_file)
    print( 'loaded edge_features size', edge_features.size() )
    edge_features = edge_features[training_index_start:training_index_end,:,:,:]

    # We want to use the mean SC values of edges to a given node as an additional node feature.
    # It is easier to calculate it here, before we have converted edge_features from square to upper triangular form.
    mean_sc = torch.mean(edge_features, dim=2)
    print( 'mean SC size', mean_sc.size() )

    # Select the part of edge features above the diagonal.
    num_nodes = edge_features.size(dim=2)
    triu_rows, triu_cols = isingmodel.get_triu_indices_for_products(num_nodes=num_nodes, device=device)
    edge_features = edge_features[:,triu_rows,triu_cols,:]
    print( 'triu edge_features size', edge_features.size() )

    # Extend node features with mean SC and spherical coordinates.
    rectangular_coords = node_features[:,:,:3]
    print( 'rectangular_coords size', rectangular_coords.size() )
    spherical_coords = get_spherical_coords(rectangular_coords=rectangular_coords)
    print( 'spherical_coords size', spherical_coords.size() )
    other_node_features = node_features[:,:,3:]
    print( 'other_node_features size', other_node_features.size() )
    extended_node_features = torch.cat( (rectangular_coords, spherical_coords, other_node_features, mean_sc), dim=-1 )
    print( 'extended_node_features size', extended_node_features.size() )

    # Extend edge features with Euclidian distances between nodes and absolute differences between node features.
    node_feature_diffs = torch.abs(extended_node_features[:,triu_rows,:] - extended_node_features[:,triu_cols,:])
    print( 'node_feature_diffs size', node_feature_diffs.size() )
    distances = node_feature_diffs[:,:,:3].square().sum(dim=-1, keepdim=True).sqrt()
    print( 'distances size', distances.size() )
    extended_edge_features = torch.cat( (distances, node_feature_diffs, edge_features), dim=-1 )
    print( 'extended_edge_features size', extended_edge_features.size() )

    # clone() so that we do not keep the larger underlying Tensor of which node_features is a view.
    # We do not need to do this for extended_edge_features, since it is a new Tensor allocated at the right size.
    return extended_node_features, extended_edge_features

def load_and_simulate_ising_model(input_directory:str, file_name_fragment:str, training_index_start:int, training_index_end:int, sim_length:int):
    ising_model_file = os.path.join(input_directory, f'ising_model_{file_name_fragment}_fold_1_betas_5_steps_1200_lr_0.01.pt')
    ising_model = torch.load(f=ising_model_file)
    print(f'time {time.time()-code_start_time:.3f}, simulating ising model after beta updates {ising_model.num_beta_updates}, param updates {ising_model.num_param_updates}...')
    sim_mean_state, sim_mean_state_product = ising_model.simulate_and_record_means(num_steps=sim_length)
    print(f'time {time.time()-code_start_time:.3f}, simulation complete.')
    num_models = sim_mean_state.size(dim=0)
    models_per_subject = ising_model.num_betas_per_target
    num_subjects = num_models//models_per_subject
    node_variance = 1 - sim_mean_state.unflatten( dim=0, sizes=(models_per_subject, num_subjects) )[:,training_index_start:training_index_end,:].square()
    edge_variance = 1 - sim_mean_state_product.unflatten( dim=0, sizes=(models_per_subject, num_subjects) )[:,training_index_start:training_index_end,:].square()
    print( 'node variance size', node_variance.size() )
    print( f'min {node_variance.min():.3g}, mean {node_variance.mean():.3g}, max {node_variance.max():.3g}' )
    print( 'edge_variance size', edge_variance.size() )
    print( f'min {edge_variance.min():.3g}, mean {edge_variance.mean():.3g}, max {edge_variance.max():.3g}' )
    return node_variance.clone(), edge_variance.clone()

def load_features_and_variances(input_directory:str, file_name_fragment:str, training_index_start:int, training_index_end:int, sim_length:int):
    node_features, edge_features = load_structural_features(input_directory=input_directory, file_name_fragment=file_name_fragment, training_index_start=training_index_start, training_index_end=training_index_end)
    node_variance_file = os.path.join(output_directory, f'node_variance_{file_name_fragment}.pt')
    edge_variance_file = os.path.join(output_directory, f'edge_variance_{file_name_fragment}.pt')
    if os.path.exists(node_variance_file) and os.path.exists(edge_variance_file):
        node_variances = torch.load(node_variance_file)
        edge_variances = torch.load(edge_variance_file)
        num_total_subjects = node_variances.size(dim=1)
        num_training_subjects = training_index_end-training_index_start
        if num_total_subjects > num_training_subjects:
            print(f'found variances for {num_total_subjects} subjects, extracting indices {training_index_start} up to {training_index_end}')
            node_variances = node_variances[:,training_index_start:training_index_end,:]
            edge_variances = edge_variances[:,training_index_start:training_index_end,:]
    else:
        node_variances, edge_variances = load_and_simulate_ising_model(input_directory=input_directory, file_name_fragment=file_name_fragment, training_index_start=training_index_start, training_index_end=training_index_end, sim_length=sim_length)
        torch.save(obj=node_variances, f=node_variance_file)
        print(f'{time.time() - code_start_time:.3f}, saved {node_variance_file}')
        torch.save(obj=edge_variances, f=edge_variance_file)
        print(f'{time.time() - code_start_time:.3f}, saved {edge_variance_file}')
    node_feature_names = ['x', 'y', 'z'] + ['radius', 'inclination', 'azimuth'] + ['thickness', 'myelination', 'curvature', 'sulcus depth'] + ['mean SC']
    print( 'node_feature_names size', len(node_feature_names) )
    edge_feature_names = ['distance'] + [f'|{pname} difference|' for pname in node_feature_names] + ['SC']
    print( 'edge_feature_names size', len(edge_feature_names) )
    correlation_names = [f'{fname} vs node variance' for fname in node_feature_names] + [f'{fname} vs edge variance' for fname in edge_feature_names]
    return node_features, edge_features, node_variances, edge_variances, correlation_names

# model_param has size models_per_subject x num_subjects x num_nodes (or num_pairs)
# features has size num_subjects x num_nodes (or num_pairs) x num_features
# We want to replicate features across models_per_subject and replicate model_param across num_features
# and take the correlation over models per subject, subjects, and nodes (or node pairs)
# so that we end up with a correlation matrix that is 1D with num_features elements.
def model_feature_correlation(model_param:torch.Tensor, feature:torch.Tensor, epsilon:torch.float=0.0):
    std_1, mean_1 = torch.std_mean( model_param, dim=(0,1,2) )# Take std and mean over model instance, subject, and node/pair.
    std_2, mean_2 = torch.std_mean( feature, dim=(0,1) )# Take std and mean over subject and node/pair
    return ( torch.mean( model_param.unsqueeze(dim=-1) * feature.unsqueeze(dim=0), dim=(0,1,2) ) - mean_1 * mean_2 + epsilon )/(std_1 * std_2 + epsilon)

def get_correlations(node_features:torch.Tensor, edge_features:torch.Tensor, node_variances:torch.Tensor, edge_variances:torch.Tensor, epsilon:torch.float=0.0):
    h_correlations = model_feature_correlation(model_param=node_variances, feature=node_features, epsilon=epsilon)
    J_correlations = model_feature_correlation(model_param=edge_variances, feature=edge_features, epsilon=epsilon)
    return torch.cat( (h_correlations, J_correlations), dim=0 )

def compare_correlation_to_permuted(node_features:torch.Tensor, edge_features:torch.Tensor, node_variances:torch.Tensor, edge_variances:torch.Tensor, num_permutations:int, shuffle_subjects:bool, shuffle_nodes:bool, correlation_names:list, true_correlations:torch.Tensor, file_name:str):
    num_subjects = node_features.size(dim=0)
    num_nodes = node_features.size(dim=1)
    num_pairs = edge_features.size(dim=1)
    num_correlations = len(correlation_names)
    if os.path.exists(file_name):
        perm_correlations = torch.load(f=file_name)
        print(f'loaded correlations from {file_name}')
    else:
        perm_correlations = torch.zeros( (num_permutations, num_correlations), dtype=float_type, device=device )
        # Because of how indexing works in PyTorch, we have to permute one index at a time.
        for perm in range(num_permutations):
            # print(f'time {time.time() - code_start_time:.3f},\tpermutation {perm} of {num_permutations}...')
            node_features_perm = node_features
            edge_features_perm = edge_features
            if shuffle_subjects:
                subject_order = torch.randperm(n=num_subjects, dtype=int_type, device=device)
                node_features_perm = node_features_perm[subject_order,:,:]
                edge_features_perm = edge_features_perm[subject_order,:,:]
            if shuffle_nodes:
                node_order = torch.randperm(n=num_nodes, dtype=int_type, device=device)
                node_features_perm = node_features_perm[:,node_order,:]
                pair_order = torch.randperm(n=num_pairs, dtype=int_type, device=device)
                edge_features_perm = edge_features_perm[:,pair_order,:]
            perm_correlations[perm,:] = get_correlations(node_features=node_features_perm, edge_features=edge_features_perm, node_variances=node_variances, edge_variances=edge_variances)
        torch.save(obj=perm_correlations, f=file_name)
        print(f'saved correlations to {file_name}')
    p_bigger = torch.count_nonzero( perm_correlations.abs() >= true_correlations.abs().unsqueeze(dim=0), dim=0 )/num_permutations
    return p_bigger, perm_correlations

def summarize_distribution(values:torch.Tensor):
    values = values.flatten()
    quantile_cutoffs = torch.tensor([0.005, 0.5, 0.995], dtype=float_type, device=device)
    quantiles = torch.quantile(values, quantile_cutoffs)
    min_val = torch.min(values)
    max_val = torch.max(values)
    return f'median\t{quantiles[1].item():.3g}\t0.5%ile\t{quantiles[0].item():.3g}\t99.5%ile\t{quantiles[2].item():.3g}\tmin\t{min_val.item():.3g}\tmax\t{max_val.item():.3g}'

def try_all_permutation_cases(node_features:torch.Tensor, edge_features:torch.Tensor, node_variances:torch.Tensor, edge_variances:torch.Tensor, num_permutations:int, correlation_names:list, output_directory:str, file_name_fragment:str):
    true_correlations_file = os.path.join(output_directory, f'true_correlations_{file_name_fragment}.pt')
    if os.path.exists(true_correlations_file):
        true_correlations = torch.load(true_correlations_file)
        print(f'loaded true correlations from {true_correlations_file}')
    else:
        print(f'time {time.time() - code_start_time:.3f},\tstarting true correlations...')
        true_correlations = get_correlations(node_features=node_features, edge_features=edge_features, node_variances=node_variances, edge_variances=edge_variances)
        torch.save(obj=true_correlations, f=true_correlations_file)
        print(f'saved true correlations to {true_correlations_file}')
    print(f'time {time.time() - code_start_time:.3f},\tstarting subject permutations...')
    perm_correlations_subjects_file = os.path.join(output_directory, f'perm_correlations_subjects_{file_name_fragment}.pt')
    p_bigger_subjects, perm_correlations_subjects = compare_correlation_to_permuted(node_features=node_features, edge_features=edge_features, node_variances=node_variances, edge_variances=edge_variances, num_permutations=num_permutations, shuffle_subjects=True, shuffle_nodes=False, correlation_names=correlation_names, true_correlations=true_correlations, file_name=perm_correlations_subjects_file)
    print(f'time {time.time() - code_start_time:.3f},\tstarting node and node-pair permutations...')
    perm_correlations_nodes_file = os.path.join(output_directory, f'perm_correlations_nodes_{file_name_fragment}.pt')
    p_bigger_nodes, perm_correlations_nodes = compare_correlation_to_permuted(node_features=node_features, edge_features=edge_features, node_variances=node_variances, edge_variances=edge_variances, num_permutations=num_permutations, shuffle_subjects=False, shuffle_nodes=True, correlation_names=correlation_names, true_correlations=true_correlations, file_name=perm_correlations_nodes_file)
    print(f'time {time.time() - code_start_time:.3f},\tstarting (subject, node) and (subject, node-pair) permutations...')
    perm_correlations_both_file = os.path.join(output_directory, f'perm_correlations_both_{file_name_fragment}.pt')
    p_bigger_both, perm_correlations_both = compare_correlation_to_permuted(node_features=node_features, edge_features=edge_features, node_variances=node_variances, edge_variances=edge_variances, num_permutations=num_permutations, shuffle_subjects=True, shuffle_nodes=True, correlation_names=correlation_names, true_correlations=true_correlations, file_name=perm_correlations_both_file)
    num_correlations = len(correlation_names)
    for correlation_index in range(num_correlations):
        subjects_distribution_str = summarize_distribution(perm_correlations_subjects[:,correlation_index])
        nodes_distribution_str = summarize_distribution(perm_correlations_nodes[:,correlation_index])
        both_distribution_str = summarize_distribution(perm_correlations_both[:,correlation_index])
        print(f'time\t{time.time() - code_start_time:.3f}\t{correlation_names[correlation_index]}\tcorrelation\t{true_correlations[correlation_index]:.3g}\tprobability abs(correlation) with randomized subjects >= actual\t{p_bigger_subjects[correlation_index]:.3g}\t{subjects_distribution_str}\tprobability abs(correlation) with randomized nodes >= actual\t{p_bigger_nodes[correlation_index]:.3g}\t{nodes_distribution_str}\tprobability abs(correlation) with randomized subjects and nodes >= actual\t{p_bigger_both[correlation_index]:.3g}\t{both_distribution_str}')
    return true_correlations, perm_correlations_subjects, perm_correlations_nodes, perm_correlations_both

features_file_name_fragment = f'variance_{file_name_fragment}_sim_length_{sim_length}'
perms_file_name_fragment = f'{features_file_name_fragment}_permutations_{num_permutations}'
node_features, edge_features, node_variances, edge_variances, correlation_names = load_features_and_variances(input_directory=input_directory, file_name_fragment=file_name_fragment, training_index_start=training_index_start, training_index_end=training_index_end, sim_length=sim_length)
true_correlations, perm_correlations_subjects, perm_correlations_nodes, perm_correlations_both = try_all_permutation_cases(node_features=node_features, edge_features=edge_features, node_variances=node_variances, edge_variances=edge_variances, num_permutations=num_permutations, correlation_names=correlation_names, output_directory=output_directory, file_name_fragment=perms_file_name_fragment)
print(f'time {time.time()-code_start_time:.3f}, done')