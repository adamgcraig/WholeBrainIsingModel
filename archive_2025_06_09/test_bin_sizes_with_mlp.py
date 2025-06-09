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

parser = argparse.ArgumentParser(description="Group (structural feature, Ising model parameter) pairs with similar feature values into bins, and train an MLP to predict mean and variance of the param bin given those of the feature bin. Try it with different bin sizes.")
parser.add_argument("-a", "--input_directory", type=str, default='E:\\Ising_model_results_daai', help="directory from which we read the training examples")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the trained models and pandas DataFrame of loss results")
parser.add_argument("-c", "--file_name_fragment", type=str, default='group_training_and_individual_all', help="part of the input example files between example_[index]_ and _[validation|training]_example.pt, will also be part of the output file names")
parser.add_argument("-d", "--training_index_start", type=int, default=1, help="first index of training subjects")
parser.add_argument("-e", "--training_index_end", type=int, default=670, help="last index of training subjects + 1")
parser.add_argument("-p", "--validation_index_start", type=int, default=670, help="first index of validation subjects")
parser.add_argument("-q", "--validation_index_end", type=int, default=754, help="last index of validation subjects + 1")
parser.add_argument("-f", "--num_permutations", type=int, default=10, help="number of permutations of permuted pairings to try")
parser.add_argument("-g", "--abs_params", action='store_true', default=False, help="Set this flag in order to take the absolute values of parameters.")
parser.add_argument("-j", "--multiply_beta", action='store_true', default=False, help="Set this flag in order to multiply beta into the h and J parameters before taking the correlations.")
parser.add_argument("-k", "--num_epochs", type=int, default=1000, help="number of epochs for which to train the h and J prediction models")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate with which to train the h and J prediction models")
parser.add_argument("-m", "--hidden_layer_width", type=int, default=13, help="number of nodes per layer to use in the h and J prediction models")
parser.add_argument("-n", "--num_hidden_layers", type=int, default=1, help="number of hidden layers to use in the h and J prediction models")
parser.add_argument("-o", "--neighborhood_in_std_devs_increment", type=float, default=0.1, help="width of the neighborhood around each point in each feature direction in terms of standard deviations of that feature")
# parser.add_argument("-p", "--points_per_bin", type=int, default=100, help="number of individual samples to use in each bin (We drop bins with too few and sample randomly from bins with too many.)")
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
validation_index_start = args.validation_index_start
print(f'validation_index_start={validation_index_start}')
validation_index_end = args.validation_index_end
print(f'validation_index_end={validation_index_end}')
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
neighborhood_in_std_devs_increment = args.neighborhood_in_std_devs_increment
print(f'neighborhood_in_std_devs_increment={neighborhood_in_std_devs_increment}')
# points_per_bin = args.points_per_bin
# print(f'points_per_bin={points_per_bin}')

if abs_params:
    abs_str = 'abs_params'
else:
    abs_str = 'signed_params'

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

def load_structural_features(input_directory:str, file_name_fragment:str):

    node_features_file = os.path.join(input_directory, f'node_features_{file_name_fragment}.pt')
    node_features = torch.load(f=node_features_file)
    print( 'loaded node_features size', node_features.size() )
    # node_features = node_features[training_index_start:training_index_end,:,:]

    edge_features_file = os.path.join(input_directory, f'edge_features_{file_name_fragment}.pt')
    edge_features = torch.load(f=edge_features_file)
    print( 'loaded edge_features size', edge_features.size() )
    # edge_features = edge_features[training_index_start:training_index_end,:,:,:]

    # We want to use the mean SC values of edges to a given node as an additional node feature.
    # It is easier to calculate it here, before we have converted edge_features from square to upper triangular form.
    node_features = torch.cat(  ( node_features, torch.mean(edge_features, dim=2) ), dim=-1  )

    # Select the part of edge features above the diagonal.
    num_nodes = edge_features.size(dim=2)
    triu_rows, triu_cols = isingmodel.get_triu_indices_for_products(num_nodes=num_nodes, device=device)
    edge_features = edge_features[:,triu_rows,triu_cols,:]

    # Append some derived features.
    # We start with node features
    # 0: mean SC, 1: thickness, 2: myelination, 3: curvature, 4: sulcus depth
    # 5: x, 6: y, 7: z
    # and may replace x, y, z with
    # 5: radius, 6: inclination, 7: azimuth.
    # if spherical_coords:
    #     node_features[:,:,:3] = get_spherical_coords(rectangular_coords=node_features[:,:,:3])
    #     coord_names = ['radius', 'inclination', 'azimuth']
    # else:
    #     coord_names = ['x', 'y', 'z']
    spherical_coords = get_spherical_coords(rectangular_coords=node_features[:,:,:3])
    extended_node_features = torch.cat( (spherical_coords, node_features), dim=-1 )

    # We start with edge feature 0: SC and append
    # 1: distance
    # 2-11: abs(diff(p)) where p is each of the node features for the endpoints.
    node_feature_diffs = torch.abs(extended_node_features[:,triu_rows,:]-extended_node_features[:,triu_cols,:])
    distances = torch.sqrt(  torch.sum( torch.square(node_feature_diffs[:,:,:3]), dim=-1, keepdim=True )  )
    extended_edge_features = torch.cat( (edge_features, distances, node_feature_diffs), dim=-1 )
    # num_subjects, num_pairs, original_num_edge_features = edge_features.size()
    # num_node_features = node_features.size(dim=-1)
    # new_num_edge_features = original_num_edge_features + 1 + num_node_features
    # extended_edge_features = torch.zeros( size=(num_subjects, num_pairs, new_num_edge_features), dtype=edge_features.dtype, device=edge_features.device )
    # extended_edge_features[:,:,:original_num_edge_features] = edge_features
    # extended_edge_features[:,:,-num_node_features:] = torch.abs(node_features[:,triu_rows,:]-node_features[:,triu_cols,:])
    # extended_edge_features[:,:,original_num_edge_features] = torch.sqrt(  torch.sum( torch.square(extended_edge_features[:,:,2:5]), dim=-1 )  )
    
    # node_feature_names = coord_names + ['thickness', 'myelination', 'curvature', 'sulcus depth'] + ['mean SC']
    # edge_feature_names = ['SC', 'distance'] + [f'|{pname} difference|' for pname in node_feature_names]
    

    # clone() so that we do not keep the larger underlying Tensor of which node_features is a view.
    # We do not need to do this for extended_edge_features, since it is a new Tensor allocated at the right size.
    return extended_node_features, extended_edge_features

def load_ising_model(input_directory:str, file_name_fragment:str, abs_params:bool):

    ising_model_file = os.path.join(input_directory, f'ising_model_{file_name_fragment}_fold_1_betas_5_steps_1200_lr_0.01.pt')
    ising_model = torch.load(f=ising_model_file)
    print(f'ising model after beta updates {ising_model.num_beta_updates}, param updates {ising_model.num_param_updates}')
    beta = ising_model.beta
    print( 'loaded beta size', beta.size() )
    h = ising_model.h
    print( 'loaded h size', h.size() )
    J = ising_model.J
    print( 'loaded J size', J.size() )
    # target_state_means = ising_model.target_state_means
    # print( 'loaded target_state_mean size', target_state_means.size() )
    target_state_product_means = ising_model.target_state_product_means
    print( 'target_state_product_mean size', target_state_product_means.size() )
    
    triu_rows, triu_cols = ising_model.get_triu_indices_for_products()
    J = J[:,triu_rows,triu_cols]
    target_state_product_means= target_state_product_means[:,triu_rows,triu_cols]

    num_models = h.size(dim=0)
    models_per_subject = ising_model.num_betas_per_target
    num_subjects = num_models//models_per_subject
    beta = beta.unflatten( dim=0, sizes=(models_per_subject, num_subjects) )
    h = h.unflatten( dim=0, sizes=(models_per_subject, num_subjects) )
    J = J.unflatten( dim=0, sizes=(models_per_subject, num_subjects) )
    # target_state_means = target_state_means.unflatten( dim=0, sizes=(models_per_subject, num_subjects) )
    target_state_product_means = target_state_product_means.unflatten( dim=0, sizes=(models_per_subject, num_subjects) )

    # beta = beta[:,training_index_start:training_index_end,:]
    # h = h[:,training_index_start:training_index_end,:]
    # J = J[:,training_index_start:training_index_end,:]
    # target_state_means = target_state_means[:,training_index_start:training_index_end,:]
    # target_state_product_means = target_state_product_means[:,training_index_start:training_index_end,:]

    if abs_params:
        h = torch.abs(h)
        J = torch.abs(J)
    
    print('After taking upper triangular parts of square matrices, unsqueezing the batch dimension, and selecting training subjects')
    print( 'beta size', beta.size() )
    print( 'h size', h.size() )
    print( 'J size', J.size() )
    print( 'target_state_product_means size', target_state_product_means.size() )

    # clone() so that we do not keep the larger underlying Tensors of which these are views.
    return beta.clone(), h.clone(), J.clone()

# This is likely to run out of memory.
# features: our num_subjects x num_nodes/num_pairs x num_features feature Tensor
# range_over_neighborhood_width a factor to determine the size of the neighborhood around each center point
# The larger the value, the narrower the neighborhood around each feature.
# is_neighbor: a num_points x num_points boolean Tensor where num_points = num_subjects * num_nodes/num_pairs.
# is_neighbor[i,j] is True if samples i and j are close enough along all feature axes.
def get_neighbors(features:torch.Tensor, range_over_neighborhood_width:int):
    features_flat = features.flatten(start_dim=0, end_dim=1)
    feature_mins, _ = torch.min(features_flat, dim=0)
    feature_maxs, _ = torch.max(features_flat, dim=0)
    feature_ranges = feature_maxs - feature_mins
    half_neighborhood_widths = feature_ranges/(2*range_over_neighborhood_width)
    is_neighbor = torch.all( torch.abs(features_flat[:,None,:] - features_flat[None,:,:]) <= half_neighborhood_widths[None,None,:], dim=-1 )
    return is_neighbor

# features: our num_subjects x num_nodes/num_pairs x num_features structural feature Tensor
# param: our models_per_subject x num_subjects x num_nodes/num_pairs Ising model parameter Tensor 
# neighborhood_in_std_devs: the width of the neighborhood around each point in each direction in terms of the standard deviation of the relevant parameter
# neighbor_std_mean_features: num_subjects x num_nodes/num_pairs x 2*num_features Tensor where we alternate between std.dev.s and means of features in neighborhoods
# neighbor_std_mean_param: num_subjects x num_nodes/num_pairs x 2 Tensor where the first feature is the std.dev. and the second is the mean of the param value
# num_neighbors: 1D int-valued Tensor of length num_points where num_neighbors[i] is the number of neighbors of node/pair i
def get_neighborhood_std_mean(features:torch.Tensor, param:torch.Tensor, neighborhood_in_std_devs:float):
    num_subjects, points_per_subject, _ = features.size()
    features_flat = features.flatten(start_dim=0, end_dim=1)
    param_flat = param.flatten(start_dim=1, end_dim=2)
    half_neighborhood_widths = torch.std(features_flat, dim=0)*neighborhood_in_std_devs/2.0
    num_points = num_subjects * points_per_subject
    neighbor_std_features = torch.zeros_like(features_flat)
    neighbor_mean_features = torch.zeros_like(features_flat)
    neighbor_std_param = torch.zeros( (num_points,), dtype=param_flat.dtype, device=param_flat.device )
    neighbor_mean_param = torch.zeros( (num_points,), dtype=param_flat.dtype, device=param_flat.device )
    num_neighbors = torch.zeros( (num_points,), dtype=int_type, device=features_flat.device )
    for center_point in range(num_points):
        is_neighbor = torch.all( torch.abs(features_flat - features_flat[center_point,:]) <= half_neighborhood_widths, dim=-1 )
        num_neighbors[center_point] = torch.count_nonzero(is_neighbor)
        neighbor_std_features[center_point,:], neighbor_mean_features[center_point,:] = torch.std_mean(features_flat[is_neighbor,:], dim=0)
        neighbor_std_param[center_point], neighbor_mean_param[center_point] = torch.std_mean(param_flat[:,is_neighbor])
    # std_mean() returns NaN for the standard deviation of a singleton set, but we want to use 0.
    is_singleton = num_neighbors < 2
    neighbor_std_features[is_singleton,:] = 0.0
    neighbor_std_param[is_singleton] = 0.0
    neighbor_std_mean_features = torch.stack( (neighbor_std_features, neighbor_mean_features), dim=-1 ).flatten(start_dim=-2, end_dim=-1).unflatten( dim=0, sizes=(num_subjects, points_per_subject) )
    neighbor_std_mean_param = torch.stack( (neighbor_std_param, neighbor_mean_param), dim=-1 ).unflatten( dim=0, sizes=(num_subjects, points_per_subject) )
    return neighbor_std_mean_features, neighbor_std_mean_param, num_neighbors

def load_features(input_directory:str, file_name_fragment:str, abs_params:bool):
    beta, h, J = load_ising_model(input_directory=input_directory, file_name_fragment=file_name_fragment, abs_params=abs_params)
    node_features, edge_features = load_structural_features(input_directory=input_directory, file_name_fragment=file_name_fragment)
    return node_features, edge_features, beta, h, J

def get_mlp_predictions(training_features:torch.Tensor, training_param:torch.Tensor, validation_features:torch.Tensor, num_epochs:int=1000, learning_rate:float=0.001, num_hidden_layers:int=1, hidden_layer_width:int=7):
    model = UniformMultiLayerPerceptron( num_in_features=training_features.size(dim=-1), num_out_features=training_param.size(dim=-1), hidden_layer_width=hidden_layer_width, num_hidden_layers=num_hidden_layers, dtype=training_features.dtype, device=training_features.device )
    optimizer = torch.optim.Adam( params=model.parameters(), lr=learning_rate )
    loss_fn = torch.nn.MSELoss()
    # param_prediction = model(features)
    # loss = loss_fn(param_prediction, param)
    # print(f'initial loss {loss:.3g}')
    for _ in range(num_epochs):
        optimizer.zero_grad()
        param_prediction = model(training_features)
        loss = loss_fn(param_prediction, training_param)
        loss.backward()
        optimizer.step()
        # print(f'epoch {epoch+1}, loss {loss:.3g}')
    return param_prediction.detach(), model(validation_features).detach()

# model_param has size models_per_subject x num_subjects x num_nodes (or num_pairs)
# features has size num_subjects x num_nodes (or num_pairs) x num_features
# We want to replicate features across models_per_subject and replicate model_param across num_features
# and take the correlation over models per subject, subjects, and nodes (or node pairs)
# so that we end up with a correlation matrix that is 1D with num_features elements.
def get_param_std_mean_correlations(true_param:torch.Tensor, predicted_param:torch.Tensor, epsilon:torch.float=0.0):
    std_1, mean_1 = torch.std_mean( true_param, dim=(0,1) )# Take std and mean over subject and node/pair.
    std_2, mean_2 = torch.std_mean( predicted_param, dim=(0,1) )# Take std and mean over subject and node/pair.
    return ( torch.mean( true_param * predicted_param, dim=(0,1) ) - mean_1 * mean_2 + epsilon )/(std_1 * std_2 + epsilon)

def get_correlations(training_node_features:torch.Tensor, training_edge_features:torch.Tensor, training_h:torch.Tensor, training_J:torch.Tensor, validation_node_features:torch.Tensor, validation_edge_features:torch.Tensor, validation_h:torch.Tensor, validation_J:torch.Tensor, num_epochs:int=1000, learning_rate:float=0.001, num_hidden_layers:int=1, hidden_layer_width:int=7, epsilon:torch.float=0.0):
    print(f'time {time.time()-code_start_time:.3f}, training node model...')
    training_h_prediction, validation_h_prediction = get_mlp_predictions(training_features=training_node_features, training_param=training_h, validation_features=validation_node_features, num_epochs=num_epochs, learning_rate=learning_rate, num_hidden_layers=num_hidden_layers, hidden_layer_width=hidden_layer_width)
    with torch.no_grad():
        training_h_correlations = get_param_std_mean_correlations(true_param=training_h, predicted_param=training_h_prediction, epsilon=epsilon)
        validation_h_correlations = get_param_std_mean_correlations(true_param=validation_h, predicted_param=validation_h_prediction, epsilon=epsilon)
    print(f'time {time.time()-code_start_time:.3f}, h prediction correlation for training std.dev.s {training_h_correlations[0]:.3g}, for training means {training_h_correlations[1]:.3g}, for validation std.dev.s {validation_h_correlations[0]:.3g}, for validation means {validation_h_correlations[1]:.3g}')
    print(f'time {time.time()-code_start_time:.3f}, training edge model...')
    training_J_prediction, validation_J_prediction = get_mlp_predictions(training_features=training_edge_features, training_param=training_J, validation_features=validation_edge_features, num_epochs=num_epochs, learning_rate=learning_rate, num_hidden_layers=num_hidden_layers, hidden_layer_width=hidden_layer_width)
    with torch.no_grad():
        training_J_correlations = get_param_std_mean_correlations(true_param=training_J, predicted_param=training_J_prediction, epsilon=epsilon)
        validation_J_correlations = get_param_std_mean_correlations(true_param=validation_J, predicted_param=validation_J_prediction, epsilon=epsilon)
    print(f'time {time.time()-code_start_time:.3f}, J prediction correlation for std.dev.s {training_J_correlations[0]:.3g}, for means {training_J_correlations[1]:.3g}, for validation std.dev.s {validation_J_correlations[0]:.3g}, for validation means {validation_J_correlations[1]:.3g}')
    return torch.cat( (training_h_correlations, validation_h_correlations, training_J_correlations, validation_J_correlations), dim=0 )

def summarize_distribution(values:torch.Tensor):
    values = values.flatten()
    quantile_cutoffs = torch.tensor([0.005, 0.5, 0.995], dtype=float_type, device=device)
    quantiles = torch.quantile(values, quantile_cutoffs)
    min_val = torch.min(values)
    max_val = torch.max(values)
    return f'median\t{quantiles[1].item():.3g}\t0.5%ile\t{quantiles[0].item():.3g}\t99.5%ile\t{quantiles[2].item():.3g}\tmin\t{min_val.item():.3g}\tmax\t{max_val.item():.3g}'

with torch.no_grad():
    node_features, edge_features, beta, h, J = load_features(input_directory=input_directory, file_name_fragment=file_name_fragment, abs_params=abs_params)
    if multiply_beta:
        h = beta*h
        J = beta*J
        multiply_beta_str = 'times_beta'
    else:
        multiply_beta_str = 'no_beta'
neighborhood_in_std_devs = 0.0
neighborhood_less_than_all = True
training_node_features = node_features[training_index_start:training_index_end,:,:]
training_edge_features = edge_features[training_index_start:training_index_end,:,:]
training_h = h[:,training_index_start:training_index_end,:]
training_J = J[:,training_index_start:training_index_end,:]
validation_node_features = node_features[validation_index_start:validation_index_end,:,:]
validation_edge_features = edge_features[validation_index_start:validation_index_end,:,:]
validation_h = h[:,validation_index_start:validation_index_end,:]
validation_J = J[:,validation_index_start:validation_index_end,:]
correlations = torch.zeros( (num_permutations,8), dtype=float_type, device=device )
while neighborhood_less_than_all:
    training_std_mean_node_features, training_std_mean_h, training_num_node_neighbors = get_neighborhood_std_mean(features=training_node_features, param=training_h, neighborhood_in_std_devs=neighborhood_in_std_devs)
    print(f'time {time.time()-code_start_time:.3f}, binned training node data, sizes min {training_num_node_neighbors.min():.3g}, max {training_num_node_neighbors.max():.3g}')
    training_std_mean_edge_features, training_std_mean_J, training_num_edge_neighbors = get_neighborhood_std_mean(features=training_edge_features, param=training_J, neighborhood_in_std_devs=neighborhood_in_std_devs)
    print(f'time {time.time()-code_start_time:.3f}, binned training edge data, sizes min {training_num_edge_neighbors.min():.3g}, max {training_num_edge_neighbors.max():.3g}')
    validation_std_mean_node_features, validation_std_mean_h, validation_num_node_neighbors = get_neighborhood_std_mean(features=validation_node_features, param=validation_h, neighborhood_in_std_devs=neighborhood_in_std_devs)
    print(f'time {time.time()-code_start_time:.3f}, binned validation node data, sizes min {validation_num_node_neighbors.min():.3g}, max {validation_num_node_neighbors.max():.3g}')
    validation_std_mean_edge_features, validation_std_mean_J, validation_num_edge_neighbors = get_neighborhood_std_mean(features=validation_edge_features, param=validation_J, neighborhood_in_std_devs=neighborhood_in_std_devs)
    print(f'time {time.time()-code_start_time:.3f}, binned validation edge data, sizes min {validation_num_edge_neighbors.min():.3g}, max {validation_num_edge_neighbors.max():.3g}')
    # correlation_names = ['std. dev. h-prediction correlation', 'mean h-prediction correlation', 'std. dev. J-prediction correlation', 'mean J-prediction correlation']
    features_file_name_fragment = f'{file_name_fragment}_{abs_str}_{multiply_beta_str}_epochs_{num_epochs}_lr_{learning_rate:.3g}_depth_{num_hidden_layers}_width_{hidden_layer_width}_neighborhood_{neighborhood_in_std_devs}'
    # training_std_mean_node_features_file = os.path.join(output_directory, f'training_std_mean_node_features_{features_file_name_fragment}.pt')
    # torch.save( obj=training_std_mean_node_features, f=training_std_mean_node_features_file )
    # training_std_mean_edge_features_file = os.path.join(output_directory, f'training_std_mean_edge_features_{features_file_name_fragment}.pt')
    # torch.save( obj=training_std_mean_edge_features, f=training_std_mean_edge_features_file )
    # training_std_mean_h_file = os.path.join(output_directory, f'training_std_mean_h_{features_file_name_fragment}.pt')
    # torch.save( obj=training_std_mean_h, f=training_std_mean_h_file )
    # training_std_mean_J_file = os.path.join(output_directory, f'training_std_mean_node_J_{features_file_name_fragment}.pt')
    # torch.save( obj=training_std_mean_J, f=training_std_mean_J_file )
    # training_num_node_neighbors_file = os.path.join(output_directory, f'training_num_node_neighbors_{features_file_name_fragment}.pt')
    # torch.save( obj=training_num_node_neighbors, f=training_num_node_neighbors_file )
    # training_num_edge_neighbors_file = os.path.join(output_directory, f'training_num_edge_neighbors_{features_file_name_fragment}.pt')
    # torch.save( obj=training_num_edge_neighbors, f=training_num_edge_neighbors_file )
    # true_correlations, perm_correlations_subjects, perm_correlations_nodes, perm_correlations_both = try_all_permutation_cases(neighbor_std_mean_node_features=training_std_mean_node_features, neighbor_std_mean_edge_features=training_std_mean_edge_features, neighbor_std_mean_h=training_std_mean_h, neighbor_std_mean_J=training_std_mean_J, num_permutations=num_permutations, correlation_names=correlation_names, output_directory=output_directory, file_name_fragment=perms_file_name_fragment, num_epochs=num_epochs, learning_rate=learning_rate, num_hidden_layers=num_hidden_layers, hidden_layer_width=hidden_layer_width)
    for rep in range(num_permutations):
        print(f'time {time.time()-code_start_time:.3f}, rep {rep+1}')
        correlations[rep,:] = get_correlations(training_node_features=training_std_mean_node_features, training_edge_features=training_std_mean_edge_features, training_h=training_std_mean_h, training_J=training_std_mean_J, validation_node_features=validation_std_mean_node_features, validation_edge_features=validation_std_mean_edge_features, validation_h=validation_std_mean_h, validation_J=validation_std_mean_J, num_epochs=num_epochs, learning_rate=learning_rate, num_hidden_layers=num_hidden_layers, hidden_layer_width=hidden_layer_width)
    correlations_file_name = os.path.join(output_directory, f'correlations_{features_file_name_fragment}_reps_{num_permutations}_neighborhood_{neighborhood_in_std_devs:.3g}.pt')
    torch.save(obj=correlations, f=correlations_file_name)
    print(f'time {time.time()-code_start_time:.3f}, saved {correlations_file_name}')
    # Stop when all points are in a single neighborhood.
    neighborhood_less_than_all = ( training_num_node_neighbors.min() < training_num_node_neighbors.numel() ) or ( training_num_edge_neighbors.min() < training_num_edge_neighbors.numel() )
    if neighborhood_less_than_all:
        neighborhood_in_std_devs += neighborhood_in_std_devs_increment
print(f'time {time.time()-code_start_time:.3f}, done')