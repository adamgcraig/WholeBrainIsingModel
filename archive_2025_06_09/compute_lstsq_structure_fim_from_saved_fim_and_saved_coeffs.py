import os
import torch
import time
import argparse
import isingmodellight
from isingmodellight import IsingModelLight

code_start_time = time.time()
float_type = torch.float
int_type = torch.int

parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the expected values of their observables. Compare to expected values from data. Do an Euler step update. Repeat.")
parser.add_argument("--data_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the Ising model file")
parser.add_argument("--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
parser.add_argument("--model_file_fragment", type=str, default='ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_63000', help="Ising model file name, except .pt extension")
parser.add_argument("--region_feature_file_part", type=str, default='node_features_all_as_is', help='region feature file name except for the .pt file extension')
parser.add_argument("--sc_file_part", type=str, default='edge_features_all_as_is', help='SC file name except for the .pt file extension')
parser.add_argument("--individual_model_file_part", type=str, default='ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_to_thresh_1_reps_5_subj_837_individual_updates_50000', help="Ising model file name for individual models, except .pt extension")
parser.add_argument("--sim_length", type=int, default=64980, help="number of simulation steps to run")
parser.add_argument("--rep_index", type=int, default=89, help="index of selected replica")
parser.add_argument("--target_index", type=int, default=10, help="index of selected threshold or subject")
parser.add_argument("--model_device", type=str, default='cuda', help="device on which the model Tensors existed when the other script passed it to torch.save()")
parser.add_argument("--device", type=str, default='cpu', help="device to which we want to load the model now")
parser.add_argument("--pca_iterations", type=int, default=10, help="number of iterations of lowrank PCA")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
model_file_fragment = args.model_file_fragment
print(f'model_file_fragment={model_file_fragment}')
region_feature_file_part = args.region_feature_file_part
print(f'region_feature_file_part={region_feature_file_part}')
sc_file_part = args.sc_file_part
print(f'sc_file_part={sc_file_part}')
individual_model_file_part = args.individual_model_file_part
print(f'individual_model_file_part={individual_model_file_part}')
sim_length = args.sim_length
print(f'sim_length={sim_length}')
rep_index = args.rep_index
print(f'rep_index={rep_index}')
target_index = args.target_index
print(f'target_index={target_index}')
model_device_str = args.model_device
print(f'model_device={model_device_str}')
model_device = torch.device(model_device_str)
device_str = args.device
print(f'device={device_str}')
device = torch.device(device_str)
pca_iterations = args.pca_iterations
print(f'pca_iterations={pca_iterations}')

output_file_fragment = f'{model_file_fragment}_target_{target_index}_rep_{rep_index}_test_length_{sim_length}'

def save_and_print(mat:torch.Tensor, mat_name:str):
    mat_file = os.path.join(output_directory, f'{mat_name}_{output_file_fragment}.pt')
    torch.save(obj=mat, f=mat_file)
    print( f'time {time.time()-code_start_time:.3f}, saved {mat_file}, size', mat.size(), f'min {mat.min():.3g}, mean {mat.mean():.3g}, max {mat.max():.3g}' )
    return 0

def downsample_save_and_print(mat:torch.Tensor, mat_name:str, downsample_factor:int=10):
    mat = mat.unflatten( dim=-1, sizes=(-1, downsample_factor) ).mean(dim=-1).transpose(dim0=0, dim1=1).unflatten( dim=-1, sizes=(-1, downsample_factor) ).mean(dim=-1).transpose(dim0=0, dim1=1)
    mat_file = os.path.join(output_directory, f'{mat_name}_{output_file_fragment}_downsample_{downsample_factor}.pt')
    torch.save(obj=mat, f=mat_file)
    print( f'time {time.time()-code_start_time:.3f}, saved {mat_file}, size', mat.size(), f'min {mat.min():.3g}, mean {mat.mean():.3g}, max {mat.max():.3g}' )
    return 0

def save_to_cpu_and_print(mat:torch.Tensor, mat_name:str):
    mat_file = os.path.join(output_directory, f'{mat_name}_{output_file_fragment}_cpu.pt')
    torch.save( obj=mat.detach().cpu(), f=mat_file )
    print( f'time {time.time()-code_start_time:.3f}, saved {mat_file}, size', mat.size(), f'min {mat.min():.3g}, mean {mat.mean():.3g}, max {mat.max():.3g}' )
    return 0

def load_and_print(mat_name:str):
    mat_file = os.path.join(output_directory, f'{mat_name}_{output_file_fragment}.pt')
    mat = torch.load(f=mat_file, weights_only=False, map_location=device_str)
    print( f'time {time.time()-code_start_time:.3f}, loaded {mat_file}, size', mat.size(), f'min {mat.min():.3g}, mean {mat.mean():.3g}, max {mat.max():.3g}' )
    return mat

def get_structural_features():
    region_feature_file = os.path.join(data_directory, f'{region_feature_file_part}.pt')
    region_features = torch.flatten( input=torch.load(f=region_feature_file, weights_only=False, map_location=device)[:,:,:4], start_dim=-2, end_dim=-1 )
    print( f'time {time.time()-code_start_time:.3f}, loaded {region_feature_file}, size', region_features.size(), f'min {region_features.min():.3g}, mean {region_features.mean():.3g}, max {region_features.max():.3g}' )
    sc_file = os.path.join(data_directory, f'{sc_file_part}.pt')
    sc = torch.load(f=sc_file, weights_only=False, map_location=device)[:,:,0]
    print( f'time {time.time()-code_start_time:.3f}, loaded {sc_file}, size', sc.size(), f'min {sc.min():.3g}, mean {sc.mean():.3g}, max {sc.max():.3g}' )
    return torch.cat( tensors=(region_features, sc), dim=-1 )

def get_individual_model_parameters():
    model_file = os.path.join(data_directory, f'{individual_model_file_part}.pt')
    model = torch.load(f=model_file, weights_only=False, map_location=device)
    h = torch.mean(input=model.h, dim=0)
    print( f'time {time.time()-code_start_time:.3f}, loaded {model_file} h, size', h.size(), f'min {h.min():.3g}, mean {h.mean():.3g}, max {h.max():.3g}' )
    triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=h.size(dim=-1), device=device )
    J = torch.mean(input=model.J[:,:,triu_rows,triu_cols], dim=0)
    print( f'time {time.time()-code_start_time:.3f}, loaded {model_file} J, size', J.size(), f'min {J.min():.3g}, mean {J.mean():.3g}, max {J.max():.3g}' )
    num_subjects = h.size(dim=0)
    one_col = torch.ones( size=(num_subjects,1), dtype=h.dtype, device=h.device )
    return torch.cat( tensors=(h, J, one_col), dim=-1 )

def get_all_lstsq_model():
    params = get_individual_model_parameters()
    features = get_structural_features()
    coeffs = torch.linalg.lstsq(features, params).solution
    coeffs_file = os.path.join(output_directory, f'coeffs_lstsq_all_to_all_{individual_model_file_part}_mean_over_reps.pt')
    torch.save(obj=coeffs, f=coeffs_file)
    print( f'time {time.time()-code_start_time:.3f}, saved {coeffs_file} size', coeffs.size(), f'min {coeffs.min():.3g}, mean {coeffs.mean():.3g}, max {coeffs.max():.3g}' )
    return coeffs[:-1,:]

def get_one_model():
    model_file = os.path.join(data_directory, f'{model_file_fragment}.pt')
    model = torch.load(f=model_file, weights_only=False, map_location={model_device_str:device_str})
    # model = torch.load(f=model_file, weights_only=False)
    print(f'time {time.time()-code_start_time:.3f}, loaded {model_file}')
    # Select one model, but create singleton dimensions, since the simulation code assumes that these dimensions exist.
    # Make copies instead of just slices so that we can free the memory from the larger ensemble of models.
    num_nodes = model.h.size(dim=-1)
    model_dtype = model.h.dtype
    new_h = torch.zeros( size=(1, 1, num_nodes), dtype=model_dtype, device=device )
    new_h[0, 0, :] = model.h[rep_index, target_index, :]
    model.h = new_h
    print( 'selected h, size', model.h.size(), f'min {model.h.min():.3g}, mean {model.h.mean():.3g}, max {model.h.max():.3g}' )
    new_J = torch.zeros( size=(1, 1, num_nodes, num_nodes), dtype=model_dtype, device=device )
    new_J[0, 0, :, :] = model.J[rep_index, target_index, :, :]
    model.J = new_J
    print( 'selected J, size', model.J.size(), f'min {model.J.min():.3g}, mean {model.J.mean():.3g}, max {model.J.max():.3g}' )
    new_beta = torch.zeros( size=(1, 1), dtype=model_dtype, device=device )
    new_beta[0, 0] = model.beta[rep_index, target_index]
    model.beta = new_beta
    print( 'selected beta, size', model.beta.size(), f'min {model.beta.min():.3g}, mean {model.beta.mean():.3g}, max {model.beta.max():.3g}' )
    new_s = torch.zeros( size=(1, 1, num_nodes), dtype=model_dtype, device=device )
    new_s[0, 0, :] = model.s[rep_index, target_index, :]
    model.s = new_s
    print( 'selected s, size', model.s.size(), f'min {model.s.min():.3g}, mean {model.s.mean():.3g}, max {model.s.max():.3g}' )
    return model
    
def get_time_series():
    # Squeeze out the singleton dimensions from the time series, since we no longer need them.
    ts = torch.squeeze( input=get_one_model().simulate_and_record_time_series_pmb(num_steps=sim_length), dim=0 ).squeeze(dim=0)
    save_and_print(mat=ts, mat_name='ts')
    return ts

def get_augmented_time_series():
    ts = torch.transpose( input=load_and_print(mat_name='ts'), dim0=0, dim1=1 )
    num_nodes = ts.size(dim=1)
    triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products(num_nodes=num_nodes, device=ts.device)
    augmented_ts = torch.cat( tensors=(ts, ts[:,triu_rows] * ts[:,triu_cols]), dim=1 )
    print( f'time {time.time()-code_start_time:.3f}, computed augmented time series, size', augmented_ts.size(), f'min {augmented_ts.min():.3g}, mean {augmented_ts.mean():.3g}, max {augmented_ts.max():.3g}' )
    # Do not save this. It is too big but easy to recompute from the original ts.
    augmented_ts -= augmented_ts.mean(dim=0,keepdim=True)
    print( f'time {time.time()-code_start_time:.3f}, centered augmented time series, size', augmented_ts.size(), f'min {augmented_ts.min():.3g}, mean {augmented_ts.mean():.3g}, max {augmented_ts.max():.3g}' )
    return augmented_ts

def save_augmented_ts_pca_and_return_augmented_ts():
    augmented_ts = get_augmented_time_series()
    num_samples, num_observables = augmented_ts.size()
    (U, S, V) = torch.pca_lowrank( A=augmented_ts, q=min(num_samples, num_observables), center=False, niter=pca_iterations )
    save_and_print(mat=U, mat_name='U')
    save_and_print(mat=S, mat_name='S')
    save_and_print(mat=V, mat_name='V')
    return augmented_ts


def check_fim_for_nans(fim:torch.Tensor):
    num_rows = fim.size(dim=0)
    num_nans = 0
    for row_index in range(num_rows):
        num_nans += torch.count_nonzero( torch.isnan(fim[row_index,:]) )
    return num_nans

def check_fim_for_infs(fim:torch.Tensor):
    num_rows = fim.size(dim=0)
    num_infs = 0
    for row_index in range(num_rows):
        num_infs += torch.count_nonzero( torch.isinf(fim[row_index,:]) )
    return num_infs

def get_fim():
    augmented_ts = get_augmented_time_series()# save_augmented_ts_pca_and_return_augmented_ts()
    mean_agumented_ts = augmented_ts.mean(dim=-1)
    fim = torch.matmul( augmented_ts.transpose(dim0=0, dim1=1), augmented_ts )
    fim /= augmented_ts.size(dim=-1)
    fim -= ( mean_agumented_ts.unsqueeze(dim=-1) * mean_agumented_ts.unsqueeze(dim=-2) )
    save_and_print(mat=fim, mat_name='fim')
    downsample_save_and_print(mat=fim, mat_name='fim')
    # save_to_cpu_and_print(mat=fim, mat_name='fim')
    num_nans = check_fim_for_nans(fim=fim)
    num_infs = check_fim_for_infs(fim=fim)
    print( f'time {time.time()-code_start_time:.3f}, done counting NaNs {num_nans}, Infs {num_infs}' )
    return fim

def load_jacobian():
    jacobian_file = os.path.join(output_directory, f'coeffs_lstsq_all_to_all_ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_to_thresh_1_reps_5_subj_837_individual_updates_50000_mean_over_reps.pt')
    jacobian = torch.clone( torch.load(f=jacobian_file, weights_only=False, map_location=device)[:-1,:] )
    print( f'time {time.time()-code_start_time:.3f}, loaded {jacobian_file}, size', jacobian.size(), f'min {jacobian.min():.3g}, mean {jacobian.mean():.3g}, max {jacobian.max():.3g}' )
    return jacobian

def load_fim():
    fim_file = os.path.join(output_directory, 'fim_ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_63000_target_10_rep_89_test_length_64980.pt')
    fim = torch.load(f=fim_file, weights_only=False, map_location=device)
    print( f'time {time.time()-code_start_time:.3f}, loaded {fim_file}, size', fim.size(), f'min {fim.min():.3g}, mean {fim.mean():.3g}, max {fim.max():.3g}' )
    return fim

def get_jacobian_times_fim(jacobian:torch.Tensor):
    j_fim = torch.matmul( input=jacobian, other=load_fim() )
    print( f'time {time.time()-code_start_time:.3f}, J*FIM, size', j_fim.size(), f'min {j_fim.min():.3g}, mean {j_fim.mean():.3g}, max {j_fim.max():.3g}' )
    save_and_print(mat=j_fim, mat_name='j_fim')
    return j_fim

def get_structure_fim_prerequisites():
    # jacobian = load_jacobian()
    j_fim = load_and_print(mat_name='j_fim')# get_jacobian_times_fim(jacobian=jacobian)
    j_trans = load_and_print(mat_name='j_trans')# torch.transpose(input=jacobian, dim0=0, dim1=1)
    # print( f'time {time.time()-code_start_time:.3f}, J^T, size', j_trans.size(), f'min {j_trans.min():.3g}, mean {j_trans.mean():.3g}, max {j_trans.max():.3g}' )
    # save_and_print(mat=j_trans, mat_name='j_trans')
    return j_fim, j_trans

def get_structure_fim():
    j_fim, j_trans = get_structure_fim_prerequisites()
    # downsample_save_and_print(mat=jacobian, mat_name='coeffs_lstsq_all_to_all_wo_intercept', downsample_factor=10)
    structure_fim = torch.matmul(input=j_fim, other=j_trans)
    print( f'time {time.time()-code_start_time:.3f}, structure FIM = J*FIM*J^T, size', structure_fim.size(), f'min {structure_fim.min():.3g}, mean {structure_fim.mean():.3g}, max {structure_fim.max():.3g}' )
    sf_name = 'struct_fim_lstsq_all_to_all'
    save_and_print(mat=structure_fim, mat_name=sf_name)
    downsample_save_and_print(mat=structure_fim, mat_name=sf_name)
    return structure_fim

def get_fim_eigs():
    eigenvalues, eigenvectors = torch.linalg.eigh( get_fim() )
    eigenvalues_file = os.path.join(output_directory, f'eigenvalues_{output_file_fragment}.pt')
    torch.save(obj=eigenvalues, f=eigenvalues_file)
    print( f'time {time.time()-code_start_time:.3f}, saved {eigenvalues_file}, size', eigenvalues.size(), f'min {eigenvalues.min():.3g}, mean {eigenvalues.mean():.3g}, max {eigenvalues.max():.3g}' )
    eigenvectors_file = os.path.join(output_directory, f'eigenvectors_{output_file_fragment}.pt')
    torch.save(obj=eigenvectors, f=eigenvectors_file)
    print( f'time {time.time()-code_start_time:.3f}, saved {eigenvectors_file}, size', eigenvectors.size(), f'min {eigenvectors.min():.3g}, mean {eigenvectors.mean():.3g}, max {eigenvectors.max():.3g}' )
    return 0

def get_structure_fim_eigs():
    eigenvalues, eigenvectors = torch.linalg.eigh( get_structure_fim() )
    eigenvalues_file = os.path.join(output_directory, f'eigenvalues_{output_file_fragment}.pt')
    torch.save(obj=eigenvalues, f=eigenvalues_file)
    print( f'time {time.time()-code_start_time:.3f}, saved {eigenvalues_file}, size', eigenvalues.size(), f'min {eigenvalues.min():.3g}, mean {eigenvalues.mean():.3g}, max {eigenvalues.max():.3g}' )
    eigenvectors_file = os.path.join(output_directory, f'eigenvectors_{output_file_fragment}.pt')
    torch.save(obj=eigenvectors, f=eigenvectors_file)
    print( f'time {time.time()-code_start_time:.3f}, saved {eigenvectors_file}, size', eigenvectors.size(), f'min {eigenvectors.min():.3g}, mean {eigenvectors.mean():.3g}, max {eigenvectors.max():.3g}' )
    return 0

with torch.no_grad():
    get_structure_fim_eigs()
    print(f'time {time.time()-code_start_time:.3f}, done')