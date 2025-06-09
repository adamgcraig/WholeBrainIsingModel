import os
import torch
import time
import argparse
import hcpdatautilsnopandas as hcp
import isingmodellight
from isingmodellight import IsingModelLight

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Test the effect of binarization covariance and FC for both individual and group fMRI data.")
    parser.add_argument("-a", "--input_directory", type=str, default='E:\\HCP_data', help="containing folder of fMRI_ts_binaries/ts_[6-digit subject ID]_[1|2]_[RL|LR].bin")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the mean state and mean state product PyTorch pickle files")
    parser.add_argument("-c", "--training_subject_start", type=int, default=0, help="first training subject index")
    parser.add_argument("-e", "--training_subject_end", type=int, default=670, help="1 past last training subject index")
    parser.add_argument("-f", "--num_thresholds", type=int, default=13, help="number of binarization thresholds in std. dev.s above the mean to try")
    parser.add_argument("-g", "--min_threshold", type=float, default=-6, help="minimum threshold in std. dev.s")
    parser.add_argument("-i", "--max_threshold", type=float, default=6, help="minimum threshold in std. dev.s")
    args = parser.parse_args()
    print('getting arguments...')
    input_directory = args.input_directory
    print(f'input_directory={input_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    training_subject_start = args.training_subject_start
    print(f'training_subject_start={training_subject_start}')
    training_subject_end = args.training_subject_end
    print(f'training_subject_end={training_subject_end}')
    num_thresholds = args.num_thresholds
    print(f'num_thresholds={num_thresholds}')
    min_threshold = args.min_threshold
    print(f'min_threshold={min_threshold}')
    max_threshold = args.max_threshold
    print(f'max_threshold={max_threshold}')

    def get_cov_and_fc(mean_state:torch.Tensor, mean_state_product:torch.Tensor):
        covariance =  mean_state_product - mean_state.unsqueeze(dim=-1) * mean_state.unsqueeze(dim=-2)
        std_dev = torch.sqrt( torch.diagonal(mean_state_product, offset=0, dim1=-2, dim2=-1).square() - mean_state.square() )
        fc = covariance/( std_dev.unsqueeze(dim=-1) * std_dev.unsqueeze(dim=-2) )
        # If there is no variance for one or the other variable, the values effectively do not co-vary.
        return covariance, fc.nan_to_num(nan=0, posinf=0, neginf=0)
    
    def get_means_binarized(ts:torch.Tensor, threshold:float):
        ts_per_subject, num_subjects, num_nodes, num_steps = ts.size()
        mean_state = torch.zeros( size=(ts_per_subject, num_subjects, num_nodes), dtype=ts.dtype, device=ts.device )
        mean_state_product = torch.zeros( size=(ts_per_subject, num_subjects, num_nodes, num_nodes), dtype=ts.dtype, device=ts.device )
        for step_index in range(num_steps):
            step = 2.0*(ts[:,:,:,step_index] >= threshold).float() - 1.0
            mean_state += step
            mean_state_product += step.unsqueeze(dim=-1) * step.unsqueeze(dim=-2)
        # Divide the sums by number of steps, and then average over scans so that we have one set of means per subject.
        return torch.mean(mean_state/num_steps, dim=0), torch.mean(mean_state_product/num_steps, dim=0)
    
    def get_individual_and_group_cov_and_fc_from_means(mean_state:torch.Tensor, mean_state_product:torch.Tensor):
        covariance, fc = get_cov_and_fc(mean_state=mean_state, mean_state_product=mean_state_product)
        group_mean_state = mean_state.mean(dim=0)
        group_mean_state_product = mean_state_product.mean(dim=0)
        group_covariance, group_fc = get_cov_and_fc(mean_state=group_mean_state, mean_state_product=group_mean_state_product)
        return covariance, fc, group_covariance, group_fc
    
    def get_individual_and_group_cov_and_fc_from_ts(ts:torch.Tensor):
        ts_per_subject, num_subjects, num_nodes, num_steps = ts.size()
        mean_state = torch.zeros( size=(ts_per_subject, num_subjects, num_nodes), dtype=ts.dtype, device=ts.device )
        mean_state_product = torch.zeros( size=(ts_per_subject, num_subjects, num_nodes, num_nodes), dtype=ts.dtype, device=ts.device )
        for step_index in range(num_steps):
            step = ts[:,:,:,step_index]
            mean_state += step
            mean_state_product += step.unsqueeze(dim=-1) * step.unsqueeze(dim=-2)
        # Divide the sums by number of steps, and then average over scans so that we have one set of means per subject.
        mean_state = torch.mean(mean_state/num_steps, dim=0)
        mean_state_product = torch.mean(mean_state_product/num_steps, dim=0)
        covariance, fc, group_covariance, group_fc = get_individual_and_group_cov_and_fc_from_means(mean_state=mean_state, mean_state_product=mean_state_product)
        return covariance, fc, group_covariance, group_fc
    
    def get_rmse(mat1:torch.Tensor, mat2:torch.Tensor):
        return isingmodellight.get_pairwise_rmse_ut( mat1=mat1.unsqueeze(dim=0), mat2=mat2.unsqueeze(dim=0) ).squeeze(dim=0)
    
    def get_corr(mat1:torch.Tensor, mat2:torch.Tensor):
        return isingmodellight.get_pairwise_correlation_ut( mat1=mat1.unsqueeze(dim=0), mat2=mat2.unsqueeze(dim=0), epsilon=0.0 ).squeeze(dim=0)
    
    def get_rmse_group(mat1:torch.Tensor, mat2:torch.Tensor):
        return isingmodellight.get_pairwise_rmse_ut( mat1=mat1.unsqueeze(dim=0).unsqueeze(dim=0), mat2=mat2.unsqueeze(dim=0).unsqueeze(dim=0) ).squeeze(dim=0).squeeze(dim=0)
    
    def get_corr_group(mat1:torch.Tensor, mat2:torch.Tensor):
        return isingmodellight.get_pairwise_correlation_ut( mat1=mat1.unsqueeze(dim=0).unsqueeze(dim=0), mat2=mat2.unsqueeze(dim=0).unsqueeze(dim=0), epsilon=0.0 ).squeeze(dim=0).unsqueeze(dim=0)
    
    data_ts_file = os.path.join(output_directory, 'data_ts_all_as_is.pt')
    data_ts = torch.load(data_ts_file)[:,training_subject_start:training_subject_end,:,:]
    data_std, data_mean = torch.std_mean(input=data_ts, dim=-1, keepdim=True)
    data_ts -= data_mean
    data_ts /= data_std
    print( f'time {time.time()-code_start_time:.3f}, loaded {data_ts_file}, size after selecting training subjects', data_ts.size() )
    covariance, fc, group_covariance, group_fc = get_individual_and_group_cov_and_fc_from_ts(ts=data_ts)
    
    thresholds = torch.linspace(start=min_threshold, end=max_threshold, steps=num_thresholds, dtype=data_ts.dtype, device=data_ts.device)

    num_subjects = training_subject_end - training_subject_start
    cov_rmse = torch.zeros( size=(num_thresholds, num_subjects), dtype=data_ts.dtype, device=data_ts.device )
    cov_corr = torch.zeros( size=(num_thresholds, num_subjects), dtype=data_ts.dtype, device=data_ts.device )
    fc_rmse = torch.zeros( size=(num_thresholds, num_subjects), dtype=data_ts.dtype, device=data_ts.device )
    fc_corr = torch.zeros( size=(num_thresholds, num_subjects), dtype=data_ts.dtype, device=data_ts.device )
    group_cov_rmse = torch.zeros( size=(num_thresholds,), dtype=data_ts.dtype, device=data_ts.device )
    group_cov_corr = torch.zeros( size=(num_thresholds,), dtype=data_ts.dtype, device=data_ts.device )
    group_fc_rmse = torch.zeros( size=(num_thresholds,), dtype=data_ts.dtype, device=data_ts.device )
    group_fc_corr = torch.zeros( size=(num_thresholds,), dtype=data_ts.dtype, device=data_ts.device )
    for threshold_index in range(num_thresholds):
        threshold = thresholds[threshold_index]
        print(f'time {time.time()-code_start_time:.3f}, threshold {threshold_index+1} of {num_thresholds}, {threshold:.3g}')
        binarized_mean_state, binarized_mean_state_product = get_means_binarized(ts=data_ts, threshold=threshold)
        binarized_covariance, binarized_fc, binarized_group_covariance, binarized_group_fc = get_individual_and_group_cov_and_fc_from_means(mean_state=binarized_mean_state, mean_state_product=binarized_mean_state_product)
        cov_rmse[threshold_index,:] = get_rmse(mat1=covariance, mat2=binarized_covariance)
        cov_corr[threshold_index,:] = get_corr(mat1=covariance, mat2=binarized_covariance)
        fc_rmse[threshold_index,:] = get_rmse(mat1=fc, mat2=binarized_fc)
        fc_corr[threshold_index,:] = get_corr(mat1=fc, mat2=binarized_fc)
        group_cov_rmse[threshold_index] = get_rmse_group(mat1=group_covariance, mat2=binarized_group_covariance)
        group_cov_corr[threshold_index] = get_corr_group(mat1=group_covariance, mat2=binarized_group_covariance)
        group_fc_rmse[threshold_index] = get_rmse_group(mat1=group_fc, mat2=binarized_group_fc)
        group_fc_corr[threshold_index] = get_corr_group(mat1=group_fc, mat2=binarized_group_fc)
    
    param_string = f'data_ts_training_vs_binarized_thresholds_{num_thresholds}_min_{min_threshold:.3g}_max_{max_threshold:.3g}'
    cov_rmse_file = os.path.join(output_directory, f'cov_rmse_{param_string}.pt')
    torch.save(obj=cov_rmse, f=cov_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, {cov_rmse_file}')
    cov_corr_file = os.path.join(output_directory, f'cov_corr_{param_string}.pt')
    torch.save(obj=cov_corr, f=cov_corr_file)
    print(f'time {time.time()-code_start_time:.3f}, {cov_corr_file}')
    fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{param_string}.pt')
    torch.save(obj=fc_rmse, f=fc_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, {fc_rmse_file}')
    fc_corr_file = os.path.join(output_directory, f'fc_corr_{param_string}.pt')
    torch.save(obj=fc_corr, f=fc_corr_file)
    print(f'time {time.time()-code_start_time:.3f}, {fc_corr_file}')
    group_cov_rmse_file = os.path.join(output_directory, f'group_cov_rmse_{param_string}.pt')
    torch.save(obj=group_cov_rmse, f=group_cov_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, {group_cov_rmse_file}')
    group_cov_corr_file = os.path.join(output_directory, f'group_cov_corr_{param_string}.pt')
    torch.save(obj=group_cov_corr, f=group_cov_corr_file)
    print(f'time {time.time()-code_start_time:.3f}, {group_cov_corr_file}')
    group_fc_rmse_file = os.path.join(output_directory, f'group_fc_rmse_{param_string}.pt')
    torch.save(obj=group_fc_rmse, f=group_fc_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, {group_fc_rmse_file}')
    group_fc_corr_file = os.path.join(output_directory, f'group_fc_corr_{param_string}.pt')
    torch.save(obj=group_fc_corr, f=group_fc_corr_file)
    print(f'time {time.time()-code_start_time:.3f}, {group_fc_corr_file}')
print(f'time {time.time()-code_start_time:.3f}, done')