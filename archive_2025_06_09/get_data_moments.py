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

    parser = argparse.ArgumentParser(description="Test the effect of binarization threshold on our ability to fit a group Ising model.")
    parser.add_argument("-a", "--input_directory", type=str, default='E:\\HCP_data', help="containing folder of fMRI_ts_binaries/ts_[6-digit subject ID]_[1|2]_[RL|LR].bin")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the mean state and mean state product PyTorch pickle files")
    parser.add_argument("-c", "--training_subject_start", type=int, default=0, help="first training subject index")
    parser.add_argument("-e", "--training_subject_end", type=int, default=670, help="1 past last training subject index")
    args = parser.parse_args()
    print('getting arguments...')
    input_directory = args.input_directory
    print(f'input_directory={input_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    training_subject_start = args.training_subject_start
    print(f'training_subject_start={training_subject_start}')
    training_subject_end = args.training_subject_end

    def moment(data_minus_mean:torch.Tensor, power:float):
        return torch.mean( torch.pow(input=data_minus_mean, exponent=power), dim=-1, keepdim=False )
    
    def print_stats(data:torch.Tensor, name:str):
        data = data.flatten()
        quantile_cutoffs = torch.tensor([0.025, 0.5, 0.975], dtype=float_type, device=device)
        quantiles = torch.quantile(data, quantile_cutoffs)
        min_val = torch.min(data)
        max_val = torch.max(data)
        print(f'The distribution of {name} values has median {quantiles[1].item():.3g} with 95% CI [{quantiles[0].item():.3g}, {quantiles[2].item():.3g}] and range [{min_val.item():.3g}, {max_val.item():.3g}].')

    def print_and_save(data:torch.Tensor, name:str):
        # print(f'{name} min {data.min():.3g} mean {data.mean():.3g} max {data.max():.3g}')
        print_stats(data=data, name=name)
        file = os.path.join(output_directory, f'{name}_data_ts_training_as_is.pt')
        torch.save(obj=data, f=file)
        print(f'time {time.time()-code_start_time:.3f} saved {file}')
        return file

    data_ts_file = os.path.join(output_directory, f'data_ts_all_as_is.pt')
    data_ts = torch.load(data_ts_file)[:,training_subject_start:training_subject_end,:,:]
    print( 'data_ts size', data_ts.size() )
    num_ts, num_subjects, num_nodes, num_steps = data_ts.size()
    print(f'time {time.time()-code_start_time:.3f}, computing unbinarized means and flip rates...')
    # data_ts /= data_ts.abs().max(dim=-1).values# Divide the whole time series by the largest absolute value so that the FC stays the same but the means and mean products are <= 1.
    data_ts_mean = torch.mean(data_ts, dim=-1, keepdim=True)
    centered_data_ts = data_ts - data_ts_mean
    first_moment = data_ts_mean.squeeze(dim=-1)
    second_moment = moment(data_minus_mean=centered_data_ts, power=2)
    third_moment = moment(data_minus_mean=centered_data_ts, power=3)
    fourth_moment = moment(data_minus_mean=centered_data_ts, power=4)

    mean = first_moment
    print_and_save(data=mean, name='mean')
    variance = second_moment
    print_and_save(data=variance, name='variance')
    skewness = third_moment/torch.pow(input=second_moment, exponent=3/2)
    print_and_save(data=skewness, name='skewness')
    kurtosis = fourth_moment/torch.pow(input=second_moment, exponent=2)
    print_and_save(data=kurtosis, name='kurtosis')

    z_scored_data_ts = centered_data_ts/torch.sqrt(variance).unsqueeze(dim=-1)
    group_z_scored_data_ts = torch.permute( z_scored_data_ts, dims=(2,3,0,1) ).flatten(start_dim=1, end_dim=-1)
    # The means of all individual time series are now 0, so the means of the concatenated time series are also guaranteed to be 0.
    # The variance should also still be 1, but we re-compute anyway just to check.
    group_second_moment = moment(data_minus_mean=group_z_scored_data_ts, power=2)
    group_third_moment = moment(data_minus_mean=group_z_scored_data_ts, power=3)
    group_fourth_moment = moment(data_minus_mean=group_z_scored_data_ts, power=4)
    group_variance = group_second_moment
    print_and_save(data=group_variance, name='group_variance')
    group_skewness = group_third_moment/torch.pow(input=group_second_moment, exponent=3/2)
    print_and_save(data=group_skewness, name='group_skewness')
    group_kurtosis = group_fourth_moment/torch.pow(input=group_second_moment, exponent=2)
    print_and_save(data=group_kurtosis, name='group_kurtosis')

    # Compute the covariances.
    # Since we already z-scored the data,
    # the uncentered covariance, centered covariance, and Pearson correlation
    # should all be equivalent.
    covariance_triu = isingmodellight.square_to_triu_pairs(  square_pairs=torch.matmul( input=z_scored_data_ts, other=torch.transpose(z_scored_data_ts, dim0=-2, dim1=-1) )/z_scored_data_ts.size(dim=-1)  )
    cov_first_moment = covariance_triu.mean(dim=-1, keepdim=True)
    cov_centered = covariance_triu - cov_first_moment
    cov_second_moment = moment(data_minus_mean=cov_centered, power=2)
    cov_third_moment = moment(data_minus_mean=cov_centered, power=3)
    cov_fourth_moment = moment(data_minus_mean=cov_centered, power=4)
    cov_mean = cov_first_moment.squeeze(dim=-1)
    print_and_save(data=cov_mean, name='cov_mean')
    cov_variance = cov_second_moment
    print_and_save(data=cov_variance, name='cov_variance')
    cov_skewness = cov_third_moment/torch.pow(input=cov_second_moment, exponent=3/2)
    print_and_save(data=cov_skewness, name='cov_skewness')
    cov_kurtosis = cov_fourth_moment/torch.pow(input=cov_second_moment, exponent=2)
    print_and_save(data=cov_kurtosis, name='cov_kurtosis')

    # Since each covariance is already a mean over time points of the product of a pair of states,
    # we can take the mean of means to get the group-level mean for that pair.
    group_covariance_triu = torch.mean( input=covariance_triu, dim=(0,1) )
    group_cov_first_moment = group_covariance_triu.mean(dim=-1, keepdim=True)
    group_cov_centered = group_covariance_triu - group_cov_first_moment
    group_cov_second_moment = moment(data_minus_mean=group_cov_centered, power=2)
    group_cov_third_moment = moment(data_minus_mean=group_cov_centered, power=3)
    group_cov_fourth_moment = moment(data_minus_mean=group_cov_centered, power=4)
    group_cov_mean = group_cov_first_moment.squeeze(dim=-1)
    print_and_save(data=group_cov_mean, name='group_cov_mean')
    group_cov_variance = group_cov_second_moment
    print_and_save(data=group_cov_variance, name='group_cov_variance')
    group_cov_skewness = group_cov_third_moment/torch.pow(input=group_cov_second_moment, exponent=3/2)
    print_and_save(data=group_cov_skewness, name='group_cov_skewness')
    group_cov_kurtosis = group_cov_fourth_moment/torch.pow(input=group_cov_second_moment, exponent=2)
    print_and_save(data=group_cov_kurtosis, name='group_cov_kurtosis')

    print(f'time {time.time()-code_start_time:.3f}, done')