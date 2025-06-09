import os
import torch
import time
import argparse
import hcpdatautilsnopandas as hcp
import isingmodellight
from isingmodellight import IsingModelLight

def get_binarized_info(data_ts:torch.Tensor, threshold_z:float):
    data_ts_std, data_ts_mean = torch.std_mean(data_ts, dim=-1, keepdim=True)
    binarized_ts = 2*( data_ts > (data_ts_mean + threshold_z*data_ts_std) ).float() - 1
    # We want to average over all scans and all subjects but then the model expects a singleton subject batch dimension.
    return binarized_ts.mean(dim=-1).mean( dim=(0,1) ).unsqueeze(dim=0), torch.matmul( binarized_ts, binarized_ts.transpose(dim0=-2, dim1=-1) ).mean( dim=(0,1) ).unsqueeze(dim=0)/binarized_ts.size(dim=-1)

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
    parser.add_argument("-d", "--training_subject_end", type=int, default=670, help="1 past last training subject index")
    parser.add_argument("-e", "--threshold_z", type=float, default=1.0, help="binarization threshold in std. dev.s above the mean")
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
    threshold_z = args.threshold_z
    print(f'threshold_z={threshold_z:.3g}')

    data_ts_file = os.path.join(output_directory, f'data_ts_all_as_is.pt')
    data_ts = torch.load(data_ts_file)
    num_ts, num_subjects, num_nodes, num_steps = data_ts.size()
    print(f'time {time.time()-code_start_time:.3f}, computing binarized mean states and state products...')
    # Use only the training subjects to get the group means.
    target_state_mean, target_state_product_mean = get_binarized_info(data_ts=data_ts[:,training_subject_start:training_subject_end,:,:], threshold_z=threshold_z)
    target_state_mean_file = os.path.join(output_directory, f'mean_state_group_threshold_{threshold_z:.3g}.pt')
    torch.save(obj=target_state_mean, f=target_state_mean_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {target_state_mean_file}')
    target_state_mean_product_file = os.path.join(output_directory, f'mean_state_product_group_threshold_{threshold_z:.3g}.pt')
    torch.save(obj=target_state_product_mean, f=target_state_mean_product_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {target_state_mean_product_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')
