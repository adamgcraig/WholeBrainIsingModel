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
    parser.add_argument("-c", "--model_file_part", type=str, default='ising_model_light_group_init_means_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_1_steps_1200_lr_0.01_beta_updates_8_v2_param_updates_40000', help="Ising model file except for the .pt file extension")
    parser.add_argument("-d", "--fc_corr_file_part", type=str, default='fc_corr_ising_model_light_group_init_means_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_1_steps_1200_lr_0.01_beta_updates_8_v2_param_updates_40000_test_length_120000', help="fc corr file except for the .pt file extension, used to select the best group model")
    parser.add_argument("-e", "--num_thresholds", type=int, default=31, help="number of thresholds used with the multithreshold group models")
    parser.add_argument("-f", "--min_threshold", type=float, default=0.0, help="lowest threshold used")
    parser.add_argument("-g", "--max_threshold", type=float, default=3.0, help="highest threshold used")
    parser.add_argument("-i", "--target_threshold", type=float, default=1.0, help="select the threshold closest to this one")
    parser.add_argument("-j", "--num_subjects", type=int, default=837, help="number of copies of the best group model to make along the subject dimension")
    parser.add_argument("-k", "--models_per_subject", type=int, default=5, help="number of copies of the best group model to make along the replica dimension")
    args = parser.parse_args()
    print('getting arguments...')
    input_directory = args.input_directory
    print(f'input_directory={input_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    model_file_part = args.model_file_part
    print(f'model_file_part={model_file_part}')
    fc_corr_file_part = args.fc_corr_file_part
    print(f'fc_corr_file_part={fc_corr_file_part}')
    num_thresholds = args.num_thresholds
    print(f'num_thresholds={num_thresholds}')
    min_threshold = args.min_threshold
    print(f'min_threshold={min_threshold}')
    max_threshold = args.max_threshold
    print(f'max_threshold={max_threshold}')
    target_threshold = args.target_threshold
    print(f'target_threshold={target_threshold}')
    num_subjects = args.num_subjects
    print(f'num_subjects={num_subjects}')
    models_per_subject = args.models_per_subject
    print(f'models_per_subject={models_per_subject}')

    group_model_file = os.path.join(output_directory, f'{model_file_part}.pt')
    group_model = torch.load(f=group_model_file, weights_only=False)
    print(f'time {time.time()-code_start_time:.3f}, loaded group models from file {group_model_file}')
    fc_corr_file = os.path.join(output_directory, f'{fc_corr_file_part}.pt')
    fc_corr = torch.load(f=fc_corr_file, weights_only=False)
    print(f'time {time.time()-code_start_time:.3f}, loaded fc correlations from file {fc_corr_file}')
    print( 'FC corr size', fc_corr.size(), f'min {fc_corr.min():.3g}, mean {fc_corr.mean():.3g}, max {fc_corr.max():.3g}' )
    thresholds = torch.linspace(start=min_threshold, end=max_threshold, steps=num_thresholds, dtype=fc_corr.dtype, device=fc_corr.device)
    diffs_from_target = torch.abs(thresholds - target_threshold)
    threshold_index = torch.argmin(diffs_from_target)
    selected_threshold = thresholds[threshold_index]
    selected_diff = diffs_from_target[threshold_index]
    print(f'selected threshold {selected_threshold:.3g} at index {threshold_index}, difference from target {target_threshold:.3g} is {selected_diff:.3g}')
    fc_corr_at_threshold = fc_corr[:,threshold_index]
    rep_index = torch.argmax(fc_corr_at_threshold)
    best_fc_corr = fc_corr_at_threshold[rep_index]
    print(f'at this threshold FC corr has min {fc_corr_at_threshold.min():.3g}, mean {fc_corr_at_threshold.mean():.3g}, max {best_fc_corr:.3g}')
    selected_beta = group_model.beta[rep_index, threshold_index]
    selected_h = group_model.h[rep_index, threshold_index, :]
    selected_J = group_model.J[rep_index, threshold_index, :, :]
    selected_s = group_model.s[rep_index, threshold_index, :]
    print(f'selected model has beta {selected_beta:.3g}, h min {selected_h.min():.3g}, h mean {selected_h.mean():.3g}, h max {selected_h.max():.3g}, J min {selected_J.min():.3g}, J mean {selected_J.mean():.3g}, J max {selected_J.max():.3g}, s min {selected_s.min():.3g}, s mean {selected_s.mean():.3g}, s min {selected_s.max():.3g}')
    beta = torch.full( size=(models_per_subject, num_subjects), fill_value=selected_beta, dtype=selected_beta.dtype, device=selected_beta.device )
    print( 'beta size', beta.size() )
    h = selected_h.unsqueeze(dim=0).unsqueeze(dim=0).repeat( (models_per_subject, num_subjects, 1) )
    print( 'h size', h.size() )
    J = selected_J.unsqueeze(dim=0).unsqueeze(dim=0).repeat( (models_per_subject, num_subjects, 1, 1) )
    print( 'J size', J.size() )
    s = selected_s.unsqueeze(dim=0).unsqueeze(dim=0).repeat( (models_per_subject, num_subjects, 1) )
    print( 's size', s.size() )
    model = IsingModelLight(beta=beta, J=J, h=h, s=s)
    starting_model_file = os.path.join(output_directory, f'{model_file_part}_to_thresh_{selected_threshold:.3g}_reps_{models_per_subject}_subj_{num_subjects}.pt')
    torch.save(obj=model, f=starting_model_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {starting_model_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')