import os
import numpy as np
import torch
import nibabel as nib
import scipy.signal as sig
import time

with torch.no_grad():

    code_start_time = time.time()

    int_type = torch.int
    float_type = torch.float
    device = torch.device('cuda')

    def depytorch(t:torch.Tensor):
        return t.detach().cpu().numpy()

    def load_voxel_to_roi_mat(): 
        atlas_file = 'D:\\aal_from_france\\aal\\ext-d000035_AAL1Atlas_pub-Release2018_SPM12\\aal_for_SPM12\\atlas\\AAL.nii'
        atlas = nib.load(atlas_file)
        print(f'{time.time() - code_start_time:.3f}, loaded {atlas_file}')
        print(atlas)
        atlas_flat = torch.from_numpy( atlas.get_fdata() ).to(device=device).flatten(start_dim=0, end_dim=2)
        print(f'total number of voxels {atlas_flat.numel()}')
        roi_list = torch.unique(atlas_flat)
        print(f'number of ROIs {roi_list.numel()}')
        voxel_to_roi_mat = ( roi_list.unsqueeze(dim=1) == atlas_flat.unsqueeze(dim=0) ).double()
        voxel_to_roi_mat /= voxel_to_roi_mat.sum(dim=-1, keepdim=True)
        print( 'voxel-to-ROI map size', voxel_to_roi_mat.size() )
        return voxel_to_roi_mat

    def parcellate_and_bandpass_fmri_data(data, voxel_to_roi_mat:torch.Tensor):
        print(data)
        data_flat = torch.from_numpy( data.get_fdata() ).flatten(start_dim=0, end_dim=2).to(device=device)
        print( 'voxels x timepoints flattened data size', data_flat.size() )

        data_by_roi = torch.matmul(voxel_to_roi_mat, data_flat)
        print( 'data parcellated by ROI size', data_by_roi.size() )

        b, a = sig.cheby2(N=5, rs=40, Wn=[0.01, 0.1], btype='bandpass', analog=False, output='ba', fs=1/0.72)
        data_by_roi_bp = torch.from_numpy(  sig.filtfilt( b=b, a=a, x=depytorch(data_by_roi), axis=-1 ).copy()  ).to(device=device)
        print( 'after band-pass filtering size', data_by_roi_bp.size() )

        data_std, data_mean = torch.std_mean(input=data_by_roi_bp, dim=-1, keepdim=True)
        data_z = (data_by_roi_bp - data_mean)/data_std
        print( 'after z-scoring', data_z.size() )
        return data_z
    
    voxel_to_roi_mat = load_voxel_to_roi_mat()
    file_check_delay = 180
    # subject_list = 'D:\\HCP_data\\Subj_list.txt'
    subject_list = 'C:\\Users\\agcraig\\Documents\\GitHub\MachineLearningNeuralMassModel\\IsingModel\\missing_subjects.txt'
    with open(file=subject_list, mode='r') as subject_list_file:
        for subject_id in subject_list_file.readlines():
            for ts_suffix in ['rest_1_lr', 'rest_1_rl']:# , 'rest_2_lr', 'rest_2_rl'
                subject_id = subject_id.rstrip()
                data_file = os.path.join('D:\\hcp_from_s3_second_try', f'{subject_id}_{ts_suffix}.nii.gz')
                output_file = os.path.join('D:\\fmri_aal', f'{subject_id}_{ts_suffix}.pt')
                print(f'{time.time() - code_start_time:.3f}, checking for file {data_file}...')
                while not os.path.isfile(path=data_file):
                    print(f'file not found, will check again in {file_check_delay/60:.3g} minutes...')
                    time.sleep(file_check_delay)# Wait 3 minutes.
                print(f'{time.time() - code_start_time:.3f}, loading from {data_file}...')
                torch.save(  obj=parcellate_and_bandpass_fmri_data( data=nib.load(data_file), voxel_to_roi_mat=voxel_to_roi_mat ), f=output_file  )
                print(f'{time.time() - code_start_time:.3f}, saved to {output_file}')
                os.remove(path=data_file)
                print(f'{time.time() - code_start_time:.3f}, removed {data_file}')

print(f'{time.time() - code_start_time:.3f}, done')