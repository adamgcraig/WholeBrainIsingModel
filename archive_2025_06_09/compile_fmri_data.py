import os
import numpy as np
import torch
import time

with torch.no_grad():
    code_start_time = time.time()
    int_type = torch.int
    float_type = torch.float
    device = torch.device('cuda')
    subject_list = 'D:\\HCP_data\\concat_subjects.txt'
    # subject_list = 'C:\\Users\\agcraig\\Documents\\GitHub\MachineLearningNeuralMassModel\\IsingModel\\Subj_list_redo.txt'
    ts_suffixes = ['rest_1_lr', 'rest_1_rl', 'rest_2_lr', 'rest_2_rl']
    with open(file=subject_list, mode='r') as subject_list_file:
        subjects = [l.rstrip() for l in subject_list_file.readlines()]
        num_subjects = len(subjects)
        num_ts_per_subject = len(ts_suffixes)
        print(f'{subject_list} has {num_subjects} lines.')
        data_file = os.path.join('D:\\fmri_aal', f'{subjects[0]}_{ts_suffixes[0]}.pt')
        data_ts_init = torch.load(data_file, weights_only=False)
        print( f'loaded init test file {data_file}, dtype {data_ts_init.dtype}, device {data_ts_init.device}, size', data_ts_init.size() )
        data_ts = torch.zeros_like( torch.unsqueeze(input=data_ts_init, dim=0).unsqueeze(dim=0).repeat( repeats=(num_ts_per_subject, num_subjects, 1, 1) ) )
        for subject_index in range(num_subjects):
            for ts_index in range(num_ts_per_subject):
                data_file = os.path.join('D:\\fmri_aal', f'{subjects[subject_index]}_{ts_suffixes[ts_index]}.pt')
                data_ts[ts_index, subject_index, :, :] = torch.load(f=data_file, weights_only=False)
                print(f'time {time.time()-code_start_time:.3f}, loaded {data_file}')
        print( 'final data_ts size', data_ts.size() )
        compiled_file = os.path.join('E:\\Ising_model_results_daai', 'data_ts_all_as_is_aal.pt')
        torch.save(obj=data_ts, f=compiled_file)
        print(f'{time.time() - code_start_time:.3f}, saved {compiled_file}')
print(f'{time.time() - code_start_time:.3f}, done')