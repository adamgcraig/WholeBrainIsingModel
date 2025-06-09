import os
import torch
import time

code_start_time = time.time()

print('loading subject lists...')
subj_list_dir = 'D:\\HCP_data'
def load_subj_list(list_name:str):
    with open( file=os.path.join(subj_list_dir, list_name), mode='r' ) as more_subjects_file:
        more_subjects = more_subjects_file.readlines()
    return more_subjects
more_subj = load_subj_list(list_name='Subj_list.txt')
print(f'input subjects: {len(more_subj)}')
training_subj = load_subj_list(list_name='sc_training_subject_ids.txt')
print(f'training subjects: {len(training_subj)}')
validation_subj = load_subj_list(list_name='sc_validation_subject_ids.txt')
print(f'validation subjects: {len(validation_subj)}')
testing_subj = load_subj_list(list_name='sc_testing_subject_ids.txt')
print(f'testing subjects: {len(testing_subj)}')
concat_subj = training_subj + validation_subj + testing_subj
concat_subjects_file_name = os.path.join(subj_list_dir, 'concat_subjects.txt')
with open( file=concat_subjects_file_name, mode='w' ) as concat_subjects_file:
    concat_subjects_file.writelines(concat_subj)
print( f'time {time.time()-code_start_time:.3f}, saved {concat_subjects_file}' )
missing_subj = set(concat_subj).difference( set(more_subj) )
print(f'missing subjects: {len(missing_subj)}')
missing_subjects_file_name = os.path.join(subj_list_dir, 'missing_subjects.txt')
with open( file=missing_subjects_file_name, mode='w' ) as missing_subjects_file:
    missing_subjects_file.writelines(missing_subj)
print( f'time {time.time()-code_start_time:.3f}, saved {missing_subjects_file}' )

index_map = [ more_subj.index(s) for s in concat_subj ]

with torch.no_grad():
    print(f'time {time.time()-code_start_time:.3f}, loading data...')
    data_dir = 'E:\Ising_model_results_daai'
    data_ts_more_file = os.path.join(data_dir, 'data_ts_more_as_is_aal.pt')
    data_ts_more = torch.load(f=data_ts_more_file, weights_only=False)
    print( f'time {time.time()-code_start_time:.3f}, loaded {data_ts_more_file}, size', data_ts_more.size() )
    data_ts_all = data_ts_more[index_map].clone()
    data_ts_all_file = os.path.join(data_dir, 'data_ts_all_as_is_aal.pt')
    torch.save(obj=data_ts_all, f=data_ts_all_file)
    print( f'time {time.time()-code_start_time:.3f}, saved {data_ts_all_file}, size', data_ts_all_file.size() )
print('done')