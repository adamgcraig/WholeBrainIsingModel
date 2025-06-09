import os
import torch
import time
import argparse
import hcpdatautilsnopandas as hcp
import isingmodel
from isingmodel import IsingModel

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')

parser = argparse.ArgumentParser(description="Filter lists of subject IDs to make sure each subject has an SC data file.")
parser.add_argument("-a", "--input_directory", type=str, default='E:\\HCP_data', help="directory from which we read the lists of subject IDs, must contain a folder dtMRI_binaries with the SC data files")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\HCP_data', help="directory to which we write the filtered lists of subject IDs")
args = parser.parse_args()
print('getting arguments...')
input_directory = args.input_directory
print(f'input_directory={input_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')

for subset in ['training', 'validation', 'testing']:
    subject_ids = hcp.load_subject_subset(directory_path=input_directory, subject_subset=subset, require_sc=True)
    subject_ids_str = [f'{id_int}\n' for id_int in subject_ids]
    id_file_name = os.path.join(output_directory, f'{subset}_subject_ids.txt')
    with open(file=id_file_name, mode='w') as id_file:
        id_file.writelines(subject_ids_str)
    print(f'time {time.time()-code_start_time:.3f}, saved {subset} subject IDs to {id_file_name}')
print(f'time {time.time()-code_start_time:.3f}, done')