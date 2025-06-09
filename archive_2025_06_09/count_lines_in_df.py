import os
import time
import argparse
import pandas
import torch

code_start_time = time.time()

parser = argparse.ArgumentParser(description="Count the number of lines with losses and training times filled in in our hyperparameter comparison save file.")
parser.add_argument("-a", "--file_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can read and write files")
parser.add_argument("-2", "--results_file_name", type=str, default='compare_g2gcnn_hyperparameters.pkl', help="We will save a record of the training and validation errors for all combinations of hyperparameters to this file as a pandas DataFrame pickle file.")
args = parser.parse_args()
print('getting arguments...')
file_directory = args.file_directory
results_file_name = args.results_file_name
print(f'results_file_name={results_file_name}')

# Check whether we have already created the results pickle file in a previous run.
results_file = os.path.join(file_directory, results_file_name)
if os.path.exists(results_file):
    print(f'loading results table from {results_file}...')
    losses_df = pandas.read_pickle(results_file)
    start_index = len( losses_df.loc[ losses_df['time'] != -1.0 ].index )
else:
    print('file not found')
num_cases = len(losses_df.index)
print(f'placeholder lines start from condition {start_index} of {num_cases}')