import os
import pandas
import numpy as np
import torch
import math
import isingmodellight
from isingmodellight import IsingModelLight
from scipy import stats
import time
import hcpdatautils as hcp

int_type = torch.int
float_type = torch.float
device = torch.device('cuda')
code_start_time = time.time()

def load_data_ts_z():
    # data_ts_file = os.path.join('results', 'ising_model', 'data_ts_all_as_is.pt')
    data_ts_file = os.path.join('D:', 'ising_model_results_daai', 'data_ts_all_as_is.pt')
    data_ts_all_as_is = torch.load( f=data_ts_file, weights_only=False )
    print( f'time {time.time()-code_start_time:.3f} loaded {data_ts_file}, size', data_ts_all_as_is.size() )
    data_ts_all_as_is = torch.permute( input=data_ts_all_as_is, dims=(1,2,0,3) ).flatten(start_dim=-2, end_dim=-1)
    print( f'time {time.time()-code_start_time:.3f} resized to, size', data_ts_all_as_is.size() )
    data_ts_all_as_is_std, data_ts_all_as_is_mean = torch.std_mean(data_ts_all_as_is, dim=-1, keepdim=True)
    data_ts_all_as_is -= data_ts_all_as_is_mean
    data_ts_all_as_is /= data_ts_all_as_is_std
    print( f'time {time.time()-code_start_time:.3f} z-scored' )
    return data_ts_all_as_is

def count_all_or_none(data_ts_all_as_is:torch.Tensor, threshold:float):
    num_time_points = data_ts_all_as_is.size(dim=-1)
    num_high = torch.count_nonzero(data_ts_all_as_is > threshold, dim=-1)
    num_none_high = torch.count_nonzero(num_high == 0)
    num_all_high = torch.count_nonzero(num_high == num_time_points)
    print(f'{time.time()-code_start_time:.3f}\t{threshold:.3g}\t{num_none_high}\t{num_all_high}')
    return 0

with torch.no_grad():
    data_ts_all_as_is = load_data_ts_z()
    data_ts_max_z = torch.max(input=data_ts_all_as_is, dim=-1).values
    data_ts_min_max_z = torch.min(data_ts_max_z)
    print(f'The maximum binarization threshold at which no region of any 4-scan single-subject time series is all -1 is {data_ts_min_max_z:3g}.')
    print('The rest of the maximum thresholds for subjects:')
    print( data_ts_max_z.tolist() )
    # print(f'time\tthreshold\tnum_all_low\tnum_all_high')
    # for threshold in torch.arange(start=0.0, end=6.0, step=0.0001):
    #     count_all_or_none(data_ts_all_as_is=data_ts_all_as_is, threshold=threshold)
print(f'time {time.time()-code_start_time:.3f}, done')