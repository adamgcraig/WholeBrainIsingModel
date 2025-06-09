import argparse
import os
import numpy as np
import torch

parser = argparse.ArgumentParser(description="Load a series of .bin binary data files for fMRI time series, binarize each, compute their means and uncentered covariances, and save the results in a pair of PyTroch Tensors in pickle .pt files.")
parser.add_argument("-i", "--binary_data_dir", type=str, default='E:\\aal90_short_binaries', help="directory where we can find the time series binary files")
parser.add_argument("-o", "--pt_data_dir", type=str, default='E:\\aal90_short_pytorch', help="directory to which we write the PyTorch pickle files")
parser.add_argument("-n", "--num_nodes", type=int, default=116, help="number of ROIs in each time series")
parser.add_argument("-t", "--threshold", type=float, default=1.0, help="binarization threshold in standard deviations above the mean")
parser.add_argument("-d", "--device", type=str, default='cuda', help="device string for PyTorch")
args = parser.parse_args()
binary_data_dir = args.binary_data_dir
print(f'binary_data_dir={binary_data_dir}')
pt_data_dir = args.pt_data_dir
print(f'pt_data_dir={pt_data_dir}')
num_nodes = args.num_nodes
print(f'num_nodes={num_nodes}')
device = torch.device(args.device)
print(f'device={device}')
threshold = args.threshold
print(f'threshold={threshold}')
dtype = torch.float
print(f'dtype={dtype}')

def get_mean_tensor(dlist:list):
    return torch.stack( [ts.mean(dim=-1) for ts in dlist], dim=0 )

def get_std_tensor(dlist:list):
    return torch.stack( [ts.std(dim=-1) for ts in dlist], dim=0 )

def get_mean_product_tensor(dlist:list):
    return torch.stack(  [torch.matmul( ts, ts.transpose(dim0=0,dim1=1) )/ts.size(dim=-1) for ts in dlist], dim=0  )

binary_ts_files = [os.path.join(binary_data_dir, file) for file in os.listdir(path=binary_data_dir)]

data_list_np = [np.fromfile( file, np.float64 ).reshape( (num_nodes, -1), order='C' ) for file in binary_ts_files]
print( 'num time series', len(data_list_np) )
data_list_torch = [torch.from_numpy(data).to(device, dtype=dtype) for data in data_list_np]

data_mean = get_mean_tensor(dlist=data_list_torch)
print( 'mean size', data_mean.size() )
data_mean_file = os.path.join(pt_data_dir, f'data_mean_all_as_is.pt')
torch.save(obj=data_mean, f=data_mean_file)
print(f'saved {data_mean_file}')

data_std = get_std_tensor(dlist=data_list_torch)
print( 'SD size', data_std.size() )
data_std_file = os.path.join(pt_data_dir, f'data_std_all_as_is.pt')
torch.save(obj=data_std, f=data_std_file)
print(f'saved {data_std_file}')

data_mean_product = get_mean_product_tensor(dlist=data_list_torch)
print( 'uncentered covariances size', data_mean_product.size() )
data_mean_product_file = os.path.join(pt_data_dir, f'data_mean_product_all_as_is.pt')
torch.save(obj=data_mean_product, f=data_mean_product_file)
print(f'saved {data_mean_product_file}')

data_cov = data_mean_product - data_mean.unsqueeze(dim=-2) * data_mean.unsqueeze(dim=-1)
print( 'centered covariances size', data_cov.size() )
data_cov_file = os.path.join(pt_data_dir, f'data_cov_all_as_is.pt')
torch.save(obj=data_cov, f=data_cov_file)
print(f'saved {data_cov_file}')

data_fc = data_mean_product /( data_std.unsqueeze(dim=-2) * data_std.unsqueeze(dim=-1) )
print( 'FCs size', data_fc.size() )
data_fc_file = os.path.join(pt_data_dir, f'data_fc_all_as_is.pt')
torch.save(obj=data_fc, f=data_fc_file)
print(f'saved {data_fc_file}')

data_list_binary = [ 2.0*(  ts >= ( ts.mean(dim=-1, keepdim=True) + threshold * ts.std(dim=-1, keepdim=True) )  ).float() - 1.0 for ts in data_list_torch ]

data_mean_binary = get_mean_tensor(dlist=data_list_binary)
print( 'binarized data means size', data_mean_binary.size() )
data_mean_binary_file = os.path.join(pt_data_dir, f'data_mean_all_mean_std_{threshold:.3g}.pt')
torch.save(obj=data_mean_binary, f=data_mean_binary_file)
print(f'saved {data_mean_binary_file}')

data_std_binary = get_std_tensor(dlist=data_list_binary)
print( 'binarized data SD size', data_std_binary.size() )
data_std_binary_file = os.path.join(pt_data_dir, f'data_std_all_mean_std_{threshold:.3g}.pt')
torch.save(obj=data_std_binary, f=data_std_binary_file)
print(f'saved {data_std_binary_file}')

data_mean_product_binary = get_mean_product_tensor(dlist=data_list_binary)
print( 'binarized data uncentered covariances size', data_mean_product_binary.size() )
data_mean_product_binary_file = os.path.join(pt_data_dir, f'data_mean_product_all_mean_std_{threshold:.3g}.pt')
torch.save(obj=data_mean_product_binary, f=data_mean_product_binary_file)
print(f'saved {data_mean_product_binary_file}')

data_cov_binary = data_mean_product_binary - data_mean_binary.unsqueeze(dim=-2) * data_mean_binary.unsqueeze(dim=-1)
print( 'binarized centered covariances size', data_cov_binary.size() )
data_cov_binary_file = os.path.join(pt_data_dir, f'data_cov_all_mean_std_{threshold:.3g}.pt')
torch.save(obj=data_cov_binary, f=data_cov_binary_file)
print(f'saved {data_cov_binary_file}')

data_fc_binary = data_mean_product_binary /( data_std_binary.unsqueeze(dim=-2) * data_std_binary.unsqueeze(dim=-1) )
print( 'binarized FCs size', data_fc_binary.size() )
data_fc_binary_file = os.path.join(pt_data_dir, f'data_fc_all_mean_std_{threshold:.3g}.pt')
torch.save(obj=data_fc_binary, f=data_fc_binary_file)
print(f'saved {data_fc_binary_file}')

print('done')