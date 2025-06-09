import os
import torch
import time
import argparse
import isingmodellight

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Compute the covariances of pairs of regions and pairs of products of pairs of regions over the set of all training time series.")
    parser.add_argument("-a", "--input_directory", type=str, default='E:\\HCP_data', help="containing folder of data_ts_all_as_is.pt")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the output file data_fim_group_threshold_[t].pt")
    parser.add_argument("-c", "--training_subject_start", type=int, default=0, help="first training subject index")
    parser.add_argument("-d", "--training_subject_end", type=int, default=670, help="1 past last training subject index")
    parser.add_argument("-e", "--threshold", type=float, default=1.0, help="threshold at which to binarize the data")
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
    threshold = args.threshold
    print(f'threshold={threshold}')

    def load_data_ts(input_directory:str, threshold:float):
        data_ts_file = os.path.join(input_directory, f'data_ts_all_as_is.pt')
        data_ts = torch.load(f=data_ts_file)[:,training_subject_start:training_subject_end,:,:]
        data_ts_std, data_ts_mean = torch.std_mean(input=data_ts, dim=-1, keepdim=True)
        data_ts -= data_ts_mean
        data_ts /= data_ts_std
        return torch.flatten(   torch.permute(  input=( 2.0*(data_ts >= threshold).float() - 1.0 ), dims=(2,3,0,1)  ), start_dim=1, end_dim=-1   )
    
    def extend_observables(data_ts:torch.Tensor):
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=data_ts.size(dim=0), device=data_ts.device )
        return torch.cat( (data_ts, data_ts[triu_rows,:]*data_ts[triu_cols,:]), dim=0 )
    
    # FIM(i,j) = <x[i,t]x[j,t]>_t - <x[i,t]>_t<x[j,t]>_t
    # where x[0,t], ..., x[N-1,t] are the node states n[0,t], ..., n[N-1,t]
    # and x[N,t], ..., x[M,t] are the products of pairs of node states n[0,t]n[1,t], n[0,t]n[2,t], ..., n[N-2,t]n[N-1,t]
    # with expected values taken over times t=1, ..., T.
    def make_fim(observables:torch.Tensor):
        mean_observables = torch.mean(observables, dim=-1)
        return torch.matmul( input=observables, other=observables.transpose(dim0=-2, dim1=-1) )/observables.size(dim=-1) - mean_observables.unsqueeze(dim=-1) * mean_observables.unsqueeze(dim=-2)
    
    def march_through_fim(observables:torch.Tensor):
        num_observables, num_steps = observables.size()
        fim = torch.zeros( size=(num_observables, num_observables), dtype=observables.dtype, device=observables.device )
        for step in range(num_steps):
            observables_now = observables[:,step]
            fim += ( observables_now.unsqueeze(dim=-1) * observables_now.unsqueeze(dim=-2) )
        fim /= num_steps
        mean_observables = torch.mean(observables, dim=-1)
        fim -= ( mean_observables.unsqueeze(dim=-1) * mean_observables.unsqueeze(dim=-2) )
        return fim
    
    def fim_from_ts(data_ts:torch.Tensor):
        float_type = data_ts.dtype
        device = data_ts.device
        num_nodes, num_steps = data_ts.size()
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products(num_nodes=num_nodes, device=device)
        num_pairs = triu_rows.numel()
        num_observables = num_nodes + num_pairs
        fim = torch.zeros( size=(num_observables, num_observables), dtype=float_type, device=device )
        for step in range(num_steps):
            nodes = data_ts[:,step]
            nodes_row = nodes.unsqueeze(dim=0)
            nodes_col = nodes.unsqueeze(dim=1)
            products = nodes[triu_rows] * nodes[triu_cols]
            products_row = products.unsqueeze(dim=0)
            products_col = products.unsqueeze(dim=1)
            nodes_by_products = nodes_col * products_row
            fim[:num_nodes,:num_nodes] += nodes_col * nodes_row
            fim[:num_nodes,num_nodes:] += nodes_by_products
            fim[num_nodes:,:num_nodes] += nodes_by_products.transpose(dim0=0, dim1=1)
            fim[num_nodes:,num_nodes:] += products_col * products_row
        fim /= num_steps
        nodes = data_ts.mean(dim=-1)
        nodes_row = nodes.unsqueeze(dim=0)
        nodes_col = nodes.unsqueeze(dim=1)
        fim[:num_nodes,:num_nodes] -= nodes_col * nodes_row
        # The upper-left-most num_nodes x num_nodes square already has the means of the products of the states. 
        mean_products_triu = fim[triu_rows,triu_cols]
        fim[num_nodes:,num_nodes:] -= mean_products_triu.unsqueeze(dim=0) * mean_products_triu.unsqueeze(dim=1)
        return fim

    def save_fim_tiles(fim:torch.Tensor, tile_side_length:int, threshold:float):
        num_observables = fim.size(dim=0)
        num_tile_sides = num_observables % tile_side_length
        for row_block in range(num_tile_sides):
            row_start = row_block*tile_side_length
            row_end = row_start + tile_side_length
            for col_block in range(num_tile_sides):
                col_start = col_block*tile_side_length
                col_end = col_start + tile_side_length
                block_file = os.path.join(output_directory, f'data_fim_group_threshold_{threshold:.3g}_block_{row_block}_{col_block}.pt')
                torch.save( obj=fim[row_start:row_end,col_start:col_end].clone(), f=block_file )

    fim_file = os.path.join(output_directory, f'data_fim_group_threshold_{threshold:.3g}.pt')
    # torch.save(   obj=make_fim(  observables=extend_observables( data_ts=load_data_ts(input_directory=input_directory, threshold=threshold) )  ), f=fim_file   )
    torch.save(  obj=fim_from_ts( data_ts=load_data_ts(input_directory=input_directory, threshold=threshold) ), f=fim_file  )
    print(f'saved {fim_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')