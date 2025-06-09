# functions we use in multiple Ising-model-related scripts

import torch

def get_random_state(batch_size:int, num_nodes:int, device='cpu'):
    return torch.randint( 2, (batch_size, num_nodes), dtype=torch.bool, device=device )

def get_batched_ising_models(batch_size:int, num_nodes:int, dtype=torch.float, device='cpu'):
    J = torch.randn( (batch_size, num_nodes, num_nodes), dtype=dtype, device=device )
    J = 0.5*( J + J.transpose(dim0=-2, dim1=-1) )
    J = J - torch.diag_embed( torch.diagonal(J,dim1=-2,dim2=-1) )
    h = torch.zeros( (batch_size, num_nodes), dtype=dtype, device=device )
    s = torch.randint( 2, (batch_size, num_nodes), dtype=torch.bool, device=device )
    return J, h, s

def standardize_and_binarize_ts_data(ts:torch.Tensor, threshold:float=0.1, time_dim:int=-2):
    ts_std, ts_mean = torch.std_mean(ts, dim=time_dim, keepdim=True)
    return ( (ts - ts_mean)/ts_std ) < threshold

def median_binarize_ts_data(ts:torch.Tensor, time_dim:int=-2):
    return ts < ts.median(dim=time_dim, keepdim=True).values

# B: batch size
# N: num nodes
# s size (B,N), type bool
# h size (B,N), type float
# J size (B,N,N), type float
def get_batched_delta_h(J:torch.Tensor, h:torch.Tensor, s:torch.Tensor, node_index:int):
    s_i = s[:,node_index]# (B,), bool
    s_i_float = torch.logical_not(s_i).float() - s_i.float()# (B,), float
    h_i = h[:,node_index]# (B,), float
    s_is_j = torch.logical_xor( s_i.unsqueeze(dim=1), s )# (B,N), bool
    s_is_j_float = torch.logical_not(s_is_j).float() - s_is_j.float()# (B,N), float
    J_ij = J[:,:,node_index]# (B,N), float
    return 2.0 * (  torch.sum(J_ij*s_is_j_float, dim=-1) + h_i*s_i_float  )

def run_batched_balanced_metropolis_sim_step(J:torch.Tensor, h:torch.Tensor, s:torch.Tensor, beta:float=0.5):
    num_nodes = s.size(dim=-1)
    # batch_size, num_nodes = s.size()
    # if not torch.is_tensor(sim_ts):
    #     sim_ts = torch.zeros( (batch_size, num_steps, num_nodes), dtype=s.dtype, device=s.device )
    node_order = torch.randperm(n=num_nodes, device=s.device)
    for i in range(num_nodes):
        node_index = node_order[i].item()
        deltaH = get_batched_delta_h(J=J, h=h, s=s, node_index=node_index)
        prob_accept = torch.clamp( torch.exp(-beta*deltaH), min=0.0, max=1.0 )
        flip = torch.bernoulli(prob_accept).bool()
        s[:,node_index] = torch.logical_xor(s[:,node_index], flip)
    return s

def run_batched_balanced_metropolis_sim(sim_ts:torch.Tensor, J:torch.Tensor, h:torch.Tensor, s:torch.Tensor, num_steps:int, beta:float=0.5):
    # batch_size, num_nodes = s.size()
    # if not torch.is_tensor(sim_ts):
    #     sim_ts = torch.zeros( (batch_size, num_steps, num_nodes), dtype=s.dtype, device=s.device )
    for t in range(num_steps):
        s = run_batched_balanced_metropolis_sim_step(J=J, h=h, s=s, beta=beta)
        sim_ts[:,t,:] = s
    return sim_ts, s

# We frequently just want to work with the functional connectivity, not the full time series.
# For long simulation times, just keeping running totals we need in order to calculate the FC has a lower memory footprint than storing the full time series.
def run_batched_balanced_metropolis_sim_for_fc(J:torch.Tensor, h:torch.Tensor, s:torch.Tensor, num_steps:int, beta:float=0.5, epsilon=0.0000000001):
    # batch_size, num_nodes = s.size()
    # if not torch.is_tensor(sim_ts):
    #     sim_ts = torch.zeros( (batch_size, num_steps, num_nodes), dtype=s.dtype, device=s.device )
    s_int = s.int()
    s_sum = torch.zeros_like(s_int)
    s_product_sum = s_sum[:,:,None] * s_sum[:,None,:]
    for _ in range(num_steps):
        s = run_batched_balanced_metropolis_sim_step(J=J, h=h, s=s, beta=beta)
        s_int = s.int()
        s_sum += s_int
        s_product_sum += (s_int[:,:,None] * s_int[:,None,:])
    s_mean = ( (1 - 2*s_sum).float() )/num_steps
    s_product_mean = (1 - 2*s_product_sum).float()/num_steps
    s_squared_mean = torch.diagonal(s_product_mean, dim1=-2, dim2=-1)
    s_std = torch.sqrt( s_squared_mean - s_mean * s_mean )
    s_cov = s_product_mean - s_mean[:,:,None] * s_mean[:,None,:]
    s_std_prod = s_std[:,:,None] * s_std[:,None,:]
    if torch.any(s_std_prod == 0.0):
        s_cov += epsilon
        s_std_prod += epsilon
    sim_fc = s_cov/s_std_prod
    return sim_fc, s
