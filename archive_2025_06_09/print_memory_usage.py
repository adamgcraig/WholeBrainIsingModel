import torch

device = torch.device('cuda')
print( torch.cuda.memory_summary(device=device) )
print( torch.cuda.memory_stats(device=device) )
print( 'total', torch.cuda.get_device_properties(device=device).total_memory )
print( 'reserved', torch.cuda.memory_reserved(device=device) )
print( 'allocated', torch.cuda.memory_allocated(device=device) )