import torch
num_gpus = torch.cuda.device_count()
for device_index in range(num_gpus):
    device = torch.cuda.device(device_index)
    print( device, torch.cuda.get_device_properties(device_index).name )
