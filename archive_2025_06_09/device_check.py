# based on https://stackoverflow.com/questions/64776822/how-do-i-list-all-currently-available-gpus-with-pytorch
# and https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
import torch
num_devices = torch.cuda.device_count()
for device_index in range(num_devices):
   properties = torch.cuda.get_device_properties(device_index)
   name = properties.name
   total_memory = properties.total_memory
   reserved_memory = torch.cuda.memory_reserved(device_index)
   allocated_memory = torch.cuda.memory_allocated(device_index)
   print(f'{device_index} name {name}, total memory {total_memory}, reserved memory {reserved_memory}, allocated memory {allocated_memory}')
big_chunks = [torch.zeros( size=(1000//32, 1000, 1000), dtype=torch.float32, device=torch.device(f'cuda:{device_index}') ) for device_index in range(num_devices)]
print('after allocation')
for device_index in range(num_devices):
   big_chunks += [torch.zeros( size=(1000//32, 1000, 1000), dtype=torch.float32, device=torch.device(f'cuda:{device_index}') )]
   properties = torch.cuda.get_device_properties(device_index)
   name = properties.name
   total_memory = properties.total_memory
   reserved_memory = torch.cuda.memory_reserved(device_index)
   allocated_memory = torch.cuda.memory_allocated(device_index)
   print(f'{device_index} name {name}, total memory {total_memory}, reserved memory {reserved_memory}, allocated memory {allocated_memory}')   