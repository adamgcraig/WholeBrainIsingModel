import os
import torch
import time
import argparse
import isingmodellight
from isingmodellight import IsingModelLight

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')
    file_dir = 'E:\\Ising_model_results_daai'
    models_per_subject = 101
    num_subjects = 1
    num_nodes = 360

    beta = isingmodellight.get_linspace_beta(models_per_subject=models_per_subject, num_subjects=num_subjects, dtype=float_type, device=device)
    mean_state_product = torch.zeros( size=(num_subjects, num_nodes, num_nodes), dtype=float_type, device=device )
    mean_state_product_file = os.path.join(file_dir, 'mean_state_product_zero.pt')
    torch.save(obj=mean_state_product, f=mean_state_product_file)
    print(f'saved {mean_state_product_file}')
    J = isingmodellight.get_J_from_means(models_per_subject=models_per_subject, mean_state_product=mean_state_product)
    mean_state = torch.zeros( size=(num_subjects, num_nodes), dtype=float_type, device=device )
    mean_state_file = os.path.join(file_dir, 'mean_state_zero.pt')
    torch.save(obj=mean_state, f=mean_state_file)
    print(f'saved {mean_state_file}')
    h = isingmodellight.get_h_from_means(models_per_subject=models_per_subject, mean_state=mean_state)
    s= isingmodellight.get_neg_state_like(input=h)
    model = IsingModelLight(beta=beta, J=J, h=h, s=s)
    model_file = os.path.join(file_dir,'ising_model_zero.pt')
    torch.save(obj=model, f=model_file)
    print(f'saved {model_file}')
    print('done')