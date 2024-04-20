import torch
import numpy as np
import math
from timeit import default_timer
from argparse import ArgumentParser

from data_storage.loss_recording import total_model_dict, save_checkpoint
from ddp_utils.data_utils import *

import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributed.optim import ZeroRedundancyOptimizer

from torch.nn.parallel import DistributedDataParallel as DDP

parser = ArgumentParser(description='GNOT Artemis Training Study')
parser.add_argument('--name', type=str, default='test')
parser.add_argument('--path', type=str, default= r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\GNOT\data\steady_cavity_case_b200_maxU100ms_simple_normalized.npy')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--sub_x', type=int, default=4)
parser.add_argument('--inference', type=str, default='True')
parser.add_argument('--n_hidden', type=int, default=128)
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--pde_weight', type=float, default=0.0)
parser.add_argument('--warm_up', type=int, default=5)
global ARGS 
ARGS = parser.parse_args()

def check_gradients(model):
    nan_flag = False
    inf_flag = False
    for param in model.parameters():
        if param.grad.data.isnan().any():
            nan_flag = True
        if param.grad.data.isfinite().any():
            inf_flag = True
    
    return nan_flag, inf_flag

def demo_basic(rank, world_size=1):
    print(f"Running basic Single Processor example on rank {rank}.")
    torch.manual_seed(42)
    # create model and move it to GPU with id rank
    dataset_args, model_args, training_args = get_default_args()
    model = get_model(model_args).to(rank)
    
    dataset_args['sub_x']           = ARGS.sub_x
    dataset_args['batchsize']       = ARGS.batch_size
    training_args['epochs']         = ARGS.epochs
    training_args["save_name"]      = ARGS.name
    training_args['PDE_weight']     = ARGS.pde_weight
    training_args['warmup_epochs']  = ARGS.warm_up
    dataset_args['file_path']       = ARGS.path #'/project/MLFluids/steady_cavity_case_b200_maxU100ms_simple_normalized.npy'

    train_loader, val_loader, batch_size, output_normalizer, input_f_normalizer = get_dataset(dataset_args, ddp=False)

    loss_fn = LpLoss_custom()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  betas=(0.9, 0.999), 
                                  lr=training_args['base_lr'],
                                  weight_decay=training_args['weight-decay']
                                  )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr=training_args['base_lr'], 
                                                    div_factor=1e4, 
                                                    pct_start=0.2, 
                                                    final_div_factor=1e4, 
                                                    steps_per_epoch=len(train_loader), 
                                                    epochs=training_args['epochs']
                                                    )
    
    print(f'[Rank {rank}] Length of Train Loader: {len(train_loader)} for world size {world_size} and batchsize {ARGS.batch_size} produces trainloader batch size of {batch_size}')
    
    training_run_results = total_model_dict(model_config=model_args, training_config=training_args, data_config=dataset_args)
    

    print(f"Started training on rank {rank}.")
    for epoch in range(training_args['epochs']):
        torch.cuda.empty_cache()
        epoch_start_time = default_timer()
        for batch_n, batch in enumerate(train_loader):
            in_queries, in_keys, out_truth, reverse_indices = batch
            in_queries, in_keys, out_truth = in_queries.to(rank), in_keys.to(rank), out_truth.to(rank)
            optimizer.zero_grad()
            output = model(x=in_queries,inputs = in_keys)
            
            # Pointwise Loss
            train_loss = loss_fn(output, out_truth)

            # PDE Loss
            if training_args['PDE_weight'] > 0: # and epoch >= training_args['warmup_epochs']:
                outputs, input_keys = output_realiser(output, in_keys, output_normalizer.to(rank), input_f_normalizer.to(rank), reverse_indices.to(rank))
                Du_dx, Dv_dy, continuity_eq,__ = NS_FDM_cavity_internal_vertex_non_dim(U=outputs, lid_velocity=input_keys, nu=0.01, L=1.0)
                pde_loss_1 = loss_fn(Du_dx, torch.zeros_like(Du_dx))
                pde_loss_2 = loss_fn(Dv_dy, torch.zeros_like(Dv_dy))
                pde_loss_3 = loss_fn(continuity_eq, torch.zeros_like(continuity_eq))
                pde_loss = (pde_loss_1 + pde_loss_2 + pde_loss_3)/3
                total_loss = train_loss + training_args['PDE_weight']*pde_loss
            else:
                total_loss = train_loss
                pde_loss_1 = torch.tensor([0])
                pde_loss_2 = torch.tensor([0])
                pde_loss_3 = torch.tensor([0])

            total_loss.backward()
            #if rank == 0: 
            nan_flag, inf_flag = check_gradients(model)
            print(f'[Epoch{epoch:4.0f}][Batch{batch_n:2.0f}] Before mean(grad): Total Loss: {total_loss.item():7.4f} ' +
                  f'LR:{scheduler.get_lr()[0]:7.6f} NaN Grads: {nan_flag} Inf Grads: {inf_flag} Model Output NaNs: {output.isnan().any()} '+
                  f'|Loss: {train_loss.item():7.4f} | PDE Losses: {pde_loss_1.item():7.4f}|{pde_loss_2.item():7.4f}|{pde_loss_3.item():7.4f}')
            
            torch.nn.utils.clip_grad_norm_(model.parameters(),training_args['grad-clip'])
            optimizer.step()
            #if epoch >= training_args['warmup_epochs']:
            scheduler.step()

        epoch_end_time = default_timer()

        with torch.no_grad():
            val_loss = 0
            for in_queries, in_keys, out_truth,__ in val_loader:
                in_queries, in_keys, out_truth = in_queries.to(rank), in_keys.to(rank), out_truth.to(rank)
                output = model(x=in_queries,inputs = in_keys)
                val_loss += loss_fn(output, out_truth)
            val_loss = val_loss/len(val_loader)

        training_run_results.update_loss({'Epoch Time': epoch_end_time - epoch_start_time})
        training_run_results.update_loss({'Training L2 Loss': train_loss.item()})
        training_run_results.update_loss({'Evaluation L2 Loss': val_loss.item()})

        if training_args['PDE_weight'] > 0:
            training_run_results.update_loss({'X-Momentum': pde_loss_1.item()})
            training_run_results.update_loss({'Y-Momentum': pde_loss_2.item()})
            training_run_results.update_loss({'Continuity': pde_loss_3.item()})

        #print(f"[Epoch{epoch}]: Training/Validation Loss on Rank {rank} is {train_loss.item():7.4f}/{val_loss.item():7.4f} with memory reserved ({rank}): {torch.cuda.memory_reserved(rank) / 1024**3:8.4f}GB ")
    
    save_checkpoint(training_args["save_dir"], training_args["save_name"], model=model, loss_dict=training_run_results.dictionary, optimizer=optimizer)

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    demo_basic(rank=device)
