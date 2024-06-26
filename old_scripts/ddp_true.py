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
global ARGS 
ARGS = parser.parse_args()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group "gloo" nccl
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def average_loss(loss):
    size = float(dist.get_world_size())
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    loss /= size
    return loss

def check_gradients(model):
    nan_flag = False
    inf_flag = False
    for param in model.parameters():
        if param.grad.data.isnan().any():
            nan_flag = True
        if param.grad.data.isfinite().any():
            inf_flag = True
    
    return nan_flag, inf_flag

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    torch.manual_seed(42)
    # create model and move it to GPU with id rank
    dataset_args, model_args, training_args = get_default_args()
    model = get_model(model_args).to(rank)
    #model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    dataset_args['sub_x']           = ARGS.sub_x
    dataset_args['batchsize']       = ARGS.batch_size
    training_args['epochs']         = ARGS.epochs
    training_args["save_name"]      = ARGS.name
    training_args['PDE_weight']     = ARGS.pde_weight
    dataset_args['file_path']       = ARGS.path #'/project/MLFluids/steady_cavity_case_b200_maxU100ms_simple_normalized.npy'

    train_loader, val_loader, batch_size, output_normalizer, input_f_normalizer = get_dataset(dataset_args)

    loss_fn = LpLoss_custom()
    # optimizer = torch.optim.AdamW(ddp_model.parameters(), 
    #                               betas=(0.9, 0.999), 
    #                               lr=training_args['base_lr'],
    #                               weight_decay=training_args['weight-decay']
    #                               )
    optimizer = ZeroRedundancyOptimizer(ddp_model.parameters(),
                                        optimizer_class=torch.optim.AdamW,
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
    
    if rank == 0:
        training_run_results = total_model_dict(model_config=model_args, training_config=training_args, data_config=dataset_args)
    string = f"cuda:{rank}"
    

    print(f"Started training on rank {rank}.")
    for epoch in range(training_args['epochs']):
        dist.barrier()
        torch.cuda.empty_cache()
        epoch_start_time = default_timer()
        for in_queries, in_keys, out_truth, reverse_indices in train_loader:
            in_queries, in_keys, out_truth = in_queries.to(rank), in_keys.to(rank), out_truth.to(rank)
            optimizer.zero_grad()
            output = ddp_model(x=in_queries,inputs = in_keys)
            
            # Pointwise Loss
            train_loss = loss_fn(output, out_truth)

            # PDE Loss
            if training_args['PDE_weight'] > 0:
                outputs, input_keys = output_realiser(output, in_keys, output_normalizer.to(rank), input_f_normalizer.to(rank), reverse_indices.to(rank))
                Du_dx, Dv_dy, continuity_eq,__ = NS_FDM_cavity_internal_vertex_non_dim(U=outputs, lid_velocity=input_keys, nu=0.01, L=1.0)
                pde_loss_1 = loss_fn(Du_dx)
                pde_loss_2 = loss_fn(Dv_dy)
                pde_loss_3 = loss_fn(continuity_eq)
                pde_loss = (pde_loss_1 + pde_loss_2 + pde_loss_3)/3
                total_loss = 5*train_loss + pde_loss
            else:
                total_loss = 5*train_loss

            total_loss.backward()
            #if rank == 0: 
                #nan_flag, inf_flag = check_gradients(model)
                #print(f'[Epoch{epoch}][Rank{rank}] Before mean(grad): Loss: {train_loss.item():7.4f} LR:{scheduler.get_lr()} NaN Grads: {nan_flag} Inf Grads: {inf_flag} Model Output NaNs: {output.isnan().any()}')
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(),training_args['grad-clip'])
            optimizer.step()
            scheduler.step()

        epoch_end_time = default_timer()

        dist.barrier()
        with torch.no_grad():
            val_loss = 0
            for in_queries, in_keys, out_truth,__ in val_loader:
                in_queries, in_keys, out_truth = in_queries.to(rank), in_keys.to(rank), out_truth.to(rank)
                output = ddp_model(x=in_queries,inputs = in_keys)
                val_loss += loss_fn(output, out_truth)
            val_loss = val_loss/len(val_loader)

        if rank == 0:
            training_run_results.update_loss({'Epoch Time': epoch_end_time - epoch_start_time})
            training_run_results.update_loss({'Training L2 Loss': train_loss.item()})
            training_run_results.update_loss({'Evaluation L2 Loss': val_loss.item()})

        print(f"[Epoch{epoch}]: Training/Validation Loss on Rank {rank} is {train_loss.item():7.4f}/{val_loss.item():7.4f} with memory reserved ({string}): {torch.cuda.memory_reserved(torch.device(string)) / 1024**3:8.4f}GB ")
    
    dist.barrier()
    if rank == 0:
        save_checkpoint(training_args["save_dir"], training_args["save_name"], model=model, loss_dict=training_run_results.dictionary, optimizer=optimizer)
    dist.barrier()

    cleanup()


def run(fn, world_size):
    mp.spawn(
        fn,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":

    run(demo_basic, ARGS.world_size)
