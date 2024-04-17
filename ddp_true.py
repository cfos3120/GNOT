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

from torch.nn.parallel import DistributedDataParallel as DDP


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

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    dataset_args, model_args, training_args = get_default_args()
    model = get_model(model_args).to(rank)
    #model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    dataset_args['sub_x']           = os.environ['ARGS'].sub_x
    dataset_args['batchsize']       = os.environ['ARGS'].batch_size
    training_args['epochs']         = os.environ['ARGS'].epochs
    training_args["save_name"]      = os.environ['ARGS'].name
    dataset_args['file_path'] = '/project/MLFluids/steady_cavity_case_b200_maxU100ms_simple_normalized.npy'

    train_loader, val_loader, batch_size = get_dataset(dataset_args)

    loss_fn = LpLoss_custom()
    optimizer = torch.optim.AdamW(ddp_model.parameters(), 
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
    
    if rank == 0:
        training_run_results = total_model_dict(model_config=model_args, training_config=training_args, data_config=dataset_args)

    print(f"Started training on rank {rank}.")
    for epoch in range(training_args['epochs']):

        for in_queries, in_keys, out_truth in train_loader:
            in_queries, in_keys, out_truth = in_queries.to(rank), in_keys.to(rank), out_truth.to(rank)
            optimizer.zero_grad()
            output = ddp_model(x=in_queries,inputs = in_keys)
            
            loss = loss_fn(output, out_truth)
            loss.backward()
            average_gradients(ddp_model)
            torch.nn.utils.clip_grad_norm_(model.parameters(),training_args['grad-clip'])
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            for in_queries, in_keys, out_truth in val_loader:
                in_queries, in_keys, out_truth = in_queries.to(rank), in_keys.to(rank), out_truth.to(rank)
                output = ddp_model(x=in_queries,inputs = in_keys)
                loss = loss_fn(output, out_truth)

    string = f"cuda:{rank}"
    print(f"Loss on Rank {rank} is {loss.item()}. and device({string}) {torch.cuda.memory_reserved(torch.device(string)) / 1024**3:8.4f}GB ")

    cleanup()


def run(fn, world_size):
    mp.spawn(
        fn,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":

    parser = ArgumentParser(description='GNOT Artemis Training Study')
    parser.add_argument('--name', type=str, default='test')
    #parser.add_argument('--path', type=str, default= r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\GNOT\data\steady_cavity_case_b200_maxU100ms_simple_normalized.npy')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--sub_x', type=int, default=4)
    parser.add_argument('--inference', type=str, default='True')
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4)
    global ARGS 
    ARGS = parser.parse_args()
    os.environ['ARGS'] = parser.parse_args()
    
    run(demo_basic, 2)
