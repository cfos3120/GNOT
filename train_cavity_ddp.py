import sys
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from timeit import default_timer

from models.noahs_model import CGPTNO
from utils import UnitTransformer
from data_storage.loss_recording import total_model_dict, save_checkpoint
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from train_cavity_artemis_v2 import get_default_args, get_model, LpLoss_custom
import torch.multiprocessing as mp

def setup(rank, world_size):

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()



def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    dataset_args, model_args, training_args = get_default_args()

    # create model and move it to GPU with id rank
    model = get_model(model_args).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_function = LpLoss_custom()
    
    # Default optimizer and scheduler from paper
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
                                                    steps_per_epoch=10, 
                                                    epochs=training_args['epochs']
                                                    )

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(4, 10,10,2))
    labels = torch.randn(4, 10,10,3).to(rank)
    loss_function(outputs, labels).backward()
    optimizer.step()

    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)