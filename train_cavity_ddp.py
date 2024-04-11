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

def demo_basic():

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank'
    dataset_args, model_args, training_args = get_default_args()

    device_id = rank % torch.cuda.device_count()
    model = get_model(model_args).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

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
    in_queries = torch.randn(4, 100,2)
    in_keys = torch.randn(4, 1,1)
    outputs = ddp_model(x=in_queries,inputs = in_keys)
    labels = torch.randn(4, 100,3).to(device_id)
    loss_function(outputs, labels).backward()
    optimizer.step()
    dist.destroy_process_group()

if __name__ == "__main__":
    demo_basic()
    