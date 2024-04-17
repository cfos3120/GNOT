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

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    dataset_args, model_args, training_args = get_default_args()
    model = get_model(model_args).to(rank)
    #model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(x=torch.randn(4, 10, 2),inputs = torch.randn(4, 1, 1))
    labels = torch.randn(4, 10, 3).to(rank)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    string = f"cuda:{rank}"
    print(f"Loss on Rank {rank} is {loss.item()}. and device({string}) {torch.cuda.memory_reserved(torch.device(string))}")

    cleanup()


def run(fn, world_size):
    mp.spawn(
        fn,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    run(demo_basic, 2)
