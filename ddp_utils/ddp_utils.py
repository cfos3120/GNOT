import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

'''
To run DDP:
define a function to run in parallel ('fn')

in __main__ run 'run' which features MP Spawn

1. in the function 'fn' include 'setup()'
2. wrap the model in DDP
3. at the end of the function: 'cleanup()'
'''

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
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
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

def get_gpu_resources():
    if torch.cuda.is_available():
        print_string1 = '   Memory Reserved:'
        print_string2 = '   Memory Allocated:'
        for gpu_n in range(torch.cuda.device_count()):
            gpu_n_str = f"cuda:{gpu_n}"
            print_string1 += f'GPU{gpu_n} {torch.cuda.memory_reserved(torch.device(gpu_n_str)) / 1024**3:8.4f}GB '
            print_string2 += f'GPU{gpu_n} {torch.cuda.memory_allocated(torch.device(gpu_n_str)) / 1024**3:8.4f}GB '
    else:
        print_string1, print_string2 = '        No GPUs', '        No GPUs'

    return print_string1, print_string2

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def reduce_dict(input_dict, average=True):
    world_size = float(dist.get_world_size())
    names, values = [], []
    for k in sorted(input_dict.keys()):
        names.append(k)
        values.append(input_dict[k])
    values = torch.stack(values, dim=0)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if average:
        values /= world_size
    reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict