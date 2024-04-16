from argparse import ArgumentParser
import numpy as np
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from models.noahs_model import CGPTNO
from utils import UnitTransformer
from timeit import default_timer

from data_storage.loss_recording import total_model_dict, save_checkpoint

class Cavity_2D_dataset_for_GNOT():
    def __init__(self, 
                 data_path, 
                 L=1.0, 
                 sub_x = 1, 
                 normalize_y=False, 
                 normalize_x = False, 
                 vertex = False, 
                 boundaries = False):

        print('\n Creating Dataset:')
        # Normalizer settings:
        self.normalize_y = normalize_y
        self.normalize_x = normalize_x
        self.sub_x = sub_x
        self.vertex = vertex
        self.boundaries = boundaries

        # Load in Dataset and apply transforms
        self.data_out   = np.load(data_path)
        if self.sub_x > 1: 
            self.subsampler()
        if self.vertex: 
            self.cell_to_vertex_converter()
        if self.boundaries: 
            self.add_boundaries()

        # Dataset shapes
        self.n_batches  = self.data_out.shape[0]
        self.nx         = int(self.data_out.shape[1])
        self.num_nodes  = self.nx**2

        # Flatten 2D to 1D
        self.data_out = torch.tensor(self.data_out).reshape(self.n_batches,self.num_nodes,3)

        # Input Functions (Lid Velocities)
        self.data_lid_v = torch.tensor(np.round(np.arange(0.5,100.5,0.5),1))# * 0.1/0.01 #<- apply for Reynolds Number

        # Input Queries (Coordinates)
        self.queries = self.get_query_grid(L=L)

        # Get Normalizers
        if self.normalize_x:
            self.query_normalizer = self.create_a_normalizer(self.queries)
            self.queries = self.query_normalizer.transform(self.queries, inverse=False)
            print(f'    Queries Normalized with Means: {self.query_normalizer.mean} and Stds: {self.query_normalizer.std}')
            
        if self.normalize_y:
            self.output_normalizer = self.create_a_normalizer(self.data_out)
            self.data_out = self.output_normalizer.transform(self.data_out, inverse=False)
            self.input_f_normalizer = UnitTransformer(self.data_lid_v)
            self.data_lid_v = self.input_f_normalizer.transform(self.data_lid_v, inverse=False)
            print(f'    Datset Normalized with Means: {self.output_normalizer.mean} and Stds: {self.output_normalizer.std}')
            
        self.__update_dataset_config()

    def subsampler(self):
        self.data_out = torch.nn.functional.avg_pool2d(torch.tensor(self.data_out).permute(0,3,1,2), self.sub_x).permute(0,2,3,1).numpy()
        print(f'    Dataset Subsampled by factor of {self.sub_x}')

    def cell_to_vertex_converter(self):
        self.data_out = torch.nn.functional.avg_pool2d(torch.tensor(self.data_out).permute(0,3,1,2),2,stride=1).permute(0,2,3,1).numpy()
        print('     Cell Centres Converted to Cell Verticies using average pooling (this reduces the size of each dimension by one)')

    def add_boundaries(self):
        self.data_out = torch.nn.functional.pad(torch.tensor(self.data_out).permute(0,3,1,2),(1, 1, 1, 1)).permute(0,2,3,1).numpy()

        # Lid Velocity
        self.data_out[:,-1 ,:,0] = 1

        # Pressure
        self.data_out[:,  0 ,1:-1, 2] = self.data_out[:,  1 ,1:-1, 2]  # Bottom Wall
        self.data_out[:, -1 ,1:-1, 2] = self.data_out[:, -2 ,1:-1, 2]  # Lid (y-vel)
        self.data_out[:,1:-1,  0 , 2] = self.data_out[:,1:-1,  1 , 2]  # Left Wall
        self.data_out[:,1:-1, -1 , 2] = self.data_out[:,1:-1, -2 , 2]  # Right Wall
        print('     Boundary Data added by padding and adding extra cells (this increases the size of each dimension by two)')    

    def get_query_grid(self, L):
    
        # Get grid coordinates based on vertx/volume boundary/no boundary
        divisor = self.nx - 2*int(self.boundaries) + 1*int(self.vertex)
        dx = L/divisor
        offset = dx/2 - dx*int(self.boundaries) + dx/2*int(self.vertex)
        x = torch.arange(self.nx)/divisor + offset
        y = x

        # take note of the indexing. Best for this to match the output
        [X, Y] = torch.meshgrid(x, y, indexing = 'ij')
        X = X.reshape(self.num_nodes,1)
        Y = Y.reshape(self.num_nodes,1)

        return torch.concat([Y,X],dim=-1)
    
    def create_a_normalizer(self,un_normalized_data):

        # Flatten
        n_channels = un_normalized_data.shape[-1]
        batches_and_nodes = torch.prod(torch.tensor(un_normalized_data.shape[:-1])).item()
        all_features = un_normalized_data.reshape(batches_and_nodes,n_channels)
        
        return UnitTransformer(all_features)

    def __update_dataset_config(self): 
        self.config = {
            'input_dim': self.queries.shape[-1],
            'theta_dim': 0,
            'output_dim': self.data_out.shape[-1],
            'branch_sizes': [1]
        }

class CavityDataset(Dataset):
    def __init__(self,dataset, train=True, inference=True, train_ratio=0.7, seed=42):
        print('\nCreating Dataloader:')
        self.in_queries = dataset.queries
        self.in_keys_all = dataset.data_lid_v
        self.out_truth_all = dataset.data_out
        self.train = train
        self.data_splitter(train,inference,train_ratio,seed)
        
    def data_splitter(self, train, inference, train_ratio, seed):
        n_batches = self.out_truth_all.shape[0]
        train_size = int(train_ratio * n_batches)
        test_size = n_batches - train_size

        if inference:
            seed_generator = torch.Generator().manual_seed(seed)

            # we want to pin the end points as training (to ensure complete inference)
            train_split,  test_split        = torch.utils.data.random_split(self.out_truth_all[1:-1,...], [train_size-2, test_size], generator=seed_generator)
            train_split.indices.append(0)
            train_split.indices.append(-1)
        
            # The torch.utils.data.random_split() only gives objects with the whole datset or a integers, so we need to override these variables with the indexed datset split
            train_dataset,  test_dataset    = self.out_truth_all[train_split.indices,...], self.out_truth_all[test_split.indices,...]
            train_lid_v,    test_lid_v      = self.in_keys_all[train_split.indices], self.in_keys_all[test_split.indices]
            print(f'    Dataset Split up for inference using torch generator seed: {seed_generator.initial_seed()}')
        else:
            train_dataset,  test_dataset    = self.out_truth_all[:train_size,...],  self.out_truth_all[train_size:,...]
            train_lid_v,    test_lid_v      = self.in_keys_all[:train_size,...],     self.in_keys_all[train_size:,...]
            print(f'    Dataset Split up for High reynolds number extrapolation')

        if train:
            self.out_truth_all = train_dataset
            self.in_keys_all = train_lid_v
            print(f'    Training Datset Selected')
        else:
            self.out_truth_all = test_dataset
            self.in_keys_all = test_lid_v
            print(f'    Testing Datset Selected')

    def __len__(self):
        return len(self.in_keys_all)

    def __getitem__(self, idx):
        
        # randomize input coordinates
        if self.train:
            indexes = torch.randperm(self.in_queries.shape[0])
            in_queries  = self.in_queries[indexes,...].float()
            out_truth   = self.out_truth_all[idx,indexes,...].float()
        else:
            in_queries  = self.in_queries.float()
            out_truth   = self.out_truth_all[idx,...].float()

        in_keys     = self.in_keys_all[idx].float().reshape(1,1)
        
        return in_queries, in_keys, out_truth

def get_cavity_dataset(args):

    dataset = Cavity_2D_dataset_for_GNOT(data_path=args['file_path'], 
                                        L=args['L'], 
                                        sub_x=args['sub_x'], 
                                        normalize_y=args['normalize_y'], 
                                        normalize_x=args['normalize_x'], 
                                        vertex=args['vertex'], 
                                        boundaries=args['boundaries']
                                        )
    torch_dataset = CavityDataset(dataset, 
                                    train=args['train'], 
                                    inference=args['inference'],
                                    train_ratio=args['train_ratio'], 
                                    seed=args['seed']
                                    )
    
    return torch_dataset

class LpLoss_custom(object):
    # Loss function is based on the DGL weighted loss function (only it is dgl package free)
    
    def __init__(self):
        super(LpLoss_custom, self).__init__()

    def avg_pool(self,input):
        #r^{(i)} = \frac{1}{N_i}\sum_{k=1}^{N_i} x^{(i)}_k
        batch_size = input.shape[0] # shape: Batch, Nodes, Channels
        num_nodes = input.shape[1]
        pooled_value = (1/num_nodes)*torch.sum(input,dim=1)

        return pooled_value
    
    def __call__(self, x, y):
        
        losses = (self.avg_pool(((x - y).abs() ** 2)) + 1e-8) ** (1 / 2)
        loss = losses.mean()
        return loss

def get_default_args():
    
    dataset_args = dict()
    dataset_args['L']           = 1.0
    dataset_args['sub_x']       = 8
    dataset_args['normalize_y'] = True
    dataset_args['normalize_x'] = True
    dataset_args['vertex']      = True
    dataset_args['boundaries']  = True
    dataset_args['train']       = True
    dataset_args['inference']   = True
    dataset_args['train_ratio'] = 0.7
    dataset_args['seed']        = 42

    model_args = dict()
    model_args['trunk_size']        = 2
    model_args['theta_size']        = 0
    model_args['branch_sizes']      = [1]
    model_args['output_size']       = 3
    model_args['n_layers']          = 3
    model_args['n_hidden']          = 128  
    model_args['n_head']            = 1
    model_args['attn_type']         = 'linear'
    model_args['ffn_dropout']       = 0.0
    model_args['attn_dropout']      = 0.0
    model_args['mlp_layers']        = 2
    model_args['act']               = 'gelu'
    model_args['hfourier_dim']      = 0

    training_args = dict()
    training_args['epochs']                 = 1
    training_args['base_lr']                = 0.001
    training_args['weight-decay']           = 0.00005
    #training_args['grad-clip']              = 1000.0    
    training_args['batchsize']              = 4
    training_args["save_dir"]               = 'gnot_artemis'
    training_args["save_name"]              = 'test'
    training_args['warmup_epochs']          = 5

    return dataset_args, model_args, training_args

def get_model(args):
    model = CGPTNO(
                trunk_size          = args['trunk_size'] + args['theta_size'],
                branch_sizes        = args['branch_sizes'], 
                output_size         = args['output_size'],
                n_layers            = args['n_layers'],
                n_hidden            = args['n_hidden'],
                n_head              = args['n_head'],
                attn_type           = args['attn_type'],
                ffn_dropout         = args['ffn_dropout'],
                attn_dropout        = args['attn_dropout'],
                mlp_layers          = args['mlp_layers'],
                act                 = args['act'],
                horiz_fourier_dim   = args['hfourier_dim']
                )
    
    return model

# credits:
# how to use DDP module with DDP sampler: https://gist.github.com/sgraaf/5b0caa3a320f28c27c12b5efeb35aa4c
# how to setup a basic DDP example from scratch: https://pytorch.org/tutorials/intermediate/dist_tuto.html
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import random

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import math

def get_dataset(args):
    world_size = dist.get_world_size()
    train_set = val_set = get_cavity_dataset(args)
    args['train'] = False
    val_set = get_cavity_dataset(args)
    
    train_sampler = DistributedSampler(train_set,num_replicas=world_size)
    val_sampler = DistributedSampler(val_set,num_replicas=world_size)
    batch_size = int(args['batchsize'] / float(world_size))
    
    print(world_size, batch_size)
    
    train_loader = DataLoader(
        dataset=train_set,
        sampler=train_sampler,
        batch_size=batch_size
    )
    val_loader = DataLoader(
        dataset=val_set,
        sampler=val_sampler,
        batch_size=batch_size
    )

    return train_loader, val_loader, batch_size

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

def train(model,train_loader,optimizer,scheduler,batch_size):
    device = torch.device(f"cuda:{dist.get_rank()}")
    train_num_batches = int(math.ceil(len(train_loader.dataset) / float(batch_size)))
    model.train()
    # let all processes sync up before starting with a new epoch of training
    # dist.barrier()
    criterion = LpLoss_custom()#.to(device)
    train_loss = 0.0
    for in_queries, in_keys, out_truth in train_loader:
        
        in_queries, in_keys, out_truth = in_queries.to(device), in_keys.to(device), out_truth.to(device)
        optimizer.zero_grad()
        output = model(x=in_queries,inputs = in_keys)
        loss = criterion(output, out_truth)
        loss.backward()
        # average gradient as DDP doesn't do it correctly
        average_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)#training_args['grad-clip'])
        optimizer.step()
        scheduler.step()

        loss_ = {'loss': torch.tensor(loss.item()).to(device)}
        train_loss += reduce_dict(loss_)['loss'].item()
        # cleanup
        # dist.barrier()
        # data, target, output = data.cpu(), target.cpu(), output.cpu()
    train_loss_val = train_loss / train_num_batches
    return train_loss_val
    
def val(model, val_loader,batch_size):
    device = torch.device(f"cuda:{dist.get_rank()}")
    val_num_batches = int(math.ceil(len(val_loader.dataset) / float(batch_size)))
    model.eval()
    # let all processes sync up before starting with a new epoch of training
    # dist.barrier()
    criterion = LpLoss_custom()#.to(device)
    val_loss = 0.0
    with torch.no_grad():
        for in_queries, in_keys, out_truth in val_loader:
            in_queries, in_keys, out_truth = in_queries.to(device), in_keys.to(device), out_truth.to(device)
            output = model(x=in_queries,inputs = in_keys)
            loss = criterion(output, out_truth)
            loss_ = {'loss': torch.tensor(loss.item()).to(device)}
            val_loss += reduce_dict(loss_)['loss'].item()
    val_loss_val = val_loss / val_num_batches
    return val_loss_val

def run(rank, world_size, args):
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42)

    dataset_args, model_args, training_args = get_default_args()
    
    # manual override:
    #dataset_args['file_path']       = args.path
    dataset_args['sub_x']           = args.sub_x
    dataset_args['batchsize']       = args.batchsize
    training_args['epochs']         = args.epochs
    training_args["save_name"]      = args.name
    dataset_args['file_path'] = '/project/MLFluids/steady_cavity_case_b200_maxU100ms_simple_normalized.npy'

    train_loader, val_loader, batch_size = get_dataset(dataset_args)

    model = get_model(model_args).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # use if model contains batchnorm.
    model = DDP(model,device_ids=[rank],output_device=rank)
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.5)
    
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
    
    if rank == 0:
        training_run_results = total_model_dict(model_config=model_args, training_config=training_args, data_config=dataset_args)

    for epoch in range(training_args['epochs']):
        
        epoch_start_time = default_timer()
        train_loss_val = train(model,train_loader,optimizer,scheduler,batch_size)
        val_loss_val = val(model,val_loader,batch_size)

        epoch_end_time = default_timer()
        if rank == 0:
            print(f'Rank {rank} epoch {epoch}: {train_loss_val:.2f}')/{val_loss_val:.2f}')

            training_run_results.update_loss({'Epoch Time': epoch_end_time - epoch_start_time})
            training_run_results.update_loss({'Training L2 Loss': train_loss_val.item()})
            training_run_results.update_loss({'Evaluation L2 Loss': val_loss_val.item()})
      
    print(f'Rank {rank} finished training')

    dist.barrier()
    if rank == 0:
        save_checkpoint(args["save_dir"], args["save_name"], model=model, loss_dict=training_run_results.dictionary, optimizer=optimizer)
    dist.barrier()
    cleanup(rank)  

def cleanup(rank):
    # dist.cleanup()  
    dist.destroy_process_group()
    print(f"Rank {rank} is done.")

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_process(
        rank, # rank of the process
        world_size, # number of workers
        fn, # function to be run
        args,
        # backend='gloo',# good for single node
        # backend='nccl' # the best for CUDA
        backend='gloo'
    ):
    # information used for rank 0
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    dist.barrier()
    setup_for_distributed(rank == 0)
    fn(rank, world_size, args)


if __name__ == "__main__":

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
    args = parser.parse_args()

    # DDP
    world_size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()