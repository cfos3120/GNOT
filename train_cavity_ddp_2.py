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

SEED = 42
BATCH_SIZE = 4
NUM_EPOCHS = 3

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

            train_split,  test_split        = torch.utils.data.random_split(self.out_truth_all, [train_size, test_size], generator=seed_generator)
        
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
    dataset_args['sub_x']       = 1
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
    training_args['grad-clip']              = 1000.0    
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

if __name__ == '__main__':
    parser = ArgumentParser('DDP usage example')
    #parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')  # you need this argument in your scripts for DDP to work
    parser.add_argument('--path', type=str, default= r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\GNOT\data\steady_cavity_case_b200_maxU100ms_simple_normalized.npy')
    args = parser.parse_args()
    args.local_rank = os.environ['LOCAL_RANK']
    # keep track of whether the current process is the `master` process (totally optional, but I find it useful for data laoding, logging, etc.)
    args.is_master = args.local_rank == 0

    # set the device
    args.device = torch.cuda.device(args.local_rank)

    # initialize PyTorch distributed using environment variables (you could also do this more explicitly by specifying `rank` and `world_size`, but I find using environment variables makes it so that you can easily use the same script on different machines)
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)

    # set the seed for all GPUs (also make sure to set the seed for random, numpy, etc.)
    torch.cuda.manual_seed_all(SEED)

    # get args and override
    dataset_args, model_args, training_args = get_default_args()
    dataset_args['file_path']       = args.path
    #dataset_args['sub_x']           = 1#args.sub_x

    # initialize your model (GNOT in this example)
    model = get_model(model_args)

    # send your model to GPU
    device = torch.device(f"cuda:{dist.get_rank()}")
    model = model.to(device)

    # initialize distributed data parallel (DDP)
    model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank
    )

    # initialize your dataset
    dataset = get_cavity_dataset(dataset_args)

    # initialize the DistributedSampler
    sampler = DistributedSampler(dataset)

    # initialize the dataloader
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=BATCH_SIZE
    )

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
                                                    epochs=NUM_EPOCHS
                                                    )

    # start your training!
    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)
        
        # put model in train mode
        model.train()

        # let all processes sync up before starting with a new epoch of training
        dist.barrier()

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            in_queries, in_keys, out_truth = batch
            in_queries, in_keys, out_truth = in_queries.to(device), in_keys.to(device), out_truth.to(device)
            
            # forward pass
            out = model(x=in_queries,inputs = in_keys)
            
            # compute loss
            loss = loss_function(out,out_truth)

            # etc.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_args['grad-clip'])
            optimizer.step()

            scheduler.step()

        print(f'Epoch: {epoch :8} L2 Training Loss {loss :12.7f}')