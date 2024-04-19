import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from models.noahs_model import CGPTNO
from utils import UnitTransformer

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
            test_split.indices = list(1 + np.array(test_split.indices))
            train_split.indices = list(1 + np.array(train_split.indices))
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
        
        in_keys     = self.in_keys_all[idx].float().reshape(1,1)

        # randomize input coordinates
        if self.train:
            indexes = torch.randperm(self.in_queries.shape[0])
            in_queries  = self.in_queries[indexes,...].float()
            out_truth   = self.out_truth_all[idx,indexes,...].float()
            reverse_indices = torch.argsort(indexes)
            return in_queries, in_keys, out_truth, reverse_indices
        else:
            in_queries  = self.in_queries.float()
            out_truth   = self.out_truth_all[idx,...].float()
            reverse_indices = torch.arange(self.in_queries.shape[0]) # <- just normal order
            return in_queries, in_keys, out_truth, reverse_indices

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
    
    return torch_dataset, dataset.output_normalizer, dataset.input_f_normalizer

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
    
    def __call__(self, x, y=None):

        if y is not None:
            losses = (self.avg_pool(((x - y).abs() ** 2)) + 1e-8) ** (1 / 2)
        else:
            losses = (self.avg_pool((x.abs() ** 2)) + 1e-8) ** (1 / 2)
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
    dataset_args['batchsize']   = 4

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
    training_args["save_dir"]               = 'gnot_artemis'
    training_args["save_name"]              = 'test'
    training_args['warmup_epochs']          = 5
    training_args['PDE_weight']             = 0.25

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

## FOR DDP:
# this may need 'self.train_data.sampler.set_epoch(epoch)' at the start of each epoch
def get_dataset(args):
    world_size = dist.get_world_size()
    train_set, output_normalizer, input_f_normalizer = get_cavity_dataset(args)
    args['train'] = False
    val_set = get_cavity_dataset(args)
    
    train_sampler = DistributedSampler(train_set,num_replicas=world_size)
    val_sampler = DistributedSampler(val_set,num_replicas=world_size)
    batch_size = int(args['batchsize'] / float(world_size))
    
    #print(world_size, batch_size)
    
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

    return train_loader, val_loader, batch_size, output_normalizer, input_f_normalizer

def NS_FDM_cavity_internal_vertex_non_dim(U, lid_velocity, nu, L, pressure_overide=False):

    # with boundary conditions

    batchsize = U.size(0)
    nx = U.size(1)
    ny = U.size(2)
    
    device = U.device

    # assign Reynolds Number array:
    Re = (lid_velocity * L/nu).repeat(1,nx-2,ny-2)

    # create isotropic grid (non-dimensional i.e. L=1.0)
    y = torch.tensor(np.linspace(0, 1, nx), dtype=torch.float, device=device)
    x = y

    # initialize Storage of derivatives as zeros
    ux = torch.zeros([batchsize, nx-2, ny-2])
    uy = torch.zeros_like(ux)
    vx = torch.zeros_like(ux)
    vy = torch.zeros_like(ux)
    uxx = torch.zeros_like(ux)
    uyy = torch.zeros_like(ux)
    vxx = torch.zeros_like(ux)
    vyy = torch.zeros_like(ux)
    px = torch.zeros_like(ux)
    py = torch.zeros_like(ux)
    
    # second order first derivative scheme
    dx = abs(x[1]-x[0])
    dy = dx
       
    u = torch.zeros([batchsize, nx-2, ny-2])
    v = torch.zeros_like(u)
    p = torch.zeros_like(u)
    
    # assign internal field
    u = U[...,0]
    v = U[...,1]
    p = U[...,2]

    if pressure_overide:
        p = (u**2 + v**2)*1/2           #density is one
        
    # gradients in internal zone
    uy  = (u[:, 2:  , 1:-1] -   u[:,  :-2, 1:-1]) / (2*dy)
    ux  = (u[:, 1:-1, 2:  ] -   u[:, 1:-1,  :-2]) / (2*dx)
    uyy = (u[:, 2:  , 1:-1] - 2*u[:, 1:-1, 1:-1] + u[:,  :-2, 1:-1]) / (dy**2)
    uxx = (u[:, 1:-1, 2:  ] - 2*u[:, 1:-1, 1:-1] + u[:, 1:-1,  :-2]) / (dx**2)

    vy  = (v[:, 2:  , 1:-1] -   v[:,  :-2, 1:-1]) / (2*dy)
    vx  = (v[:, 1:-1, 2:  ] -   v[:, 1:-1,  :-2]) / (2*dx)
    vyy = (v[:, 2:  , 1:-1] - 2*v[:, 1:-1, 1:-1] + v[:,  :-2, 1:-1]) / (dy**2)
    vxx = (v[:, 1:-1, 2:  ] - 2*v[:, 1:-1, 1:-1] + v[:, 1:-1,  :-2]) / (dx**2)

    py  = (p[:, 2:  , 1:-1] - p[:,  :-2, 1:-1]) / (2*dy)
    px  = (p[:, 1:-1, 2:  ] - p[:, 1:-1,  :-2]) / (2*dx)

    # No time derivative as we are assuming steady state solution
    Du_dx = U[...,1:-1,1:-1 ,0]*ux + U[...,1:-1,1:-1, 1]*uy - (1/Re) * (uxx + uyy) + px
    Dv_dy = U[...,1:-1,1:-1 ,0]*vx + U[...,1:-1,1:-1 ,1]*vy - (1/Re) * (vxx + vyy) + py
    continuity_eq = (ux + vy)

    fdm_derivatives = tuple([ux, uy, vx, vy, px, py, uxx, uyy, vxx, vyy, Du_dx, Dv_dy, continuity_eq])
    
    return Du_dx, Dv_dy, continuity_eq, fdm_derivatives

def output_realiser(model_output, model_input_key, output_normalizer, input_key_normalizer, reverse_indices=None):
    output = model_output.clone()
    input_key = model_input_key.clone()

    # rearrange (only makes a difference if input query coordinates was shuffled)
    if reverse_indices is not None:
        output = torch.concat([output[batch,index_order,:].unsqueeze(0) for batch, index_order in enumerate(reverse_indices)],dim=0)

    output = output_normalizer.transform(output, inverse=True)
    input_key = input_key_normalizer.transform(input_key, inverse=True)

    # assuming isotropic grid:
    dim = int(np.sqrt(model_output.shape[1]))
    batches = int(model_output.shape[0])
    channels = batches = int(model_output.shape[-1])
    output = output.reshape(batches, dim, dim, channels)
    return output, input_key


