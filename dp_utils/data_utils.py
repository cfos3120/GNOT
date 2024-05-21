import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from models.noahs_model import CGPTNO
from models.noahs_model_2 import GNOT
from utils import UnitTransformer

class Cavity_2D_dataset_for_GNOT():
    def __init__(self, 
                 data_path, 
                 L=1.0, 
                 sub_x = 1, 
                 normalize_y=False, 
                 normalize_x = False,
                 normalize_f = False, 
                 vertex = False, 
                 boundaries = False):

        print('\n Creating Dataset:')
        # Normalizer settings:
        self.normalize_y = normalize_y
        self.normalize_x = normalize_x
        self.normalize_f = normalize_f
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
        else:
            self.query_normalizer = None
            
        if self.normalize_y:
            self.output_normalizer = self.create_a_normalizer(self.data_out)
            self.data_out = self.output_normalizer.transform(self.data_out, inverse=False)
            print(f'    Datset Normalized with Means: {self.output_normalizer.mean} and Stds: {self.output_normalizer.std}')
        else:
            self.output_normalizer = None
        
        if self.normalize_f:
            self.input_f_normalizer = UnitTransformer(self.data_lid_v)
            self.data_lid_v = self.input_f_normalizer.transform(self.data_lid_v, inverse=False)
            print(f'    Keys Normalized with Means: {self.input_f_normalizer.mean} and Stds: {self.input_f_normalizer.std}')
        else:
            self.input_f_normalizer = None
        
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
    def __init__(self,dataset, train=True, inference=True, train_ratio=0.7, seed=42, random_coords = True):
        print('\nCreating Dataloader:')
        self.in_queries = dataset.queries
        self.in_keys_all = dataset.data_lid_v
        self.out_truth_all = dataset.data_out
        self.random_coords = random_coords
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
        
        # randomize input coordinates
        if self.random_coords:
            indices = torch.randperm(self.in_queries.shape[0])
            reverse_indices = torch.argsort(indices)
        else:
            indices = torch.arange(self.in_queries.shape[0])
            reverse_indices = indices
        
        in_keys     = self.in_keys_all[idx].float().reshape(1,1)
        in_queries  = self.in_queries[indices,...].float()
        out_truth   = self.out_truth_all[idx,indices,...].float()
        return in_queries, in_keys, out_truth, reverse_indices

def prepare_dataset(args):
    torch_dataset = Cavity_2D_dataset_for_GNOT(data_path=args['file_path'], 
                                                L=args['L'], 
                                                sub_x=args['sub_x'], 
                                                normalize_y=args['normalize_y'], 
                                                normalize_x=args['normalize_x'], 
                                                normalize_f=args['normalize_f'], 
                                                vertex=args['vertex'], 
                                                boundaries=args['boundaries']
                                                )
    return torch_dataset

def create_torch_dataset(dataset,args,train,random_coords):
    torch_dataset = CavityDataset(dataset, 
                                    train=train, 
                                    inference=args['inference'],
                                    train_ratio=args['train_ratio'], 
                                    seed=args['seed'],
                                    random_coords=random_coords
                                    )
    
    return torch_dataset

def get_default_args():
    
    dataset_args = dict()
    dataset_args['L']           = 1.0
    dataset_args['sub_x']       = 8
    dataset_args['normalize_y'] = True
    dataset_args['normalize_x'] = True
    dataset_args['normalize_f'] = False
    dataset_args['vertex']      = True
    dataset_args['boundaries']  = True
    dataset_args['inference']   = True
    dataset_args['train_ratio'] = 0.7
    dataset_args['seed']        = 42
    dataset_args['batchsize']   = 4
    dataset_args['random_coords'] = False

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
    model_args['model_name']        = 'CGPTNO'

    training_args = dict()
    training_args['epochs']                 = 1
    training_args['base_lr']                = 0.001
    training_args['weight-decay']           = 0.00005
    training_args['grad-clip']              = 1000.0    
    training_args["save_dir"]               = 'gnot_artemis_dp_3'
    training_args["save_name"]              = 'test'
    training_args['warmup_epochs']          = 5
    training_args['PDE_weight']             = 0.0

    return dataset_args, model_args, training_args

def get_model(args):

    if args['model_name'] == 'CGPTNO':
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
    elif args['model_name'] == 'GNOT':
        model = GNOT(
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
                    horiz_fourier_dim   = args['hfourier_dim'],
                    space_dim=2
                    )
    else: raise NotImplemented("model name does not exist")
    
    return model

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

def cuda_get_all_memory_reserved():
    
    if torch.cuda.device_count() > 0:
        string = '\n|'
        for i in range(torch.cuda.device_count()):
            GPU_name = f"cuda:{i}"
            memory = torch.cuda.memory_reserved(torch.device(GPU_name))
            string += f"{GPU_name} {memory / 1024**3:8.4f}GB | "
        string += '\n'
    else:
        string = '\n| No GPUs Available |\n'

    print(string)

def NS_FDM_cavity_internal_vertex_non_dim(U, lid_velocity, nu, L, pressure_overide=False):

    # with boundary conditions

    batchsize = U.size(0)
    nx = U.size(1)
    ny = U.size(2)
    
    device = U.device

    # assign Reynolds Number array:
    Re = (lid_velocity * L/nu).repeat(1,nx,ny)

    # create isotropic grid (non-dimensional i.e. L=1.0)
    y = torch.tensor(np.linspace(0.0, 1.0, nx), dtype=torch.float, device=device)
    x = y

    # initialize Storage of derivatives as zeros
    ux = torch.zeros([batchsize, nx, ny])
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
    
    # Create padding
    u = torch.zeros([batchsize, nx+2, ny+2], dtype=torch.float, device=device)
    v = torch.zeros_like(u, dtype=torch.float, device=device)
    p = torch.zeros_like(u, dtype=torch.float, device=device)
    
    # assign internal field
    u[...,1:-1,1:-1] = U[...,0]
    v[...,1:-1,1:-1] = U[...,1]
    p[...,1:-1,1:-1] = U[...,2]

    #Fill in padding
    u[:,-1,:]     = 1.0
    p[:,-1,:]     = p[:,-2,:]
    p[:,1:-1,0]   = p[:,1:-1,1]
    p[:,1:-1,-1]  = p[:,1:-1,-2]
    p[:,0,:]      = p[:,1,:]

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
    Du_dx = U[...,0]*ux + U[...,1]*uy - (1/Re) * (uxx + uyy) + px
    Dv_dy = U[...,0]*vx + U[...,1]*vy - (1/Re) * (vxx + vyy) + py
    continuity_eq = (ux + vy)

    fdm_derivatives = tuple([ux, uy, vx, vy, px, py, uxx, uyy, vxx, vyy, Du_dx, Dv_dy, continuity_eq])
    
    return Du_dx.float(), Dv_dy.float(), continuity_eq.float()#, fdm_derivatives

def NS_FDM_cavity_boundary_vertex_non_dim(output, loss_fn):

    # velocity
    lid     = output[:,-1,:,:]
    wall_l  = output[:,1:-1,0,:]
    wall_r  = output[:,1:-1,-1,:]
    wall_b  = output[:,0,:,:]

    bc1 = (loss_fn(lid[...,0], torch.ones_like(lid[...,0])) + 
           loss_fn(wall_l[...,0]) + 
           loss_fn(wall_r[...,0]) + 
           loss_fn(wall_b[...,0])
           )/4
    bc2 = (loss_fn(lid[...,1]) + 
           loss_fn(wall_l[...,1]) + 
           loss_fn(wall_r[...,1]) + 
           loss_fn(wall_b[...,1])
           )/4

    # Pressure
    bc3 = (loss_fn(lid[...,2], output[:,-2,:,2]) + 
           loss_fn(wall_l[...,2], output[:,1:-1,1,2]) + 
           loss_fn(wall_r[...,2], output[:,1:-1,-2,2]) + 
           loss_fn(wall_b[...,2], output[:,1,:,2])
           )/4
    
    return bc1, bc2, bc3





def output_realiser(output, input_key, output_normalizer, input_key_normalizer, reverse_indices=None):

    # rearrange (only makes a difference if input query coordinates was shuffled)
    if reverse_indices is not None:
        output = torch.concat([output[batch,index_order,:].unsqueeze(0) for batch, index_order in enumerate(reverse_indices)],dim=0)

    output = output_normalizer.transform(output, inverse=True)
    if input_key_normalizer is not None:
        input_key = input_key_normalizer.transform(input_key, inverse=True)

    # assuming isotropic grid:
    dim = int(np.sqrt(output.shape[1]))
    batches = output.shape[0]
    channels = output.shape[-1]
    output = output.reshape(batches, dim, dim, channels)
    return output, input_key