import sys
import argparse
import numpy as np

import torch
import torch.nn.functional as F

from timeit import default_timer

from models.noahs_model import CGPTNO
from utils import UnitTransformer
from data_storage.loss_recording import total_model_dict, save_checkpoint
from data_storage.loss_recording import total_model_dict, save_checkpoint

class Cavity_2D_dataset_for_GNOT():
    def __init__(self, data_path, L=1.0, sub_x = 1, train=True, normalize_y=False, y_normalizer=None, normalize_x = False, x_normalizer = None, up_normalizer =None, vertex = False, boundaries = False):

        '''
        This class takes in a dataset structure used by FNO and PINO models. 
        The dataset is typically stored as [Batch, Timesteps, X-coords, Y-coords]
        This function creates a Mesh Grid of coordinates (which is the same for all batches)
        and reshapes it into a linear with shape [Batch, number of nodes, dims(coordinates)].
        NOTE This array will then be the Transformer Queries Array (X or g).

        Similarly, the ground truth voriticity profiles are taken and split by ic_t_steps into
        both the output shape/ training validation loss dataset (Y) and the initial conditions 
        which will be the first ic_t_steps time steps in the dataset.
        NOTE This ic array will be nested in a tuple and called as an input function
        i.e. The Transformer Keys and Values Array (f or g_u).

        Finally, Theta is set a as a small zero array as the current model handling does not take
        a null input. This should not impact the model training (although it does make the datasets
        slighly larger which is not ideal for memory)
        '''
        # Normalizer settings:
        self.normalize_y = normalize_y
        self.normalize_x = normalize_x
        self.y_normalizer = y_normalizer
        self.x_normalizer = x_normalizer
        self.up_normalizer = up_normalizer
        self.sub_x = sub_x
        self.vertex = vertex
        self.boundaries = boundaries

        # Load in Dataset and retrieve shape
        self.data_out   = np.load(data_path)
        if self.sub_x > 1: self.subsampler()
        if self.vertex: self.cell_to_vertex_converter()
        if self.boundaries: self.add_boundaries()

        print(f'Dataset Shape: {self.data_out.shape}, subsampled by {self.sub_x}')
        # NOTE this can also be in the form of reynolds number 
        self.data_lid_v = np.round(np.arange(0.5,100.5,0.5),1) * 0.1/0.01 #<- apply for Reynolds Number
        self.n_batches  = self.data_out.shape[0]
        self.nx         = int(self.data_out.shape[1])
        self.num_nodes  = self.nx**2

        self.L = L
        self.train = train

    def assign_data_split_type(self, inference=True, train_ratio=0.7, seed=42):
        self.seed = seed
        self.data_split = train_ratio
        self.inference = inference

    def process(self,theta=False):
        
        # Should the input function be self-referenced (theta) or cross-referenced?
        self.theta = theta

        # SECTION 0: Split into train or test (Same as for FNO training)
        train_size = int(self.data_split * self.n_batches)
        test_size = self.n_batches - train_size

        seed_generator = torch.Generator().manual_seed(self.seed)

        # Perform Inference or Extrapolation (Inference is randomly sampled)
        if self.inference:
            train_dataset,  test_dataset    = torch.utils.data.random_split(torch.from_numpy(self.data_out),    [train_size, test_size], generator=seed_generator)
            train_lid_v,    test_lid_v      = torch.utils.data.random_split(torch.from_numpy(self.data_lid_v),  [train_size, test_size], generator=seed_generator)
            
            # The torch.utils.data.random_split() only gives objects with the whole datset or a integers, so we need to override these variables with the indexed datset split
            train_dataset,  test_dataset    = train_dataset.dataset[train_dataset.indices,...], test_dataset.dataset[test_dataset.indices,...]
            train_lid_v,    test_lid_v      = train_lid_v.dataset[train_lid_v.indices], test_lid_v.dataset[test_lid_v.indices]
            print(f'''Dataset Split up using torch generator seed: {seed_generator.initial_seed()}
              This can be replicated e.g.
                generator_object = torch.Generator().manual_seed({seed_generator.initial_seed()})\n ''')
        else:
            train_dataset,  test_dataset    = torch.from_numpy(self.data_out[:train_size,...]), torch.from_numpy(self.data_out[train_size:,...])
            train_lid_v,    test_lid_v      = torch.from_numpy(self.data_lid_v[:test_size,...]), torch.from_numpy(self.data_lid_v[test_size:,...])

        if self.train:
            self.data_out   = train_dataset
            self.data_lid_v = train_lid_v
            self.n_batches  = train_size
        else:
            self.data_out   = test_dataset
            self.data_lid_v = test_lid_v
            self.n_batches  = test_size

        # SECTION 1: Transformer Queries
        # Assume Isotropic Grid adjusting coordinates for cell centered or vertex points accordingly.
        # Also includes boundaries if stated (note boundaries + cell-centered will cause boundary coordinates to be 0-dx, 1+dx overflow)
        # this is to maintain isotropic property
        divisor = self.nx - 2*int(self.boundaries) + 1*int(self.vertex)
        dx = self.L/divisor
        offset = dx/2 - dx*int(self.boundaries) + dx/2*int(self.vertex)
        x = torch.arange(self.nx)/divisor + offset
        y = x

        # take note of the indexing. Best for this to match the output
        [X, Y] = torch.meshgrid(x, y, indexing = 'ij')
        print(X.shape)
        X = X.reshape(self.num_nodes,1)
        Y = Y.reshape(self.num_nodes,1)

        # we need to linearize these matrices.
        self.X_for_queries = torch.concat([Y,X],dim=-1)
        print('Queries', self.X_for_queries.shape, 'Coordinates', X.shape)
        
        # SECTION 3: Transform to be MIOdataset Loader Compatible
        self.normalizer()
        self.__update_dataset_config()

    def subsampler(self):
        self.data_out = torch.nn.functional.avg_pool2d(torch.tensor(self.data_out).permute(0,3,1,2), self.sub_x).permute(0,2,3,1).numpy()

    def cell_to_vertex_converter(self):
        self.data_out = torch.nn.functional.avg_pool2d(torch.tensor(self.data_out).permute(0,3,1,2),2,stride=1).permute(0,2,3,1).numpy()
    
    def add_boundaries(self):
        self.data_out = torch.nn.functional.pad(torch.tensor(self.data_out).permute(0,3,1,2),(1, 1, 1, 1)).permute(0,2,3,1).numpy()

        # Lid Velocity
        self.data_out[:,-1 ,:,0] = 1

        # Pressure
        self.data_out[:,  0 ,1:-1, 2] = self.data_out[:,  1 ,1:-1, 2]  # Bottom Wall
        self.data_out[:, -1 ,1:-1, 2] = self.data_out[:, -2 ,1:-1, 2]  # Lid (y-vel)
        self.data_out[:,1:-1,  0 , 2] = self.data_out[:,1:-1,  1 , 2]  # Left Wall
        self.data_out[:,1:-1, -1 , 2] = self.data_out[:,1:-1, -2 , 2]  # Right Wall

    def normalizer(self):
        if self.normalize_y:
            self.__normalize_y()
        if self.normalize_x:
            self.__normalize_x()

        self.__update_dataset_config()
        

    def __normalize_y(self):
        if self.y_normalizer is None:
            if self.normalize_y == 'unit':
                self.y_normalizer = UnitTransformer(self.data_out)
                print('Target features are normalized using unit transformer')
            else: 
                raise NotImplementedError
        else:
            self.data_out = self.y_normalizer.transform(self.data_out, inverse=False)  # a torch quantile transformer
            print('Target features are normalized using unit transformer')

    def __normalize_x(self):
        if self.x_normalizer is None:
            if self.normalize_x == 'unit':
                self.x_normalizer = UnitTransformer(self.X_for_queries)
                self.up_normalizer = UnitTransformer(self.data_lid_v)
            else: 
                raise NotImplementedError

    def __update_dataset_config(self):
        
        if self.theta:
            branch_sizes = None
            theta_dim = 1
        else:
            branch_sizes = [1]
            theta_dim = 0

        self.config = {
            'input_dim': self.X_for_queries.shape[-1],
            'theta_dim': theta_dim,
            'output_dim': self.data_out.shape[-1],
            'branch_sizes': branch_sizes
        }

class UnitTransformer():
    def __init__(self, X):
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True) + 1e-8


    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def transform(self, X, inverse=True,component='all'):
        if component == 'all' or 'all-reduce':
            if inverse:
                orig_shape = X.shape
                return (X*(self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                return (X-self.mean)/self.std
        else:
            if inverse:
                orig_shape = X.shape
                return (X*(self.std[:,component] - 1e-8)+ self.mean[:,component]).view(orig_shape)
            else:
                return (X - self.mean[:,component])/self.std[:,component]

if __name__ == '__main__': 
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    parser = argparse.ArgumentParser(description='MLP PINN Training Study')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--path', type=str, default= r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\GNOT\data\steady_cavity_case_b200_maxU100ms_simple_normalized.npy')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--sub_x', type=int, default=4)
    parser.add_argument('--theta', type=bool, default=True)
    
    args = parser.parse_args()

    model_args = dict()
    dataset_args = dict()
    training_args = dict()

    # 1. Prepare Data
    dataset_args['file']                    = args.path
    dataset_args['percent split (decimal)'] = 0.7
    dataset_args['randomizer seed']         = 42
    dataset_args['Interpolate (instead of Extrapolate)'] = True
    dataset_args['use-normalizer']          = 'unit'
    dataset_args['normalize_x']             = 'unit'
    dataset_args['cell to pointwise']       = True
    dataset_args['add boundaries']          = True
    dataset_args['sub_x']                   = args.sub_x

    dataset = Cavity_2D_dataset_for_GNOT(data_path=dataset_args['file'], 
                                        L=1.0, 
                                        sub_x = dataset_args['sub_x'], 
                                        train=True, 
                                        normalize_y=dataset_args['use-normalizer'], 
                                        y_normalizer=None, 
                                        normalize_x = dataset_args['normalize_x'], 
                                        x_normalizer = None, 
                                        up_normalizer =None, 
                                        vertex = dataset_args['cell to pointwise'], 
                                        boundaries = dataset_args['add boundaries'])
    
    # Process dataset
    dataset.assign_data_split_type(inference=dataset_args['Interpolate (instead of Extrapolate)'], 
                                   train_ratio=dataset_args['percent split (decimal)'], 
                                   seed=dataset_args['randomizer seed'])
    dataset.process(theta=args.theta)

    

    # 2. Construct Model
    model_args['trunk_size']        = dataset.config['input_dim']
    model_args['theta_size']        = dataset.config['theta_dim']
    model_args['branch_sizes']      = dataset.config['branch_sizes'] 

    model_args['output_size']         = 3
    model_args['n_layers']            = 3
    model_args['n_hidden']            = 64 #128  
    model_args['n_head']              = 1
    model_args['attn_type']           = 'linear'
    model_args['ffn_dropout']         = 0.0
    model_args['attn_dropout']        = 0.0
    model_args['mlp_layers']          = 2
    model_args['act']                 = 'gelu'
    model_args['hfourier_dim']        = 0

    model = None
    model = CGPTNO(
                trunk_size          = model_args['trunk_size'] + model_args['theta_size'],
                branch_sizes        = model_args['branch_sizes'],     # No input function means no branches
                output_size         = model_args['output_size'],
                n_layers            = model_args['n_layers'],
                n_hidden            = model_args['n_hidden'],
                n_head              = model_args['n_head'],
                attn_type           = model_args['attn_type'],
                ffn_dropout         = model_args['ffn_dropout'],
                attn_dropout        = model_args['attn_dropout'],
                mlp_layers          = model_args['mlp_layers'],
                act                 = model_args['act'],
                horiz_fourier_dim   = model_args['hfourier_dim']
                ).to(device)
    
    # 3. Training Settings
    training_args['epochs']                 = args.epochs
    training_args["save_dir"]               = 'gnot_artemis'
    #training_args['eval_while_training']    = True
    #training_args['milestones']             = None
    training_args['base_lr']                = 0.001
    #training_args['lr_method']              = 'cycle'
    #training_args['scheduler_gamma']        = None  
    #training_args['xy_loss']                = 0.0
    #training_args['pde_loss']               = 1.0
    training_args['weight-decay']           = 0.00005
    training_args['grad-clip']              = 1000.0    
    #training_args['batchsize']              = 4
    #training_args['component']              = 'all'
    #training_args['normalizer']             = None
    training_args["save_name"]              = args.name

    loss_f = torch.nn.MSELoss(reduction='mean')
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
                                                    steps_per_epoch=dataset.data_out.shape[0], 
                                                    epochs=training_args['epochs']
                                                    )
    
    # Initialize Results Storage: 
    training_run_results = total_model_dict(model_config=model_args, training_config=training_args, data_config=dataset_args)

    # queries are universal for all batches in this case
    in_queries = dataset.X_for_queries.unsqueeze(0).float().to(device)#.requires_grad_(True)

    for epoch in range(training_args['epochs']):
        epoch_start_time = default_timer()

        # Set Model to Train
        model.train()
        torch.cuda.empty_cache()

        # Initialize loss storage per epoch
        loss_total_list             = list()
        
        batch_average_loss = 0
        for batch_n in range(dataset.data_out.shape[0]):
            
            print(f'Epoch: {epoch} Batch: {batch_n} Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f}GB Memory Cached: {torch.cuda.memory_reserved(device) / 1024**3:.2f}GB')
            optimizer.zero_grad()

            out_truth   = dataset.data_out[batch_n,...].clone().float().to(device)

            if args.theta:
                in_keys = dataset.data_lid_v[batch_n].clone().reshape(1,1).float().to(device)
                out = model(x=in_queries,u_p = in_keys)
            else:
                in_keys = dataset.data_lid_v[batch_n].clone().reshape(1,1,1).float().to(device)
                out = model(x=in_queries,inputs = in_keys)
                
            loss = loss_f(out.reshape(65,65,3),out_truth)
            batch_average_loss += loss

            # Sudo implementation of batch training of 4 samples
            if batch_n+1 % 4 == 0 or batch_n+1 == dataset.data_out.shape[0]: 
                batch_average_loss = batch_average_loss/(batch_n+1)
                batch_average_loss.backward()#(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(model.parameters(), training_args['grad-clip'])
                optimizer.step()
                scheduler.step()
                
                batch_average_loss = 0
            
            if training_args['epochs'] == 1: break

        epoch_end_time = default_timer()
        training_run_results.update_loss({'Epoch Time': epoch_end_time - epoch_start_time})
        training_run_results.update_loss({'Training L2 Loss': batch_average_loss.item()})

    if training_args['epochs'] != 1: 
        save_checkpoint(training_args["save_dir"], 
                        training_args["save_name"], 
                        model=model, 
                        loss_dict=training_run_results.dictionary, 
                        optimizer=optimizer)