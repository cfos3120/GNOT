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
        self.data_lid_v = np.round(np.arange(0.5,100.5,0.5),1)# * 0.1/0.01 #<- apply for Reynolds Number
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
        
        # Final Data:
        self.queries = self.X_for_queries
        #self.theta = torch.zeros([self.n_batches])
        self.input_f = self.data_lid_v
        self.output_truth = self.data_out.reshape(self.n_batches, self.num_nodes,3)
        
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
            #y_feats_all = torch.cat([g.ndata['y'] for g in self.graphs],dim=0)
            y_feats_all = self.output_truth.reshape(self.n_batches * self.num_nodes,3)
            if self.normalize_y == 'unit':
                self.y_normalizer = UnitTransformer(y_feats_all)
                print('Target features are normalized using unit transformer')
                print(self.y_normalizer.mean, self.y_normalizer.std)
            else: 
                raise NotImplementedError
        
        self.output_truth = self.y_normalizer.transform(self.output_truth, inverse=False)
        #for g in self.graphs:
        #    g.ndata['y'] = self.y_normalizer.transform(g.ndata["y"], inverse=False)  # a torch quantile transformer
        print('Target features are normalized using unit transformer')

    def __normalize_x(self):
        if self.x_normalizer is None:
            # X features are the same for all cases (same grid coords)
            #x_feats_all = torch.cat([g.ndata["x"] for g in self.graphs],dim=0)
            x_feats_all = self.queries
            if self.normalize_x == 'unit':
                self.x_normalizer = UnitTransformer(x_feats_all)
                #self.up_normalizer = UnitTransformer(self.theta)
            else: 
                raise NotImplementedError

        #for g in self.graphs:
        #    g.ndata['x'] = self.x_normalizer.transform(g.ndata['x'], inverse=False)
        self.queries = self.x_normalizer.transform(self.queries, inverse=False)
        #self.u_p_list = self.up_normalizer.transform(self.theta, inverse=False)
        print('Input features are normalized using unit transformer')

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

class CavityDataset(Dataset):
    def __init__(self,dataset,theta=True):
        self.theta = theta
        self.in_queries = dataset.queries
        self.in_keys_all = dataset.input_f
        self.out_truth_all = dataset.output_truth

    def __len__(self):
        return len(self.in_keys_all)

    def __getitem__(self, idx):
        in_queries  = self.in_queries.float()
        in_keys     = self.in_keys_all[idx].float()
        out_truth   = self.out_truth_all[idx,...].float()
        
        if self.theta:
            in_keys = in_keys.reshape(1)
        else:
            in_keys = in_keys.reshape(1,1)

        return in_queries, in_keys, out_truth
    
class custom_l2_loss(object):
    def __init__(self):
        super(custom_l2_loss, self).__init__()
        self.p = 2
    
    def __call__(self, x, y):
        batches = x.shape[0]
        num_nodes = x.shape[1]
        #losses = ((1/num_nodes)*(pred - target).abs() ** p)) ** (1 / p)

        return torch.mean(torch.norm(x.reshape(batches,-1) - y.reshape(batches,-1), self.p, 1))

class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
       
if __name__ == '__main__': 
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    parser = argparse.ArgumentParser(description='MLP PINN Training Study')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--path', type=str, default= r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\GNOT\data\steady_cavity_case_b200_maxU100ms_simple_normalized.npy')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--sub_x', type=int, default=4)
    parser.add_argument('--theta', type=bool, default=False)
    
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
    print(dataset.config)
    
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

    net = None
    net = CGPTNO(
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
    
    # parallize
    model = torch.nn.DataParallel(net)
    
    # 3. Training Settings
    training_args['epochs']                 = args.epochs
    training_args["save_dir"]               = 'gnot_artemis'
    training_args['base_lr']                = 0.001
    training_args['weight-decay']           = 0.00005
    training_args['grad-clip']              = 1000.0    
    training_args['batchsize']              = 4
    training_args["save_name"]              = args.name
    
    
    dataset = CavityDataset(dataset=dataset)
    train_dataloader = DataLoader(dataset, batch_size=training_args['batchsize'], shuffle=True)  

    #loss_f = torch.nn.MSELoss(reduction='sum')
    #loss_f = custom_l2_loss()
    loss_f = LpLoss()
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
                                                    steps_per_epoch=len(train_dataloader), 
                                                    epochs=training_args['epochs']
                                                    )
    
    # Initialize Results Storage: 
    training_run_results = total_model_dict(model_config=model_args, training_config=training_args, data_config=dataset_args)

    for epoch in range(training_args['epochs']):
        epoch_start_time = default_timer()

        # Set Model to Train
        model.train()
        torch.cuda.empty_cache()

        # Initialize loss storage per epoch
        loss_total_list = list()
        
        for batch_n, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            in_queries, in_keys, out_truth = batch
            in_queries, in_keys, out_truth = in_queries.to(device), in_keys.to(device), out_truth.to(device)

            if args.theta:
                out = model(x=in_queries,u_p = in_keys)
            else:
                out = model(x=in_queries,inputs = in_keys)
                
            loss = loss_f(out,out_truth)

            loss.backward()#(retain_graph=True)
        
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_args['grad-clip'])
            optimizer.step()
            scheduler.step()
            
            if training_args['epochs'] == 10: break

        print(f'Epoch: {epoch :8} Batch: {batch_n :3} L2 Loss: {loss :12.7f}, Memory Allocated: GPU1 {torch.cuda.memory_allocated(torch.device("cuda:0")) / 1024**3:5.2f}GB ' + 
                  f'GPU2 {torch.cuda.memory_allocated(torch.device("cuda:1")) / 1024**3:5.2f}GB ' +
                  f'Memory Cached: GPU1 {torch.cuda.memory_reserved(torch.device("cuda:0")) / 1024**3:5.2f}GB ' +
                  f'GPU2 {torch.cuda.memory_reserved(torch.device("cuda:1")) / 1024**3:5.2f}GB '
                  )
        
        epoch_end_time = default_timer()
        training_run_results.update_loss({'Epoch Time': epoch_end_time - epoch_start_time})
        training_run_results.update_loss({'Training L2 Loss': loss.item()})

    if training_args['epochs'] != 1: 
        save_checkpoint(training_args["save_dir"], 
                        training_args["save_name"], 
                        model=model, 
                        loss_dict=training_run_results.dictionary, 
                        optimizer=optimizer)