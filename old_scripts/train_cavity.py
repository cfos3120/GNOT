import torch
import torch.nn.functional as F
import numpy as np
import sys
from dgl.data import DGLDataset
import dgl
from timeit import default_timer

from data_utils import MultipleTensors, MIODataLoader, get_dataset
from models.cgpt import CGPTNO
from data_utils import WeightedLpRelLoss, WeightedLpLoss

sys.path.append(r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\Lasso')
from training_utils.optimizers import Adam
from training_utils.loss_functions_2 import LpLoss, loss_selector
from data_handling.data_utils import load_dataset, sample_data, total_loss_list
from training_utils.save_checkpoint import save_checkpoint
#from training_utils.train_cavity import Navier_Stokes_FDM_cavity_internal

import numpy as np
import torch
from dgl.data import DGLDataset
import dgl
import sys
sys.path.append(r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\GNOT')
from data_utils import MultipleTensors
from utils import UnitTransformer

def Navier_Stokes_FDM_cavity_internal(U, lid_velocity, nu=0.01, L=0.1, order=2):

    batchsize = U.size(0)
    nx = U.size(1)
    ny = U.size(2)
    
    device = U.device

    # create isotropic grid
    y = torch.tensor(np.linspace(0, L, nx+1)[:-1], dtype=torch.float, device=device)
    x = y

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
    
    # scheme selector
    if order == 2:
        
        u = torch.zeros([batchsize, nx+2, ny+2])
        v = torch.zeros_like(u)
        p = torch.zeros_like(u)
        
        # assign internal field
        u[:,1:-1,1:-1] = U[...,0]
        v[:,1:-1,1:-1] = U[...,1]
        p[:,1:-1,1:-1] = U[...,2]

        # hard set boundaries
        u[:,  0 ,1:-1], v[:,  0 ,1:-1] = -u[:,  1 ,1:-1], -v[:,  1 ,1:-1]   # Bottom Wall
        v[:, -1 ,1:-1] = -v[:, -2 ,1:-1]                                    # Lid (y-vel)
        u[:,1:-1,  0 ], v[:,1:-1,  0 ] = -u[:,1:-1,  1 ], -v[:,1:-1,  1 ]   # Left Wall
        u[:,1:-1, -1 ], v[:,1:-1, -1 ] = -u[:,1:-1, -2 ], -v[:,1:-1, -2 ]   # Right Wall
        
        u[:, -1 ,1:-1] = 2*lid_velocity.repeat(1,ny) - u[:, -2 ,1:-1]   # Lid (x-vel)

        # Pressure Boundaries
        p[:,  0 ,1:-1] = p[:,  1 ,1:-1]  # Bottom Wall
        p[:, -1 ,1:-1] = p[:, -2 ,1:-1]  # Lid (y-vel)
        p[:,1:-1,  0 ] = p[:,1:-1,  1 ]  # Left Wall
        p[:,1:-1, -1 ] = p[:,1:-1, -2 ]  # Right Wall


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

    elif order == 4:
        
        u = torch.zeros([batchsize, nx+4, ny+4])
        v = torch.zeros_like(u)
        p = torch.zeros_like(u)

        # assign internal field
        u[:,2:-2,2:-2] = U[...,0]
        v[:,2:-2,2:-2] = U[...,1]
        p[:,2:-2,2:-2] = U[...,2]

        # hard set boundaries
        u[:,  1 ,2:-2], v[:,  1 ,2:-2] = -u[:,  2 ,2:-2], -v[:,  2 ,2:-2]   # Bottom Wall (First Ghost Nodes)
        u[:,  0 ,2:-2], v[:,  0 ,2:-2] =  u[:,  1 ,2:-2], v[:,  1 ,2:-2]    # Bottom Wall (Second Ghost Nodes)
        v[:, -2 ,2:-2] = -v[:, -3 ,2:-2]                                    # Lid (y-vel) (First Ghost Nodes)
        v[:, -1 ,2:-2] =  v[:, -2 ,2:-2]                                    # Lid (y-vel) (Second Ghost Nodes)
        u[:,2:-2,  1 ], v[:,2:-2,  1 ] = -u[:,2:-2,  2 ], -v[:,2:-2,  2 ]   # Left Wall (First Ghost Nodes)
        u[:,2:-2,  0 ], v[:,2:-2,  0 ] =  u[:,2:-2,  1 ], v[:,2:-2,  1 ]    # Left Wall (Second Ghost Nodes)
        u[:,2:-2, -2 ], v[:,2:-2, -2 ] = -u[:,2:-2, -3 ], -v[:,2:-2, -3 ]   # Right Wall (First Ghost Nodes)
        u[:,2:-2, -1 ], v[:,2:-2, -1 ] =  u[:,2:-2, -2 ], v[:,2:-2, -2 ]    # Right Wall (Second Ghost Nodes)
        u[:, -2 ,2:-2] = 2*lid_velocity.repeat(1,ny) - u[:, -3 ,2:-2]       # Lid (x-vel) (First Ghost Nodes)
        u[:, -1 ,2:-2] = u[:, -2 ,2:-2]                                     # Lid (x-vel) (Second Ghost Nodes)
        
        # Pressure Boundaries
        p[:,  0 ,2:-2], p[:,  1 ,2:-2] = p[:,  2 ,2:-2], p[:,  2 ,2:-2] # Bottom Wall (Both Ghost Nodes)
        p[:, -1 ,2:-2], p[:, -2 ,2:-2] = p[:, -3 ,2:-2], p[:, -3 ,2:-2] # Lid (y-vel) (Both Ghost Nodes)
        p[:,2:-2,  0 ], p[:,2:-2,  1 ] = p[:,2:-2,  2 ], p[:,2:-2,  2 ] # Left Wall   (Both Ghost Nodes)
        p[:,2:-2, -1 ], p[:,2:-2, -2 ] = p[:,2:-2, -3 ], p[:,2:-2, -3 ] # Right Wall  (Both Ghost Nodes)

        # gradients in internal zone
        uy  = (-u[:, 4:  , 2:-2 ] +  8*u[:, 3:-1, 2:-2 ] -  8*u[:, 1:-3, 2:-2 ] +    u[:,  :-4, 2:-2 ]) / (12*dy)
        ux  = (-u[:, 2:-2, 4:   ] +  8*u[:, 2:-2, 3:-1 ] -  8*u[:, 2:-2, 1:-3 ] +    u[:, 2:-2,  :-4 ]) / (12*dx)
        uyy = (-u[:, 4:  , 2:-2 ] + 16*u[:, 3:-1, 2:-2 ] - 30*u[:, 2:-2, 2:-2 ] + 16*u[:, 1:-3, 2:-2 ] - u[:,  :-4, 2:-2 ]) / (12*dy**2)
        uxx = (-u[:, 2:-2, 4:   ] + 16*u[:, 2:-2, 3:-1 ] - 30*u[:, 2:-2, 2:-2 ] + 16*u[:, 2:-2, 1:-3 ] - u[:, 2:-2,  :-4 ]) / (12*dx**2)
        
        vy  = (-v[:, 4:  , 2:-2 ] +  8*v[:, 3:-1, 2:-2 ] -  8*v[:, 1:-3, 2:-2 ] +    v[:,  :-4, 2:-2 ]) / (12*dy)
        vx  = (-v[:, 2:-2, 4:   ] +  8*v[:, 2:-2, 3:-1 ] -  8*v[:, 2:-2, 1:-3 ] +    v[:, 2:-2,  :-4 ]) / (12*dx)
        vyy = (-v[:, 4:  , 2:-2 ] + 16*v[:, 3:-1, 2:-2 ] - 30*v[:, 2:-2, 2:-2 ] + 16*v[:, 1:-3, 2:-2 ] - v[:,  :-4, 2:-2 ]) / (12*dy**2)
        vxx = (-v[:, 2:-2, 4:   ] + 16*v[:, 2:-2, 3:-1 ] - 30*v[:, 2:-2, 2:-2 ] + 16*v[:, 2:-2, 1:-3 ] - v[:, 2:-2,  :-4 ]) / (12*dx**2)

        py  = (-p[:, 4:  , 2:-2 ] +  8*p[:, 3:-1, 2:-2 ] -  8*p[:, 1:-3, 2:-2 ] +    p[:,  :-4, 2:-2 ]) / (12*dy)
        px  = (-p[:, 2:-2, 4:   ] +  8*p[:, 2:-2, 3:-1 ] -  8*p[:, 2:-2, 1:-3 ] +    p[:, 2:-2,  :-4 ]) / (12*dx) # / (rho*U^2) <- for non-dimensional

    else: raise(NameError)

    # No time derivative as we are assuming steady state solution
    Du_dx = U[...,0]*ux + U[...,1]*uy - nu * (uxx + uyy) + px
    Dv_dy = U[...,0]*vx + U[...,1]*vy - nu * (vxx + vyy) + py
    continuity_eq = (ux + vy)

    # loss_mx = F.mse_loss(Du_dx, torch.zeros_like(Du_dx))
    # loss_my = F.mse_loss(Dv_dy, torch.zeros_like(Dv_dy))
    # loss_c  = F.mse_loss(continuity_eq, torch.zeros_like(continuity_eq))

    fdm_derivatives = tuple([ux, uy, vx, vy, px, py, uxx, uyy, vxx, vyy, Du_dx, Dv_dy, continuity_eq])
    

    #total_pde_loss = ((loss_mx*1.0) + (loss_my*1.0) + (loss_c*1.0))/3
    return Du_dx, Dv_dy, continuity_eq, fdm_derivatives

class Cavity_2D_dataset_handling_v1(DGLDataset):
    def __init__(self, data_path, L=0.1, name=' ', train=True, normalize_y=False, y_normalizer=None, normalize_x = False, x_normalizer = None, up_normalizer =None):

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

        # Load in Dataset and retrieve shape
        cavity_data     = np.load(data_path)
        self.data_out   = cavity_data[:, 1:-1, 1:-1, 1,:]
        self.data_lid_v = cavity_data[:,0,0,0,0]
        self.n_batches  = self.data_out.shape[0]
        self.nx         = self.data_out.shape[1]
        self.num_nodes  = self.nx**2

        self.L = L
        self.train = train

        super(Cavity_2D_dataset_handling_v1, self).__init__(name)   # invoke super method after read data


    def process(self):
        
        # SECTION 0: Split into train or test (Same as for FNO training)
        train_size = int(0.7 * self.n_batches)
        test_size = self.n_batches - train_size

        train_dataset, test_dataset = torch.split(torch.from_numpy(self.data_out), [train_size, test_size])
        train_lid_v, test_lid_v     = torch.split(torch.from_numpy(self.data_lid_v), [train_size, test_size])
        if self.train:
            self.data_out = train_dataset
            self.data_lid_v = train_lid_v
            self.n_batches = train_size
        else:
            self.data_out = test_dataset
            self.data_lid_v = test_lid_v
            self.n_batches = test_size
        
        # SECTION 1: Transformer Queries
        # Assume Isotropic Grid 
        x = torch.arange(self.nx)*self.L/self.nx
        y = x

        # take note of the indexing. Best for this to match the output
        [X, Y] = torch.meshgrid(x, y, indexing = 'ij')

        X = X.reshape(self.num_nodes,1)
        Y = Y.reshape(self.num_nodes,1)

        # we need to linearize these matrices.
        self.X_for_queries = torch.concat([Y,X],dim=-1)
        print('Queries', self.X_for_queries.shape, 'Coordinates', X.shape)
        
        # SECTION 3: Transform to be MIOdataset Loader Compatible
        self.MIOdataset_structure()
        self.__update_dataset_config()


    def MIOdataset_structure(self):
        '''
        NOTE here for FNO, X is constant because the grid is the same for all batches
        Theta is [0] because it requires a non-null input in the current model config
        Output (Y) and initial conditions (g_u) change per batch)
        g_u is the input condition in tuple structure as the model can accept multiple input functions.
        '''

        # Query Coordinates are the same so we keep these constant:
        x = self.X_for_queries
        u_p = torch.tensor([0])

        # initialize storage
        self.graphs = []
        self.inputs_f = []
        self.u_p_list = []

        for i in range(self.n_batches):

            y = self.data_out[i,...].reshape(self.num_nodes,3)
            input_f = self.data_lid_v[[i]].float()

            # Construct Graph object
            g = dgl.DGLGraph()
            g.add_nodes(x.shape[0])
            g.ndata['x'] = x.float()
            g.ndata['y'] = y.float()
            up = u_p.float()
            self.graphs.append(g)
            self.u_p_list.append(up) # global input parameters
            if input_f is not None:
                input_f = MultipleTensors([torch.tensor([f]).unsqueeze(-1) for f in input_f])
                self.inputs_f.append(input_f)
                self.num_inputs = len(input_f)
            
            #if len(inputs_f) == 0:
            #    inputs_f = torch.zeros([self.n_batches])  # pad values, tensor of 0, not list

        self.u_p_list = torch.stack(self.u_p_list)

        if self.normalize_y:
            self.__normalize_y()
        if self.normalize_x:
            self.__normalize_x()

        self.__update_dataset_config()

    def __normalize_y(self):
        if self.y_normalizer is None:
            y_feats_all = torch.cat([g.ndata['y'] for g in self.graphs],dim=0)
            if self.normalize_y == 'unit':
                self.y_normalizer = UnitTransformer(y_feats_all)
                print('Target features are normalized using unit transformer')
                print(self.y_normalizer.mean, self.y_normalizer.std)
            else: 
                raise NotImplementedError


        for g in self.graphs:
            g.ndata['y'] = self.y_normalizer.transform(g.ndata["y"], inverse=False)  # a torch quantile transformer
        print('Target features are normalized using unit transformer')

    def __normalize_x(self):
        if self.x_normalizer is None:
            x_feats_all = torch.cat([g.ndata["x"] for g in self.graphs],dim=0)
            if self.normalize_x == 'unit':
                self.x_normalizer = UnitTransformer(x_feats_all)
                self.up_normalizer = UnitTransformer(self.u_p_list)
            else: 
                raise NotImplementedError

        for g in self.graphs:
            g.ndata['x'] = self.x_normalizer.transform(g.ndata['x'], inverse=False)
        self.u_p_list = self.up_normalizer.transform(self.u_p_list, inverse=False)
        print('Input features are normalized using unit transformer')

    def __update_dataset_config(self):
        self.config = {
            'input_dim': self.graphs[0].ndata['x'].shape[1],
            'theta_dim': self.u_p_list.shape[1],
            'output_dim': self.graphs[0].ndata['y'].shape[1],
            'branch_sizes': [x.shape[1] for x in self.inputs_f[0]] if isinstance(self.inputs_f, list) else 0
        }

    def __getitem__(self, idx):
        return self.graphs[idx], self.u_p_list[idx], self.inputs_f[idx]
    
    def __len__(self):
        return self.n_batches
    
class Cavity_2D_dataset_handling_v0(DGLDataset):
    def __init__(self, data_path, L=0.1, name=' ', train=True):

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

        # Load in Dataset and retrieve shape
        cavity_data     = np.load(data_path)
        self.data_out   = cavity_data[:, 1:-1, 1:-1, 1,:]
        self.data_lid_v = cavity_data[:,0,0,0,0]
        self.n_batches  = self.data_out.shape[0]
        self.nx         = self.data_out.shape[1]
        self.num_nodes  = self.nx**2

        self.L = L
        self.train = train

        super(Cavity_2D_dataset_handling_v0, self).__init__(name)   # invoke super method after read data


    def process(self):
        
        # SECTION 0: Split into train or test (Same as for FNO training)
        train_size = int(0.7 * self.n_batches)
        test_size = self.n_batches - train_size

        train_dataset, test_dataset = torch.split(torch.from_numpy(self.data_out), [train_size, test_size])
        train_lid_v, test_lid_v     = torch.split(torch.from_numpy(self.data_lid_v), [train_size, test_size])
        if self.train:
            self.data_out = train_dataset
            self.data_lid_v = train_lid_v
            self.n_batches = train_size
        else:
            self.data_out = test_dataset
            self.data_lid_v = test_lid_v
            self.n_batches = test_size
        
        # SECTION 1: Transformer Queries
        # Assume Isotropic Grid 
        x = torch.arange(self.nx)*self.L/self.nx
        y = x

        # take note of the indexing. Best for this to match the output
        [X, Y] = torch.meshgrid(x, y, indexing = 'ij')

        X = X.reshape(self.num_nodes,1)
        Y = Y.reshape(self.num_nodes,1)

        # we need to linearize these matrices.
        self.X_for_queries = torch.concat([Y,X],dim=-1)
        print('Queries', self.X_for_queries.shape, 'Coordinates', X.shape)
        
        # SECTION 3: Transform to be MIOdataset Loader Compatible
        self.MIOdataset_structure()
        self.__update_dataset_config()


    def MIOdataset_structure(self):
        '''
        NOTE here for FNO, X is constant because the grid is the same for all batches
        Theta is [0] because it requires a non-null input in the current model config
        Output (Y) and initial conditions (g_u) change per batch)
        g_u is the input condition in tuple structure as the model can accept multiple input functions.
        '''

        # Query Coordinates are the same so we keep these constant:
        x = self.X_for_queries
        input_f = None
        
        # initialize storage
        self.graphs = []
        self.inputs_f = []
        self.u_p_list = []

        for i in range(self.n_batches):

            y = self.data_out[i,...].reshape(self.num_nodes,3)
            u_p = self.data_lid_v[[i]]

            # Construct Graph object
            g = dgl.DGLGraph()
            g.add_nodes(x.shape[0])
            g.ndata['x'] = x.float()
            g.ndata['y'] = y.float()
            up = u_p.float()
            self.graphs.append(g)
            self.u_p_list.append(up) # global input parameters
            if input_f is not None:
                input_f = MultipleTensors([torch.from_numpy(f).float() for f in input_f])
                self.inputs_f.append(input_f)
                num_inputs = len(input_f)
            
            #if len(inputs_f) == 0:
            #    inputs_f = torch.zeros([self.n_batches])  # pad values, tensor of 0, not list

        self.u_p_list = torch.stack(self.u_p_list)

    def __update_dataset_config(self):
        self.config = {
            'input_dim': self.graphs[0].ndata['x'].shape[1],
            'theta_dim': self.u_p_list.shape[1],
            'output_dim': self.graphs[0].ndata['y'].shape[1],
            'branch_sizes': 0 #[x.shape[1] for x in self.inputs_f[0]] if isinstance(self.inputs_f, list) else 0
        }

    def __getitem__(self, idx):
        return self.graphs[idx], self.u_p_list[idx]#, self.inputs_f[idx]
    
    def __len__(self):
        return self.n_batches

def model_routine_cavity(args, model, training_dataset, testing_dataset=None):
    
    # 0. Print training configurations
    device = next(model.parameters()).device

    # 1. Data and training parameters
    training_dataloader = MIODataLoader(training_dataset, batch_size=args['batchsize'], shuffle=True, drop_last=False)
    if args['eval_while_training']:
        testing_dataloader = MIODataLoader(testing_dataset, batch_size=args['batchsize'], shuffle=False, drop_last=False)
    
    xy_weight = args['xy_loss']
    pde_weight = args['pde_loss']
    epochs = args['epochs']

    # 2. Training optimizer and learning rate scheduler
    if 'base_lr' in args:
        #optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=args['base_lr'])
        optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=args['base_lr'], weight_decay=args['weight-decay'])

    if args['lr_method'] == 'cycle':
        print('Using cycle learning rate schedule')
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args['base_lr'], div_factor=1e4, pct_start=0.2, final_div_factor=1e4, steps_per_epoch=len(training_dataloader), epochs=epochs)
    elif args['lr_method'] == 'step':
        print('Using step learning rate schedule')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['lr_step_size']*len(training_dataloader), gamma=0.7)
    elif args['lr_method'] == 'warmup':
        print('Using warmup learning rate schedule')
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1)/(args['warmup_epochs'] * len(training_dataloader)), np.power(args['warmup_epochs'] * len(training_dataloader)/float(steps + 1), 0.5)))
    elif args['lr_method'] == 'milestones':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['milestones'], gamma=args['scheduler_gamma'])
    
    # 3.1. Initialize Loss Function (from FNO) and Recording Dictionary
    loss_function_FNO = loss_selector(args['fno_loss_type'])
    total_loss_dictionary = total_loss_list()
    epoch_end_time = 0

    # 3.2. Initialize Loss Function (From GNOT) for backwards Pass
    #loss_func = WeightedLpRelLoss(p=2,component=args['component'], normalizer=args['normalizer'])
    loss_func = WeightedLpLoss(p=2,component=args['component'], normalizer=args['normalizer'])
    # 4. Run Model Routine (Training)  
    model.train()

    print(f'Commencing training with {len(training_dataloader)} batches of size {args["batchsize"]}')

    for epoch in range(epochs): 
        epoch_start_time = default_timer()

        # Set Model to Train
        model.train()
        torch.cuda.empty_cache()

        # 5. Train all batches

        for batch_n, batch in enumerate(training_dataloader):
            batch_start_time = default_timer()
            optimizer.zero_grad()

            g, u_p, g_u = batch
            g, u_p, g_u = g.to(device), u_p.to(device), g_u.to(device)

            # 5.1. Forward Pass in Model
            out = model(g, u_p, g_u)
            y_pred, y = out.squeeze(), g.ndata['y'].squeeze()
            
            # 5.2. Calculate Loss for Backwards Pass
            loss, reg,  _ = loss_func(g, y_pred, y)
            loss_total = loss + reg
            
            # 5.3. Calculate Monitoring Losses (reshape too)
            y_pred  = y_pred.reshape(len(u_p), training_dataset.nx, training_dataset.nx, 3)
            y       = y.reshape(len(u_p), training_dataset.nx, training_dataset.nx, 3)
            
            loss_l2 = loss_function_FNO(y_pred, y)
            loss_mse = F.mse_loss(y_pred, y)

            Du_dx, Dv_dy, continuity_eq,__ = Navier_Stokes_FDM_cavity_internal(training_dataset.y_normalizer.transform(y_pred,inverse=True), g_u[0].squeeze(-1), nu=0.01)
            Du_dx, Dv_dy, continuity_eq = Du_dx.reshape(len(u_p)*training_dataset.nx**2, 1), Dv_dy.reshape(len(u_p)*training_dataset.nx**2, 1), continuity_eq.reshape(len(u_p)*training_dataset.nx**2, 1)
            pde_loss1, pde_reg1,_ = loss_func(g, Du_dx, torch.zeros_like(Du_dx))
            pde_loss2, pde_reg2,_ = loss_func(g, Dv_dy, torch.zeros_like(Dv_dy))
            pde_loss3, pde_reg3,_ = loss_func(g, continuity_eq, torch.zeros_like(continuity_eq))
            total_avg_pde_loss = (pde_loss1 + pde_loss2 + pde_loss3)/3
            #total_avg_pde_loss = max([pde_loss1, pde_loss2, pde_loss3])

            total_weighted_loss = xy_weight*loss_total + pde_weight*total_avg_pde_loss

            # 5.4. Backwards Pass to Optimizing and Scheduling
            total_weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['grad-clip'])
            optimizer.step()

            scheduler.step()
        
            # Print to Console:
            batch_end_time = default_timer()
            if (epoch == 0 and batch_n == 0) or (epoch == epochs-1 and batch_n ==  len(training_dataloader)-1): print_end = '\n'
            else: print_end = '\r'
            print(f'Epoch: {epoch}, Batch: {batch_n + 1}, Last LP Training Loss: {loss_total.item()}, Estimated Time Left {(batch_end_time - batch_start_time)*(epochs-epoch+1)*(len(training_dataloader)-batch_n+1)/3600:.2f}hrs', end=print_end)
            
            if epochs == 1 and batch_n == 1: break # debug control

        epoch_end_time = default_timer()

        # 5.5. Store Losses to Dictionary
        total_loss_dictionary.update({'Epoch Time': epoch_end_time - epoch_start_time})
        total_loss_dictionary.update({'Total Weighted Loss': total_weighted_loss.item()})
        total_loss_dictionary.update({'GNOT Loss Type': loss.item()})
        total_loss_dictionary.update({'GNOT Loss Type Regression': reg.item()})

        total_loss_dictionary.update({'FNO Loss Type (LP Loss)': loss_l2.item()})
        total_loss_dictionary.update({'FNO Loss Type (MSE Loss)': loss_mse.item()})
        total_loss_dictionary.update({'Internal Weighted PDE Loss': total_avg_pde_loss.item()})
        total_loss_dictionary.update({'Training X-Momentum': pde_loss1.item()})
        total_loss_dictionary.update({'Training Y-Momentum': pde_loss2.item()})
        total_loss_dictionary.update({'Training Continuity': pde_loss3.item()})

        # 6. Validate Model
        if args['eval_while_training']:
        
            model.eval()
            
            for batch_n, batch in enumerate(testing_dataloader):
                
                g_eval, u_p_eval, g_u_eval = batch
                g_eval, u_p_eval, g_u_eval = g_eval.to(device), u_p_eval.to(device), g_u_eval.to(device)

                # 5.1. Forward Pass in Model
                out_eval = model(g_eval, u_p_eval, g_u_eval)
                y_pred_eval, y_eval = out_eval.squeeze(), g_eval.ndata['y'].squeeze()
                
                # 5.2. Calculate Loss for Backwards Pass
                loss, reg,  _ = loss_func(g_eval, y_pred_eval, y_eval)
                loss_total = loss + reg
                
                # 5.2. Calculate FNO Loss for Monitoring
                y_pred_eval  = y_pred_eval.reshape(len(u_p_eval), testing_dataset.nx, testing_dataset.nx, 3)
                y_eval       = y_eval.reshape(len(u_p_eval), testing_dataset.nx, testing_dataset.nx, 3)
                loss_l2 = loss_function_FNO(y_pred_eval, y_eval)
                
            # 6.8 Calculate the Average Loss and Store to Dictionary
            loss_total = loss_total/(batch_n+1)
            loss_total_fno = loss_l2/(batch_n+1)
            total_loss_dictionary.update({'Average Validation Loss (GNOT Type)': loss_total.item()})
            total_loss_dictionary.update({'Average Validation Loss (FNO Type)': loss_total_fno.item()})
        
        #______________________________________________________________________________________________________________________

    # 7. Save Model Checkpoint and Losses
    save_checkpoint(args["save_dir"], args["save_name"], model, 
                loss_dict=total_loss_dictionary.fetch_list(), optimizer=optimizer,
                input_sample=u_p, output_sample=y_pred, ground_truth_sample=y)
    
    # 8. Save Evaluation
    if args['eval_while_training']:
        save_checkpoint(args["save_dir"], args["save_name"]+'eval',
                        input_sample=u_p_eval, output_sample=y_pred_eval, ground_truth_sample=y_eval)
    
    print('Model Routine Complete with final training Loss: ', loss_l2.item())
    return

def model_routine_cavity_fine_one_case(args, model, training_dataset, testing_dataset=None):
    
    # 0. Print training configurations
    device = next(model.parameters()).device

    # 1. Data and training parameters
    training_dataloader = MIODataLoader(training_dataset, batch_size=args['batchsize'], shuffle=False, drop_last=False)
    
    xy_weight = args['xy_loss']
    pde_weight = args['pde_loss']
    epochs = args['epochs']

    # 2. Training optimizer and learning rate scheduler
    if 'base_lr' in args:
        #optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=args['base_lr'])
        optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=args['base_lr'], weight_decay=args['weight-decay'])

    if args['lr_method'] == 'cycle':
        print('Using cycle learning rate schedule')
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args['base_lr'], div_factor=1e4, pct_start=0.2, final_div_factor=1e4, steps_per_epoch=len(training_dataloader), epochs=epochs)
    elif args['lr_method'] == 'step':
        print('Using step learning rate schedule')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['lr_step_size']*len(training_dataloader), gamma=0.7)
    elif args['lr_method'] == 'warmup':
        print('Using warmup learning rate schedule')
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1)/(args['warmup_epochs'] * len(training_dataloader)), np.power(args['warmup_epochs'] * len(training_dataloader)/float(steps + 1), 0.5)))
    elif args['lr_method'] == 'milestones':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['milestones'], gamma=args['scheduler_gamma'])
    
    # 3.1. Initialize Loss Function (from FNO) and Recording Dictionary
    loss_function_FNO = loss_selector(args['fno_loss_type'])
    total_loss_dictionary = total_loss_list()
    epoch_end_time = 0

    # 3.2. Initialize Loss Function (From GNOT) for backwards Pass
    #loss_func = WeightedLpRelLoss(p=2,component=args['component'], normalizer=args['normalizer'])
    loss_func = WeightedLpLoss(p=2,component=args['component'], normalizer=args['normalizer'])
    # 4. Run Model Routine (Training)  
    model.train()

    print(f'Commencing fine-tuning with {1} batches of size {args["batchsize"]}')

    for epoch in range(epochs): 
        epoch_start_time = default_timer()

        # Set Model to Train
        model.train()
        torch.cuda.empty_cache()

        # 5. Train all batches

        for batch_n, batch in enumerate(training_dataloader):
            batch_start_time = default_timer()
            optimizer.zero_grad()

            g, u_p, g_u = batch
            g, u_p, g_u = g.to(device), u_p.to(device), g_u.to(device)

            # 5.1. Forward Pass in Model
            out = model(g, u_p, g_u)
            y_pred, y = out.squeeze(), g.ndata['y'].squeeze()
            
            # 5.2. Calculate Loss for Backwards Pass
            loss, reg,  _ = loss_func(g, y_pred, y)
            loss_total = loss + reg
            
            # 5.3. Calculate Monitoring Losses (reshape too)
            y_pred  = y_pred.reshape(len(u_p), training_dataset.nx, training_dataset.nx, 3)
            y       = y.reshape(len(u_p), training_dataset.nx, training_dataset.nx, 3)
            
            loss_l2 = loss_function_FNO(y_pred, y)
            loss_mse = F.mse_loss(y_pred, y)

            Du_dx, Dv_dy, continuity_eq,__ = Navier_Stokes_FDM_cavity_internal(training_dataset.y_normalizer.transform(y_pred,inverse=True), g_u[0].squeeze(-1), nu=0.01, order=4)
            Du_dx, Dv_dy, continuity_eq = Du_dx.reshape(len(u_p)*training_dataset.nx**2, 1), Dv_dy.reshape(len(u_p)*training_dataset.nx**2, 1), continuity_eq.reshape(len(u_p)*training_dataset.nx**2, 1)
            pde_loss1, pde_reg1,_ = loss_func(g, Du_dx, torch.zeros_like(Du_dx))
            pde_loss2, pde_reg2,_ = loss_func(g, Dv_dy, torch.zeros_like(Dv_dy))
            pde_loss3, pde_reg3,_ = loss_func(g, continuity_eq, torch.zeros_like(continuity_eq))
            #total_avg_pde_loss = (pde_loss1 + pde_loss2 + pde_loss3)/3
            #total_avg_pde_loss = max([pde_loss1, pde_loss2, pde_loss3])
            total_avg_pde_loss = 0.01*(pde_loss1 + pde_loss2) + pde_loss3
            #total_avg_pde_loss = np.log(np.exp(pde_loss1) + np.exp(pde_loss2) + np.exp(pde_loss3))
            total_weighted_loss = xy_weight*loss_total + pde_weight*total_avg_pde_loss

            # 5.4. Backwards Pass to Optimizing and Scheduling
            total_weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['grad-clip'])
            optimizer.step()

            scheduler.step()
        
            # Print to Console:
            batch_end_time = default_timer()
            if (epoch == 0 and batch_n == 0) or (epoch == epochs-1 and batch_n ==  len(training_dataloader)-1): print_end = '\n'
            else: print_end = '\r'
            print(f'Epoch: {epoch + 1}, Last LP Training Loss: {loss_total.item()}, Estimated Time Left {(batch_end_time - batch_start_time)*(epochs-epoch+1)*(len(training_dataloader)-batch_n+1)/3600:.2f}hrs', end=print_end)
            
            break # only do first batch

        epoch_end_time = default_timer()

        # 5.5. Store Losses to Dictionary
        total_loss_dictionary.update({'Epoch Time': epoch_end_time - epoch_start_time})
        total_loss_dictionary.update({'Total Weighted Loss': total_weighted_loss.item()})
        total_loss_dictionary.update({'GNOT Loss Type': loss.item()})
        total_loss_dictionary.update({'GNOT Loss Type Regression': reg.item()})

        total_loss_dictionary.update({'FNO Loss Type (LP Loss)': loss_l2.item()})
        total_loss_dictionary.update({'FNO Loss Type (MSE Loss)': loss_mse.item()})
        total_loss_dictionary.update({'Internal Weighted PDE Loss': total_avg_pde_loss.item()})
        total_loss_dictionary.update({'Training X-Momentum': pde_loss1.item()})
        total_loss_dictionary.update({'Training Y-Momentum': pde_loss2.item()})
        total_loss_dictionary.update({'Training Continuity': pde_loss3.item()})
        
        #______________________________________________________________________________________________________________________

    # 7. Save Model Checkpoint and Losses
    save_checkpoint(args["save_dir"], args["save_name"], model, 
                loss_dict=total_loss_dictionary.fetch_list(), optimizer=optimizer,
                input_sample=u_p, output_sample=y_pred, ground_truth_sample=y)
    
    print('Model Routine Complete with final fine-tuning Loss: ', loss_l2.item())
    return

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 0. Configs
    dataset_path = r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\PINO datasets\sample_steady_cavity_case_b200_maxU100ms1d_t0.npy'

    # Model Parameters (Commented out are the ones for 2dNS with objects and steady state)
    output_size     = 3         #default is 3 (U,V and Pressure)
    n_layers        = 3
    n_hidden        = 128
    n_head          = 1         #default
    attn_type       = 'linear'  #default
    ffn_dropout     = 0.0       #default
    attn_dropout    = 0.0       #default
    mlp_layers      = 2         #default
    act             = 'gelu'    #default
    hfourier_dim    = 0         #default
    
    # 1. Prepare Data
    dataset_args = dict()      
    dataset_args['use-normalizer']         = 'unit'
    dataset_args['normalize_x']            = 'unit'

    dataset = Cavity_2D_dataset_handling_v1(dataset_path, name='cavity', train=True, 
                                        normalize_y=dataset_args['use-normalizer'], normalize_x = dataset_args['normalize_x'])
    eval_dataset = Cavity_2D_dataset_handling_v1(dataset_path, name='cavity_eval', train=False,
                                        normalize_y=dataset_args['use-normalizer'], 
                                        y_normalizer=dataset.y_normalizer, 
                                        x_normalizer=dataset.x_normalizer, 
                                        up_normalizer=dataset.up_normalizer, 
                                        normalize_x = dataset_args['normalize_x'])
    
    # Data Specific Model Parameters
    trunk_size      = dataset.config['input_dim']
    theta_size      = dataset.config['theta_dim']
    branch_sizes    = dataset.config['branch_sizes']

    # 2. Create model and load in model version
    model = CGPTNO(trunk_size=trunk_size + theta_size,
                branch_sizes=branch_sizes,         # No input function means no branches
                output_size=output_size,
                n_layers=n_layers,                  # from args
                n_hidden=n_hidden,                  # from args
                n_head=n_head,                      # from args
                attn_type=attn_type,                # from args
                ffn_dropout=ffn_dropout,            # from args
                attn_dropout=attn_dropout,          # from args
                mlp_layers=mlp_layers,              # from args
                act=act,                            # from args
                horiz_fourier_dim=hfourier_dim      # from args
                ).to(device)

    # 3. Assign Training Parameters
    training_args = dict()
    training_args['epochs']                 = 750
    training_args["save_dir"]               = 'gnot_cavity'
    training_args["save_name"]              = 'attempt_8_fine'
    training_args['eval_while_training']    = True
    training_args['milestones']             = None
    training_args['fno_loss_type']          = 'LPLoss Relative'
    training_args['base_lr']                = 0.001
    training_args['lr_method']              = 'cycle'
    training_args['scheduler_gamma']        = None  
    training_args['xy_loss']                = 5.0
    training_args['pde_loss']               = 1.0
    training_args['weight-decay']           = 0.00005
    training_args['grad-clip']              = 1000.0    
    training_args['batchsize']              = 1
    training_args['component']              = 'all'
    training_args['normalizer']             = None
    training_args['ckpt']                   = r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\GNOT\checkpoints\gnot_cavity\attempt_3.pt'
    training_args['fine_tuning']            = True

    # 4. Load Model Checkpoint
    if 'ckpt' in training_args:
        ckpt_path = training_args['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

        if 'fine_tuning' in training_args:
            if training_args['fine_tuning']:

                # Set Evaluation to False
                training_args['eval_while_training'] = False
                training_args['xy_loss']             = 0.0
                
                # Set Test data to Training Data
                dataset = eval_dataset
                eval_dataset = None

                # 5. Run scripts based on configurations
                model_routine_cavity_fine_one_case(training_args, model, training_dataset=dataset, testing_dataset=eval_dataset)
                print('Fine-Tuning Complete \n\n')
        else:
            model_routine_cavity(training_args, model, training_dataset=dataset, testing_dataset=eval_dataset)
            print('Training Complete \n\n')