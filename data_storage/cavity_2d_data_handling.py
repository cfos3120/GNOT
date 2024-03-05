from dgl.data import DGLDataset
import dgl
import torch
import numpy as np
from data_utils import MultipleTensors
from utils import UnitTransformer

class Cavity_2D_dataset_handling_v2(DGLDataset):
    def __init__(self, data_path, L=1.0, name=' ', sub_x = 1, train=True, normalize_y=False, y_normalizer=None, normalize_x = False, x_normalizer = None, up_normalizer =None, 
                 data_split = 0.7, seed = 42, vertex = False, boundaries = False):

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
        self.sub_x = sub_x # <- not implemented yet
        self.seed = seed
        self.data_split = data_split
        self.vertex = vertex
        self.boundaries = boundaries

        # Load in Dataset and retrieve shape
        self.data_out   = np.load(data_path)
        if self.sub_x > 1: self.subsampler()
        if self.vertex: self.cell_to_vertex_converter()
        if self.boundaries: self.add_boundaries()

        print(f'Dataset Shape: {self.data_out.shape}, subsampled by {self.sub_x}')
        # NOTE this can also be in the form of reynolds number 
        self.data_lid_v = np.round(np.arange(0.5,100.5,0.5),1) # * 0.1/0.01 <- apply for Reynolds Number
        self.n_batches  = self.data_out.shape[0]
        self.nx         = int(self.data_out.shape[1])
        self.num_nodes  = self.nx**2

        self.L = L
        self.train = train

        super(Cavity_2D_dataset_handling_v2, self).__init__(name)   # invoke super method after read data


    def process(self):
        
        # SECTION 0: Split into train or test (Same as for FNO training)
        train_size = int(self.data_split * self.n_batches)
        test_size = self.n_batches - train_size

        seed_generator = torch.Generator().manual_seed(self.seed)

        train_dataset,  test_dataset    = torch.utils.data.random_split(torch.from_numpy(self.data_out),    [train_size, test_size], generator=seed_generator)
        train_lid_v,    test_lid_v      = torch.utils.data.random_split(torch.from_numpy(self.data_lid_v),  [train_size, test_size], generator=seed_generator)
        
        print(f'''Dataset Split up using torch generator seed: {seed_generator.initial_seed()}
              This can be replicated e.g.
                generator_object = torch.Generator().manual_seed({seed_generator.initial_seed()})\n ''')
        
        # The torch.utils.data.random_split() only gives objects with the whole datset or a integers, so we need to override these variables with the indexed datset split
        train_dataset,  test_dataset    = train_dataset.dataset[train_dataset.indices,...], test_dataset.dataset[test_dataset.indices,...]
        train_lid_v,    test_lid_v      = train_lid_v.dataset[train_lid_v.indices], test_lid_v.dataset[test_lid_v.indices]

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
        self.MIOdataset_structure()
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

def NS_FDM_cavity_internal_cell_non_dim(U, lid_velocity, nu, L):

    batchsize = U.size(0)
    nx = U.size(1)
    ny = U.size(2)
    
    device = U.device

    # assign Reynolds Number array:
    Re = (lid_velocity * L/nu).repeat(1,nx,ny)

    # create isotropic grid (non-dimensional i.e. L=1.0)
    y = torch.tensor(np.linspace(0, 1, nx+1)[:-1], dtype=torch.float, device=device)
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
    
    u[:, -1 ,1:-1] = - u[:, -2 ,1:-1] + 2*(1)   # Lid (x-vel) (here the (1) is the lid velocity)

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

    # No time derivative as we are assuming steady state solution
    Du_dx = U[...,0]*ux + U[...,1]*uy - (1/Re) * (uxx + uyy) + px
    Dv_dy = U[...,0]*vx + U[...,1]*vy - (1/Re) * (vxx + vyy) + py
    continuity_eq = (ux + vy)

    fdm_derivatives = tuple([ux, uy, vx, vy, px, py, uxx, uyy, vxx, vyy, Du_dx, Dv_dy, continuity_eq])
    
    return Du_dx, Dv_dy, continuity_eq, fdm_derivatives

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