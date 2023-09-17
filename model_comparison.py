import numpy as np
import torch
import time
from dgl.data import DGLDataset
import dgl
from data_utils import MultipleTensors, MIODataLoader
from utils import get_num_params

from models.FNO_3D import FNO3d
from models.cgpt import CGPTNO

# Create Dataset handling Function for GNOT
class PINO_dataset_handling(DGLDataset):
    def __init__(self, data=None, data_path=None, ic_t_steps=5, name=' '):

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
        if data == None and data_path != None:
            self.data = np.load(data_path, allow_pickle=True)[:1,...]
        elif data != None and data_path == None:
            self.data = data
        else: raise(ValueError)
       
        self.ic_t_steps = ic_t_steps    
        self.n_batches  = self.data.shape[0]
        self.nx         = self.data.shape[2]
        self.nt         = self.data.shape[1]
        self.num_nodes  = self.nx*self.nx*(self.nt-ic_t_steps)

        super(PINO_dataset_handling, self).__init__(name)   # invoke super method after read data


    def process(self):

        # Section 1: Transformer Queries
        x = torch.arange(self.nx)
        y = x                           # Assume Isotropic Grid     
        t = torch.arange(self.nt-self.ic_t_steps)    # exclude initial condition in this case

        [X, Y, T] = torch.meshgrid(x, y, t)

        # the order of these are not intuitive, but as long as features Y are mapped the same way
        # and we reshape the output backwards in a similar manner. It should work.
        X = X.reshape(self.num_nodes,1)
        Y = Y.reshape(self.num_nodes,1)
        T = T.reshape(self.num_nodes,1)

        # Linearize these matrices.
        self.X_for_queries = torch.concat([T,Y,X],axis=-1)
        print('Shape of Queries Input: ', self.X_for_queries.shape)

        # Section 2: Output and Initial conditions
        self.outputs_ground_truth    = self.data[:,self.ic_t_steps:,...].reshape(self.n_batches,self.num_nodes,1)
        self.inital_conditions       = self.data[:,:self.ic_t_steps,...].reshape(self.n_batches,self.nx*self.nx*self.ic_t_steps,1)

        print('Shape of Keys and Values Input: ', self.inital_conditions.shape)
        print('Shape of Expected Model Output: ', self.outputs_ground_truth.shape)
        
        # Section 3: Transform to be MIOdataset Loader Compatible
        self.MIOdataset_structure()
        self.__update_dataset_config()


    def MIOdataset_structure(self):
        '''
        NOTE here for FNO, X is constant because the grid is the same for all batches
        Theta is [0] because it requires a non-null input in the current model config
        Output (Y) and initial conditions (g_u) change per batch)
        g_u is the input condition in tuple structure as the model can accept multiple input functions.
        '''
        # initialize storage
        self.graphs = []
        self.inputs_f = []
        self.u_p_list = []

        for i in range(self.n_batches):

            x, y    = self.X_for_queries, self.outputs_ground_truth[i,...]
            u_p     = np.array([0])
            input_f = tuple([self.inital_conditions[i,...]])

            # Construct Graph object
            g = dgl.DGLGraph()
            g.add_nodes(x.shape[0])
            g.ndata['x'] = x.float()
            g.ndata['y'] = y.float()
            up = torch.from_numpy(u_p).float()
            self.graphs.append(g)
            self.u_p_list.append(up) # global input parameters
            if input_f is not None:
                input_f = MultipleTensors([f.float() for f in input_f])
                self.inputs_f.append(input_f)
                num_inputs = len(input_f)

        self.u_p_list = torch.stack(self.u_p_list)


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
    
# Create Dataset handling Function for FNO


# First we try the transformer

if __name__ == "__main__":

    ''' 
    Create dataset:
       1. Batch
       2. Time Steps
       3. X-dim
       4. Y-dim
       5. Channels (Vorticity)
    
    FNO Dataset has coordinates already embedded (e.g. 4 channels)
    '''

    print('_____________________________________\n FNO Model Study:')
    data_sample_FNO = torch.rand([1,65,64,64,4])

    model_fno = FNO3d(modes1=[8, 8, 8, 8],
                        modes2=[8, 8, 8, 8],
                        modes3=[8, 8, 8, 8],
                        fc_dim=128,
                        layers=[64, 64, 64, 64, 64],
                        in_dim=1 + 3,
                        out_dim=1)

    
    print('FNO Model has ', get_num_params(model_fno), 'parameters')

    print('Running FNO Forward Pass')
    print(f'    Input Shape:', data_sample_FNO.shape)
    t0 = time.time()
    out = model_fno(data_sample_FNO)
    t1 = time.time()
    print(f'    Total Forward Pass Inference Time: {t1-t0:.4f}s')
    print(f'    Output Shape:', out.shape)




    ## GNOT Model Pass through
    print('\n\n_____________________________________\n GNOT Model Study:')
    data_sample_GNOT = torch.rand([1,65,64,64,1])
    
    t0 = time.time()
    train_dataset = PINO_dataset_handling(data_sample_GNOT)
    dataloader = MIODataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=False)
    t1 = time.time()
    print(f'Total Dataset transformation time: {t1-t0:.4f}s')

    for batch_fno in dataloader:
        break
    
    print('training dataset configurations for model: ', train_dataset.config)
    model_gnot = CGPTNO(trunk_size=train_dataset.config['input_dim'] + train_dataset.config['theta_dim'],
                        branch_sizes=train_dataset.config['branch_sizes'], 
                        output_size=1,
                        n_layers=3,
                        n_hidden=128,
                        n_head=1,
                        attn_type='linear',
                        ffn_dropout=0.0,
                        attn_dropout=0.0,
                        mlp_layers=2,
                        act='gelu',
                        horiz_fourier_dim=0
                        )

    print('GNOT Model has ', get_num_params(model_gnot), 'parameters')

    print('Running GNOT Forward Pass')
    g, u_p, g_u = batch_fno
    t0 = time.time()
    out = model_gnot(g, u_p, g_u)
    t1 = time.time()
    print(f'    Total Forward Pass Inference Time: {t1-t0:.4f}s')
    print('     Output Shape:', out.shape)