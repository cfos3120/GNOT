import numpy as np

import torch
import torch.nn.functional as F
import argparse
import dgl
from dgl.data import DGLDataset

from timeit import default_timer
from data_utils import MultipleTensors, MIODataLoader, get_dataset, WeightedLpRelLoss, WeightedLpLoss, LpLoss
from models.cgpt import CGPTNO
from utils import UnitTransformer, get_num_params
from data_storage.loss_recording import total_model_dict, save_checkpoint
from data_storage.cavity_2d_data_handling import  *
from train_cavity_4 import *

class fake_isotropic_data(DGLDataset):
    def __init__(self, resolution, dimensions=2, channels=1, batches=4, input_functions=None):
        
        # Queries are just the cell coordinates:
        x = torch.arange(resolution)

        if dimensions == 3:
            [X, Y, Z] = torch.meshgrid(x, x, x,  indexing = 'ij')
            X = X.flatten().unsqueeze(-1)
            Y = Y.flatten().unsqueeze(-1)
            Z = Z.flatten().unsqueeze(-1)
            queries = torch.concat([Y,X,Z],dim=-1)
            output = torch.rand([resolution**3,channels])
            
        elif dimensions == 2:
            [X, Y] = torch.meshgrid(x, x,  indexing = 'ij')
            X = X.flatten().unsqueeze(-1)
            Y = Y.flatten().unsqueeze(-1)
            queries = torch.concat([Y,X],dim=-1)
            output = torch.rand([resolution**2,channels])

        else: raise ValueError('Wrong number of Dimensions, must be 2D or 3D')

        
        # we can test for various combinations of input functions

        # NOTE: This does not work currently. Only single values are given
        if input_functions is not None:
            input_functions_list = []
            for input_function_type in input_functions:
                if input_function_type == 'Single Value':
                    f = torch.rand([1])
                elif input_function_type == '1D Same Dim':
                    f = torch.rand([resolution])
                elif input_function_type == '2D Same Dim':
                    f = torch.rand([resolution,resolution])
                elif input_function_type == '3D Same Dim':
                    f = torch.rand([resolution,resolution])
                else: raise ValueError("Input function types need to be either 'Single Value', '1D Same Dim', '2D Same Dim', '3D Same Dim'")

                input_functions_list.append(f)

            input_f = MultipleTensors([torch.tensor([f]).unsqueeze(-1) for f in input_functions_list])
            #input_f = MultipleTensors([f.float() for f in input_functions_list])
        else:
            # Pad cells
            input_f = torch.zeros([batches])

        ##______________________________________________________________________________________##
        # Creating DGL dataset object
        
        # initialize storage
        graphs = []
        inputs_f = []
        u_p_list = []

        for i in range(batches):

            # Construct Graph object
            g = dgl.DGLGraph()
            g.add_nodes(queries.shape[0])
            g.ndata['x'] = queries.float()
            g.ndata['y'] = output.float()
            up = torch.tensor([0]).float()
            graphs.append(g)
            u_p_list.append(up) # global input parameters
            inputs_f.append(input_f)

        u_p_list = torch.stack(u_p_list)

        config = {
            'input_dim': graphs[0].ndata['x'].shape[1],
            'theta_dim': u_p_list.shape[1],
            'output_dim': graphs[0].ndata['y'].shape[1],
            'branch_sizes': [x.shape[1] for x in inputs_f[0]] if input_functions is not None else 0
        }

        self.config = config
        self.graphs = graphs
        self.u_p_list = u_p_list
        self.inputs_f = inputs_f
        self.n_batches = batches

    def __getitem__(self, idx):
        return self.graphs[idx], self.u_p_list[idx], self.inputs_f[idx]
    
    def __len__(self):
        return self.n_batches
    

if __name__ == '__main__':
    
    # 0. Set Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_args = dict()


    parser = argparse.ArgumentParser(description='GNOT Parameter Study')
    parser.add_argument('--output_sizes', type=int, default=1)
    parser.add_argument('--resolutions', type=int, default=64)
    parser.add_argument('--dimensions', type=int, default=2)
    parser.add_argument('--n_layerss', type=int, default=3)
    parser.add_argument('--n_iterations', type=int, default=1)
    args = parser.parse_args()

    # Here we need to iterate over the following studies:
    #   - Change in Resolution for both 2D and 3D
    #   - Change in Channels for Both 2D and 3D

    # These can be combined into two charts with channels being an extra tracer

    #   - Change in Layers
    #   - Change in attention

    # NOTE: You need to change the output_size to match the channel numbers

    # Fixed Arguments (including variables that will be overwritten)
    model_args['output_size']         = 3
    model_args['n_layers']            = 3
    model_args['n_hidden']            = 128
    model_args['n_head']              = 1
    model_args['attn_type']           = 'linear'
    model_args['ffn_dropout']         = 0.0
    model_args['attn_dropout']        = 0.0
    model_args['mlp_layers']          = 2
    model_args['act']                 = 'gelu'
    model_args['hfourier_dim']        = 0

    # Variable Arguments
    output_sizes = [args.output_sizes]
    n_layerss = [args.n_layerss]
    attn_types = []

    # for averaging compute time
    n_iterations = args.n_iterations

    dimensions = [args.dimensions]
    resolutions = [args.resolutions]

    cases = list()
    for d in dimensions:
        for r in resolutions:
            for c in output_sizes:
                for l in n_layerss:
                    cases.append({'serial':f'case_d{d}r{r}c{c}l{l}', 
                                'dimension':d, 
                                'resolution':r,
                                'channels':c,
                                'layers':l,
                                })
    print(cases)

    for case in cases:
        
        torch.cuda.empty_cache()

        print(f'For Case {case["serial"]}:')
        # 0. Overwrite Cases
        model_args['output_size']         = case['channels']
        model_args['n_layers']            = case['layers']

        # 1. Prepare Data
        dataset = fake_isotropic_data(case['resolution'], dimensions=case['dimension'])

        # 2. Construct Model
        model_args['trunk_size']        = dataset.config['input_dim']
        model_args['theta_size']        = dataset.config['theta_dim']
        model_args['branch_sizes']      = dataset.config['branch_sizes']

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

        model = model.cuda()

        dataloader = MIODataLoader(dataset, batch_size=4, shuffle=False, drop_last=False)

        param_count = get_num_params(model)

        #Iterate over how much
        torch.no_grad()
        model.eval()
        model.zero_grad()
        time_storage = 0
        for i in range(n_iterations):
            print(torch.cuda.memory_allocated())
            for batch in dataloader:
                g, u_p, g_u = batch
                g, u_p, g_u = g.to(device), u_p.to(device), g_u.to(device)

            # 5.1. Forward Pass in Model
            inference_time = default_timer()
            out = model(g, u_p, g_u).cpu()
            inference_time = default_timer() - inference_time
            time_storage += inference_time
            print(torch.cuda.memory_allocated())
        time_storage = time_storage / n_iterations 

        # And print parameters

        print(f'    Mean Inference Time: {time_storage}')
        print(f'    Total Number of Parameters: {param_count}')
        del out, model, g, u_p, g_u, dataloader, dataset
        torch.cuda.empty_cache()