import sys
import argparse
import numpy as np

import torch
import torch.nn.functional as F

from timeit import default_timer

from models.model_memory_study import CGPTNO, check_cuda_memory
from utils import UnitTransformer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GNOT MODEL MEMORY STUDY')
    parser.add_argument('--res', type=int, default=64)
    parser.add_argument('--theta', type=bool, default=True)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_hidden', type=int, default=64)
    
    args = parser.parse_args()
    print(f'CURRENT TESTING SETUP: --res {args.res} --theta {args.theta} --n_layers {args.n_layers} --n_hidden {args.n_hidden}')
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")

    # 2. Construct Model
    model_args = dict()

    model_args['output_size']         = 3
    #model_args['n_layers']            = 3
    #model_args['n_hidden']            = 64 #128  
    model_args['n_head']              = 1
    model_args['attn_type']           = 'linear'
    model_args['ffn_dropout']         = 0.0
    model_args['attn_dropout']        = 0.0
    model_args['mlp_layers']          = 2
    model_args['act']                 = 'gelu'
    model_args['hfourier_dim']        = 0
    
    model_args['trunk_size']        = 2

    if args.theta:
        model_args['theta_size']        = 1
        model_args['branch_sizes']      = None
    else: 
        model_args['theta_size']        = 0
        model_args['branch_sizes']      = [1]

    model = None
    model = CGPTNO(
                trunk_size          = model_args['trunk_size'] + model_args['theta_size'],
                branch_sizes        = model_args['branch_sizes'],     # No input function means no branches
                output_size         = model_args['output_size'],
                n_layers            = args.n_layers,
                n_hidden            = args.n_hidden,
                n_head              = model_args['n_head'],
                attn_type           = model_args['attn_type'],
                ffn_dropout         = model_args['ffn_dropout'],
                attn_dropout        = model_args['attn_dropout'],
                mlp_layers          = model_args['mlp_layers'],
                act                 = model_args['act'],
                horiz_fourier_dim   = model_args['hfourier_dim']
                ).to(device)
    
    check_cuda_memory(device, status_line='After model Creation')

    x = torch.rand([1,args.res**2,2])
    x_2 = torch.rand([1,args.res**2,3])
    y = torch.rand([1,1])
    loss_f = torch.nn.MSELoss(reduction='mean')
    
    for epoch in range(2):
        print(f'EPOCH: {epoch}')
        if args.theta:
            out = model(x=x.clone().float().to(device),u_p = y.clone().float().to(device))
        else:
            out = model(x=x.clone().float().to(device),inputs = y.clone().float().to(device))

        check_cuda_memory(device, status_line='After model Output now backwards pass')

        loss = loss_f(out,x_2)
        loss.backward()
        check_cuda_memory(device, status_line='After Backwards pass')