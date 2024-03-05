import sys
import numpy as np
import shutil
import argparse

import torch
import torch.nn.functional as F

import dgl
from dgl.data import DGLDataset

from timeit import default_timer

from data_utils import MultipleTensors, MIODataLoader, get_dataset, WeightedLpRelLoss, WeightedLpLoss, LpLoss
from models.cgpt import CGPTNO
from utils import UnitTransformer
from data_storage.loss_recording import total_model_dict, save_checkpoint
from data_storage.cavity_2d_data_handling import  *
from train_cavity_4 import *

if __name__ == '__main__':
    
    print('Fine Tuning off checkpoint not set up')
    parser = argparse.ArgumentParser(description='GNOT Training Study')
    parser.add_argument('--name', type=str)
    parser.add_argument('--subsample', type=int, default=1)
    parser.add_argument('--weight_l2', type=int, default=5)
    parser.add_argument('--weight_pinn', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--fine', type=bool, default=False)
    parser.add_argument('--weight_balance', type=str, default=None, choices=[None,'dynamic','scaled'])
    args = parser.parse_args()

    # 0. Set Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #names = ['ansatz_pinn_dynamic_sub_4']#, 'ansatz_sub_4', 'ansatz_sub_2']
    #subs = [4]#,4,2]

    model_args = dict()
    dataset_args = dict()
    training_args = dict()
    training_args["save_name"] = args.name
    dataset_args['subsampler'] = args.subsample

    # 1. Prepare Data
    dataset_args['file']                    = r'/content/drive/MyDrive/Data/steady_cavity_case_b200_maxU100ms_simple_normalized.npy'
    dataset_args['percent split (decimal)'] = 0.7
    dataset_args['randomizer seed']         = 42
    dataset_args['use-normalizer']          = 'unit'
    dataset_args['normalize_x']             = 'unit'
    #dataset_args['subsampler']              = 4
    dataset_args['cell to pointwise']       = True
    dataset_args['add boundaries']          = True

    dataset = Cavity_2D_dataset_handling_v2(dataset_args['file'], name='cavity', train=True, sub_x = dataset_args['subsampler'],
                                        normalize_y=dataset_args['use-normalizer'], normalize_x = dataset_args['normalize_x'],
                                        data_split = dataset_args['percent split (decimal)'], seed = dataset_args['randomizer seed'],
                                        vertex = dataset_args['cell to pointwise'], boundaries = dataset_args['add boundaries']
                                        )

    eval_dataset = Cavity_2D_dataset_handling_v2(dataset_args['file'], name='cavity_eval', train=False, sub_x = dataset_args['subsampler'],
                                        normalize_y=dataset_args['use-normalizer'],
                                        data_split = dataset_args['percent split (decimal)'], seed = dataset_args['randomizer seed'],
                                        vertex = dataset_args['cell to pointwise'], boundaries = dataset_args['add boundaries'],
                                        y_normalizer=dataset.y_normalizer,
                                        x_normalizer=dataset.x_normalizer,
                                        up_normalizer=dataset.up_normalizer,
                                        normalize_x = dataset_args['normalize_x'])

    # 2. Construct Model
    model_args['trunk_size']        = dataset.config['input_dim']
    model_args['theta_size']        = dataset.config['theta_dim']
    model_args['branch_sizes']      = dataset.config['branch_sizes']

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
    training_args["save_dir"]               = 'gnot_cavity_v4'
    #training_args["save_name"]              = 'scaled_v1'
    training_args['eval_while_training']    = True
    training_args['milestones']             = None
    training_args['base_lr']                = 0.001
    training_args['lr_method']              = 'cycle'
    training_args['scheduler_gamma']        = None
    training_args['xy_loss']                = args.weight_l2
    training_args['pde_loss']               = args.weight_pinn
    training_args['weight-decay']           = 0.00005
    training_args['grad-clip']              = 1000.0
    training_args['batchsize']              = 4
    training_args['component']              = 'all'
    training_args['normalizer']             = None
    training_args['loss_weighting']         = args.weight_balance #'dynamic' #'scaled' #scaled #scaled, dynamic or none
    #training_args['ckpt']                   = r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\GNOT\checkpoints\gnot_cavity_v2\baseline_ansatz_.pt'
    training_args['fine_tuning']            = args.fine
    training_args['boundaries']             = 'hard'
    # Override any duplicate settings
    if training_args['fine_tuning']:
        if 'ckpt' not in training_args:
            print('NOTE: NO CHECKPOINT LOADED')
            #raise ValueError('Can not fine-tune without Checkpoint')

        training_args['xy_loss']            = 0.0
        training_args['batchsize']          = 1
        training_args['eval_while_training']= False

    # Initialize Results Storage:
    training_run_results = total_model_dict(model_config=model_args, training_config=training_args, data_config=dataset_args)


    model_training_routine(device, model, training_args, dataset, eval_dataset, training_run_results)

    path = 'checkpoints/'+training_args["save_dir"] + '/' + training_args["save_name"] + '_results.npy'
    print(path)

    # Save Training Log
    path = 'checkpoints/'+training_args["save_dir"] + '/' + training_args["save_name"] + '_results.npy'
    shutil.copyfile(path, '/content/drive/MyDrive/Results/'+training_args["save_name"] + '_results.npy')

    # Save Model
    path = 'checkpoints/'+training_args["save_dir"] + '/' + training_args["save_name"] + '.pt'
    shutil.copyfile(path, '/content/drive/MyDrive/Results/'+training_args["save_name"] + '.pt')