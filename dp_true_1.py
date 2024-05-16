'''
This Code Uses DP. One Machine - Multi-GPUs
Main differences:
- Choice to normalize input keys or not.
- Choice to shuffle coordinates or not.
'''

import torch
import numpy as np
import math
from timeit import default_timer
from argparse import ArgumentParser
import shutil

from data_storage.loss_recording import total_model_dict, save_checkpoint
from dp_utils.data_utils import *

import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from models.optimizer import Adam, AdamW
from torch.distributed.optim import ZeroRedundancyOptimizer

from torch.nn.parallel import DistributedDataParallel as DDP
#torch.backends.cuda.matmul.allow_tf32 = False



parser = ArgumentParser(description='GNOT Artemis Training Study')
parser.add_argument('--name'        , type=str  , default='test')
parser.add_argument('--dir'         , type=str  , default='test_dir')
parser.add_argument('--path'        , type=str  , default= r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\GNOT\data\steady_cavity_case_b200_maxU100ms_simple_normalized.npy')
parser.add_argument('--epochs'      , type=int  , default=1)
parser.add_argument('--sub_x'       , type=int  , default=4)
parser.add_argument('--inference'   , type=int  , default=1)
parser.add_argument('--n_hidden'    , type=int  , default=128)
parser.add_argument('--train_ratio' , type=float, default=0.7)
parser.add_argument('--seed'        , type=int  , default=42)
parser.add_argument('--lr'          , type=float, default=0.001)
parser.add_argument('--batch_size'  , type=int  , default=4)
parser.add_argument('--rand_cood'   , type=int  , default=0)
parser.add_argument('--normalize_f' , type=int  , default=1)
parser.add_argument('--DP'          , type=int  , default=0)
parser.add_argument('--Optim'       , type=str  , default='Adamw')
parser.add_argument('--Hybrid'      , type=int  , default=0)
parser.add_argument('--scheduler'   , type=str  , default='Cycle')
parser.add_argument('--step_size'   , type=int  , default=50)
parser.add_argument('--init_w'      , type=int  , default=0)
parser.add_argument('--datasplit'   , type=float, default=0.7)
parser.add_argument('--ckpt_path'   , type=str  , default='None')

global ARGS 
ARGS = parser.parse_args()

torch.manual_seed(ARGS.seed)
torch.cuda.manual_seed(ARGS.seed)
np.random.seed(ARGS.seed)
torch.cuda.manual_seed_all(ARGS.seed)

def validation(model, dataloader):
    #device = model.get_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for in_queries, in_keys, out_truth,__ in dataloader:
            in_queries, in_keys, out_truth = in_queries.to(device), in_keys.to(device), out_truth.to(device)
            output = model(x=in_queries,inputs = in_keys)
            val_loss += loss_fn(output, out_truth).item()
        val_loss = val_loss/len(dataloader)
    return val_loss

def train_model(model, train_loader, training_args, loss_fn, recorder, eval_loader=None, output_normalizer=None, input_key_normalizer=None):

    #device = model.get_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(training_args['epochs']):
        torch.cuda.empty_cache()
        model.train()
        epoch_start_time = default_timer()
        mean_train_loss = 0
        mean_train_l2_loss = 0
        mean_train_pde1_loss = 0
        mean_train_pde2_loss = 0
        mean_train_pde3_loss = 0
        mean_train_bc1_loss = 0
        mean_train_bc2_loss = 0
        mean_train_bc3_loss = 0

        for in_queries, in_keys, out_truth, reverse_indices in train_loader:
            optimizer.zero_grad()
            in_queries, in_keys, out_truth = in_queries.to(device), in_keys.to(device), out_truth.to(device)

            #with torch.autograd.detect_anomaly():
            output = model(x=in_queries, inputs=in_keys)

            # Pointwise Loss
            l2_loss = loss_fn(output, out_truth)

            if training_args['Hybrid']:
                output, in_keys = output_realiser(output, in_keys, output_normalizer, input_key_normalizer)
                pde1, pde2, pde3 = NS_FDM_cavity_internal_vertex_non_dim(output, in_keys, nu=0.01, L=0.1)
                pde1 = loss_fn(pde1)
                pde2 = loss_fn(pde2)
                pde3 = loss_fn(pde3)

                total_pde_loss = (pde1 + pde2 + pde3)/3
                mean_train_pde1_loss += pde1.item()
                mean_train_pde2_loss += pde2.item()
                mean_train_pde3_loss += pde3.item()

                # add boundary loss
                bc1, bc2, bc3 = NS_FDM_cavity_boundary_vertex_non_dim(output, loss_fn)
                total_bc_loss = (bc1 + bc2 + bc3)/3
                mean_train_bc1_loss += bc1.item()
                mean_train_bc2_loss += bc2.item()
                mean_train_bc3_loss += bc3.item()
            else:
                total_pde_loss = 0.0
                total_bc_loss = 0.0

            train_loss = l2_loss + total_pde_loss + total_bc_loss

            mean_train_loss     += train_loss.item()
            mean_train_l2_loss  += l2_loss.item()

            if train_loss.isnan(): raise ValueError('Training loss was NaN') #train_loss = 1e-6#raise ValueError('training loss is nan')

            train_loss.backward()

            # Ignore gradient step if there are nulls
            #if torch.stack([torch.isnan(p).any() for p in model.parameters()]).any(): pass

            torch.nn.utils.clip_grad_norm_(model.parameters(),training_args['grad-clip'])
            optimizer.step()
            if training_args['scheduler'] != 'adj_Cycle':
                scheduler.step()

        if training_args['scheduler'] == 'adj_Cycle':
            scheduler.step()

        
        mean_train_loss         = mean_train_loss       /len(train_loader)
        mean_train_l2_loss      = mean_train_l2_loss    /len(train_loader)
        mean_train_pde1_loss    = mean_train_pde1_loss  /len(train_loader)
        mean_train_pde2_loss    = mean_train_pde2_loss  /len(train_loader)
        mean_train_pde3_loss    = mean_train_pde3_loss  /len(train_loader)
        mean_train_bc1_loss     = mean_train_bc1_loss   /len(train_loader)
        mean_train_bc2_loss     = mean_train_bc2_loss   /len(train_loader)
        mean_train_bc3_loss     = mean_train_bc3_loss   /len(train_loader)

        epoch_end_time = default_timer()

        recorder.update_loss({'Epoch Time': epoch_end_time - epoch_start_time})
        recorder.update_loss({'Total Training Loss': mean_train_loss})
        recorder.update_loss({'Training L2 Loss': mean_train_l2_loss})

        if training_args['Hybrid']:
            recorder.update_loss({'Training X-Momentum': mean_train_pde1_loss})
            recorder.update_loss({'Training Y-Momentum': mean_train_pde2_loss})
            recorder.update_loss({'Training Continuity': mean_train_pde3_loss})
            recorder.update_loss({'U Boundary': mean_train_bc1_loss})
            recorder.update_loss({'V Boundary': mean_train_bc2_loss})
            recorder.update_loss({'P Boundary': mean_train_bc3_loss})

        if eval_loader is not None:
            val_loss = validation(model, eval_loader)
            
            recorder.update_loss({'Evaluation L2 Loss': val_loss})
        else: 
            val_loss = np.nan

        if epoch == 0: cuda_get_all_memory_reserved()

        print(f"[Epoch{epoch:4.0f}]: Training Loss {mean_train_loss:7.4f} | Train/Val Loss {mean_train_l2_loss:7.4f}/{val_loss:7.4f} | PDE Loss {(mean_train_pde1_loss+mean_train_pde2_loss+mean_train_pde3_loss)/3:7.4f}")
    
    print('Training Complete')
    save_checkpoint(training_args["save_dir"], training_args["save_name"], model=model, loss_dict=training_run_results.dictionary, optimizer=optimizer)

    try:
        shutil.copyfile(f'checkpoints/{training_args["save_dir"]}/{training_args["save_name"]}.pt'         , f'/content/drive/MyDrive/Results/{training_args["save_name"]}.pt')
        shutil.copyfile(f'checkpoints/{training_args["save_dir"]}/{training_args["save_name"]}_results.npy' , f'/content/drive/MyDrive/Results/{training_args["save_name"]}_results.npy')
        print('saved to Google Drive directory')
    except:
        pass

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_generator = torch.Generator().manual_seed(42)

    # Get Args
    dataset_args, model_args, training_args = get_default_args()

    # Make adjustments based on ArgParser inputs
    dataset_args['sub_x']           = ARGS.sub_x
    dataset_args['batchsize']       = ARGS.batch_size
    dataset_args['file_path']       = ARGS.path
    dataset_args['random_coords']   = ARGS.rand_cood == 1
    dataset_args['normalize_f']     = ARGS.normalize_f == 1
    dataset_args['inference']       = ARGS.inference == 1
    dataset_args['train_ratio']     = ARGS.datasplit

    training_args['DP']             = ARGS.DP == 1
    training_args['Hybrid']         = ARGS.Hybrid == 1
    training_args['base_lr']        = ARGS.lr
    training_args['step_size']      = ARGS.step_size
    training_args["save_name"]      = ARGS.name
    training_args["save_dir"]       = ARGS.dir
    training_args['epochs']         = ARGS.epochs
    training_args['scheduler']      = ARGS.scheduler
    training_args['ckpt']           = ARGS.ckpt_path

    model_args['init_w']            = ARGS.init_w == 1
    

    # Dataset Creation
    dataset = prepare_dataset(dataset_args)
    train_torch_dataset = create_torch_dataset(dataset,
                                               dataset_args,
                                               train=True,
                                               random_coords=dataset_args['random_coords']
                                               )
    val_torch_dataset = create_torch_dataset(dataset,
                                             dataset_args,
                                             train=False,
                                             random_coords=dataset_args['random_coords']
                                             )
    train_loader = DataLoader(
            dataset=train_torch_dataset,
            batch_size=dataset_args['batchsize'],
            shuffle=True,
            generator=seed_generator
        )
    val_loader = DataLoader(
            dataset=val_torch_dataset,
            batch_size=dataset_args['batchsize'],
            shuffle=False
        )
    
    print(f'Number of Training/Validation Batches: {len(train_loader)}/{len(val_loader)}')
    
    # Model Setup
    model = get_model(model_args)
    
    if model_args['init_w']:
        model.apply(model._init_weights)

    if ARGS.DP:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Use checkpoint
    if training_args['ckpt'] != 'None':
        ckpt_path = training_args['ckpt']
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)


    # Training Settings:
    loss_fn = LpLoss_custom()
    #loss_fn = torch.nn.MSELoss()

    if ARGS.Optim == 'Adamw':
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    betas=(0.9, 0.999), 
                                    lr=training_args['base_lr'],
                                    weight_decay=training_args['weight-decay']
                                    )
    elif ARGS.Optim == 'Adamw_custom':
        optimizer = AdamW(model.parameters(), 
                        betas=(0.9, 0.999), 
                        lr=training_args['base_lr'],
                        weight_decay=training_args['weight-decay']
                        )
    elif ARGS.Optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                    betas=(0.9, 0.999), 
                                    lr=training_args['base_lr'],
                                    weight_decay=training_args['weight-decay']
                                    )
    elif ARGS.Optim == 'Adam_custom':
        optimizer = Adam(model.parameters(), 
                        betas=(0.9, 0.999), 
                        lr=training_args['base_lr'],
                        weight_decay=training_args['weight-decay']
                        )
    else: raise NotImplementedError
    
    if ARGS.scheduler == 'Cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr=training_args['base_lr'], 
                                                        div_factor=1e4, 
                                                        pct_start=0.2, 
                                                        final_div_factor=1e4, 
                                                        steps_per_epoch=len(train_loader), 
                                                        epochs=training_args['epochs']
                                                        )
    elif ARGS.scheduler == 'adj_Cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr=training_args['base_lr'], 
                                                        div_factor=1e4, 
                                                        pct_start=0.2, 
                                                        final_div_factor=1e4, 
                                                        steps_per_epoch=1, 
                                                        epochs=training_args['epochs']
                                                        )
    elif ARGS.scheduler == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training_args['step_size']*len(train_loader), gamma=0.7) #default step size 50
    else: raise NotImplementedError
    
    # Initialize Model Training Recorder
    training_run_results = total_model_dict(model_config=model_args, training_config=training_args, data_config=dataset_args)
    
    if training_args['Hybrid']: print('Training in Hybrid mode with PDEs\n')
    else: print('Training in Pure mode with only L2 Loss\n')
    
    train_model(model, 
                train_loader, 
                training_args,  
                loss_fn, 
                training_run_results, 
                val_loader,
                output_normalizer=dataset.output_normalizer.to(device), 
                input_key_normalizer=dataset.input_f_normalizer.to(device))