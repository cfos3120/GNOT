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
from torch.distributed.optim import ZeroRedundancyOptimizer

from torch.nn.parallel import DistributedDataParallel as DDP



parser = ArgumentParser(description='GNOT Artemis Training Study')
parser.add_argument('--name'        , type=str  , default='test')
parser.add_argument('--path'        , type=str  , default= r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\GNOT\data\steady_cavity_case_b200_maxU100ms_simple_normalized.npy')
parser.add_argument('--epochs'      , type=int  , default=1)
parser.add_argument('--sub_x'       , type=int  , default=4)
parser.add_argument('--inference'   , type=str  , default='True')
parser.add_argument('--n_hidden'    , type=int  , default=128)
parser.add_argument('--train_ratio' , type=float, default=0.7)
parser.add_argument('--seed'        , type=int  , default=42)
parser.add_argument('--lr'          , type=float, default=0.001)
parser.add_argument('--batch_size'  , type=int  , default=4)
parser.add_argument('--rand_cood'   , type=int  , default=0)
parser.add_argument('--normalize_f' , type=int  , default=0)
parser.add_argument('--DP'         , type=int  , default=0)
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

def train_model(model, train_loader, training_args, loss_fn, recorder, eval_loader=None):
    
    #device = model.get_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(training_args['epochs']):
        torch.cuda.empty_cache()
        model.train()
        epoch_start_time = default_timer()
        mean_train_loss = 0
        for in_queries, in_keys, out_truth, reverse_indices in train_loader:
            optimizer.zero_grad()
            in_queries, in_keys, out_truth = in_queries.to(device), in_keys.to(device), out_truth.to(device)
            output = model(x=in_queries, inputs=in_keys)

            # Pointwise Loss
            train_loss = loss_fn(output, out_truth)
            mean_train_loss += train_loss.item()

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),training_args['grad-clip'])
            optimizer.step()
            scheduler.step()

        mean_train_loss = mean_train_loss/len(train_loader)
        epoch_end_time = default_timer()

        recorder.update_loss({'Epoch Time': epoch_end_time - epoch_start_time})
        recorder.update_loss({'Training L2 Loss': mean_train_loss})

        if eval_loader is not None:
            val_loss = validation(model, eval_loader)
            
            recorder.update_loss({'Evaluation L2 Loss': val_loss})
        else: 
            val_loss = np.nan

        if epoch == 0: cuda_get_all_memory_reserved()

        print(f"[Epoch{epoch:4.0f}]: Train/Val Loss {mean_train_loss:7.4f}/{val_loss:7.4f} | Last Lid Vels: {torch.flatten(in_keys)}")
    
    print('Training Complete')
    save_checkpoint(training_args["save_dir"], training_args["save_name"], model=model, loss_dict=training_run_results.dictionary, optimizer=optimizer)

    try:
        shutil.copyfile(f'checkpoints/{training_args["save_dir"]}/{training_args["save_name"]}.pt'         , f'/content/drive/MyDrive/Results/{training_args["save_name"]}.pt')
        shutil.copyfile(f'checkpoints/{training_args["save_dir"]}/{training_args["save_name"]}_results.npy' , f'/content/drive/MyDrive/Results/{training_args["save_name"]}_results.pt')
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
    training_args['epochs']         = ARGS.epochs
    training_args["save_name"]      = ARGS.name
    dataset_args['file_path']       = ARGS.path
    dataset_args['random_coords']   = ARGS.rand_cood == 1
    dataset_args['normalize_f']     = ARGS.normalize_f == 1
    training_args['DP']            = ARGS.DP == 1

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
    
    # Model Setup
    model = get_model(model_args)
    if ARGS.DP:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Training Settings:
    loss_fn = LpLoss_custom()

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
                                                    steps_per_epoch=len(train_loader), 
                                                    epochs=training_args['epochs']
                                                    )
    
    # Initialize Model Training Recorder
    training_run_results = total_model_dict(model_config=model_args, training_config=training_args, data_config=dataset_args)
    

    train_model(model, train_loader, training_args, loss_fn, training_run_results, val_loader)