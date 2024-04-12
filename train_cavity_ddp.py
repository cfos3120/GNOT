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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from train_cavity_artemis_v2 import get_default_args, get_model, LpLoss_custom, get_cavity_dataset

def demo_basic():

    parser = argparse.ArgumentParser(description='GNOT Artemis Training Study')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--path', type=str, default= r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\GNOT\data\steady_cavity_case_b200_maxU100ms_simple_normalized.npy')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--sub_x', type=int, default=4)
    parser.add_argument('--inference', type=str, default='True')
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    # DDP Initialize
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank'
    dataset_args, model_args, training_args = get_default_args()

    # adjustments to defaults
    dataset_args['file_path']       = args.path
    dataset_args['sub_x']           = args.sub_x
    if args.inference == 'False': dataset_args['inference'] = False
    training_args['epochs']         = args.epochs
    training_args['save_name']      = args.name
    model_args['n_hidden']          = args.n_hidden
    dataset_args['train_ratio']     = args.train_ratio
    dataset_args['seed']            = args.seed
    training_args['base_lr']        = args.lr
    training_args['batchsize']      = args.batch_size

    ## DDP Model
    device_id = rank % torch.cuda.device_count()
    model = get_model(model_args).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    loss_function = LpLoss_custom()
    # Default optimizer and scheduler from paper
    optimizer = torch.optim.AdamW(ddp_model.parameters(), 
                                  betas=(0.9, 0.999), 
                                  lr=training_args['base_lr'],
                                  weight_decay=training_args['weight-decay']
                                  )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr=training_args['base_lr'], 
                                                    div_factor=1e4, 
                                                    pct_start=0.2, 
                                                    final_div_factor=1e4, 
                                                    steps_per_epoch=10, 
                                                    epochs=training_args['epochs']
                                                    )

    # DDP Dataset
    dataset = get_cavity_dataset(dataset_args)
    ddp_sampler = dist.DistributedSampler(dataset, shuffle=True)
    train_dataloader = DataLoader(ddp_sampler, batch_size=training_args['batchsize'], shuffle=False) 

    # also get testing datset
    dataset_args_eval = dataset_args
    dataset_args_eval['train'] = False
    dataset_eval = get_cavity_dataset(dataset_args_eval)
    ddp_sampler_eval = dist.DistributedSampler(dataset_eval, shuffle=False)
    eval_dataloader = DataLoader(ddp_sampler_eval, batch_size=training_args['batchsize'], shuffle=False)

    if device_id == 0:
        # Initialize Results Storage: 
        training_run_results = total_model_dict(model_config=model_args, training_config=training_args, data_config=dataset_args)

    torch.cuda.set_device(rank)

    for epoch in range(training_args['epochs']):
        epoch_start_time = default_timer()

        # Set Model to Train
        ddp_model.train()
        torch.cuda.empty_cache()

        # Initialize loss storage per epoch
        loss_total_list = list()
        
        for batch_n, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            in_queries, in_keys, out_truth = batch
            in_queries, in_keys, out_truth = in_queries.to(device), in_keys.to(device), out_truth.to(device)
            
            out = ddp_model(x=in_queries,inputs = in_keys)
            
            loss = loss_function(out,out_truth)

            loss.backward()#(retain_graph=True)
            torch.cuda.synchronize(device=rank)

            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), training_args['grad-clip'])
            optimizer.step()

            scheduler.step()

        epoch_end_time = default_timer()
        
        if device_id == 0:
            training_run_results.update_loss({'Epoch Time': epoch_end_time - epoch_start_time})
            training_run_results.update_loss({'Training L2 Loss': loss.item()})

        # Now lets evaluate the model at each epoch too
        with torch.no_grad():
            loss_eval = 0
            for batch_n, batch in enumerate(eval_dataloader):
                in_queries, in_keys, out_truth = batch
                in_queries, in_keys, out_truth = in_queries.to(device), in_keys.to(device), out_truth.to(device)

                out = ddp_model(x=in_queries,inputs = in_keys)
                
                loss_eval += loss_function(out,out_truth).item()

            loss_eval = loss_eval/(batch_n+1)

            if device_id == 0:
                training_run_results.update_loss({'Evaluation L2 Loss': loss_eval})

        
        print(f'[GPU{device_id}] Epoch: {epoch :8} L2 Training Loss {loss :12.7f}, L2 Evaluation Loss: {loss_eval :12.7f}')

        sys.stdout.flush()
    
    if training_args['epochs'] != 1 and device_id == 0:
        torch.cuda.synchronize(device=rank) 
        save_checkpoint(training_args["save_dir"], 
                        training_args["save_name"], 
                        model=ddp_model, 
                        loss_dict=training_run_results.dictionary, 
                        optimizer=optimizer)
        
    dist.destroy_process_group()

if __name__ == "__main__":
    demo_basic()
    