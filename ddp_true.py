import torch
import numpy as np
import math
from timeit import default_timer
from argparse import ArgumentParser

from ddp_utils.ddp_utils import *
from ddp_utils.data_utils import *

from data_storage.loss_recording import total_model_dict, save_checkpoint

def train(model,train_loader,optimizer,scheduler,batch_size):
    device = torch.device(f"cuda:{dist.get_rank()}")
    train_num_batches = int(math.ceil(len(train_loader.dataset) / float(batch_size)))
    model.train()
    # let all processes sync up before starting with a new epoch of training
    # dist.barrier()
    criterion = LpLoss_custom()#.to(device)
    train_loss = 0.0
    for in_queries, in_keys, out_truth in train_loader:
        
        in_queries, in_keys, out_truth = in_queries.to(device), in_keys.to(device), out_truth.to(device)
        optimizer.zero_grad()
        output = model(x=in_queries,inputs = in_keys)
        loss = criterion(output, out_truth)
        loss.backward()
        # average gradient as DDP doesn't do it correctly
        average_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)#training_args['grad-clip'])
        optimizer.step()
        scheduler.step()

        loss_ = {'loss': torch.tensor(loss.item()).to(device)}
        train_loss += reduce_dict(loss_)['loss'].item()
        # cleanup
        # dist.barrier()
        # data, target, output = data.cpu(), target.cpu(), output.cpu()
    train_loss_val = train_loss / train_num_batches
    return train_loss_val
    
def val(model, val_loader,batch_size):
    device = torch.device(f"cuda:{dist.get_rank()}")
    val_num_batches = int(math.ceil(len(val_loader.dataset) / float(batch_size)))
    model.eval()
    # let all processes sync up before starting with a new epoch of training
    # dist.barrier()
    criterion = LpLoss_custom()#.to(device)
    val_loss = 0.0
    with torch.no_grad():
        for in_queries, in_keys, out_truth in val_loader:
            in_queries, in_keys, out_truth = in_queries.to(device), in_keys.to(device), out_truth.to(device)
            output = model(x=in_queries,inputs = in_keys)
            loss = criterion(output, out_truth)
            loss_ = {'loss': torch.tensor(loss.item()).to(device)}
            val_loss += reduce_dict(loss_)['loss'].item()
    val_loss_val = val_loss / val_num_batches
    return val_loss_val



def function_on_rank(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    dataset_args, model_args, training_args = get_default_args()

    dataset_args['file_path'] = '/project/MLFluids/steady_cavity_case_b200_maxU100ms_simple_normalized.npy'
    
    model = get_model(model_args).to(rank)
    model = DDP(model,device_ids=[rank])#,output_device=rank)

    if rank == 0:
        print('after model allocation:')
        gpu_print_string,__ = get_gpu_resources()
        print(gpu_print_string)

    train_loader, val_loader, batch_size = get_dataset(dataset_args)   
    
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

    if rank == 0:
        training_run_results = total_model_dict(model_config=model_args, training_config=training_args, data_config=dataset_args)

    for epoch in range(training_args['epochs']):
            
        epoch_start_time = default_timer()
        train_loss_val = train(model,train_loader,optimizer,scheduler,batch_size)
        val_loss_val = val(model,val_loader,batch_size)
        epoch_end_time = default_timer()
            
        if rank == 0:
            print(f'Rank {rank} epoch {epoch}: {train_loss_val:.2f})/{val_loss_val:.2f}')
            training_run_results.update_loss({'Epoch Time': epoch_end_time - epoch_start_time})
            training_run_results.update_loss({'Training L2 Loss': train_loss_val})
            training_run_results.update_loss({'Evaluation L2 Loss': val_loss_val})
                
        print(f'Rank {rank} finished training')
        dist.barrier()
        
        if rank == 0:
            print('Saving Model...')
            save_checkpoint(training_args["save_dir"], training_args["save_name"], model=model, loss_dict=training_run_results.dictionary, optimizer=optimizer)
        
        dist.barrier()
        cleanup(rank)

if __name__ == "__main__":
    parser = ArgumentParser(description='GNOT Artemis Training Study')
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

    run(function_on_rank, 2)
