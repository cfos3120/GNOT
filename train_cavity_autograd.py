import torch
import numpy as np
import argparse
import shutil
from timeit import default_timer

from models.cgpt import CGPTNO
from data_storage.cavity_2d_data_handling import  *
from data_storage.loss_recording import total_model_dict, save_checkpoint
from data_utils import MIODataLoader, WeightedLpLoss, LpLoss

# 0. Set Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_dataset_file(dataset_args):
    train_dataset = Cavity_2D_dataset_handling_v2(dataset_args['file'], name='cavity', train=True, sub_x = dataset_args['subsampler'],
                                                normalize_y=dataset_args['use-normalizer'], normalize_x = dataset_args['normalize_x'],
                                                data_split = dataset_args['percent split (decimal)'], seed = dataset_args['randomizer seed'],
                                                vertex = dataset_args['cell to pointwise'], boundaries = dataset_args['add boundaries'])
    
    eval_dataset = Cavity_2D_dataset_handling_v2(dataset_args['file'], name='cavity_eval', train=False, sub_x = dataset_args['subsampler'],
                                                normalize_y=dataset_args['use-normalizer'],
                                                data_split = dataset_args['percent split (decimal)'], seed = dataset_args['randomizer seed'], 
                                                vertex = dataset_args['cell to pointwise'], boundaries = dataset_args['add boundaries'],
                                                y_normalizer=train_dataset.y_normalizer, 
                                                x_normalizer=train_dataset.x_normalizer, 
                                                up_normalizer=train_dataset.up_normalizer, 
                                                normalize_x = dataset_args['normalize_x'])
    
    return train_dataset, eval_dataset, train_dataset.y_normalizer

def get_model(model_args):    
    model = CGPTNO(trunk_size          = model_args['trunk_size'] + model_args['theta_size'],
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
    return model

def hard_enforce_boundaries(y_pred):
    
    # Velocities
    y_pred[:,  0 ,  :, :] = 0                           # Bottom wall
    y_pred[:,  : ,  0, :] = 0                           # Left wall
    y_pred[:,  : , -1, :] = 0                           # Right wall
    y_pred[:, -1 ,  :, 0], y_pred[:, -1 ,:,1] = 1, 0    # Lid

    # Pressure:
    y_pred[:,  0 ,  : ,2] = y_pred[:,  1 ,  : ,2]  # Bottom Wall
    y_pred[:,  : ,  0 ,2] = y_pred[:,  : ,  1 ,2]  # Left Wall
    y_pred[:,  : , -1 ,2] = y_pred[:,  : , -2 ,2]  # Right Wall
    y_pred[:, -1 ,  : ,2] = y_pred[:, -2 ,  : ,2]  # Lid (y-vel)
    return y_pred

def pde_autograd(model_input_coords, model_out, Re, y_normalizer=None, hard_enforce_bc=False):

    # Stack and Repeat Re for tensor multiplication
    Re = Re.squeeze(-1).repeat(1,int(model_out.shape[0]/g_u[0].shape[0])).reshape(model_out.shape[0],1)

    # Un-normalize model for real derivatives
    if y_normalizer is not None:
        model_out = y_normalizer.transform(model_out.to('cpu'),inverse=True)

    if hard_enforce_bc:
        raise NotImplementedError('Hard enforced Boundaries not Supported yet')

    u = model_out[..., 0:1]
    v = model_out[..., 1:2]
    p = model_out[..., 2:3]

    # First Derivatives
    u_out = torch.autograd.grad(u.sum(), model_input_coords, create_graph=True)[0]
    v_out = torch.autograd.grad(v.sum(), model_input_coords, create_graph=True)[0]
    p_out = torch.autograd.grad(p.sum(), model_input_coords, create_graph=True)[0]

    u_x = u_out[..., 0:1]
    u_y = u_out[..., 1:2]

    v_x = v_out[..., 0:1]
    v_y = v_out[..., 1:2]

    p_x = p_out[..., 0:1]
    p_y = p_out[..., 1:2]
    
    # Second Derivatives
    u_xx = torch.autograd.grad(u_x.sum(), model_input_coords, create_graph=True)[0][..., 0:1]
    u_yy = torch.autograd.grad(u_y.sum(), model_input_coords, create_graph=True)[0][..., 1:2]
    v_xx = torch.autograd.grad(v_x.sum(), model_input_coords, create_graph=True)[0][..., 0:1]
    v_yy = torch.autograd.grad(v_y.sum(), model_input_coords, create_graph=True)[0][..., 1:2]

    # Continuity equation
    f0 = u_x + v_y

    # Navier-Stokes equation
    f1 = u*u_x + v*u_y - (1/Re) * (u_xx + u_yy) + p_x
    f2 = u*v_x + v*v_y - (1/Re) * (v_xx + v_yy) + p_y

    mse_f0 = torch.mean(torch.square(f0))
    mse_f1 = torch.mean(torch.square(f1))
    mse_f2 = torch.mean(torch.square(f2))
    return mse_f0, mse_f1, mse_f2


# If we want to use something like LBFGS as an optimizer, we need to create a closure function
# NOTE: This closure function cannot in itself take any inputs and returns only one loss!
# IDEAS:
#   1. It looks like if we want to do general training, all batches need to be iterated within this loop
#   2. Alternatively we could use this just for fine-tuning/zero-shot
#      In which case it would be better to set up a class with a closure function
def closure():
    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GNOT Training Study')
    parser.add_argument('--name', type=str)
    parser.add_argument('--subsample', type=int, default=8)
    parser.add_argument('--weight_l2', type=int, default=5)
    parser.add_argument('--weight_pinn', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--fine_tuning', type=bool, default=False)
    parser.add_argument('--weight_balance', type=str, default=None, choices=[None,'dynamic','scaled'])
    args = parser.parse_args()

    model_args = dict()
    dataset_args = dict()
    training_args = dict()
    
    # 1. Prepare Data
    dataset_args['file']                    = r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\GNOT\data\steady_cavity_case_b200_maxU100ms_simple_normalized.npy'
    dataset_args['percent split (decimal)'] = 0.7
    dataset_args['randomizer seed']         = 42
    dataset_args['use-normalizer']          = 'unit'
    dataset_args['normalize_x']             = 'unit'
    dataset_args['subsampler']              = args.subsample
    dataset_args['cell to pointwise']       = True
    dataset_args['add boundaries']          = True

    if args.fine_tuning:
        __, dataset, y_normalizer = get_dataset_file(dataset_args)
    else:
        dataset, eval_dataset, y_normalizer = get_dataset_file(dataset_args)
    
    # 2. Model Hyper-Parameters
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

    # 3. Training Settings
    training_args['epochs']                 = 1
    training_args["save_dir"]               = 'gnot_cavity_v5_collab'
    training_args["save_name"]              = args.name
    training_args['eval_while_training']    = True
    training_args['milestones']             = None
    training_args['base_lr']                = 0.001
    training_args['lr_method']              = 'cycle'
    training_args['scheduler_gamma']        = None  
    training_args['xy_loss']                = 0.0
    training_args['pde_loss']               = 1.0
    training_args['weight-decay']           = 0.00005
    training_args['grad-clip']              = 1000.0    
    training_args['batchsize']              = 4
    training_args['component']              = 'all'
    training_args['normalizer']             = None
    #training_args['loss_weighting']         = 'dynamic' #'scaled' #scaled #scaled, dynamic or none
    #training_args['ckpt']                   = r'C:\Users\Noahc\Documents\USYD\PHD\0 - Work Space\analytics_v2_GNOT\google colab results\dynamic_sub_4.pt'
    training_args['fine_tuning']            = args.fine_tuning
    training_args['boundaries']             = 'hard'

    # Override any duplicate settings
    # if training_args['fine_tuning']: 
    #     if 'ckpt' not in training_args:
    #         raise ValueError('Can not fine-tune without Checkpoint')

    #     ckpt_path = training_args['ckpt']
    #     ckpt = torch.load(ckpt_path, map_location=device)
    #     model.load_state_dict(ckpt['model'])
    #     print('Weights loaded from %s' % ckpt_path)

    #     training_args['xy_loss']            = 0.0
    #     training_args['batchsize']          = 1
    #     training_args['eval_while_training']= False

    # Initialize Results Storage:
    training_run_results = total_model_dict(model_config=model_args, training_config=training_args, data_config=dataset_args)

    # Create Model:
    model = get_model(model_args)

    # TRAINING ROUTINE:
    training_dataloader = MIODataLoader(dataset, batch_size=training_args['batchsize'], shuffle=True, drop_last=False)
    
    if training_args['eval_while_training']:
        testing_dataloader = MIODataLoader(eval_dataset, batch_size=training_args['batchsize'], shuffle=True, drop_last=False)

    # 2. Training optimizer and learning rate scheduler
    if 'base_lr' in training_args:
        #optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=args['base_lr'])
        optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=training_args['base_lr'], weight_decay=training_args['weight-decay'])

    if training_args['lr_method'] == 'cycle':
        print('Using cycle learning rate schedule')
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=training_args['base_lr'], div_factor=1e4, pct_start=0.2, final_div_factor=1e4, steps_per_epoch=len(training_dataloader), epochs=training_args['epochs'])
    elif training_args['lr_method'] == 'step':
        print('Using step learning rate schedule')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training_args['lr_step_size']*len(training_dataloader), gamma=0.7)
    elif training_args['lr_method'] == 'warmup':
        print('Using warmup learning rate schedule')
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1)/(training_args['warmup_epochs'] * len(training_dataloader)), np.power(training_args['warmup_epochs'] * len(training_dataloader)/float(steps + 1), 0.5)))

    loss_func = WeightedLpLoss(p=2,component=training_args['component'], normalizer=training_args['normalizer'])
    loss_function_no_graph = LpLoss()

    print(f'Commencing training with {len(training_dataloader)} batches of size {training_args["batchsize"]}')

    for epoch in range(training_args['epochs']): 
        epoch_start_time = default_timer()

        # Set Model to Train
        model.train()
        torch.cuda.empty_cache()

        # Initialize loss storage per epoch
        loss_total_list             = list()
        loss_total_eval_list        = list()
        pde_l1_list                 = list()
        pde_l2_list                 = list()
        pde_l3_list                 = list()
        total_weighted_loss_list    = list()

        # 5. Train all batches
        for batch_n, batch in enumerate(training_dataloader):
            optimizer.zero_grad()
            batch_start_time = default_timer()
            
            g, u_p, g_u = batch
            g, u_p, g_u = g.to(device), u_p.to(device), g_u.to(device)

            # Turn on Autograd 
            g.ndata['x'].requires_grad = True

            # 5.1. Forward Pass in Model
            out = model(g, u_p, g_u)
            y_pred, y = out.squeeze(), g.ndata['y'].squeeze()
            
            # 5.2. Calculate Loss for Backwards Pass
            loss, reg,  _ = loss_func(g, y_pred, y)
            loss_total = loss + reg
            
            # 5.3. PDE Loss
            L = 0.1
            nu = 0.01
            Re = g_u[0] * L/nu
            pde_l1, pde_l2, pde_l3 = pde_autograd(g.ndata['x'], 
                                                        out, 
                                                        Re=Re, 
                                                        y_normalizer=y_normalizer, 
                                                        hard_enforce_bc=False)

            # 5.4 Total Loss
            total_avg_pde_loss = (pde_l1 + pde_l2 + pde_l3)/3
            total_weighted_loss = training_args['xy_loss']*loss_total.to(device) + training_args['pde_loss'] *total_avg_pde_loss.to(device)

            pde_l1_list.append(pde_l1.item())
            pde_l2_list.append(pde_l2.item())
            pde_l3_list.append(pde_l3.item())
            loss_total_list.append(loss_total.item())
            total_weighted_loss_list.append(total_weighted_loss.item())
            
            # 5.4. Backwards Pass to Optimizing and Scheduling
            total_weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_args['grad-clip'])
            
            optimizer.step()
            scheduler.step()

            batch_end_time = default_timer()

            print(f"\r{epoch} Weighted Loss: {total_weighted_loss.item():.3e} "+
                  f"L2 Loss: {loss_total.item():.3e}"+
                  f"Weighted PDE Loss: {total_avg_pde_loss.item():.3e} "+
                  f"PDE 1: {pde_l1.item():.3e} "+
                  f"PDE 2: {pde_l2.item():.3e} "+
                  f"PDE 3: {pde_l3.item():.3e} ",
                end="")
            if epoch+1 % 500 == 0: print("")
            
            if training_args['fine_tuning']: break

        epoch_end_time = default_timer()

        # 6. Validate Model
        if args['eval_while_training']:
            model.eval()
            for batch_n, batch in enumerate(testing_dataloader):
                
                g_eval, u_p_eval, g_u_eval = batch
                g_eval, u_p_eval, g_u_eval = g_eval.to(device), u_p_eval.to(device), g_u_eval.to(device)

                # 6.1. Forward Pass in Model
                out_eval = model(g_eval, u_p_eval, g_u_eval)
                y_pred_eval, y_eval = out_eval.squeeze(), g_eval.ndata['y'].squeeze()
                
                # 6.2. Calculate Loss for Backwards Pass
                loss_eval, reg_eval,  _ = loss_func(g_eval, y_pred_eval, y_eval)
                loss_total_eval_list.append((loss_eval + reg_eval).item())

            training_run_results.update_statistics(loss_total_eval_list, 'Evaluation L2 Loss')

        # 7.1 Calculate and Store Statistics
        training_run_results.update_loss({'Epoch Time': epoch_end_time - epoch_start_time})
        training_run_results.update_statistics(total_weighted_loss_list, 'Total Weighted Loss')
        training_run_results.update_statistics(loss_total_list, 'Training L2 Loss')
        
        training_run_results.update_statistics(pde_l1_list , 'X-Momentum Loss')
        training_run_results.update_statistics(pde_l2_list , 'Y-Momentum Loss')
        training_run_results.update_statistics(pde_l3_list , 'Continuity Loss')

    save_checkpoint(args["save_dir"], args["save_name"], model=model, loss_dict=training_run_results.dictionary, optimizer=optimizer)
    