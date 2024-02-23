import sys
import numpy as np

import torch
import torch.nn.functional as F

import dgl
from dgl.data import DGLDataset

from timeit import default_timer

from data_utils import MultipleTensors, MIODataLoader, get_dataset, WeightedLpRelLoss, WeightedLpLoss, LpLoss
from models.cgpt import CGPTNO
from utils import UnitTransformer
from data_storage.loss_recording import total_model_dict, save_checkpoint
from data_storage.cavity_2d_data_handling import Cavity_2D_dataset_handling_v2, NS_FDM_cavity_internal_vertex_non_dim, NS_FDM_cavity_internal_cell_non_dim

def console_printer(device, epoch, batch_n, batch_end_time,  batch_start_time, training_dataloader, args, epochs, pde_weights, lid_velocity, training_loss, validation_loss):
    # Print to Console:
    if (epoch == epochs-1 and batch_n == len(training_dataloader)-1) or (epoch == 0 and args['fine_tuning']): print_end = '\n\n'
    if device == 'cuda:0':
        print_end = '\n'
    else: print_end = '\r'
    
    print_end = '\r'
    print(f'Epoch: {epoch}, Batch: {batch_n + 1}, ' + \
            f'Mean L2 Training Loss: {training_loss:.3f}, ' + \
            f'Mean L2 Validation Loss: {validation_loss}, ' + \
            f'Estimated Time Left {(batch_end_time - batch_start_time)*(epochs-epoch+1)*(len(training_dataloader)-batch_n+1)/3600:.2f}hrs ' + \
            f'with PDE Loss Weights {pde_weights[0].item():.3f}, {pde_weights[1].item():.3f}, {pde_weights[2].item():.3f} ' + \
            f'and lid velocities {lid_velocity:.1f} ms-1    '
            , end=print_end)

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

class PDE_weights():
    def __init__(self, device, type = None, LR = 0.005, alpha = 0.16):
        
        self.type = type
        self.device = device

        if self.type == 'dynamic':
            self.w1 = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
            self.w2 = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
            self.w3 = torch.tensor(torch.FloatTensor([1]), requires_grad=True)
            self.params = [self.w1, self.w2, self.w3]
            self.optimizer2 = torch.optim.Adam(self.params, lr=LR)
            self.Gradloss = torch.nn.L1Loss()
        else:
            self.params = torch.tensor([1.0,1.0,1.0])

    def set_intial_loss(self, l01, l02, l03):
        if self.type == 'dynamic':
            self.l01 = l01.data  
            self.l02 = l02.data
            self.l03 = l03.data
        elif self.type == 'scaled':
            print('reset')
            self.params = torch.tensor(l01.item() + l02.item() + l03.item())/(3*torch.tensor([l01.item(), l02.item(), l03.item()]))
                    
    def calculate(self, model, model_layer, l1,l2,l3):
        if  self.type == 'dynamic':
            model_param = list(model.parameters())

            G1R = torch.autograd.grad(l1, model_param[model_layer], retain_graph=True, create_graph=True)
            G1 = torch.norm(G1R[0], 2)
            G2R = torch.autograd.grad(l2, model_param[model_layer], retain_graph=True, create_graph=True)
            G2 = torch.norm(G2R[0], 2)
            G3R = torch.autograd.grad(l3, model_param[model_layer], retain_graph=True, create_graph=True)
            G3 = torch.norm(G3R[0], 2)
            G_avg = ((G1+G2+G3)/3).to(self.device)

            # Calculating relative losses 
            lhat1 = torch.div(l1,self.l01)
            lhat2 = torch.div(l2,self.l02)
            lhat3 = torch.div(l3,self.l03)
            lhat_avg = (lhat1 + lhat2 + lhat3)/3

            # Calculating relative inverse training rates for tasks 
            inv_rate1 = torch.div(lhat1,lhat_avg).to(self.device)
            inv_rate2 = torch.div(lhat2,lhat_avg).to(self.device)
            inv_rate3 = torch.div(lhat3,lhat_avg).to(self.device)

            # Calculating the constant target for Eq. 2 in the GradNorm paper
            alph = 0.16
            C1 = G_avg*(inv_rate1)**alph
            C2 = G_avg*(inv_rate2)**alph
            C3 = G_avg*(inv_rate3)**alph
            C1 = C1.detach()
            C2 = C2.detach()
            C3 = C3.detach()

            self.optimizer2.zero_grad()
            self.Lgrad = (self.Gradloss(G1, C1) + self.Gradloss(G2, C2) + self.Gradloss(G3, C3))/3
            self.Lgrad.backward()

            # Update Model Optimizer and Scheduler
            self.optimizer2.step()

    def finalise(self):
        if  self.type == 'dynamic':
            self.coef = 3/(self.w1 + self.w2 + self.w3)
            self.params = [self.coef*self.w1, self.coef*self.w2, self.coef*self.w3]

    def __getitem__(self,item):
        return self.params[item]

def model_training_routine(device, model, args, training_dataset, testing_dataset, training_run_results):

    # 1. Data and training parameters
    if args['fine_tuning']:
        training_dataloader = MIODataLoader(testing_dataset, batch_size=args['batchsize'], shuffle=False, drop_last=False)
    else:
        training_dataloader = MIODataLoader(training_dataset, batch_size=args['batchsize'], shuffle=True, drop_last=False)
    
    if args['eval_while_training']:
        testing_dataloader = MIODataLoader(testing_dataset, batch_size=args['batchsize'], shuffle=False, drop_last=False)

    xy_weight = args['xy_loss']
    pde_weight = args['pde_loss']
    epochs = args['epochs']
    loss_eval = 'Not Available'
    pde_weights = PDE_weights(device, type=args['loss_weighting'])

    # 2. Training optimizer and learning rate scheduler
    if 'base_lr' in args:
        #optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=args['base_lr'])
        optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=args['base_lr'], weight_decay=args['weight-decay'])

    if args['lr_method'] == 'cycle':
        print('Using cycle learning rate schedule')
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args['base_lr'], div_factor=1e4, pct_start=0.2, final_div_factor=1e4, steps_per_epoch=len(training_dataloader), epochs=epochs)
    elif args['lr_method'] == 'step':
        print('Using step learning rate schedule')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['lr_step_size']*len(training_dataloader), gamma=0.7)
    elif args['lr_method'] == 'warmup':
        print('Using warmup learning rate schedule')
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1)/(args['warmup_epochs'] * len(training_dataloader)), np.power(args['warmup_epochs'] * len(training_dataloader)/float(steps + 1), 0.5)))
    elif args['lr_method'] == 'milestones':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['milestones'], gamma=args['scheduler_gamma'])

    loss_func = WeightedLpLoss(p=2,component=args['component'], normalizer=args['normalizer'])
    loss_function_no_graph = LpLoss()
    
    print(f'Commencing training with {len(training_dataloader)} batches of size {args["batchsize"]}')

    for epoch in range(epochs): 
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

            # 5.1. Forward Pass in Model
            out = model(g, u_p, g_u)
            y_pred, y = out.squeeze(), g.ndata['y'].squeeze()
            
            # 5.2. Calculate Loss for Backwards Pass
            loss, reg,  _ = loss_func(g, y_pred, y)
            loss_total = loss + reg
            
            # 5.3. Calculate Monitoring Losses (reshape too)
            y_pred  = y_pred.reshape(len(u_p), training_dataset.nx, training_dataset.nx, 3)
            y_pred=training_dataset.y_normalizer.transform(y_pred.to('cpu'),inverse=True)
                                                               
            # Hard Enforce Boundaries
            if args['boundaries'] == 'hard': y_pred = hard_enforce_boundaries(y_pred)

            if not training_dataset.vertex and not training_dataset.boundaries:
                Du_dx, Dv_dy, continuity_eq,__ = NS_FDM_cavity_internal_cell_non_dim(U=y_pred, 
                                                                                       lid_velocity=g_u[0].to('cpu'), 
                                                                                       nu=0.01, 
                                                                                       L=0.1)
            elif training_dataset.vertex and training_dataset.boundaries:
                Du_dx, Dv_dy, continuity_eq,__ = NS_FDM_cavity_internal_vertex_non_dim(U=y_pred, 
                                                                                       lid_velocity=g_u[0].to('cpu'), 
                                                                                       nu=0.01, 
                                                                                       L=0.1)
                
                # need to pad so that the number of nodes match. This method does affect the averaging slightly.
                # This also means we need to create a new graph for the boundary nodes if we want to use the same loss function.
                # Du_dx = torch.nn.functional.pad(Du_dx,(1, 1, 1, 1))
                # Dv_dy = torch.nn.functional.pad(Dv_dy,(1, 1, 1, 1))
                # continuity_eq = torch.nn.functional.pad(continuity_eq,(1, 1, 1, 1))

            else: raise ValueError('Dataset not organised correctly, choose either cell centre with no boundaries or opposite')

            #Du_dx, Dv_dy, continuity_eq = torch.flatten(Du_dx), torch.flatten(Dv_dy), torch.flatten(continuity_eq)
                
            pde_l1 = pde_weights[0]*loss_function_no_graph(Du_dx, torch.zeros_like(Du_dx))#[0]
            pde_l2 = pde_weights[1]*loss_function_no_graph(Dv_dy, torch.zeros_like(Dv_dy))#[0]
            pde_l3 = pde_weights[2]*loss_function_no_graph(continuity_eq, torch.zeros_like(continuity_eq))#[0]
            
            # pde_l1 = loss_function_no_graph(Du_dx, torch.zeros_like(Du_dx))
            # pde_l2 = loss_function_no_graph(Dv_dy, torch.zeros_like(Dv_dy))
            # pde_l3 = loss_function_no_graph(continuity_eq, torch.zeros_like(continuity_eq))
            if batch_n == 0 and epoch == 0: 
                pde_weights.set_intial_loss(pde_l1,pde_l2,pde_l3)

            #total_avg_pde_loss = torch.mean(torch.tensor([pde_l1, pde_l2, pde_l3], requires_grad=True))
            total_avg_pde_loss = (pde_l1 + pde_l2 + pde_l3)/3
            total_weighted_loss = xy_weight*loss_total.to(device) + pde_weight*total_avg_pde_loss.to(device)

            pde_l1_list.append(pde_l1.item()/pde_weights[0].item())
            pde_l2_list.append(pde_l2.item()/pde_weights[1].item())
            pde_l3_list.append(pde_l3.item()/pde_weights[2].item())
            loss_total_list.append(loss_total.item())
            total_weighted_loss_list.append(total_weighted_loss.item())
            
            #print(total_weighted_loss.item(), pde_l1.item(), pde_l2.item(), pde_l3.item(), pde_weights[0].item(), pde_weights[1].item(), pde_weights[2].item())

            # 5.4. Backwards Pass to Optimizing and Scheduling
           
            total_avg_pde_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['grad-clip'])

            pde_weights.calculate(model, model_layer=-1, l1=pde_l1, l2=pde_l2, l3=pde_l3)
            
            optimizer.step()
            scheduler.step()

            pde_weights.finalise()

            batch_end_time = default_timer()

            console_printer(device, epoch, batch_n, batch_end_time,  batch_start_time, training_dataloader, args, epochs, 
                            pde_weights = pde_weights, lid_velocity=g_u[0][0][0].item(), 
                            training_loss=loss_total.item(), validation_loss=loss_eval)
            
            if args['fine_tuning']: break

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

if __name__ == '__main__':

    # 0. Set Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    names = ['ansatz_dyn_hard_sub_8']#, 'ansatz_sub_4', 'ansatz_sub_2']
    subs = [8]#,4,2]

    for i in range(len(names)):
        model_args = dict()
        dataset_args = dict()
        training_args = dict()
        training_args["save_name"] = names[i]
        dataset_args['subsampler'] = subs[i]

        # 1. Prepare Data
        dataset_args['file']                    = r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\GNOT\data\steady_cavity_case_b200_maxU100ms_simple_normalized.npy'
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
        training_args['epochs']                 = 1000
        training_args["save_dir"]               = 'gnot_cavity_v5_collab'
        #training_args["save_name"]              = 'scaled_v1'
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
        training_args['loss_weighting']         = 'dynamic' #'scaled' #scaled #scaled, dynamic or none
        training_args['ckpt']                   = r'C:\Users\Noahc\Documents\USYD\PHD\0 - Work Space\analytics_v2_GNOT\google colab results\dynamic_sub_4.pt'
        training_args['fine_tuning']            = False
        training_args['boundaries']             = 'hard'

        # Override any duplicate settings
        if training_args['fine_tuning']: 
            if 'ckpt' not in training_args:
                raise ValueError('Can not fine-tune without Checkpoint')

            ckpt_path = training_args['ckpt']
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model'])
            print('Weights loaded from %s' % ckpt_path)

            training_args['xy_loss']            = 0.0
            training_args['batchsize']          = 1
            training_args['eval_while_training']= False

        # Initialize Results Storage: 
        training_run_results = total_model_dict(model_config=model_args, training_config=training_args, data_config=dataset_args)

        
        model_training_routine(device, model, training_args, dataset, eval_dataset, training_run_results)
        

