import torch
import torch.nn.functional as F
import numpy as np
import sys
from dgl.data import DGLDataset
import dgl
from timeit import default_timer
import argparse

from data_utils import MultipleTensors, MIODataLoader, get_dataset
from models.cgpt import CGPTNO
from data_utils import WeightedLpRelLoss

sys.path.append(r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\Lasso')
from training_utils.optimizers import Adam
from training_utils.loss_functions_2 import LpLoss, loss_selector
from data_handling.data_utils import load_dataset, sample_data, total_loss_list
from training_utils.save_checkpoint import save_checkpoint
from training_utils.train_cavity import Navier_Stokes_FDM_cavity_internal
    
class Cavity_2D_dataset_handling(DGLDataset):
    def __init__(self, data_path, L=0.1, name=' ', train=True):

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
        cavity_data     = np.load(data_path)
        self.data_out   = cavity_data[:, 1:-1, 1:-1, 1,:]
        self.data_lid_v = cavity_data[:,0,0,0,0]
        self.n_batches  = self.data_out.shape[0]
        self.nx         = self.data_out.shape[1]
        self.num_nodes  = self.nx**2

        self.L = L
        self.train = train

        super(model_routine_sphere, self).__init__(name)   # invoke super method after read data


    def process(self):
        
        # SECTION 0: Split into train or test (Same as for FNO training)
        train_size = int(0.7 * self.n_batches)
        test_size = self.n_batches - train_size

        train_dataset, test_dataset = torch.split(torch.from_numpy(self.data_out), [train_size, test_size])
        train_lid_v, test_lid_v     = torch.split(torch.from_numpy(self.data_lid_v), [train_size, test_size])
        if self.train:
            self.data_out = train_dataset
            self.data_lid_v = train_lid_v
            self.n_batches = train_size
        else:
            self.data_out = test_dataset
            self.data_lid_v = test_lid_v
            self.n_batches = test_size
        
        # SECTION 1: Transformer Queries
        # Assume Isotropic Grid 
        x = torch.arange(self.nx)*self.L/self.nx
        y = x

        # take note of the indexing. Best for this to match the output
        [X, Y] = torch.meshgrid(x, y, indexing = 'ij')

        X = X.reshape(self.num_nodes,1)
        Y = Y.reshape(self.num_nodes,1)

        # we need to linearize these matrices.
        self.X_for_queries = torch.concat([Y,X],dim=-1)
        print('Queries', self.X_for_queries.shape, 'Coordinates', X.shape)
        
        # SECTION 3: Transform to be MIOdataset Loader Compatible
        self.MIOdataset_structure()
        self.__update_dataset_config()


    def MIOdataset_structure(self):
        '''
        NOTE here for FNO, X is constant because the grid is the same for all batches
        Theta is [0] because it requires a non-null input in the current model config
        Output (Y) and initial conditions (g_u) change per batch)
        g_u is the input condition in tuple structure as the model can accept multiple input functions.
        '''

        # Query Coordinates are the same so we keep these constant:
        x = self.X_for_queries
        input_f = None
        
        # initialize storage
        self.graphs = []
        self.inputs_f = []
        self.u_p_list = []

        for i in range(self.n_batches):

            y = self.data_out[i,...].reshape(self.num_nodes,3)
            u_p = self.data_lid_v[[i]]

            # Construct Graph object
            g = dgl.DGLGraph()
            g.add_nodes(x.shape[0])
            g.ndata['x'] = x.float()
            g.ndata['y'] = y.float()
            up = u_p.float()
            self.graphs.append(g)
            self.u_p_list.append(up) # global input parameters
            if input_f is not None:
                input_f = MultipleTensors([torch.from_numpy(f).float() for f in input_f])
                self.inputs_f.append(input_f)
                num_inputs = len(input_f)
            
            #if len(inputs_f) == 0:
            #    inputs_f = torch.zeros([self.n_batches])  # pad values, tensor of 0, not list

        self.u_p_list = torch.stack(self.u_p_list)

    def __update_dataset_config(self):
        self.config = {
            'input_dim': self.graphs[0].ndata['x'].shape[1],
            'theta_dim': self.u_p_list.shape[1],
            'output_dim': self.graphs[0].ndata['y'].shape[1],
            'branch_sizes': 0 #[x.shape[1] for x in self.inputs_f[0]] if isinstance(self.inputs_f, list) else 0
        }

    def __getitem__(self, idx):
        return self.graphs[idx], self.u_p_list[idx]#, self.inputs_f[idx]
    
    def __len__(self):
        return self.n_batches

def model_routine_sphere(args, model, training_dataset, testing_dataset=None):
    
    # 0. Print training configurations
    device = next(model.parameters()).device

    # 1. Data and training parameters
    training_dataloader = MIODataLoader(training_dataset, batch_size=args['batchsize'], shuffle=True, drop_last=False)
    if args['eval_while_training']:
        testing_dataloader = MIODataLoader(testing_dataset, batch_size=args['batchsize'], shuffle=False, drop_last=False)
    
    xy_weight = args['xy_loss']
    epochs = args['epochs']

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
    
    # 3.1. Initialize Loss Function (from FNO) and Recording Dictionary
    loss_function_FNO = loss_selector(args['fno_loss_type'])
    total_loss_dictionary = total_loss_list()
    epoch_end_time = 0

    # 3.2. Initialize Loss Function (From GNOT) for backwards Pass
    loss_func = WeightedLpRelLoss(p=2,component=args['component'], normalizer=args['normalizer'])
    # 4. Run Model Routine (Training)  
    model.train()

    print(f'Commencing training with {len(training_dataloader)} batches of size {args["batchsize"]}')

    for epoch in range(epochs): 
        epoch_start_time = default_timer()

        # Set Model to Train
        model.train()
        torch.cuda.empty_cache()

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
            #y_pred  = y_pred.reshape(len(u_p), training_dataset.nx, training_dataset.nx, 3)
            #y       = y.reshape(len(u_p), training_dataset.nx, training_dataset.nx, 3)
            
            #loss_l2 = loss_function_FNO(y_pred, y)
            #loss_mse = F.mse_loss(y_pred, y)

            #loss_pde,__ = Navier_Stokes_FDM_cavity_internal(y_pred, u_p.squeeze(-1), nu=0.01)
            
            # 5.4. Backwards Pass to Optimizing and Scheduling
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['grad-clip'])
            optimizer.step()

            scheduler.step()
            batch_end_time = default_timer()
        
            # Print to Console:
            print(f'Current Epoch: {epoch}, Current Batch: {batch_n + 1}, Last Training Loss: {loss_total.item()}, Estimated Time Left {(batch_end_time - batch_start_time)*(epochs-epoch+1)*(len(training_dataloader)-batch_n+1)/3600:.2f}hrs', end='\r')
            
            if epochs == 1 and batch_n == 1: break # debug control

        epoch_end_time = default_timer()

        # 5.5. Store Losses to Dictionary
        total_loss_dictionary.update({'Epoch Time': epoch_end_time - epoch_start_time})
        total_loss_dictionary.update({'Total Weighted Loss': loss_total.item()})
        total_loss_dictionary.update({'GNOT Loss Type': loss.item()})
        total_loss_dictionary.update({'GNOT Loss Type Regression': reg.item()})

        #total_loss_dictionary.update({'FNO Loss Type (LP Loss)': loss_l2.item()})
        #total_loss_dictionary.update({'FNO Loss Type (MSE Loss)': loss_mse.item()})
        #total_loss_dictionary.update({'Internal PDE Loss': loss_pde.item()})

        # 6. Validate Model
        if args['eval_while_training']:
        
            model.eval()
            
            for batch_n, batch in enumerate(testing_dataloader):
                
                g_eval, u_p_eval, g_u_eval = batch
                g_eval, u_p_eval, g_u_eval = g_eval.to(device), u_p_eval.to(device), g_u_eval.to(device)

                # 5.1. Forward Pass in Model
                out_eval = model(g_eval, u_p_eval, g_u_eval)
                y_pred_eval, y_eval = out_eval.squeeze(), g_eval.ndata['y'].squeeze()
                
                # 5.2. Calculate Loss for Backwards Pass
                loss, reg,  _ = loss_func(g_eval, y_pred_eval, y_eval)
                loss_total = loss + reg
                
                # 5.2. Calculate FNO Loss for Monitoring
                #y_pred_eval  = y_pred.reshape(len(u_p_eval), testing_dataset.nx, testing_dataset.nx, 3)
                #y_eval       = y.reshape(len(u_p_eval), testing_dataset.nx, testing_dataset.nx, 3)
                #loss_l2 = loss_function_FNO(y_pred_eval, y_eval)
                
            # 6.8 Calculate the Average Loss and Store to Dictionary
            loss_total = loss_total/(batch_n+1)
            #loss_total_fno = loss_l2/(batch_n+1)
            total_loss_dictionary.update({'Average Validation Loss (GNOT Type)': loss_total.item()})
            #total_loss_dictionary.update({'Average Validation Loss (FNO Type)': loss_total_fno.item()})
        
        #______________________________________________________________________________________________________________________

    # 7. Save Model Checkpoint and Losses
    save_checkpoint(args["save_dir"], args["save_name"], model, 
                loss_dict=total_loss_dictionary.fetch_list(), optimizer=optimizer),
                #input_sample=u_p, output_sample=y_pred, ground_truth_sample=y)
    
    # 8. Save Evaluation
    # if args['eval_while_training']:
    #     save_checkpoint(args["save_dir"], args["save_name"]+'eval',
    #                     input_sample=u_p_eval, output_sample=y_pred_eval, ground_truth_sample=y_eval)
    
    print('Model Routine Complete with final training Loss: ', loss_total.item())
    return

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 0. Configs
    dataset_path = r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\PINO datasets\sample_steady_cavity_case_b200_maxU100ms1d_t0.npy'

    # Model Parameters (Commented out are the ones for 2dNS with objects and steady state)
    output_size     = 3         #default is 3 (U,V and Pressure)
    n_layers        = 3
    n_hidden        = 128
    n_head          = 1         #default
    attn_type       = 'linear'  #default
    ffn_dropout     = 0.0       #default
    attn_dropout    = 0.0       #default
    mlp_layers      = 2         #default
    act             = 'gelu'    #default
    hfourier_dim    = 0         #default
    
    # 1. Prepare Data
    #dataset = Cavity_2D_dataset_handling(dataset_path, name='cavity', train=True)
    #eval_dataset = Cavity_2D_dataset_handling(dataset_path, name='cavity_eval', train=False)
    
    spheres_args = argparse.ArgumentParser(description='Spheres arguments')
    spheres_args.dataset        = 'ns2d'
    spheres_args.train_num      = 'all'
    spheres_args.test_num       = 'all'
    spheres_args.sort_data      = 0
    spheres_args.use_normalizer = 'unit'
    spheres_args.normalize_x    = 'unit'
    
    dataset, eval_dataset = get_dataset(spheres_args)

    # Data Specific Model Parameters
    trunk_size      = dataset.config['input_dim']
    theta_size      = dataset.config['theta_dim']
    branch_sizes    = dataset.config['branch_sizes']

    # 2. Create model and load in model version
    model = CGPTNO(trunk_size=trunk_size + theta_size,
                branch_sizes=branch_sizes,         # No input function means no branches
                output_size=output_size,
                n_layers=n_layers,                  # from args
                n_hidden=n_hidden,                  # from args
                n_head=n_head,                      # from args
                attn_type=attn_type,                # from args
                ffn_dropout=ffn_dropout,            # from args
                attn_dropout=attn_dropout,          # from args
                mlp_layers=mlp_layers,              # from args
                act=act,                            # from args
                horiz_fourier_dim=hfourier_dim      # from args
                ).to(device)

    # 3. Assign Training Parameters
    training_args = dict()
    training_args['epochs']                 = 110
    training_args["save_dir"]               = 'gnot_sphere'
    training_args["save_name"]              = 'attempt_1'
    training_args['eval_while_training']    = True
    training_args['milestones']             = None
    training_args['fno_loss_type']          = 'LPLoss Relative'
    training_args['base_lr']                = 0.001
    training_args['lr_method']              = 'cycle'
    training_args['scheduler_gamma']        = None  
    training_args['xy_loss']                = 1.0
    training_args['weight-decay']           = 0.00005
    training_args['grad-clip']              = 1000.0    
    training_args['use-normalizer']         = 'unit'
    training_args['batchsize']              = 4
    training_args['component']              = 'all'
    training_args['normalizer']             = None
    
    # 4. Run scripts based on configurations
    model_routine_sphere(training_args, model, training_dataset=dataset, testing_dataset=eval_dataset)
    print('Training Complete \n\n')

    ## TRAINING IS NOW READY TO LAUNCH. IT MAY TAKE SOME TIME HOWEVER, for 500 epochs