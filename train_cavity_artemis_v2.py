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
from accelerate import Accelerator
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Cavity_2D_dataset_for_GNOT():
    def __init__(self, 
                 data_path, 
                 L=1.0, 
                 sub_x = 1, 
                 normalize_y=False, 
                 normalize_x = False, 
                 vertex = False, 
                 boundaries = False):

        print('\n Creating Dataset:')
        # Normalizer settings:
        self.normalize_y = normalize_y
        self.normalize_x = normalize_x
        self.sub_x = sub_x
        self.vertex = vertex
        self.boundaries = boundaries

        # Load in Dataset and apply transforms
        self.data_out   = np.load(data_path)
        if self.sub_x > 1: 
            self.subsampler()
        if self.vertex: 
            self.cell_to_vertex_converter()
        if self.boundaries: 
            self.add_boundaries()

        # Dataset shapes
        self.n_batches  = self.data_out.shape[0]
        self.nx         = int(self.data_out.shape[1])
        self.num_nodes  = self.nx**2

        # Flatten 2D to 1D
        self.data_out = torch.tensor(self.data_out).reshape(self.n_batches,self.num_nodes,3)

        # Input Functions (Lid Velocities)
        self.data_lid_v = torch.tensor(np.round(np.arange(0.5,100.5,0.5),1))# * 0.1/0.01 #<- apply for Reynolds Number

        # Input Queries (Coordinates)
        self.queries = self.get_query_grid(L=L)

        # Get Normalizers
        if self.normalize_x:
            self.query_normalizer = self.create_a_normalizer(self.queries)
            self.queries = self.query_normalizer.transform(self.queries, inverse=False)
            print(f'    Queries Normalized with Means: {self.query_normalizer.mean} and Stds: {self.query_normalizer.std}')
            
        if self.normalize_y:
            self.output_normalizer = self.create_a_normalizer(self.data_out)
            self.data_out = self.output_normalizer.transform(self.data_out, inverse=False)
            self.input_f_normalizer = UnitTransformer(self.data_lid_v)
            self.data_lid_v = self.input_f_normalizer.transform(self.data_lid_v, inverse=False)
            print(f'    Datset Normalized with Means: {self.output_normalizer.mean} and Stds: {self.output_normalizer.std}')
            
        self.__update_dataset_config()

    def subsampler(self):
        self.data_out = torch.nn.functional.avg_pool2d(torch.tensor(self.data_out).permute(0,3,1,2), self.sub_x).permute(0,2,3,1).numpy()
        print(f'    Dataset Subsampled by factor of {self.sub_x}')

    def cell_to_vertex_converter(self):
        self.data_out = torch.nn.functional.avg_pool2d(torch.tensor(self.data_out).permute(0,3,1,2),2,stride=1).permute(0,2,3,1).numpy()
        print('     Cell Centres Converted to Cell Verticies using average pooling (this reduces the size of each dimension by one)')

    def add_boundaries(self):
        self.data_out = torch.nn.functional.pad(torch.tensor(self.data_out).permute(0,3,1,2),(1, 1, 1, 1)).permute(0,2,3,1).numpy()

        # Lid Velocity
        self.data_out[:,-1 ,:,0] = 1

        # Pressure
        self.data_out[:,  0 ,1:-1, 2] = self.data_out[:,  1 ,1:-1, 2]  # Bottom Wall
        self.data_out[:, -1 ,1:-1, 2] = self.data_out[:, -2 ,1:-1, 2]  # Lid (y-vel)
        self.data_out[:,1:-1,  0 , 2] = self.data_out[:,1:-1,  1 , 2]  # Left Wall
        self.data_out[:,1:-1, -1 , 2] = self.data_out[:,1:-1, -2 , 2]  # Right Wall
        print('     Boundary Data added by padding and adding extra cells (this increases the size of each dimension by two)')    

    def get_query_grid(self, L):
    
        # Get grid coordinates based on vertx/volume boundary/no boundary
        divisor = self.nx - 2*int(self.boundaries) + 1*int(self.vertex)
        dx = L/divisor
        offset = dx/2 - dx*int(self.boundaries) + dx/2*int(self.vertex)
        x = torch.arange(self.nx)/divisor + offset
        y = x

        # take note of the indexing. Best for this to match the output
        [X, Y] = torch.meshgrid(x, y, indexing = 'ij')
        X = X.reshape(self.num_nodes,1)
        Y = Y.reshape(self.num_nodes,1)

        return torch.concat([Y,X],dim=-1)
    
    def create_a_normalizer(self,un_normalized_data):

        # Flatten
        n_channels = un_normalized_data.shape[-1]
        batches_and_nodes = torch.prod(torch.tensor(un_normalized_data.shape[:-1])).item()
        all_features = un_normalized_data.reshape(batches_and_nodes,n_channels)
        
        return UnitTransformer(all_features)

    def __update_dataset_config(self): 
        self.config = {
            'input_dim': self.queries.shape[-1],
            'theta_dim': 0,
            'output_dim': self.data_out.shape[-1],
            'branch_sizes': [1]
        }

class CavityDataset(Dataset):
    def __init__(self,dataset, train=True, inference=True, train_ratio=0.7, seed=42):
        print('\nCreating Dataloader:')
        self.in_queries = dataset.queries
        self.in_keys_all = dataset.data_lid_v
        self.out_truth_all = dataset.data_out
        self.train = train
        self.data_splitter(train,inference,train_ratio,seed)
        
    def data_splitter(self, train, inference, train_ratio, seed):
        n_batches = self.out_truth_all.shape[0]
        train_size = int(train_ratio * n_batches)
        test_size = n_batches - train_size

        if inference:
            seed_generator = torch.Generator().manual_seed(seed)

            train_split,  test_split        = torch.utils.data.random_split(self.out_truth_all, [train_size, test_size], generator=seed_generator)
        
            # The torch.utils.data.random_split() only gives objects with the whole datset or a integers, so we need to override these variables with the indexed datset split
            train_dataset,  test_dataset    = self.out_truth_all[train_split.indices,...], self.out_truth_all[test_split.indices,...]
            train_lid_v,    test_lid_v      = self.in_keys_all[train_split.indices], self.in_keys_all[test_split.indices]
            print(f'    Dataset Split up for inference using torch generator seed: {seed_generator.initial_seed()}')
        else:
            train_dataset,  test_dataset    = self.out_truth_all[:train_size,...],  self.out_truth_all[train_size:,...]
            train_lid_v,    test_lid_v      = self.in_keys_all[:train_size,...],     self.in_keys_all[train_size:,...]
            print(f'    Dataset Split up for High reynolds number extrapolation')

        if train:
            self.out_truth_all = train_dataset
            self.in_keys_all = train_lid_v
            print(f'    Training Datset Selected')
        else:
            self.out_truth_all = test_dataset
            self.in_keys_all = test_lid_v
            print(f'    Testing Datset Selected')

    def __len__(self):
        return len(self.in_keys_all)

    def __getitem__(self, idx):
        
        # randomize input coordinates
        if self.train:
            indexes = torch.randperm(self.in_queries.shape[0])
            in_queries  = self.in_queries[indexes,...].float()
            out_truth   = self.out_truth_all[idx,indexes,...].float()
        else:
            in_queries  = self.in_queries.float()
            out_truth   = self.out_truth_all[idx,...].float()

        in_keys     = self.in_keys_all[idx].float().reshape(1,1)
        
        return in_queries, in_keys, out_truth

def get_default_args():
    
    dataset_args = dict()
    dataset_args['L']           = 1.0
    dataset_args['sub_x']       = 8
    dataset_args['normalize_y'] = True
    dataset_args['normalize_x'] = True
    dataset_args['vertex']      = True
    dataset_args['boundaries']  = True
    dataset_args['train']       = True
    dataset_args['inference']   = True
    dataset_args['train_ratio'] = 0.7
    dataset_args['seed']        = 42

    model_args = dict()
    model_args['trunk_size']        = 2
    model_args['theta_size']        = 0
    model_args['branch_sizes']      = [1]
    model_args['output_size']       = 3
    model_args['n_layers']          = 3
    model_args['n_hidden']          = 128  
    model_args['n_head']            = 1
    model_args['attn_type']         = 'linear'
    model_args['ffn_dropout']       = 0.0
    model_args['attn_dropout']      = 0.0
    model_args['mlp_layers']        = 2
    model_args['act']               = 'gelu'
    model_args['hfourier_dim']      = 0

    training_args = dict()
    training_args['epochs']                 = 1
    training_args['base_lr']                = 0.001
    training_args['weight-decay']           = 0.00005
    training_args['grad-clip']              = 1000.0    
    training_args['batchsize']              = 4
    training_args["save_dir"]               = 'gnot_artemis'
    training_args["save_name"]              = 'test'
    training_args['warmup_epochs']          = 5

    return dataset_args, model_args, training_args
        
def get_cavity_dataset(args):

    dataset = Cavity_2D_dataset_for_GNOT(data_path=args['file_path'], 
                                        L=args['L'], 
                                        sub_x=args['sub_x'], 
                                        normalize_y=args['normalize_y'], 
                                        normalize_x=args['normalize_x'], 
                                        vertex=args['vertex'], 
                                        boundaries=args['boundaries']
                                        )
    torch_dataset = CavityDataset(dataset, 
                                    train=args['train'], 
                                    inference=args['inference'],
                                    train_ratio=args['train_ratio'], 
                                    seed=args['seed']
                                    )
    
    return torch_dataset
    
def get_model(args):
    model = CGPTNO(
                trunk_size          = args['trunk_size'] + args['theta_size'],
                branch_sizes        = args['branch_sizes'], 
                output_size         = args['output_size'],
                n_layers            = args['n_layers'],
                n_hidden            = args['n_hidden'],
                n_head              = args['n_head'],
                attn_type           = args['attn_type'],
                ffn_dropout         = args['ffn_dropout'],
                attn_dropout        = args['attn_dropout'],
                mlp_layers          = args['mlp_layers'],
                act                 = args['act'],
                horiz_fourier_dim   = args['hfourier_dim']
                )
    
    return model

def get_gpu_resources():
    if torch.cuda.is_available():
        print_string1 = '   Memory Reserved:'
        print_string2 = '   Memory Allocated:'
        for gpu_n in range(torch.cuda.device_count()):
            gpu_n_str = f"cuda:{gpu_n}"
            print_string1 += f'GPU{gpu_n} {torch.cuda.memory_reserved(torch.device(gpu_n_str)) / 1024**3:5.2f}GB '
            print_string2 += f'GPU{gpu_n} {torch.cuda.memory_allocated(torch.device(gpu_n_str)) / 1024**3:5.2f}GB '
    else:
        print_string1, print_string2 = '        No GPUs', '        No GPUs'

    return print_string1, print_string2

class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class LpLoss_custom(object):
    # Loss function is based on the DGL weighted loss function (only it is dgl package free)
    
    def __init__(self):
        super(LpLoss_custom, self).__init__()

    def avg_pool(self,input):
        #r^{(i)} = \frac{1}{N_i}\sum_{k=1}^{N_i} x^{(i)}_k
        batch_size = input.shape[0] # shape: Batch, Nodes, Channels
        num_nodes = input.shape[1]
        pooled_value = (1/num_nodes)*torch.sum(input,dim=1)

        return pooled_value
    
    def __call__(self, x, y):
        
        losses = (self.avg_pool(((x - y).abs() ** 2)) + 1e-8) ** (1 / 2)
        loss = losses.mean()
        return loss

if __name__ == '__main__':

    '''TODO 
        add model parameter and config saver
        figure out a way to submit multiple jobs with different args using bash

        idea. Bash takes in multiple indices and we state the configurations in here.
    '''

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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #accelerator = Accelerator()
    #device = accelerator.device
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
    training_args['base_lr']         = args.lr
    training_args['batchsize']         = args.batch_size

    # get cavity data prepared for model
    workers = torch.cuda.device_count()
    dataset = get_cavity_dataset(dataset_args)
    train_dataloader = DataLoader(dataset, batch_size=training_args['batchsize'], shuffle=True, num_workers = workers)  

    # also get testing datset
    dataset_args_eval = dataset_args
    dataset_args_eval['train'] = False
    dataset_eval = get_cavity_dataset(dataset_args_eval)
    eval_dataloader = DataLoader(dataset_eval, batch_size=training_args['batchsize'], shuffle=False, num_workers = workers)  
    print(f'\n Number of Minibatches: Training Dataset {len(train_dataloader)} and Evaluation Dataset Length {len(eval_dataloader)}')
    # get model and put in parallel
    model = get_model(model_args)
    model.to(device)

    #torch.distributed.init_process_group(backend='nccl')
    #model = torch.nn.parallel.DistributedDataParallelCPU(model)

    # in-built MSE loss function (not same as paper as this is dgl free)
    #loss_function = torch.nn.MSELoss(reduction='mean')
    #loss_function = LpLoss()
    loss_function = LpLoss_custom()
    
    # Default optimizer and scheduler from paper
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
                                                    steps_per_epoch=len(train_dataloader), 
                                                    epochs=training_args['epochs']
                                                    )

    #model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

    print(f'\nModel put in parallel processing with {torch.cuda.device_count()} GPUs')
    
    # Initialize Results Storage: 
    training_run_results = total_model_dict(model_config=model_args, training_config=training_args, data_config=dataset_args)

    mem_res, mem_aloc = get_gpu_resources()
    print(f'\nIntitial Memory before training after model initialization: \n{mem_res}\n{mem_aloc}')

    print('\nStarting Training')
    for epoch in range(training_args['epochs']):
        epoch_start_time = default_timer()

        # Set Model to Train
        model.train()
        torch.cuda.empty_cache()

        # Initialize loss storage per epoch
        loss_total_list = list()
        
        for batch_n, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            #in_queries, in_keys, out_truth = batch
            in_queries, in_keys, out_truth = in_queries.to(device), in_keys.to(device), out_truth.to(device)
            
            out = model(x=in_queries,inputs = in_keys)

            loss = loss_function(out,out_truth)

            loss.backward()#(retain_graph=True)
            #accelerator.backward(loss)

            if batch_n == (len(train_dataloader)-1) and epoch == 0: mem_res1, mem_aloc1 = get_gpu_resources()

            torch.nn.utils.clip_grad_norm_(model.parameters(), training_args['grad-clip'])
            optimizer.step()

            scheduler.step()

            if batch_n == (len(train_dataloader)-1) and epoch == 0: mem_res2, mem_aloc2 = get_gpu_resources()

            if batch_n == 0 and training_args['epochs'] == 1: break

        if epoch == 0 and torch.cuda.is_available(): 
            print(f'\n  Memory After Final Batch Backwards Pass: \n{mem_res1}\n{mem_aloc1}')
            print(f'  Memory After Final Batch Optimizer Pass: \n{mem_res2}\n{mem_aloc2}\n')


        epoch_end_time = default_timer()
        training_run_results.update_loss({'Epoch Time': epoch_end_time - epoch_start_time})
        training_run_results.update_loss({'Training L2 Loss': loss.item()})

        # Now lets evaluate the model at each epoch too
        model.eval()
        loss_eval = 0
        for batch_n, batch in enumerate(eval_dataloader):
            #in_queries, in_keys, out_truth = batch
            in_queries, in_keys, out_truth = in_queries.to(device), in_keys.to(device), out_truth.to(device)

            out = model(x=in_queries,inputs = in_keys)
            
            loss_eval += loss_function(out,out_truth).item()

        loss_eval = loss_eval/(batch_n+1)
        training_run_results.update_loss({'Evaluation L2 Loss': loss_eval})

        
        print(f'Epoch: {epoch :8} L2 Training Loss {loss :12.7f}, L2 Evaluation Loss: {loss_eval :12.7f}, Learning Rate: {scheduler.get_lr()}')

        sys.stdout.flush()
    if training_args['epochs'] != 1: 
        save_checkpoint(training_args["save_dir"], 
                        training_args["save_name"], 
                        model=model, 
                        loss_dict=training_run_results.dictionary, 
                        optimizer=optimizer)