from models.noahs_model import CGPTNO
from dp_utils.data_utils import *
from timeit import default_timer
from argparse import ArgumentParser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser(description='GNOT Inference Study')
parser.add_argument('--iters'  , type=int  , default=1)
parser.add_argument('--res'    , type=int  , default=32)
parser.add_argument('--layers' , type=int  , default=3)
parser.add_argument('--dims'   , type=int  , default=2)
parser.add_argument('--linear_res', type=int  , default=128)
global ARGS 
ARGS = parser.parse_args()

# Get Args
__, model_args,__ = get_default_args()

# override
model_args
model_args['trunk_size']        = 2
model_args['theta_size']        = 0
model_args['branch_sizes']      = [1]
model_args['output_size']       = 3
model_args['n_layers']          = ARGS.layers
model_args['n_hidden']          = ARGS.linear_res  

model = get_model(model_args).to(device)
model.eval()
print('model_loaded and set to eval')

mean_time = 0

for i in range(ARGS.iters):
    torch.cuda.empty_cache()
    input_1 = torch.rand(1,ARGS.res**ARGS.dims,2).to(device)
    input_2 = torch.rand(1,1,1).to(device)

    start_time = default_timer()
    output = model(x=input_1, inputs=input_2)
    end_time = default_timer()

    mean_time += (end_time-start_time)

mean_time = mean_time/(ARGS.iters)

print(f'Mean Inference time (over {ARGS.iters} iterations) for Resolution ({ARGS.res}x{ARGS.res}): {mean_time:.6f}')