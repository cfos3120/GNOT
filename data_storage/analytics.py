import plotly.offline as py
py.init_notebook_mode(connected=False)
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
import plotly.figure_factory as ff
import numpy as np
import torch
from timeit import default_timer
from data_storage.cavity_2d_data_handling import Cavity_2D_dataset_handling_v2, NS_FDM_cavity_internal_vertex_non_dim
from models.cgpt import CGPTNO
from data_utils import MIODataLoader, WeightedLpLoss
from sklearn.metrics import r2_score

colors = plotly.colors.qualitative.Plotly

# Custom LpLoss
class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, p=2, size_average=True, reduction=True, relative=False):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert p > 0

        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.relative = relative

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)

        if self.relative:
            y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        else:
            y_norms = 1

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        
        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def reconstruct_dict(path):
    results = np.load(path, allow_pickle=True)
    dictionary = dict(results.flatten()[0])
    print(dictionary.keys())
    return dictionary

def plot_lp_losses(t1, t2=None,t3=None,t4=None, loss_names=['trace1']):
    
    # fixed parameters
    figure_1_name = 'Training L2 Loss'
    figure_2_name = 'Evaluation L2 Loss'


    if figure_2_name in t1.keys():
        plot_n = 2
    else: 
        plot_n = 1
    
    fig = make_subplots(rows=1,cols=plot_n, subplot_titles=('Training L2 Losses', 'Validation L2 Losses'))
    
    for i, result in enumerate([t1,t2,t3,t4]):
        
        if i == 0:
            first_bb = True
        else:
            first_bb = False

        if result is not None:
            
            fig.add_trace(go.Scatter(y = result[figure_1_name]['Mean'], mode="lines", name=loss_names[i], 
                                    legendgroup=i, showlegend=True, marker_color=colors[i]),1,1)
            
            # Add the Upper Bollinger Band (UB) and shade the area
            fig.add_trace(go.Scatter(y = np.array(result[figure_1_name]['Mean']) + np.array(result[figure_1_name]['Std Dev']), 
                                    mode='lines', showlegend=False, name='Bollinger Bands', legendgroup='BB', line=dict(width=0, color=colors[i])),1,1)
            
            fig.add_trace(go.Scatter(y = np.array(result[figure_1_name]['Mean']) - np.array(result[figure_1_name]['Std Dev']), 
                                    fill='tonexty', showlegend=first_bb, mode='lines', name='Bollinger Bands', legendgroup='BB', line=dict(width=0, color=colors[i])),1,1)
            
            if figure_2_name in result.keys():
                fig.add_trace(go.Scatter(y = result[figure_2_name]['Mean'], mode="lines", name=loss_names[i], 
                                    legendgroup=i, showlegend=False, marker_color=colors[i]),1,2)
            
                # Add the Upper Bollinger Band (UB) and shade the area
                fig.add_trace(go.Scatter(y = np.array(result[figure_2_name]['Mean']) + np.array(result[figure_2_name]['Std Dev']), 
                                        mode='lines', showlegend=False, name='Bollinger Bands', legendgroup='BB', line=dict(width=0, color=colors[i])),1,2)
                
                fig.add_trace(go.Scatter(y = np.array(result[figure_2_name]['Mean']) - np.array(result[figure_2_name]['Std Dev']), 
                                        fill='tonexty', showlegend=False, mode='lines', name='Bollinger Bands', legendgroup='BB', line=dict(width=0, color=colors[i])),1,2)
    
            fig.update_yaxes(title_text='Error', type="log")
            fig.update_xaxes(title_text='Epochs', range=[0, len(result[figure_1_name]['Mean'])-1])
            fig.update_layout(title_text="Training Losses for Cavity Flow Model Parameter Study")
    fig.show()

def plot_pde_losses(t1, t2=None,t3=None,t4=None, loss_names=['trace1']):
    
    # fixed parameters
    figure_1_name = 'X-Momentum Loss'
    figure_2_name = 'Y-Momentum Loss'
    figure_3_name = 'Continuity Loss'

    bolliner_line = 0.5
    
    fig = make_subplots(rows=1,cols=3, subplot_titles=('X-Momentum L2 Loss', 'Y-Momentum L2 Loss', 'Continuity L2 Loss'))
    
    for i, result in enumerate([t1,t2,t3,t4]):
        
        if i == 0:
            first_bb = True
        else:
            first_bb = False

        if result is not None:
            
            # Figure 1
            fig.add_trace(go.Scatter(y = result[figure_1_name]['Mean'], mode="lines", name=loss_names[i], 
                                    legendgroup=i, showlegend=True, marker_color=colors[i]),1,1)
            
            # Add the Upper Bollinger Band (UB) and shade the area
            fig.add_trace(go.Scatter(y = np.array(result[figure_1_name]['Mean']) + np.array(result[figure_1_name]['Std Dev']), 
                                    mode='lines', showlegend=False, name='Bollinger Bands', legendgroup='BB', line=dict(width=bolliner_line, color=colors[i])),1,1)
            
            fig.add_trace(go.Scatter(y = np.array(result[figure_1_name]['Mean']) - np.array(result[figure_1_name]['Std Dev']), 
                                    fill='tonexty', showlegend=first_bb, mode='lines', name='Bollinger Bands', legendgroup='BB', line=dict(width=bolliner_line, color=colors[i])),1,1)
            
            # Figure 2
            fig.add_trace(go.Scatter(y = result[figure_2_name]['Mean'], mode="lines", name=loss_names[i], 
                                    legendgroup=i, showlegend=False, marker_color=colors[i]),1,2)
            
            # Add the Upper Bollinger Band (UB) and shade the area
            fig.add_trace(go.Scatter(y = np.array(result[figure_2_name]['Mean']) + np.array(result[figure_2_name]['Std Dev']), 
                                    mode='lines', showlegend=False, name='Bollinger Bands', legendgroup='BB', line=dict(width=bolliner_line, color=colors[i])),1,2)
            
            fig.add_trace(go.Scatter(y = np.array(result[figure_2_name]['Mean']) - np.array(result[figure_2_name]['Std Dev']), 
                                    fill='tonexty', showlegend=False, mode='lines', name='Bollinger Bands', legendgroup='BB', line=dict(width=bolliner_line, color=colors[i])),1,2)
            
            # Figure 3
            fig.add_trace(go.Scatter(y = result[figure_3_name]['Mean'], mode="lines", name=loss_names[i], 
                                    legendgroup=i, showlegend=False, marker_color=colors[i]),1,3)
            
            # Add the Upper Bollinger Band (UB) and shade the area
            fig.add_trace(go.Scatter(y = np.array(result[figure_3_name]['Mean']) + np.array(result[figure_3_name]['Std Dev']), 
                                    mode='lines', showlegend=False, name='Bollinger Bands', legendgroup='BB', line=dict(width=bolliner_line, color=colors[i])),1,3)
            
            fig.add_trace(go.Scatter(y = np.array(result[figure_3_name]['Mean']) - np.array(result[figure_3_name]['Std Dev']), 
                                    fill='tonexty', showlegend=False, mode='lines', name='Bollinger Bands', legendgroup='BB', line=dict(width=bolliner_line, color=colors[i])),1,3)
            
    
            fig.update_yaxes(title_text='Error', type="log")
            fig.update_xaxes(title_text='Epochs', range=[0, len(result[figure_1_name]['Mean'])-1])
            fig.update_layout(title_text="Training PDE Losses for Cavity Flow Model Parameter Study")
    fig.show()

def evaluate_model(t1, checkpoint_path = None, dataset_type = 'eval', super_res = 1):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load Data and Model info
    dataset_args    = t1['Data Configuration']
    model_args      = t1['Model Configuration']
    
    # Create model
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

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % checkpoint_path)

    # Load in Data
    # Override if location was originally from Google Drive:
    try:
        if dataset_args['file'].split('/')[3] == 'MyDrive':
            file_name = dataset_args['file'].split('/')[-1]
            laptop_path = r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\GNOT\data'
            file_path = laptop_path + '/' + file_name
        else: file_path = dataset_args['file']
    except:
        file_path = dataset_args['file']
    

    # Load in Data
    dataset = Cavity_2D_dataset_handling_v2(file_path, name='cavity', train=True, sub_x = dataset_args['subsampler'],
                                            normalize_y=dataset_args['use-normalizer'], normalize_x = dataset_args['normalize_x'],
                                            data_split = dataset_args['percent split (decimal)'], seed = dataset_args['randomizer seed'],
                                            vertex = dataset_args['cell to pointwise'], boundaries = dataset_args['add boundaries']
                                            )
    
    eval_dataset = Cavity_2D_dataset_handling_v2(file_path, name='cavity_eval', train=False, sub_x = int(dataset_args['subsampler']/super_res),
                                            normalize_y=dataset_args['use-normalizer'],
                                            data_split = dataset_args['percent split (decimal)'], seed = dataset_args['randomizer seed'], 
                                            vertex = dataset_args['cell to pointwise'], boundaries = dataset_args['add boundaries'],
                                            y_normalizer=dataset.y_normalizer, 
                                            x_normalizer=dataset.x_normalizer, 
                                            up_normalizer=dataset.up_normalizer, 
                                            normalize_x = dataset_args['normalize_x'])
    
    # create loader
    if dataset_type == 'eval': 
        testing_dataset = eval_dataset
    elif dataset_type == 'train': 
        testing_dataset = dataset
    else: raise ValueError('Input for dataset_type should either be "eval" or "train"')
    
    testing_dataloader = MIODataLoader(testing_dataset, batch_size=1, shuffle=False, drop_last=False)

    # set loss function for pointwise error:
    loss_func = WeightedLpLoss(p=2,component='all', normalizer=None)
    
    for batch_n, batch in enumerate(testing_dataloader):
        
        g, u_p, g_u = batch
        g, u_p, g_u = g.to(device), u_p.to(device), g_u.to(device)
        
        batch_start_time = default_timer()
        out = model(g, u_p, g_u)
        batch_end_time = default_timer()

        y_pred, y = out.squeeze(), g.ndata['y'].squeeze()
        
        loss, reg,  _ = loss_func(g, y_pred, y)
        loss_total = loss + reg
        
        print(f'Inference time: {batch_end_time-batch_start_time:.4f}s with Weighted L2 Loss of {loss_total:.4f} and Lid Velocity: {torch.flatten(g_u[0]).item()}')

        break
    
    # transform output shape back to reality
    y_pred  = y_pred.reshape(len(u_p), testing_dataset.nx, testing_dataset.nx, 3)
    y       = y.reshape(len(u_p), testing_dataset.nx, testing_dataset.nx, 3)
    
    # transform output normalization back to reality
    y_pred  = testing_dataset.y_normalizer.transform(y_pred.to('cpu'),inverse=True)
    y       = testing_dataset.y_normalizer.transform(y.to('cpu'),inverse=True)

    return y_pred, y, torch.flatten(g_u[0]).item()

#_______________________________________________________________________________________________________________________________________
# Plotting Heatmaps
def ansatz_heat_map(prediction, ground_truth, lid_v):
    batch = 0

    line_grid = np.arange(prediction.shape[1])/(prediction.shape[1]-1)

    fig = make_subplots(rows=1,cols=3, subplot_titles=('Ground Truth', 'Ansatz', ' Difference'))
    for i in range(3):
        if i == 0: visible = True 
        else: visible = False
        fig.add_trace(go.Heatmap(z=ground_truth[batch, ..., i], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis", visible=visible),1,1)
        fig.add_trace(go.Heatmap(z=prediction[batch, ..., i], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis", visible=visible),1,2)
        fig.add_trace(go.Heatmap(z=(ground_truth-prediction)[batch, ..., i], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis", visible=visible),1,3)
        # Add dropdown 

    fig.update_layout( 
        updatemenus=[ 
            dict( 
                active=0, 
                buttons=list([ 
                    dict(label="U-Velocity", 
                        method="update", 
                        args=[{"visible": [True, True, True, False, False, False, False, False]}, 
                            {"title": f"U-Velocity with real lid velocity = {lid_v:.2f}m/s", 
                                }]), 
                    dict(label="V-Velocity", 
                        method="update", 
                        args=[{"visible": [False, False, False, True, True, True, False, False, False]}, 
                            {"title": f"V-Velocity with real lid velocity = {lid_v:.2f}m/s", 
                                }]), 
                    dict(label="Pressure", 
                        method="update", 
                        args=[{"visible": [False, False, False, False, False, False, True, True, True]}, 
                            {"title": f"Pressure with real lid velocity = {lid_v:.2f}m/s", 
                                }]),
                ]), 
            ) 
        ]) 
    
    fig.update_layout(coloraxis = {'colorscale':'RdBu', 'cmid': 0, 'reversescale': True, 'colorbar_x':-0.15}, title = f"U-Velocity with lid velocity = {lid_v:.2f}m/s")
    fig.show()
    return fig 

def fine_heat_map(ansatz_prediction, fine_prediction, ground_truth, lid_v):
    batch = 0

    line_grid = np.arange(ansatz_prediction.shape[1])/(ansatz_prediction.shape[1]-1)

    fig = make_subplots(rows=1,cols=5, subplot_titles=('Ground Truth', 'Ansatz', 'Fine', 'Ansatz Difference', 'Fine Difference'))
    for i in range(3):
        if i == 0: visible = True 
        else: visible = False
        fig.add_trace(go.Heatmap(z=ground_truth[batch, ..., i], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis", visible=visible),1,1)
        fig.add_trace(go.Heatmap(z=ansatz_prediction[batch, ..., i], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis", visible=visible),1,2)
        fig.add_trace(go.Heatmap(z=fine_prediction[batch, ..., i], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis", visible=visible),1,3)
        fig.add_trace(go.Heatmap(z=(ground_truth-ansatz_prediction)[batch, ..., i], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis", visible=visible),1,4)
        fig.add_trace(go.Heatmap(z=(ground_truth-fine_prediction)[batch, ..., i], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis", visible=visible),1,5)
        # Add dropdown 

    fig.update_layout( 
        updatemenus=[ 
            dict( 
                active=0, 
                buttons=list([ 
                    dict(label="U-Velocity", 
                        method="update", 
                        args=[{"visible": [True, True, True, True, True,
                                            False, False, False, False, False,
                                            False, False, False, False, False]}, 
                            {"title": f"U-Velocity with real lid velocity = {lid_v:.2f}m/s", 
                                }]), 
                    dict(label="V-Velocity", 
                        method="update", 
                        args=[{"visible": [False, False, False, False, False,
                                           True, True, True, True, True,
                                            False, False, False, False, False]},
                            {"title": f"V-Velocity with real lid velocity = {lid_v:.2f}m/s", 
                                }]), 
                    dict(label="Pressure", 
                        method="update", 
                        args=[{"visible": [False, False, False, False, False,
                                            False, False, False, False, False,
                                            True, True, True, True, True]},
                            {"title": f"Pressure with real lid velocity = {lid_v:.2f}m/s", 
                                }]),
                ]), 
            ) 
        ]) 
    
    fig.update_layout(coloraxis = {'colorscale':'RdBu', 'cmid': 0, 'reversescale': True, 'colorbar_x':-0.15}, title = f"Results with lid velocity = {lid_v:.2f}m/s")
    fig.show()
    return fig 

def ansatz_vorticity_map(prediction, ground_truth, lid_v):
    batch = 0
    
    line_grid = (np.arange(prediction.shape[1])/(prediction.shape[1]-1))[1:-1] 
    
    __,__,__, gt_derivatives = NS_FDM_cavity_internal_vertex_non_dim(torch.tensor(ground_truth), torch.tensor(lid_v).unsqueeze(-1), nu=0.01, L = 0.1)
    __,__,__, ansatz_derivatives = NS_FDM_cavity_internal_vertex_non_dim(torch.tensor(prediction), torch.tensor(lid_v).unsqueeze(-1), nu=0.01, L = 0.1)

    # calculate vorticity:
    w_gt = gt_derivatives[2] - gt_derivatives[1]
    w_ansatz = ansatz_derivatives[2] - ansatz_derivatives[1]

    fig = make_subplots(rows=1,cols=4, subplot_titles=('Ground Truth', 'Ansatz', ' Difference'))
    fig.add_trace(go.Heatmap(z=w_gt[batch, ...], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis"),1,1)
    fig.add_trace(go.Heatmap(z=w_ansatz[batch, ...], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis"),1,2)
    fig.add_trace(go.Heatmap(z=(w_gt-w_ansatz)[batch, ...], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis"),1,3)
    fig.update_layout(coloraxis = {'colorscale':'RdBu', 'cmid': 0, 'reversescale': True, 'colorbar_x':-0.15})
    fig.update_yaxes(title_text='y')
    fig.update_xaxes(title_text='x')
    fig.show()
    return fig

def ansatz_magnitude(prediction, ground_truth, lid_v, stream_line_density=3, arrow_scale=0.002):
    batch = 0 
    
    line_grid = np.arange(prediction.shape[1])/(prediction.shape[1]-1)
    
    vel_mag_ground = np.sqrt(ground_truth[batch,:,:,0]**2 + ground_truth[batch,:,:,1]**2)
    vel_mag_ansatz = np.sqrt(prediction[batch,:,:,0]**2 + prediction[batch,:,:,1]**2)
    fig = make_subplots(rows=1,cols=4, subplot_titles=('Ground Truth', 'Ansatz', 'Difference'))
    fig.add_trace(go.Heatmap(z=vel_mag_ground, x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis"),1,1)
    fig.add_trace(go.Heatmap(z=vel_mag_ansatz, x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis"),1,2)
    fig.add_trace(go.Heatmap(z=vel_mag_ground-vel_mag_ansatz, x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis"),1,3)

    fig_quiver = ff.create_streamline(line_grid,line_grid,u=ground_truth[batch,:,:,0],v=ground_truth[batch,:,:,1], arrow_scale=arrow_scale, density = stream_line_density, name = 'Stream-lines', legendgroup='Stream-Line', marker_color='Grey', showlegend=True)
    fig.append_trace(fig_quiver['data'][0], 1, 1)
    fig_quiver = ff.create_streamline(line_grid,line_grid,u=prediction[batch,:,:,0],v=prediction[batch,:,:,1], arrow_scale=arrow_scale, density = stream_line_density, legendgroup='Stream-Line', marker_color='Grey', showlegend=False)
    fig.append_trace(fig_quiver['data'][0], 1, 2)
    fig.update_yaxes(title_text='y', range=[0, 1])
    fig.update_xaxes(title_text='x', range=[0, 1])
    fig.update_layout(coloraxis = {'colorscale':'RdBu', 'cmid': 0, 'reversescale': True, 'colorbar_x':-0.15})
    fig.show()
    return fig

class cross_section_plots():
    def __init__(self, lid_v, L = 1):
        self.fig = make_subplots(rows=1,cols=4, subplot_titles=('U-Velocity (H-cross)', 'U-Velocity (V-cross)', 'V-Velocity (H-cross)', 'V-Velocity (V-cross)'))
        self.colors = plotly.colors.qualitative.Plotly

        sample_point_location = L/2

        self.fig['layout']['xaxis']['title'], self.fig['layout']['yaxis']['title'] = 'x', f'u(x,y={sample_point_location:.1f})'
        self.fig['layout']['xaxis2']['title'], self.fig['layout']['yaxis2']['title'] = f'u(x={sample_point_location:.1f},y)', 'y'
        self.fig['layout']['xaxis3']['title'], self.fig['layout']['yaxis3']['title'] = 'x', f'v(x,y={sample_point_location:.1f})'
        self.fig['layout']['xaxis4']['title'], self.fig['layout']['yaxis4']['title'] = f'v(x={sample_point_location:.1f},y)', 'y'
        self.fig.update_layout(title_text = f"Cross-Sectional Velocity Distributions ({lid_v:.2f}m/s)")

        # one dimension due to cross-section
        self.LpLoss = LpLoss(size_average=False, reduction=False, relative=True)
        self.l2_list = list()
        self.r2_list = list()

    def add_result(self, result, name, legend_group, colour_index, show_legend=True, mode = 'solid', baseline=False):
        
        sample_point_index = int(np.floor(result.shape[1]/2))
        line_grid = np.arange(result.shape[1])/(result.shape[1]-1)

        # if vertex centred and boundaries present:
        U_h_cross = result[0,sample_point_index,:,0].detach().numpy()
        U_v_cross = result[0,:,sample_point_index,0].detach().numpy()
        V_h_cross = result[0,sample_point_index,:,1].detach().numpy()
        V_v_cross = result[0,:,sample_point_index,1].detach().numpy()

        self.fig.add_trace(go.Scatter(x = line_grid, y = U_h_cross, mode='lines', name=name, legendgroup=legend_group, showlegend=show_legend  , marker_color=colors[colour_index], line={'dash': mode, 'color': colors[colour_index]}),1,1)
        self.fig.add_trace(go.Scatter(x = U_v_cross, y = line_grid, mode='lines', name=name, legendgroup=legend_group, showlegend=False        , marker_color=colors[colour_index], line={'dash': mode, 'color': colors[colour_index]}),1,2)
        self.fig.add_trace(go.Scatter(x = line_grid, y = V_h_cross, mode='lines', name=name, legendgroup=legend_group, showlegend=False        , marker_color=colors[colour_index], line={'dash': mode, 'color': colors[colour_index]}),1,3)
        self.fig.add_trace(go.Scatter(x = V_v_cross, y = line_grid, mode='lines', name=name, legendgroup=legend_group, showlegend=False        , marker_color=colors[colour_index], line={'dash': mode, 'color': colors[colour_index]}),1,4)

        # add L2 Scores:
        if baseline:
            self.U_h_cross_baseline = U_h_cross
            self.U_v_cross_baseline = U_v_cross
            self.V_h_cross_baseline = V_h_cross
            self.V_v_cross_baseline = V_v_cross
            self.l2_list.append({'U_h_cross':1, 'U_v_cross':1, 'V_h_cross':1, 'V_v_cross': 1})
        elif len(U_h_cross) == len(self.U_h_cross_baseline):
            U_h_cross_l2 = self.LpLoss(torch.tensor(U_h_cross).unsqueeze(0), torch.tensor(self.U_h_cross_baseline).unsqueeze(0)).item()
            U_v_cross_l2 = self.LpLoss(torch.tensor(U_v_cross).unsqueeze(0), torch.tensor(self.U_v_cross_baseline).unsqueeze(0)).item()
            V_h_cross_l2 = self.LpLoss(torch.tensor(V_h_cross).unsqueeze(0), torch.tensor(self.V_h_cross_baseline).unsqueeze(0)).item()
            V_v_cross_l2 = self.LpLoss(torch.tensor(V_v_cross).unsqueeze(0), torch.tensor(self.V_v_cross_baseline).unsqueeze(0)).item()
            self.l2_list.append({'U_h_cross':U_h_cross_l2, 'U_v_cross':U_v_cross_l2, 'V_h_cross':V_h_cross_l2, 'V_v_cross': V_v_cross_l2})
        else:
            self.l2_list.append({'U_h_cross':'Nan due to shape mismatch', 'U_v_cross':np.nan, 'V_h_cross':np.nan, 'V_v_cross': np.nan})


        # add R2 Scores:
        if baseline:
            self.r2_list.append({'U_h_cross':1, 'U_v_cross':1, 'V_h_cross':1, 'V_v_cross': 1})
        elif len(U_h_cross) == len(self.U_h_cross_baseline):
            U_h_cross_r2 = r2_score(U_h_cross, self.U_h_cross_baseline)
            U_v_cross_r2 = r2_score(U_v_cross, self.U_v_cross_baseline)
            V_h_cross_r2 = r2_score(V_h_cross, self.V_h_cross_baseline)
            V_v_cross_r2 = r2_score(V_v_cross, self.V_v_cross_baseline)
            self.r2_list.append({'U_h_cross':U_h_cross_r2, 'U_v_cross':U_v_cross_r2, 'V_h_cross':V_h_cross_r2, 'V_v_cross': V_v_cross_r2})
        else:
            self.r2_list.append({'U_h_cross':'Nan due to shape mismatch', 'U_v_cross':np.nan, 'V_h_cross':np.nan, 'V_v_cross': np.nan})
        
    def show(self):
        self.fig.show()

def fine_PDE_heat_map(ansatz_prediction, fine_prediction, ground_truth, lid_v):
    batch = 0

    line_grid = np.arange(ground_truth.shape[1])/(ground_truth.shape[1]-1)

    Du_dx, Dv_dy, continuity_eq,__ = NS_FDM_cavity_internal_vertex_non_dim(U=ansatz_prediction, lid_velocity=torch.tensor([lid_v]), nu=0.01, L=0.1)
    ansatz_pde_prediction = [Du_dx, Dv_dy, continuity_eq]
    Du_dx, Dv_dy, continuity_eq,__ = NS_FDM_cavity_internal_vertex_non_dim(U=fine_prediction, lid_velocity=torch.tensor([lid_v]), nu=0.01, L=0.1)
    fine_pde_prediction = [Du_dx, Dv_dy, continuity_eq]
    Du_dx, Dv_dy, continuity_eq,__ = NS_FDM_cavity_internal_vertex_non_dim(U=ground_truth, lid_velocity=torch.tensor([lid_v]), nu=0.01, L=0.1)
    ground_pde_prediction = [Du_dx, Dv_dy, continuity_eq]

    fig = make_subplots(rows=1,cols=5, subplot_titles=('Ground Truth', 'Ansatz', 'Fine', 'Ansatz Difference', 'Fine Difference'))
    for i in range(3):
        if i == 0: visible = True 
        else: visible = False
        fig.add_trace(go.Heatmap(z=ground_pde_prediction[i][batch, ...], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis", visible=visible),1,1)
        fig.add_trace(go.Heatmap(z=ansatz_pde_prediction[i][batch, ...], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis", visible=visible),1,2)
        fig.add_trace(go.Heatmap(z=fine_pde_prediction[i][batch, ...], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis", visible=visible),1,3)
        fig.add_trace(go.Heatmap(z=(ground_pde_prediction[i]-ansatz_pde_prediction[i])[batch, ...], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis", visible=visible),1,4)
        fig.add_trace(go.Heatmap(z=(ground_pde_prediction[i]-fine_pde_prediction[i])[batch, ...], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis", visible=visible),1,5)
        # Add dropdown 

    fig.update_layout( 
        updatemenus=[ 
            dict( 
                active=0, 
                buttons=list([ 
                    dict(label="U-Velocity", 
                        method="update", 
                        args=[{"visible": [True, True, True, True, True,
                                            False, False, False, False, False,
                                            False, False, False, False, False]}, 
                            {"title": f"X-Momentum with real lid velocity = {lid_v:.2f}m/s", 
                                }]), 
                    dict(label="V-Velocity", 
                        method="update", 
                        args=[{"visible": [False, False, False, False, False,
                                           True, True, True, True, True,
                                            False, False, False, False, False]},
                            {"title": f"Y-Momentum with real lid velocity = {lid_v:.2f}m/s", 
                                }]), 
                    dict(label="Pressure", 
                        method="update", 
                        args=[{"visible": [False, False, False, False, False,
                                            False, False, False, False, False,
                                            True, True, True, True, True]},
                            {"title": f"Continuity with real lid velocity = {lid_v:.2f}m/s", 
                                }]),
                ]), 
            ) 
        ]) 
    
    fig.update_layout(coloraxis = {'colorscale':'RdBu', 'cmid': 0, 'reversescale': True, 'colorbar_x':-0.15}, title = f"Results with lid velocity = {lid_v:.2f}m/s")
    fig.show()
    return fig

def PDE_heat_map(ansatz_prediction, ground_truth, lid_v):
    batch = 0

    line_grid = np.arange(ground_truth.shape[1])/(ground_truth.shape[1]-1)

    Du_dx, Dv_dy, continuity_eq,__ = NS_FDM_cavity_internal_vertex_non_dim(U=ansatz_prediction, lid_velocity=torch.tensor([lid_v]), nu=0.01, L=0.1)
    ansatz_pde_prediction = [Du_dx, Dv_dy, continuity_eq]
    Du_dx, Dv_dy, continuity_eq,__ = NS_FDM_cavity_internal_vertex_non_dim(U=ground_truth, lid_velocity=torch.tensor([lid_v]), nu=0.01, L=0.1)
    ground_pde_prediction = [Du_dx, Dv_dy, continuity_eq]

    fig = make_subplots(rows=1,cols=3, subplot_titles=('Ground Truth', 'Ansatz', 'Difference'))
    for i in range(3):
        if i == 0: visible = True 
        else: visible = False
        fig.add_trace(go.Heatmap(z=ground_pde_prediction[i][batch, ...], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis", visible=visible),1,1)
        fig.add_trace(go.Heatmap(z=ansatz_pde_prediction[i][batch, ...], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis", visible=visible),1,2)
        fig.add_trace(go.Heatmap(z=(ground_pde_prediction[i]-ansatz_pde_prediction[i])[batch, ...], x=line_grid, y=line_grid, showscale=False, connectgaps=True, coloraxis = "coloraxis", visible=visible),1,3)
        # Add dropdown 

    fig.update_layout( 
        updatemenus=[ 
            dict( 
                active=0, 
                buttons=list([ 
                    dict(label="U-Velocity", 
                        method="update", 
                        args=[{"visible": [True, True, True,
                                            False, False, False,
                                            False, False, False]}, 
                            {"title": f"X-Momentum with real lid velocity = {lid_v:.2f}m/s", 
                                }]), 
                    dict(label="V-Velocity", 
                        method="update", 
                        args=[{"visible": [False, False, False,
                                           True, True, True,
                                            False, False, False]},
                            {"title": f"Y-Momentum with real lid velocity = {lid_v:.2f}m/s", 
                                }]), 
                    dict(label="Pressure", 
                        method="update", 
                        args=[{"visible": [False, False, False,
                                            False, False, False,
                                            True, True, True]},
                            {"title": f"Continuity with real lid velocity = {lid_v:.2f}m/s", 
                                }]),
                ]), 
            ) 
        ]) 
    
    fig.update_layout(coloraxis = {'colorscale':'RdBu', 'cmid': 0, 'reversescale': True, 'colorbar_x':-0.15}, title = f"Results with lid velocity = {lid_v:.2f}m/s")
    fig.show()
    return fig