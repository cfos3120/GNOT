import plotly.offline as py
py.init_notebook_mode(connected=False)
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go
import plotly
import numpy as np
import plotly
colors = plotly.colors.qualitative.Plotly


def reconstruct_dict(path):
    results = np.load(path, allow_pickle=True)
    dictionary = dict(results.flatten()[0])
    print(dictionary.keys())
    return dictionary

def plot_lp_losses(t1, t2=None,t3=None,t4=None, loss_names=['trace1']):
    
    # fixed parameters
    figure_1_name = 'Training L2 Loss'
    figure_2_name = 'Validation L2 Loss'


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
                                    legendgroup=i, showlegend=False, marker_color=colors[0]),1,2)
            
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