from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go
import scipy.stats as stats
import pprint
import torch
import pandas as pd
from plotly import io as pio

from utils.hidden_state_ops import hs_to_logits, get_rank_of_token_in_vocab
from utils.globals import device

# ~~~~~~~~~~~~~ General Plotting Functions ~~~~~~~~~~~~~

def plot_hist(values, title, xlabel, ylabel, bins=10):
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # add x ticks to the histogram
    plt.xticks(np.arange(0, 1.1, 0.1))
    # add mean value to histogram
    plt.axvline(np.mean(values), color='k', linestyle='dashed', linewidth=1)
    # add label "mean value" under the line of mean
    plt.text(np.mean(values), 1, "mean value", rotation=0, verticalalignment='bottom', horizontalalignment='center')
    # add the value of the std of the cosine similarities by printing on the histogram
    plt.text(np.mean(values) + np.std(values), 1, f"std: {np.round(np.std(values), 3)}", rotation=0, verticalalignment='bottom', horizontalalignment='center')
    plt.show()


def plot_grouped_hist(data, title, xlabel, ylabel, bins=10, model_names=None):
    fig, ax = plt.subplots()
    # Plot the histogram
    ax.hist(data, bins=bins)
    # Add legend and labels
    ax.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # add x ticks to the histogram
    plt.xticks(np.arange(0, 1.1, 0.1))
    # add mean value to histogram
    plt.axvline(np.mean(data), color='k', linestyle='dashed', linewidth=1)
    # add label "mean value" under the line of mean
    plt.text(np.mean(data), 1, "mean value", rotation=0, verticalalignment='bottom', horizontalalignment='center')
    # add the value of the std of the cosine similarities by printing on the histogram
    plt.text(np.mean(data) + np.std(data), 1, f"std: {np.round(np.std(data), 3)}", rotation=0, verticalalignment='bottom', horizontalalignment='center')
    # add model names by their color labels
    ax.set_xticklabels(model_names)
    plt.show()


def plot_boxplot(values, title, ylabel):
    fig, ax = plt.subplots()

    # Add mean, median, and standard deviation to the boxplot
    mean = np.mean(values)
    median = np.median(values)
    std = np.std(values)
    ax.axhline(mean, color='r', linestyle='--', label='Mean')
    ax.axhline(median, color='g', linestyle='-.', label='Median')
    ax.axhline(mean+std, color='b', linestyle=':', label='Std Dev')
    ax.axhline(mean-std, color='b', linestyle=':')

    # Add outliers to the boxplot
    outliers = []
    for i in range(len(values)):
        if values[i] > mean + 2*std or values[i] < mean - 2*std:
            outliers.append(values[i])
    ax.scatter(np.ones(len(outliers)), outliers, color='m', marker='*', label='Outliers')

    # Plot the boxplot
    ax.boxplot(values)

    # Add legend and labels
    ax.legend()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.show()


def plot_grouped_boxplot(data, title, ylabel, model_names):
    fig, ax = plt.subplots()
    # Plot the boxplot
    ax.boxplot(data)
    # Add legend and labels
    ax.legend()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Models")
    # add model names by their color labels
    ax.set_xticklabels(model_names)
    # print mean, median and std values bottom of the boxplot
    for i in range(len(data)):
        mean = np.mean(data[i])
        median = np.median(data[i])
        std = np.std(data[i])
        ax.text(i+1, mean, f"mean: {np.round(mean, 3)}", rotation=0, verticalalignment='bottom', horizontalalignment='center')
        ax.text(i+1, median, f"median: {np.round(median, 3)}", rotation=0, verticalalignment='bottom', horizontalalignment='center')
        ax.text(i+1, mean+std, f"std: {np.round(std, 3)}", rotation=0, verticalalignment='bottom', horizontalalignment='center')
        ax.text(i+1, mean-std, f"std: {np.round(std, 3)}", rotation=0, verticalalignment='bottom', horizontalalignment='center')
    plt.show()


# ~~~~~~~~~~~~~ Model Rank Editing Plotting Functions ~~~~~~~~~~~~~

def display_dict_interactively(dict_list):
    if not isinstance(dict_list, list):
        dict_list = [dict_list]

    # Create a dropdown to select the dictionary
    select_dict = widgets.Dropdown(options=range(len(dict_list)), description='Select Dict:', layout={'width': 'max-content'})

    # Create a multiple selection list with dictionary keys
    select_multiple = widgets.SelectMultiple(description='Select Keys:', layout={'width': 'max-content', 'height': '200px'})

    # Create an Output widget
    out = widgets.Output()

    # Function to update the keys when a new dictionary is selected
    def update_keys(change):
        if change['name'] == 'value':
            select_multiple.options = dict_list[change['new']].keys()

    # Call update_keys when select_dict value changes
    select_dict.observe(update_keys, 'value')

    # Function to print the value of the selected keys
    def print_values(change):
        if change['name'] == 'value':
            keys = change['new']
            with out:
                out.clear_output()
                # Always print the "layer" value if it exists
                if 'layer' in dict_list[select_dict.value]:
                    print(f"layer:")
                    value = dict_list[select_dict.value]['layer']
                    pprint.pprint(value)
                    print("\n")
                for key in keys:
                    print(f"{key}:")
                    value = dict_list[select_dict.value][key]
                    pprint.pprint(value)
                    if torch.is_tensor(value) and value.dtype != torch.long:
                        value_float = value.float()
                        print("Tensor Statistics:")
                        print(f"Mean: {value_float.mean().item():.4e}")
                        print(f"Median: {value_float.median().item():.4e}")
                        print(f"Std: {value_float.std().item():.4e}")
                        print(f"Min: {value_float.min().item():.4e}")
                        print(f"Max: {value_float.max().item():.4e}")
                    print("\n")

    # Call print_values when select_multiple value changes
    select_multiple.observe(print_values)

    # Create a box to hold the widgets
    box = widgets.VBox([select_dict, select_multiple, out])

    # Display the widget
    display(box)

    # Initialize the keys
    update_keys({'name': 'value', 'new': select_dict.value})


def plot_token_rank_in_hs_across_sublayers(token_ranks_dict, prompt, target, title="Token Rank in HS Across Layers", log_scale=False, return_plot=False):
    title = f"{title}<br>Prompt: \"{prompt[-50:]}\" --> \"{target}\""
    # Get the keys from the first dictionary in the values of token_ranks_dict
    keys = list(next(iter(token_ranks_dict.values())).keys())
    fig = go.Figure()
    for key in keys:
        ranks = [value[key] for value in token_ranks_dict.values()]
        if log_scale:
            # Add a small constant to allow log scale to handle 0 values
            ranks = [rank if rank > 0 else 0.1 for rank in ranks]
        fig.add_trace(go.Scatter(y=ranks, mode='lines+markers', name=key))

    fig.update_layout(title=title, xaxis_title='Layer', yaxis_title='Token Rank', autosize=True)
    if log_scale:
        fig.update_yaxes(type="log")

    if return_plot:
        return fig
    else:
        fig.show()

def plot_attn_v_rank(model, tokenizer, collected_acts, prompt, target:str):
    target_token_id = tokenizer.encode(target, add_special_tokens=False)[0]
    layers_token_ranks_dict = {}
    if model.config.model_type == 'gpt-j':
        n_layers = model.config.n_layer
    elif model.config.model_type == 'llama':
        n_layers = model.config.num_hidden_layers
    raise ValueError("Model type not supported, only gpt-j and llama are supported")

    for layer in range(n_layers):
        hs_attn_v = torch.stack(collected_acts[layer]['attn_v']['output']).to(device)
        attn_v = hs_to_logits(model, hs_attn_v)
        token_ranks_dict = {}
        for token_index in range(attn_v.shape[0]):
            token_ranks_dict[token_index] = get_rank_of_token_in_vocab(attn_v[token_index], target_token_id).item()
        layers_token_ranks_dict[layer] = token_ranks_dict
    plot_token_rank_in_hs_across_sublayers(layers_token_ranks_dict, prompt, target, log_scale=True)

def plot_token_rank_across_sublayers(token_ranks_dict, prompt, target, title="Rank Edit Score Across Layers", log_scale=False, return_plot=False):
    title = f"{title}<br>Prompt: \"{prompt}\" --> \"{target}\""
    # Get the keys from the first dictionary in the values of token_ranks_dict
    keys = list(next(iter(token_ranks_dict.values())).keys())
    fig = go.Figure()
    for key in keys:
        ranks = [value[key] for value in token_ranks_dict.values()]
        if log_scale:
            # Add a small constant to allow log scale to handle 0 values
            ranks = [rank if rank > 0 else 0.1 for rank in ranks]
        fig.add_trace(go.Scatter(y=ranks, mode='lines+markers', name=key))

    fig.update_layout(title=title, xaxis_title='Layer', yaxis_title='Token Rank', autosize=True)
    if log_scale:
        fig.update_yaxes(type="log")

    if return_plot:
        return fig
    else:
        fig.show()

def plot_edit_score_statistics_bar_cf(edit_score_stats_dict, title="Edit Score Statistics Across Layers", return_plot=False, log_scale=False):
    """
    ...
    """
    keys = list(next(iter(edit_score_stats_dict.values())).keys())
    fig = go.Figure()

    color_map_lines = ['rgba(0,0,255,1)', 'rgba(255,0,0,1)', 'rgba(128,0,128,1)', 'rgba(128,0,128,1)']

    for i, key in enumerate(keys):
        mean_values = [value[key]['mean'] for value in edit_score_stats_dict.values()]
        std_values = [value[key]['std'] for value in edit_score_stats_dict.values()]
        sample_sizes = [value[key]['count'] for value in edit_score_stats_dict.values()]

        sem_values = [std / (n ** 0.5) for std, n in zip(std_values, sample_sizes)]
        ci_values = [sem * stats.t.ppf((1 + 0.95) / 2., n - 1) for sem, n in zip(sem_values, sample_sizes)]

        fig.add_trace(go.Scatter(
            y=mean_values,
            mode='lines+markers',
            name=f"{key} (mean)",
            line=dict(color=color_map_lines[i]),
            error_y=dict(
                type='data',
                symmetric=True,
                array=ci_values,
                color=color_map_lines[i]
            )
        ))

    fig.update_layout(title=title, xaxis_title='Layer', yaxis_title='Value', autosize=True)
    if log_scale:
        fig.update_yaxes(type="log")
    else:
        fig.update_yaxes(rangemode="tozero")

    if return_plot:
        return fig
    else:
        fig.show()

def plot_edit_score_statistics(edit_score_stats_dict, return_plot=False, log_scale=False, method='mean', plot_std=False, title=None):
    keys = list(next(iter(edit_score_stats_dict.values())).keys())
    fig = go.Figure()
    if title is None:
        title = f"Edit Score Statistics Across Layers, Method: {method.upper()}"
    color_map_lines = ['rgba(255,0,0,1)', 'rgba(0,0,255,1)', 'rgba(0,100,0,1)', 'rgba(0,100,0,1)']
    color_map_cis = ['rgba(255,0,0,0.2)', 'rgba(0,0,255,0.2)', 'rgba(0,100,0,0.2)', 'rgba(0,100,0,0.2)']
    assert method in ['mean', 'binary', 'median'], "method must be either 'mean' or 'binary' or 'median'"
    for i, key in enumerate(keys):
        values = [value[key][method] for value in edit_score_stats_dict.values()]
        std_values = [value[key]['std'] for value in edit_score_stats_dict.values()]
        sample_sizes = [value[key]['count'] for value in edit_score_stats_dict.values()]
        sem_values = [std / (n ** 0.5) for std, n in zip(std_values, sample_sizes)]
        ci_values = [sem * stats.t.ppf((1 + 0.95) / 2., n - 1) for sem, n in zip(sem_values, sample_sizes)]
        fig.add_trace(go.Scatter(y=values, mode='lines+markers', name=f"{key} ({method})", line=dict(color=color_map_lines[i])))
        if plot_std:
            fig.add_trace(go.Scatter(
                y=[mean - std for mean, std in zip(values, std_values)],
                mode='lines',
                line_color = color_map_cis[i],
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                y=[mean + std for mean, std in zip(values, std_values)],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                line_color = color_map_cis[i],
                fillcolor=color_map_cis[i],
                name=f"{key} (std)"
            ))
        else:
            fig.add_trace(go.Scatter(
                y=[mean - ci for mean, ci in zip(values, ci_values)],
                mode='lines',
                line_color = color_map_cis[i],
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                y=[mean + ci for mean, ci in zip(values, ci_values)],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                line_color = color_map_cis[i],
                fillcolor=color_map_cis[i],
                name=f"{key} (95% CI)"
            ))

    fig.update_layout(title=title, xaxis_title='Layer', yaxis_title='Value', autosize=True)
    if log_scale:
        fig.update_yaxes(type="log")
    else:
        fig.update_yaxes(rangemode="tozero")

    if return_plot:
        return fig
    else:
        fig.show()


def plot_multi_experiment_results_revs(res_dict, return_plot=False):
    """
    This function plots the results of multiple experiments.

    Parameters:
    res_dict (dict): A nested dictionary containing the results of the experiments.
                     The structure of the dictionary should be as follows:
                     {n_neurons: {seed: {threshold: {metric: score}}}}
                     where:
                     - n_neurons is the number of neurons used in the experiment
                     - seed is the seed used for random number generation in the experiment
                     - threshold is the threshold value used in the experiment
                     - metric is the name of the performance metric
                     - score is the score for the performance metric
    edit_method (str): The name of the method used in the experiments. This will be included in the plot title.
    return_plot (bool): If True, the function will return the plot instead of displaying it.

    Returns:
    fig (plotly.graph_objs._figure.Figure): The plotly Figure object representing the plot.
                                            Only returned if return_plot is True.
    """

    # Flatten the dictionary and convert it into a DataFrame
    df = pd.DataFrame([
        {'n_neurons': int(n_neurons), 'seed': int(seed), 'threshold': int(threshold), **score}
        for n_neurons, seed_dict in res_dict.items()
        for seed, threshold_dict in seed_dict.items()
        for threshold, score in threshold_dict.items()
    ])

    # Pivot the DataFrame so that each configuration combination becomes a column
    df = df.melt(id_vars=['n_neurons', 'seed', 'threshold'], var_name='metric', value_name='score')

    # Sort the DataFrame by descending values of threshold, n_neurons, and seed
    df = df.sort_values(by=['threshold', 'n_neurons', 'seed'], ascending=[False, False, False])

    # Now you can plot it
    fig = go.Figure([
        go.Scatter(
            x=df[(df['n_neurons'] == n_neurons) & (df['seed'] == seed) & (df['threshold'] == threshold)]['metric'],
            y=df[(df['n_neurons'] == n_neurons) & (df['seed'] == seed) & (df['threshold'] == threshold)]['score'],
            mode='lines+markers',
            name=f'threshold: {threshold}, n_neurons: {n_neurons}, seed: {seed}'
        )
        for threshold in df['threshold'].unique()
        for n_neurons in df['n_neurons'].unique()
        for seed in df['seed'].unique()
    ])

    fig.update_layout(title=f'Experiments Results SIRE', xaxis_title='Metric', yaxis_title='Score')

    if return_plot:
        return fig
    else:
        pio.show(fig)

def plot_multi_experiment_results_memit(res_dict, return_plot=False):
    """
    This function plots the results of multiple experiments.

    Parameters:
    res_dict (dict): A nested dictionary containing the results of the experiments.
                     The structure of the dictionary should be as follows:
                     {v_lr: {loss_break: {loss_pred_prob_coef: {seed: {threshold: {metric: score}}}}}}
                     where:
                     - v_lr is the learning rate used in the experiment
                     - loss_break is the loss break used in the experiment
                     - loss_pred_prob_coef is the loss prediction probability coefficient used in the experiment
                     - seed is the seed used for random number generation in the experiment
                     - threshold is the threshold value used in the experiment
                     - metric is the name of the performance metric
                     - score is the score for the performance metric
    return_plot (bool): If True, the function will return the plot instead of displaying it.

    Returns:
    fig (plotly.graph_objs._figure.Figure): The plotly Figure object representing the plot.
                                            Only returned if return_plot is True.
    """

    # Flatten the dictionary and convert it into a DataFrame
    df = pd.DataFrame([
        {'v_lr': float(v_lr), 'loss_break': float(loss_break), 'loss_pred_prob_coef': float(loss_pred_prob_coef), 'seed': int(seed), 'threshold': int(threshold), **score}
        for v_lr, loss_break_dict in res_dict.items()
        for loss_break, loss_pred_prob_coef_dict in loss_break_dict.items()
        for loss_pred_prob_coef, seed_dict in loss_pred_prob_coef_dict.items()
        for seed, threshold_dict in seed_dict.items()
        for threshold, score in threshold_dict.items()
    ])

    # Pivot the DataFrame so that each configuration combination becomes a column
    df = df.melt(id_vars=['v_lr', 'loss_break', 'loss_pred_prob_coef', 'seed', 'threshold'], var_name='metric', value_name='score')

    # Sort the DataFrame by descending values of threshold, v_lr, loss_break, loss_pred_prob_coef, and seed
    df = df.sort_values(by=['threshold', 'v_lr', 'loss_break', 'loss_pred_prob_coef', 'seed'], ascending=[False, False, False, False, False])

    # Now you can plot it
    fig = go.Figure([
        go.Scatter(
            x=df[(df['v_lr'] == v_lr) & (df['loss_break'] == loss_break) & (df['loss_pred_prob_coef'] == loss_pred_prob_coef) & (df['seed'] == seed) & (df['threshold'] == threshold)]['metric'],
            y=df[(df['v_lr'] == v_lr) & (df['loss_break'] == loss_break) & (df['loss_pred_prob_coef'] == loss_pred_prob_coef) & (df['seed'] == seed) & (df['threshold'] == threshold)]['score'],
            mode='lines+markers',
            name=f'threshold: {threshold}, v_lr: {v_lr}, loss_break: {loss_break}, loss_coef: {loss_pred_prob_coef}, seed: {seed}'
        )
        for threshold in df['threshold'].unique()
        for v_lr in df['v_lr'].unique()
        for loss_break in df['loss_break'].unique()
        for loss_pred_prob_coef in df['loss_pred_prob_coef'].unique()
        for seed in df['seed'].unique()
    ])

    fig.update_layout(title=f'Experiments Results MEMIT', xaxis_title='Metric', yaxis_title='Score')

    if return_plot:
        return fig
    else:
        pio.show(fig)

def plot_multi_experiment_results_ft(res_dict, return_plot=False):
    """
    This function plots the results of multiple experiments.

    Parameters:
    res_dict (dict): A nested dictionary containing the results of the experiments.
                     The structure of the dictionary should be as follows:
                     {lr: {loss_break: {norm_constraint: {seed: {threshold: {metric: score}}}}}}
                     where:
                     - lr is the learning rate used in the experiment
                     - loss_break is the loss break used in the experiment
                     - norm_constraint is the norm constraint used in the experiment
                     - seed is the seed used for random number generation in the experiment
                     - threshold is the threshold value used in the experiment
                     - metric is the name of the performance metric
                     - score is the score for the performance metric
    return_plot (bool): If True, the function will return the plot instead of displaying it.

    Returns:
    fig (plotly.graph_objs._figure.Figure): The plotly Figure object representing the plot.
                                            Only returned if return_plot is True.
    """

    # Flatten the dictionary and convert it into a DataFrame
    df = pd.DataFrame([
        {'lr': float(lr), 'loss_break': float(loss_break), 'norm_constraint': float(norm_constraint), 'seed': int(seed), 'threshold': int(threshold), **score}
        for lr, loss_break_dict in res_dict.items()
        for loss_break, norm_constraint_dict in loss_break_dict.items()
        for norm_constraint, seed_dict in norm_constraint_dict.items()
        for seed, threshold_dict in seed_dict.items()
        for threshold, score in threshold_dict.items()
    ])

    # Pivot the DataFrame so that each configuration combination becomes a column
    df = df.melt(id_vars=['lr', 'loss_break', 'norm_constraint', 'seed', 'threshold'], var_name='metric', value_name='score')

    # Sort the DataFrame by descending values of threshold, lr, loss_break, norm_constraint, and seed
    df = df.sort_values(by=['threshold', 'lr', 'loss_break', 'norm_constraint', 'seed'], ascending=[False, False, False, False, False])

    # Now you can plot it
    fig = go.Figure([
        go.Scatter(
            x=df[(df['lr'] == lr) & (df['loss_break'] == loss_break) & (df['norm_constraint'] == norm_constraint) & (df['seed'] == seed) & (df['threshold'] == threshold)]['metric'],
            y=df[(df['lr'] == lr) & (df['loss_break'] == loss_break) & (df['norm_constraint'] == norm_constraint) & (df['seed'] == seed) & (df['threshold'] == threshold)]['score'],
            mode='lines+markers',
            name=f'threshold: {threshold}, lr: {lr}, loss_break: {loss_break}, norm_constraint: {norm_constraint}, seed: {seed}'
        )
        for threshold in df['threshold'].unique()
        for lr in df['lr'].unique()
        for loss_break in df['loss_break'].unique()
        for norm_constraint in df['norm_constraint'].unique()
        for seed in df['seed'].unique()
    ])

    fig.update_layout(title=f'Experiments Results FTL', xaxis_title='Metric', yaxis_title='Score')

    if return_plot:
        return fig
    else:
        pio.show(fig)

def plot_multi_experiment_results_rmu(res_dict, return_plot=False):
    """
    This function plots the results of multiple RMU experiments.

    Parameters:
    res_dict (dict): A nested dictionary containing the results of the experiments.
                     The structure of the dictionary should be as follows:
                     {num_epochs: {lr: {alpha: {steering_coeff: {seed: {threshold: {metric: score}}}}}}}
                     where:
                     - num_epochs is the number of epochs used in the experiment
                     - lr is the learning rate used in the experiment
                     - alpha is the alpha value used in the experiment
                     - steering_coeff is the steering coefficient used in the experiment
                     - seed is the seed used for random number generation in the experiment
                     - threshold is the threshold value used in the experiment
                     - metric is the name of the performance metric
                     - score is the score for the performance metric
    return_plot (bool): If True, the function will return the plot instead of displaying it.

    Returns:
    fig (plotly.graph_objs._figure.Figure): The plotly Figure object representing the plot.
                                            Only returned if return_plot is True.
    """

    # Flatten the dictionary and convert it into a DataFrame
    df = pd.DataFrame([
        {'num_epochs': int(num_epochs), 'lr': float(lr), 'alpha': str(alpha), 'steering_coeff': str(steering_coeff), 'seed': int(seed), 'threshold': int(threshold), **score}
        for num_epochs, lr_dict in res_dict.items()
        for lr, alpha_dict in lr_dict.items()
        for alpha, steering_coeff_dict in alpha_dict.items()
        for steering_coeff, seed_dict in steering_coeff_dict.items()
        for seed, threshold_dict in seed_dict.items()
        for threshold, score in threshold_dict.items()
    ])

    # Pivot the DataFrame so that each configuration combination becomes a column
    df = df.melt(id_vars=['num_epochs', 'lr', 'alpha', 'steering_coeff', 'seed', 'threshold'], var_name='metric', value_name='score')

    # Sort the DataFrame by descending values of threshold, num_epochs, lr, alpha, steering_coeff, and seed
    df = df.sort_values(by=['threshold', 'num_epochs', 'lr', 'alpha', 'steering_coeff', 'seed'], ascending=[False, False, False, False, False, False])

    # Now you can plot it
    fig = go.Figure([
        go.Scatter(
            x=df[(df['num_epochs'] == num_epochs) & (df['lr'] == lr) & (df['alpha'] == alpha) & (df['steering_coeff'] == steering_coeff) & (df['seed'] == seed) & (df['threshold'] == threshold)]['metric'],
            y=df[(df['num_epochs'] == num_epochs) & (df['lr'] == lr) & (df['alpha'] == alpha) & (df['steering_coeff'] == steering_coeff) & (df['seed'] == seed) & (df['threshold'] == threshold)]['score'],
            mode='lines+markers',
            name=f'threshold: {threshold}, num_epochs: {num_epochs}, lr: {lr}, alpha: {alpha}, steering_coeff: {steering_coeff}, seed: {seed}'
        )
        for threshold in df['threshold'].unique()
        for num_epochs in df['num_epochs'].unique()
        for lr in df['lr'].unique()
        for alpha in df['alpha'].unique()
        for steering_coeff in df['steering_coeff'].unique()
        for seed in df['seed'].unique()
    ])

    fig.update_layout(title='Experiments Results RMU', xaxis_title='Metric', yaxis_title='Score')

    if return_plot:
        return fig
    else:
        pio.show(fig)

def plot_multi_experiment_results_npo(res_dict, return_plot=False):
    """
    This function plots the results of multiple NPO experiments.

    Parameters:
    res_dict (dict): A nested dictionary containing the results of the experiments.
                     The structure of the dictionary should be as follows:
                     {num_epochs: {lr: {beta: {seed: {threshold: {metric: score}}}}}}
                     where:
                     - num_epochs is the number of epochs used in the experiment
                     - lr is the learning rate used in the experiment
                     - beta is the beta parameter used in the experiment
                     - seed is the seed used for random number generation
                     - threshold is the threshold value used in the experiment
                     - metric is the name of the performance metric
                     - score is the score for the performance metric
    return_plot (bool): If True, the function will return the plot instead of displaying it.

    Returns:
    fig (plotly.graph_objs._figure.Figure): The plotly Figure object representing the plot.
                                           Only returned if return_plot is True.
    """

    # Flatten the dictionary and convert it into a DataFrame
    df = pd.DataFrame([
        {
            'num_epochs': int(num_epochs),
            'lr': float(lr),
            'beta': float(beta),
            'seed': int(seed),
            'threshold': int(threshold),
            **score
        }
        for num_epochs, lr_dict in res_dict.items()
        for lr, beta_dict in lr_dict.items()
        for beta, seed_dict in beta_dict.items()
        for seed, threshold_dict in seed_dict.items()
        for threshold, score in threshold_dict.items()
    ])

    # Pivot the DataFrame so that each configuration combination becomes a column
    df = df.melt(id_vars=['num_epochs', 'lr', 'beta', 'seed', 'threshold'],
                 var_name='metric', value_name='score')

    # Sort the DataFrame by descending values
    df = df.sort_values(
        by=['threshold', 'num_epochs', 'lr', 'beta', 'seed'],
        ascending=[False, False, False, False, False]
    )

    # Create plot with beta parameter included
    fig = go.Figure([
        go.Scatter(
            x=df[
                (df['num_epochs'] == num_epochs) &
                (df['lr'] == lr) &
                (df['beta'] == beta) &
                (df['seed'] == seed) &
                (df['threshold'] == threshold)
            ]['metric'],
            y=df[
                (df['num_epochs'] == num_epochs) &
                (df['lr'] == lr) &
                (df['beta'] == beta) &
                (df['seed'] == seed) &
                (df['threshold'] == threshold)
            ]['score'],
            mode='lines+markers',
            name=f'th:{threshold}, ep:{num_epochs}, lr:{lr}, b:{beta}, s:{seed}'
        )
        for threshold in df['threshold'].unique()
        for num_epochs in df['num_epochs'].unique()
        for lr in df['lr'].unique()
        for beta in df['beta'].unique()
        for seed in df['seed'].unique()
    ])

    fig.update_layout(
        title='Experiments Results NPO',
        xaxis_title='Metric',
        yaxis_title='Score'
    )

    if return_plot:
        return fig
    else:
        pio.show(fig)