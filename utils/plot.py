import plotly.graph_objects as go
import scipy.stats as stats
import pandas as pd
from plotly import io as pio


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

    fig.update_layout(title=f'Experiments Results REVS', xaxis_title='Metric', yaxis_title='Score')

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

    fig.update_layout(title=f'Experiments Results MEMIT', xaxis_title='Metric', yaxis_title='Score')

    if return_plot:
        return fig
    else:
        pio.show(fig)
