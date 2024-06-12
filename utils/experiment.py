from typing import List, Union
from copy import deepcopy
from collections import defaultdict
from tqdm.notebook import tqdm
import numpy as np
import torch
import pandas as pd
import wandb

# My utils:
from utils.generation import generate_from_prompts
from utils.model import load_model_tokenizer, edit_model, create_requests, MemitConfig, FTConfig, load_model_tokenizer_ssn
from revs.revs import REVSConfig, REVS, REVSScore
from utils.metrics import calculate_edit_score_statistics_squared, calculate_across_layers_score, calculate_harmonic_mean, \
    EfficacyScore, DeltaAttackScore, PerturbAttackScore, LogitLensAttackScore
from utils.hidden_state_ops import get_token_rank_across_layers
from utils.activations_collector import collect_activations_with_prompt
from utils.plot import plot_token_rank_in_hs_across_sublayers, plot_edit_score_statistics, \
    plot_multi_experiment_results_revs, plot_multi_experiment_results_memit, plot_multi_experiment_results_ft
from utils.globals import device
from utils.data import create_concat_prompts_target


# ~~~~~~~~~~~~ EXPS ~~~~~~~~~~~~


def run_revs_exp(model, tokenizer, prompts, targets, config, pinv_lm_head, specificity, generality, extraction):

    res_dict = {}
    model_editor = None
    if not config.not_unlearn:
        model_editor = run_revs_unlearn(model, tokenizer, prompts['unlearn'], targets['unlearn'], config, pinv_lm_head)

    rank_editor_scores = calc_rank_editor_scores(model, tokenizer, prompts['unlearn'], targets['unlearn'], config)

    res_dict['efficacy'] = run_efficacy(rank_editor_scores)

    if specificity:
        res_dict['specificity'] = run_specificity(model, tokenizer, prompts['retain'], targets['retain'], config)

    if generality:
        res_dict['generality'] = run_generality(model, tokenizer, prompts['generality'], targets['generality'], config)

    if extraction:
        res_dict['delta_attack'] = run_delta_attack(model, tokenizer, prompts['unlearn'], targets['unlearn'], config)
        res_dict['perturb_attack'] = run_perturbed_prompts_attack(model, tokenizer, prompts['unlearn'], targets['unlearn'], config)
        res_dict['logit_lens_attack'] = run_logit_lens_attack(rank_editor_scores)

    if config.perplexity:
        perplexities = run_perplexity(model, tokenizer, config)
        res_dict['perplexity'] = np.mean(perplexities)

    return res_dict, model_editor


def run_memit_exp(model, tokenizer, prompts, targets, config, specificity=False, generality=False, extraction=False, example_chunk_size=1000):

    res_dict = {}

    run_memit_unlearn(model, tokenizer, prompts['unlearn'], targets['unlearn'], config, example_chunk_size)

    rank_editor_scores = calc_rank_editor_scores(model, tokenizer, prompts['unlearn'], targets['unlearn'], config)

    res_dict['efficacy'] = run_efficacy(rank_editor_scores)

    if specificity:
        res_dict['specificity'] = run_specificity(model, tokenizer, prompts['retain'], targets['retain'], config)

    if generality:
        res_dict['generality'] = run_generality(model, tokenizer, prompts['generality'], targets['generality'], config)

    if extraction:
        res_dict['delta_attack'] = run_delta_attack(model, tokenizer, prompts['unlearn'], targets['unlearn'], config)
        res_dict['perturb_attack'] = run_perturbed_prompts_attack(model, tokenizer, prompts['unlearn'], targets['unlearn'], config)
        res_dict['logit_lens_attack'] = run_logit_lens_attack(rank_editor_scores)

    if config.perplexity:
        perplexities = run_perplexity(model, tokenizer, config)
        res_dict['perplexity'] = np.mean(perplexities)

    return res_dict


def run_ft_exp(model, tokenizer, prompts, targets, config, specificity=False, generality=False, extraction=False):

    res_dict = {}

    run_ft_unlearn(model, tokenizer, prompts['unlearn'], targets['unlearn'], config)

    rank_editor_scores = calc_rank_editor_scores(model, tokenizer, prompts['unlearn'], targets['unlearn'], config)

    res_dict['efficacy'] = run_efficacy(rank_editor_scores)

    if specificity:
        res_dict['specificity'] = run_specificity(model, tokenizer, prompts['retain'], targets['retain'], config)

    if generality:
        res_dict['generality'] = run_generality(model, tokenizer, prompts['retain'], targets['retain'], config)

    if extraction:
        res_dict['delta_attack'] = run_delta_attack(model, tokenizer, prompts['unlearn'], targets['unlearn'], config)
        res_dict['perturb_attack'] = run_perturbed_prompts_attack(model, tokenizer, prompts['unlearn'], targets['unlearn'], config)
        res_dict['logit_lens_attack'] = run_logit_lens_attack(rank_editor_scores)

    if config.perplexity:
        perplexities = run_perplexity(model, tokenizer, config)
        res_dict['perplexity'] = np.mean(perplexities)

    return res_dict


# ~~~~~~~~~~~~ MULTI EXPS ~~~~~~~~~~~~


def revs_multi_email_exp(config: REVSConfig, seeds: List[int], n_neurons_list: List[int], score_thresholds: List[int], message: str = None):
    """
    Run a multiple EMAIL experiment with REVS, on different seeds and n_neurons
    """
    multi_exp_config = config.to_dict()
    del multi_exp_config['n_neurons'], multi_exp_config['seed'], multi_exp_config['score_threshold']
    multi_exp_config['n_neurons_list'] = n_neurons_list
    multi_exp_config['seeds'] = seeds
    multi_exp_config['score_thresholds'] = score_thresholds

    if config.log_wandb:
        exp_name = "REVS EMAIL"
        config_name = (f"{seeds}s, {n_neurons_list}n, {score_thresholds}st, "
                f"{config.residual_top_rank_margin/1000}krtop, "
                f"{config.residual_bottom_rank_margin/1000}krbottom, "
                f"{config.act_filter}")
        name = f"{exp_name} {message + ', ' if message else ''}{config_name}"
        wandb.init(project="delpii", config=multi_exp_config, name=name)


    pinv_lm_head = None
    res_dict = defaultdict(lambda: defaultdict(dict))
    for n_neurons in tqdm(n_neurons_list, desc="Processing neurons"):
        for seed in tqdm(seeds, desc="Processing seeds", leave=False):
            exp_config = deepcopy(config)
            exp_config.n_neurons = n_neurons
            exp_config.seed = seed
            exp_config.log_wandb = False
            exp_config.save_model = False

            prompts_dict, targets_dict, df_email_dict = get_dataset_email(seed=seed, split_value=config.unlearn_num_examples, max_prompt_len=config.max_prompt_len)
            model, tokenizer = load_model_tokenizer()
            if pinv_lm_head is None:  # calculate pinv_lm_head only once
                pinv_lm_head = torch.pinverse(model.lm_head.weight).to(device)

            exp_res_dict, exp_model_editor = run_revs_exp(
                model, 
                tokenizer, 
                prompts_dict, 
                targets_dict, 
                exp_config, 
                pinv_lm_head, 
                specificity=True, 
                generality=False, 
                extraction=True
            )

            efficacy_edit_scores = exp_res_dict['efficacy']
            specificity_score = exp_res_dict['specificity']
            # extraction attacks scores
            delta_attack_edit_scores = exp_res_dict['delta_attack']
            perturbed_attack_edit_scores = exp_res_dict['perturb_attack']
            logit_lens_attack_edit_scores = exp_res_dict['logit_lens_attack']

            for threshold in tqdm(score_thresholds, desc="Processing thresholds", leave=False):
                efficacy_results = calculate_edit_score_statistics_squared(efficacy_edit_scores, threshold=threshold)
                efficacy_across_layers_score = calculate_across_layers_score(efficacy_results)
                perturbed_attack_results = calculate_edit_score_statistics_squared(perturbed_attack_edit_scores, threshold=threshold)
                perturbed_attack_across_layers_score = calculate_across_layers_score(perturbed_attack_results)
                logit_lens_attack_results = calculate_edit_score_statistics_squared(logit_lens_attack_edit_scores, threshold=threshold)
                logit_lens_attack_across_layers_score = calculate_across_layers_score(logit_lens_attack_results)

                delta_attack_mean_scores = [delta_attack_edit_score.get_delta_attack_score(threshold)['mean'] for delta_attack_edit_score in delta_attack_edit_scores]
                delta_attack_scores = {'mean': np.mean(delta_attack_mean_scores), 'min': np.min(delta_attack_mean_scores)}

                # Get the score of each sub-module (residual, mlp, attention) using the mean of residual_after range_score
                efficacy_mean = efficacy_across_layers_score['residual_after']['range_score_mean']['mean']
                efficacy_min = efficacy_across_layers_score['residual_after']['range_score_mean']['min']
                perturbed_attack_mean = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                perturbed_attack_min = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['min']
                logit_lens_attack_mean = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                logit_lens_attack_min = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['min']
                delta_attack_mean = delta_attack_scores['mean']
                delta_attack_min = delta_attack_scores['min']

                mean_core_scores = [efficacy_mean, specificity_score]
                mean_attack_scores = [delta_attack_mean, perturbed_attack_mean, logit_lens_attack_mean]
                min_core_scores = [efficacy_min, specificity_score]
                min_attack_scores = [delta_attack_min, perturbed_attack_min, logit_lens_attack_min]

                harmonic_core_mean = calculate_harmonic_mean(mean_core_scores)
                harmonic_core_min = calculate_harmonic_mean(min_core_scores)
                harmonic_attack_mean = calculate_harmonic_mean(mean_attack_scores)
                harmonic_attack_min = calculate_harmonic_mean(min_attack_scores)

                res_dict[f"{n_neurons}"][f"{seed}"][f"{threshold}"] = {
                    # 'efficacy_mean': efficacy_mean,
                    # 'delta_attack_mean': delta_attack_mean,
                    # 'perturbed_attack_mean': perturbed_attack_mean,
                    # 'logit_lens_attack_mean': logit_lens_attack_mean,
                    # 'harmonic_core_mean': harmonic_core_mean,
                    # 'harmonic_attack_mean': harmonic_attack_mean,

                    'efficacy_min': efficacy_min,
                    'delta_attack_min': delta_attack_min,
                    'perturbed_attack_min': perturbed_attack_min,
                    'logit_lens_attack_min': logit_lens_attack_min,
                    'harmonic_core_min': harmonic_core_min,
                    'harmonic_attack_min': harmonic_attack_min,
                    'specificity': specificity_score,
                }
                if config.perplexity:
                    res_dict[f"{n_neurons}"][f"{seed}"][f"{threshold}"]['perplexity'] = exp_res_dict['perplexity']

            # After each experiment, delete the model and tokenizer to reset the original weights of the model, clear the cache
            del model
            del exp_model_editor
            del tokenizer
            torch.cuda.empty_cache()

    if config.log_wandb:
        res_plot = plot_multi_experiment_results_revs(res_dict, return_plot=True)
        wandb.log({"EMAIL Aggregated Results": res_plot})
        # Add aggregated results after all iterations
        agg_res_dict = calc_aggregate_results_revs(res_dict, seeds, n_neurons_list, score_thresholds)
        wandb.log(dict(agg_res_dict))

    return dict(res_dict)


def revs_multi_ssn_exp(config: REVSConfig, seeds: List[int], n_neurons_list: List[int], score_thresholds: List[int]):
    """
    Run a multiple SSN experiment with REVS, on different seeds and n_neurons
    """
    multi_exp_config = config.to_dict()
    del multi_exp_config['n_neurons'], multi_exp_config['seed'], multi_exp_config['score_threshold']
    multi_exp_config['n_neurons_list'] = n_neurons_list
    multi_exp_config['seeds'] = seeds
    multi_exp_config['score_thresholds'] = score_thresholds

    if config.log_wandb:
        exp_name = "REVS SSN MULTI"
        num_of_examples = config.unlearn_num_examples
        if num_of_examples is not None:
            exp_name += f" {num_of_examples}e, "
        exp_name += (f"{seeds}s, {n_neurons_list}n, {score_thresholds}st, "
                    f"{config.max_prompt_len}mp_len, "
                    f"{config.residual_top_rank_margin/1000}krtop, "
                    f"{config.residual_bottom_rank_margin/1000}krbottom, "
                    f"{config.act_filter}")
        wandb.init(project="delpii", config=multi_exp_config, name=exp_name)

    pinv_lm_head = None

    res_dict = defaultdict(lambda: defaultdict(dict))
    for n_neurons in tqdm(n_neurons_list, desc="Processing neurons"):
        for seed in tqdm(seeds, desc="Processing seeds", leave=False):
            exp_config = deepcopy(config)
            exp_config.n_neurons = n_neurons
            exp_config.seed = seed
            exp_config.log_wandb = False
            exp_config.save_model = False

            prompts_dict, targets_dict, df_ssn_dict = get_dataset_ssn(seed=seed, split_value=config.unlearn_num_examples)
            model, tokenizer = load_model_tokenizer_ssn(device=device)
            if pinv_lm_head is None:  # calculate pinv_lm_head only once
                pinv_lm_head = torch.pinverse(model.lm_head.weight).to(device)

            exp_res_dict, exp_model_editor = run_revs_exp(
                model,
                tokenizer,
                prompts_dict,
                targets_dict,
                exp_config,
                pinv_lm_head,
                specificity=True,
                generality=True, 
                extraction=True
            )

            efficacy_edit_scores = exp_res_dict['efficacy']
            specificity_score = exp_res_dict['specificity']
            generality_edit_scores = exp_res_dict['generality']
            # extraction attacks scores
            delta_attack_edit_scores = exp_res_dict['delta_attack']
            perturbed_attack_edit_scores = exp_res_dict['perturb_attack']
            logit_lens_attack_edit_scores = exp_res_dict['logit_lens_attack']

            for threshold in tqdm(score_thresholds, desc="Processing thresholds", leave=False):

                efficacy_results = calculate_edit_score_statistics_squared(efficacy_edit_scores, threshold=threshold)
                generality_results = calculate_edit_score_statistics_squared(generality_edit_scores, threshold=threshold)
                perturbed_attack_results = calculate_edit_score_statistics_squared(perturbed_attack_edit_scores, threshold=threshold)
                logit_lens_attack_results = calculate_edit_score_statistics_squared(logit_lens_attack_edit_scores, threshold=threshold)

                efficacy_across_layers_score = calculate_across_layers_score(efficacy_results)
                generality_across_layers_score = calculate_across_layers_score(generality_results)
                perturbed_attack_across_layers_score = calculate_across_layers_score(perturbed_attack_results)
                logit_lens_attack_across_layers_score = calculate_across_layers_score(logit_lens_attack_results)

                delta_attack_mean_scores = [delta_attack_edit_score.get_delta_attack_score(threshold)['mean'] for delta_attack_edit_score in delta_attack_edit_scores]
                delta_attack_scores = {'mean': np.mean(delta_attack_mean_scores), 'min': np.min(delta_attack_mean_scores)}

                # Get the score of each sub-module (residual, mlp, attention) using the mean of residual_after range_score
                efficacy_mean = efficacy_across_layers_score['residual_after']['range_score_mean']['mean']
                efficacy_min = efficacy_across_layers_score['residual_after']['range_score_mean']['min']
                generality_mean = generality_across_layers_score['residual_after']['range_score_mean']['mean']
                generality_min = generality_across_layers_score['residual_after']['range_score_mean']['min']
                perturbed_attack_mean = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                perturbed_attack_min = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['min']
                logit_lens_attack_mean = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                logit_lens_attack_min = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['min']

                delta_attack_mean = delta_attack_scores['mean']
                delta_attack_min = delta_attack_scores['min']

                mean_core_scores = [efficacy_mean, generality_mean, specificity_score]
                mean_attack_scores = [delta_attack_mean, perturbed_attack_mean, logit_lens_attack_mean]
                min_core_scores = [efficacy_min, generality_min, specificity_score]
                min_attack_scores = [delta_attack_min, perturbed_attack_min, logit_lens_attack_min]

                harmonic_core_mean = calculate_harmonic_mean(mean_core_scores)
                harmonic_core_min = calculate_harmonic_mean(min_core_scores)
                harmonic_attack_mean = calculate_harmonic_mean(mean_attack_scores)
                harmonic_attack_min = calculate_harmonic_mean(min_attack_scores)

                res_dict[f"{n_neurons}"][f"{seed}"][f"{threshold}"] = {
                    # 'efficacy_mean': efficacy_mean,
                    # 'generality_mean': generality_mean,
                    # 'delta_attack_mean': delta_attack_mean,
                    # 'perturbed_attack_mean': perturbed_attack_mean,
                    # 'logit_lens_attack_mean': logit_lens_attack_mean,
                    # 'harmonic_core_mean': harmonic_core_mean,
                    # 'harmonic_attack_mean': harmonic_attack_mean,

                    'efficacy_min': efficacy_min,
                    'generality_min': generality_min,
                    'delta_attack_min': delta_attack_min,
                    'perturbed_attack_min': perturbed_attack_min,
                    'logit_lens_attack_min': logit_lens_attack_min,
                    'harmonic_core_min': harmonic_core_min,
                    'harmonic_attack_min': harmonic_attack_min,

                    'specificity': specificity_score,
                }
                if config.perplexity:
                    res_dict[f"{n_neurons}"][f"{seed}"][f"{threshold}"]['perplexity'] = exp_res_dict['perplexity']

            # After each experiment, delete the model and tokenizer to reset the original weights of the model, clear the cache
            del model
            del exp_model_editor
            del tokenizer
            torch.cuda.empty_cache()

    if config.log_wandb:
        res_plot = plot_multi_experiment_results_revs(res_dict, return_plot=True)
        wandb.log({"SSN Aggregated Results": res_plot})
        # Add aggregated results after all iterations
        agg_res_dict = calc_aggregate_results_revs(res_dict, seeds, n_neurons_list, score_thresholds)
        wandb.log(dict(agg_res_dict))

    return dict(res_dict)


def memit_multi_email_exp(config: MemitConfig, seeds: List[int], score_thresholds: List[int], v_lrs: List[float], loss_breaks: List[float], loss_pred_prob_coefs: List[float]):
    """
    Run a multiple EMAIL experiments with MEMIT, on different seeds and loss break and v_lr
    """

    def nested_dict():
        return defaultdict(nested_dict)

    multi_exp_config = config.to_dict()
    del multi_exp_config['seed'], multi_exp_config['score_threshold'], multi_exp_config['v_lr'], multi_exp_config['loss_break'], multi_exp_config['loss_pred_prob_coef']
    multi_exp_config['seeds'] = seeds
    multi_exp_config['score_thresholds'] = score_thresholds
    multi_exp_config['v_lrs'] = v_lrs
    multi_exp_config['loss_breaks'] = loss_breaks
    multi_exp_config['loss_pred_prob_coefs'] = loss_pred_prob_coefs

    if config.log_wandb:
        exp_name = "MEMIT EMAIL MULTI"
        num_of_examples = config.unlearn_num_examples
        if num_of_examples is not None:
            exp_name += f" {num_of_examples}e, "
        exp_name += (f"{seeds}s, {v_lrs}lr, {loss_breaks}lb, {loss_pred_prob_coefs}lc, "
                    f"{score_thresholds}st, {config.v_num_grad_steps}steps, "
                    f"{config.max_prompt_len}mp_len")
        wandb.init(project="delpii", config=multi_exp_config, name=exp_name)

    res_dict = nested_dict()
    for v_lr in tqdm(v_lrs, desc="Processing v_lrs"):
        for loss_break in tqdm(loss_breaks, desc="Processing loss_breaks", leave=False):
            for loss_pred_prob_coef in tqdm(loss_pred_prob_coefs, desc="Processing loss_pred_prob_coefs", leave=False):
                for seed in tqdm(seeds, desc="Processing seeds", leave=False):
                    exp_config = deepcopy(config)
                    exp_config.v_lr = v_lr
                    exp_config.seed = seed
                    exp_config.loss_break = loss_break
                    exp_config.loss_pred_prob_coef = loss_pred_prob_coef
                    exp_config.log_wandb = False
                    exp_config.save_model = False

                    prompts_dict, targets_dict, df_email_dict = get_dataset_email(seed=seed, split_value=config.unlearn_num_examples, max_prompt_len=config.max_prompt_len)
                    model, tokenizer = load_model_tokenizer(device=device)

                    exp_res_dict = run_memit_exp(model, tokenizer, prompts_dict, targets_dict, exp_config, specificity=True, generality=False, extraction=True)

                    efficacy_edit_scores = exp_res_dict['efficacy']
                    specificity_score = exp_res_dict['specificity']
                    # extraction attacks scores
                    delta_attack_edit_scores = exp_res_dict['delta_attack']
                    perturbed_attack_edit_scores = exp_res_dict['perturb_attack']
                    logit_lens_attack_edit_scores = exp_res_dict['logit_lens_attack']

                    for threshold in tqdm(score_thresholds, desc="Processing thresholds", leave=False):

                        efficacy_results = calculate_edit_score_statistics_squared(efficacy_edit_scores, threshold=threshold)
                        perturbed_attack_results = calculate_edit_score_statistics_squared(perturbed_attack_edit_scores, threshold=threshold)
                        logit_lens_attack_results = calculate_edit_score_statistics_squared(logit_lens_attack_edit_scores, threshold=threshold)

                        efficacy_across_layers_score = calculate_across_layers_score(efficacy_results)
                        perturbed_attack_across_layers_score = calculate_across_layers_score(perturbed_attack_results)
                        logit_lens_attack_across_layers_score = calculate_across_layers_score(logit_lens_attack_results)

                        delta_attack_mean_scores = [delta_attack_edit_score.get_delta_attack_score(threshold)['mean'] for delta_attack_edit_score in delta_attack_edit_scores]
                        delta_attack_scores = {'mean': np.mean(delta_attack_mean_scores), 'min': np.min(delta_attack_mean_scores)}

                        efficacy_mean = efficacy_across_layers_score['residual_after']['range_score_mean']['mean']
                        efficacy_min = efficacy_across_layers_score['residual_after']['range_score_mean']['min']
                        perturbed_attack_mean = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                        perturbed_attack_min = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['min']
                        logit_lens_attack_mean = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                        logit_lens_attack_min = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['min']
                        delta_attack_mean = delta_attack_scores['mean']
                        delta_attack_min = delta_attack_scores['min']

                        mean_core_scores = [efficacy_mean, specificity_score]
                        mean_attack_scores = [delta_attack_mean, perturbed_attack_mean, logit_lens_attack_mean]
                        min_core_scores = [efficacy_min, specificity_score]
                        min_attack_scores = [delta_attack_min, perturbed_attack_min, logit_lens_attack_min]

                        harmonic_core_mean = calculate_harmonic_mean(mean_core_scores)
                        harmonic_core_min = calculate_harmonic_mean(min_core_scores)
                        harmonic_attack_mean = calculate_harmonic_mean(mean_attack_scores)
                        harmonic_attack_min = calculate_harmonic_mean(min_attack_scores)

                        res_dict[f"{v_lr}"][f"{loss_break}"][f"{loss_pred_prob_coef}"][f"{seed}"][f"{threshold}"] = {
                            # 'efficacy_mean': efficacy_mean,
                            # 'delta_attack_mean': delta_attack_mean,
                            # 'perturbed_attack_mean': perturbed_attack_mean,
                            # 'logit_lens_attack_mean': logit_lens_attack_mean,
                            # 'harmonic_core_mean': harmonic_core_mean,
                            # 'harmonic_attack_mean': harmonic_attack_mean,

                            'efficacy_min': efficacy_min,
                            'delta_attack_min': delta_attack_min,
                            'perturbed_attack_min': perturbed_attack_min,
                            'logit_lens_attack_min': logit_lens_attack_min,
                            'harmonic_core_min': harmonic_core_min,
                            'harmonic_attack_min': harmonic_attack_min,

                            'specificity': specificity_score,
                        }
                        if config.perplexity:
                            res_dict[f"{v_lr}"][f"{loss_break}"][f"{loss_pred_prob_coef}"][f"{seed}"][f"{threshold}"]['perplexity'] = exp_res_dict['perplexity']

                    # After each experiment, delete the model and tokenizer to reset the original weights of the model, clear the cache
                    del model
                    del tokenizer
                    torch.cuda.empty_cache()

    if config.log_wandb:
        res_plot = plot_multi_experiment_results_memit(res_dict, return_plot=True)
        wandb.log({"EMAIL Aggregated Results": res_plot})
        # Add aggregated results after all iterations
        agg_res_dict = calc_aggregate_results_memit(res_dict, seeds, v_lrs, loss_breaks, loss_pred_prob_coefs, score_thresholds)
        wandb.log(dict(agg_res_dict))

    return dict(res_dict)


def memit_multi_ssn_exp(config: MemitConfig, seeds: List[int], score_thresholds: List[int], v_lrs: List[float], loss_breaks: List[float], loss_pred_prob_coefs: List[float]):
    """
    Run a multiple SSN experiment with MEMIT, on different seeds and loss break and v_lr
    """

    def nested_dict():
        return defaultdict(nested_dict)

    multi_exp_config = config.to_dict()
    del multi_exp_config['seed'], multi_exp_config['score_threshold'], multi_exp_config['v_lr'], multi_exp_config['loss_break'], multi_exp_config['loss_pred_prob_coef']
    multi_exp_config['seeds'] = seeds
    multi_exp_config['score_thresholds'] = score_thresholds
    multi_exp_config['v_lrs'] = v_lrs
    multi_exp_config['loss_breaks'] = loss_breaks

    if config.log_wandb:
        exp_name = "MEMIT SSN MULTI"
        num_of_examples = config.unlearn_num_examples
        if num_of_examples is not None:
            exp_name += f" {num_of_examples}e, "
        exp_name += (f"{seeds}s, {v_lrs}lr, {loss_breaks}lb, {loss_pred_prob_coefs}lc, "
                    f"{score_thresholds}st, {config.v_num_grad_steps}steps, "
                    f"{config.max_prompt_len}mp_len")
        wandb.init(project="delpii", config=multi_exp_config, name=exp_name)

    res_dict = nested_dict()
    for v_lr in tqdm(v_lrs, desc="Processing v_lrs"):
        for loss_break in tqdm(loss_breaks, desc="Processing loss_breaks", leave=False):
            for loss_pred_prob_coef in tqdm(loss_pred_prob_coefs, desc="Processing loss_pred_prob_coefs", leave=False):
                for seed in tqdm(seeds, desc="Processing seeds", leave=False):
                    exp_config = deepcopy(config)
                    exp_config.v_lr = v_lr
                    exp_config.seed = seed
                    exp_config.loss_break = loss_break
                    exp_config.loss_pred_prob_coef = loss_pred_prob_coef
                    exp_config.log_wandb = False
                    exp_config.save_model = False

                    prompts_dict, targets_dict, df_ssn_dict = get_dataset_ssn(seed=seed, split_value=config.unlearn_num_examples)
                    model, tokenizer = load_model_tokenizer_ssn(device=device)

                    exp_res_dict = run_memit_exp(model, tokenizer, prompts_dict, targets_dict, exp_config, specificity=True, generality=True, extraction=True)

                    efficacy_edit_scores = exp_res_dict['efficacy']
                    specificity_score = exp_res_dict['specificity']
                    generality_edit_scores = exp_res_dict['generality']
                    # extraction attacks scores
                    delta_attack_edit_scores = exp_res_dict['delta_attack']
                    perturbed_attack_edit_scores = exp_res_dict['perturb_attack']
                    logit_lens_attack_edit_scores = exp_res_dict['logit_lens_attack']

                    for threshold in tqdm(score_thresholds, desc="Processing thresholds", leave=False):

                        efficacy_results = calculate_edit_score_statistics_squared(efficacy_edit_scores, threshold=threshold)
                        generality_results = calculate_edit_score_statistics_squared(generality_edit_scores, threshold=threshold)
                        perturbed_attack_results = calculate_edit_score_statistics_squared(perturbed_attack_edit_scores, threshold=threshold)
                        logit_lens_attack_results = calculate_edit_score_statistics_squared(logit_lens_attack_edit_scores, threshold=threshold)

                        efficacy_across_layers_score = calculate_across_layers_score(efficacy_results)
                        generality_across_layers_score = calculate_across_layers_score(generality_results)
                        perturbed_attack_across_layers_score = calculate_across_layers_score(perturbed_attack_results)
                        logit_lens_attack_across_layers_score = calculate_across_layers_score(logit_lens_attack_results)

                        delta_attack_mean_scores = [delta_attack_edit_score.get_delta_attack_score(threshold)['mean'] for delta_attack_edit_score in delta_attack_edit_scores]
                        delta_attack_scores = {'mean': np.mean(delta_attack_mean_scores), 'min': np.min(delta_attack_mean_scores)}

                        efficacy_mean = efficacy_across_layers_score['residual_after']['range_score_mean']['mean']
                        efficacy_min = efficacy_across_layers_score['residual_after']['range_score_mean']['min']
                        generality_mean = generality_across_layers_score['residual_after']['range_score_mean']['mean']
                        generality_min = generality_across_layers_score['residual_after']['range_score_mean']['min']
                        perturbed_attack_mean = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                        perturbed_attack_min = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['min']
                        logit_lens_attack_mean = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                        logit_lens_attack_min = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['min']
                        delta_attack_mean = delta_attack_scores['mean']
                        delta_attack_min = delta_attack_scores['min']

                        mean_core_scores = [efficacy_mean, generality_mean, specificity_score]
                        mean_attack_scores = [delta_attack_mean, perturbed_attack_mean, logit_lens_attack_mean]
                        min_core_scores = [efficacy_min, generality_min, specificity_score]
                        min_attack_scores = [delta_attack_min, perturbed_attack_min, logit_lens_attack_min]

                        harmonic_core_mean = calculate_harmonic_mean(mean_core_scores)
                        harmonic_core_min = calculate_harmonic_mean(min_core_scores)
                        harmonic_attack_mean = calculate_harmonic_mean(mean_attack_scores)
                        harmonic_attack_min = calculate_harmonic_mean(min_attack_scores)

                        res_dict[f"{v_lr}"][f"{loss_break}"][f"{loss_pred_prob_coef}"][f"{seed}"][f"{threshold}"] = {
                            # 'efficacy_mean': efficacy_mean,
                            # 'generality_mean': generality_mean,
                            # 'delta_attack_mean': delta_attack_mean,
                            # 'perturbed_attack_mean': perturbed_attack_mean,
                            # 'logit_lens_attack_mean': logit_lens_attack_mean,
                            # 'harmonic_core_mean': harmonic_core_mean,
                            # 'harmonic_attack_mean': harmonic_attack_mean,

                            'efficacy_min': efficacy_min,
                            'generality_min': generality_min,
                            'delta_attack_min': delta_attack_min,
                            'perturbed_attack_min': perturbed_attack_min,
                            'logit_lens_attack_min': logit_lens_attack_min,
                            'harmonic_core_min': harmonic_core_min,
                            'harmonic_attack_min': harmonic_attack_min,

                            'specificity': specificity_score,
                        }
                        if config.perplexity:
                            res_dict[f"{v_lr}"][f"{loss_break}"][f"{loss_pred_prob_coef}"][f"{seed}"][f"{threshold}"]['perplexity'] = exp_res_dict['perplexity']

                    # After each experiment, delete the model and tokenizer to reset the original weights of the model, clear the cache
                    del model
                    del tokenizer
                    torch.cuda.empty_cache()

    if config.log_wandb:
        res_plot = plot_multi_experiment_results_memit(res_dict, return_plot=True)
        wandb.log({"SSN Aggregated Results": res_plot})
        # Add aggregated results after all iterations
        agg_res_dict = calc_aggregate_results_memit(res_dict, seeds, v_lrs, loss_breaks, loss_pred_prob_coefs, score_thresholds)
        wandb.log(dict(agg_res_dict))

    return dict(res_dict)


def ft_multi_email_exp(config: FTConfig, seeds: List[int], score_thresholds: List[int], lrs: List[float], loss_breaks: List[float], norm_constraints: List[float]):
    """
    Run a multiple EMAIL experiment with FT
    """

    def nested_dict():
        return defaultdict(nested_dict)

    multi_exp_config = config.to_dict()
    del multi_exp_config['seed'], multi_exp_config['score_threshold'], multi_exp_config['lr'], multi_exp_config['loss_break'], multi_exp_config['norm_constraint']
    multi_exp_config['seeds'] = seeds
    multi_exp_config['score_thresholds'] = score_thresholds
    multi_exp_config['lrs'] = lrs
    multi_exp_config['loss_breaks'] = loss_breaks
    multi_exp_config['norm_constraints'] = norm_constraints

    if config.log_wandb:
        exp_name = "FTL EMAIL MULTI"
        num_of_examples = config.unlearn_num_examples
        layers = config.layers
        if num_of_examples is not None:
            exp_name += f" {num_of_examples}e, "
        if layers is not None:
            exp_name += f" {layers}lyrs, "
        exp_name += f"{seeds}s, {lrs}lr, {loss_breaks}lb, {norm_constraints}nc, {score_thresholds}st"
        wandb.init(project="delpii", config=multi_exp_config, name=exp_name)

    res_dict = nested_dict()
    for lr in tqdm(lrs, desc="Processing lrs"):
        for loss_break in tqdm(loss_breaks, desc="Processing loss_breaks", leave=False):
            for norm_constraint in tqdm(norm_constraints, desc="Processing norm_constraints", leave=False):
                for seed in tqdm(seeds, desc="Processing seeds", leave=False):
                    exp_config = deepcopy(config)
                    exp_config.lr = lr
                    exp_config.seed = seed
                    exp_config.loss_break = loss_break
                    exp_config.norm_constraint = norm_constraint
                    exp_config.log_wandb = False
                    exp_config.save_model = False

                    prompts_dict, targets_dict, df_email_dict = get_dataset_email(seed=seed, split_value=config.unlearn_num_examples)
                    model, tokenizer = load_model_tokenizer(device='auto')  # needed for multiple GPU usage for the FT

                    exp_res_dict = run_ft_exp(model, tokenizer, prompts_dict, targets_dict, exp_config, specificity=True, generality=False, extraction=True)

                    efficacy_edit_scores = exp_res_dict['efficacy']
                    specificity_score = exp_res_dict['specificity']
                    # extraction attacks scores
                    delta_attack_edit_scores = exp_res_dict['delta_attack']
                    perturbed_attack_edit_scores = exp_res_dict['perturb_attack']
                    logit_lens_attack_edit_scores = exp_res_dict['logit_lens_attack']

                    for threshold in tqdm(score_thresholds, desc="Processing thresholds", leave=False):

                        efficacy_results = calculate_edit_score_statistics_squared(efficacy_edit_scores, threshold=threshold)
                        perturbed_attack_results = calculate_edit_score_statistics_squared(perturbed_attack_edit_scores, threshold=threshold)
                        logit_lens_attack_results = calculate_edit_score_statistics_squared(logit_lens_attack_edit_scores, threshold=threshold)

                        efficacy_across_layers_score = calculate_across_layers_score(efficacy_results)
                        perturbed_attack_across_layers_score = calculate_across_layers_score(perturbed_attack_results)
                        logit_lens_attack_across_layers_score = calculate_across_layers_score(logit_lens_attack_results)

                        delta_attack_mean_scores = [delta_attack_edit_score.get_delta_attack_score(threshold)['mean'] for delta_attack_edit_score in delta_attack_edit_scores]
                        delta_attack_scores = {'mean': np.mean(delta_attack_mean_scores), 'min': np.min(delta_attack_mean_scores)}

                        efficacy_mean = efficacy_across_layers_score['residual_after']['range_score_mean']['mean']
                        efficacy_min = efficacy_across_layers_score['residual_after']['range_score_mean']['min']
                        perturbed_attack_mean = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                        perturbed_attack_min = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['min']
                        logit_lens_attack_mean = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                        logit_lens_attack_min = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['min']
                        delta_attack_mean = delta_attack_scores['mean']
                        delta_attack_min = delta_attack_scores['min']

                        mean_core_scores = [efficacy_mean, specificity_score]
                        mean_attack_scores = [delta_attack_mean, perturbed_attack_mean, logit_lens_attack_mean]
                        min_core_scores = [efficacy_min, specificity_score]
                        min_attack_scores = [delta_attack_min, perturbed_attack_min, logit_lens_attack_min]

                        harmonic_core_mean = calculate_harmonic_mean(mean_core_scores)
                        harmonic_core_min = calculate_harmonic_mean(min_core_scores)
                        harmonic_attack_mean = calculate_harmonic_mean(mean_attack_scores)
                        harmonic_attack_min = calculate_harmonic_mean(min_attack_scores)

                        res_dict[f"{lr}"][f"{loss_break}"][f"{norm_constraint}"][f"{seed}"][f"{threshold}"] = {
                            # 'efficacy_mean': efficacy_mean,
                            # 'delta_attack_mean': delta_attack_mean,
                            # 'perturbed_attack_mean': perturbed_attack_mean,
                            # 'logit_lens_attack_mean': logit_lens_attack_mean,
                            # 'harmonic_core_mean': harmonic_core_mean,
                            # 'harmonic_attack_mean': harmonic_attack_mean,

                            'efficacy_min': efficacy_min,
                            'delta_attack_min': delta_attack_min,
                            'perturbed_attack_min': perturbed_attack_min,
                            'logit_lens_attack_min': logit_lens_attack_min,
                            'harmonic_core_min': harmonic_core_min,
                            'harmonic_attack_min': harmonic_attack_min,

                            'specificity': specificity_score,
                        }
                        if config.perplexity:
                            res_dict[f"{lr}"][f"{loss_break}"][f"{norm_constraint}"][f"{seed}"][f"{threshold}"]['perplexity'] = exp_res_dict['perplexity']

                    # After each experiment, delete the model and tokenizer to reset the original weights of the model, clear the cache
                    del model
                    del tokenizer
                    torch.cuda.empty_cache()

    if config.log_wandb:
        res_plot = plot_multi_experiment_results_ft(res_dict, return_plot=True)
        wandb.log({"EMAIL Aggregated Results": res_plot})
        # Add aggregated results after all iterations
        agg_res_dict = calc_aggregate_results_ft(res_dict, seeds, lrs, loss_breaks, norm_constraints, score_thresholds)
        wandb.log(dict(agg_res_dict))

    return dict(res_dict)


def ft_multi_ssn_exp(config: FTConfig, seeds: List[int], score_thresholds: List[int], lrs: List[float], loss_breaks: List[float], norm_constraints: List[float]):
    """
    Run a multiple SSN experiment with FT
    """

    def nested_dict():
        return defaultdict(nested_dict)

    multi_exp_config = config.to_dict()
    del multi_exp_config['seed'], multi_exp_config['score_threshold'], multi_exp_config['lr'], multi_exp_config['loss_break']
    multi_exp_config['seeds'] = seeds
    multi_exp_config['score_thresholds'] = score_thresholds
    multi_exp_config['lrs'] = lrs
    multi_exp_config['loss_breaks'] = loss_breaks
    multi_exp_config['norm_constraints'] = norm_constraints


    if config.log_wandb:
        exp_name = "FTL SSN MULTI"
        num_of_examples = config.unlearn_num_examples
        layers = config.layers
        if num_of_examples is not None:
            exp_name += f" {num_of_examples}e, "
        if layers is not None:
            exp_name += f" {layers}lyrs, "
        exp_name += f"{seeds}s, {lrs}lr, {loss_breaks}lb, {norm_constraints}nc, {score_thresholds}st"
        wandb.init(project="delpii", config=multi_exp_config, name=exp_name)


    res_dict = nested_dict()
    for lr in tqdm(lrs, desc="Processing lrs"):
        for loss_break in tqdm(loss_breaks, desc="Processing loss_breaks", leave=False):
            for norm_constraint in tqdm(norm_constraints, desc="Processing norm_constraints", leave=False):
                for seed in tqdm(seeds, desc="Processing seeds", leave=False):
                    exp_config = deepcopy(config)
                    exp_config.lr = lr
                    exp_config.seed = seed
                    exp_config.loss_break = loss_break
                    exp_config.norm_constraint = norm_constraint
                    exp_config.log_wandb = False
                    exp_config.save_model = False
                    prompts_dict, targets_dict, df_ssn_dict = get_dataset_ssn(seed=seed, split_value=config.unlearn_num_examples)
                    model, tokenizer = load_model_tokenizer_ssn(device='auto')  # needed for multiple GPU usage for the FT

                    exp_res_dict = run_ft_exp(model, tokenizer, prompts_dict, targets_dict, exp_config, specificity=True, generality=True, extraction=True)

                    efficacy_edit_scores = exp_res_dict['efficacy']
                    specificity_score = exp_res_dict['specificity']
                    generality_edit_scores = exp_res_dict['generality']
                    # extraction attacks scores
                    delta_attack_edit_scores = exp_res_dict['delta_attack']
                    perturbed_attack_edit_scores = exp_res_dict['perturb_attack']
                    logit_lens_attack_edit_scores = exp_res_dict['logit_lens_attack']

                    for threshold in tqdm(score_thresholds, desc="Processing thresholds", leave=False):

                        efficacy_results = calculate_edit_score_statistics_squared(efficacy_edit_scores, threshold=threshold)
                        generality_results = calculate_edit_score_statistics_squared(generality_edit_scores, threshold=threshold)
                        perturbed_attack_results = calculate_edit_score_statistics_squared(perturbed_attack_edit_scores, threshold=threshold)
                        logit_lens_attack_results = calculate_edit_score_statistics_squared(logit_lens_attack_edit_scores, threshold=threshold)

                        efficacy_across_layers_score = calculate_across_layers_score(efficacy_results)
                        generality_across_layers_score = calculate_across_layers_score(generality_results)
                        perturbed_attack_across_layers_score = calculate_across_layers_score(perturbed_attack_results)
                        logit_lens_attack_across_layers_score = calculate_across_layers_score(logit_lens_attack_results)

                        delta_attack_mean_scores = [delta_attack_edit_score.get_delta_attack_score(threshold)['mean'] for delta_attack_edit_score in delta_attack_edit_scores]
                        delta_attack_scores = {'mean': np.mean(delta_attack_mean_scores), 'min': np.min(delta_attack_mean_scores)}

                        efficacy_mean = efficacy_across_layers_score['residual_after']['range_score_mean']['mean']
                        efficacy_min = efficacy_across_layers_score['residual_after']['range_score_mean']['min']
                        generality_mean = generality_across_layers_score['residual_after']['range_score_mean']['mean']
                        generality_min = generality_across_layers_score['residual_after']['range_score_mean']['min']
                        perturbed_attack_mean = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                        perturbed_attack_min = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['min']
                        logit_lens_attack_mean = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                        logit_lens_attack_min = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['min']
                        delta_attack_mean = delta_attack_scores['mean']
                        delta_attack_min = delta_attack_scores['min']

                        mean_core_scores = [efficacy_mean, generality_mean, specificity_score]
                        mean_attack_scores = [delta_attack_mean, perturbed_attack_mean, logit_lens_attack_mean]
                        min_core_scores = [efficacy_min, generality_min, specificity_score]
                        min_attack_scores = [delta_attack_min, perturbed_attack_min, logit_lens_attack_min]

                        harmonic_core_mean = calculate_harmonic_mean(mean_core_scores)
                        harmonic_core_min = calculate_harmonic_mean(min_core_scores)
                        harmonic_attack_mean = calculate_harmonic_mean(mean_attack_scores)
                        harmonic_attack_min = calculate_harmonic_mean(min_attack_scores)

                        res_dict[f"{lr}"][f"{loss_break}"][f"{norm_constraint}"][f"{seed}"][f"{threshold}"] = {
                            # 'efficacy_mean': efficacy_mean,
                            # 'generality_mean': generality_mean,
                            # 'delta_attack_mean': delta_attack_mean,
                            # 'perturbed_attack_mean': perturbed_attack_mean,
                            # 'logit_lens_attack_mean': logit_lens_attack_mean,
                            # 'harmonic_core_mean': harmonic_core_mean,
                            # 'harmonic_attack_mean': harmonic_attack_mean,

                            'efficacy_min': efficacy_min,
                            'generality_min': generality_min,
                            'delta_attack_min': delta_attack_min,
                            'perturbed_attack_min': perturbed_attack_min,
                            'logit_lens_attack_min': logit_lens_attack_min,
                            'harmonic_core_min': harmonic_core_min,
                            'harmonic_attack_min': harmonic_attack_min,

                            'specificity': specificity_score,
                        }
                        if config.perplexity:
                            res_dict[f"{lr}"][f"{loss_break}"][f"{norm_constraint}"][f"{seed}"][f"{threshold}"]['perplexity'] = exp_res_dict['perplexity']

                    # After each experiment, delete the model and tokenizer to reset the original weights of the model, clear the cache
                    del model
                    del tokenizer
                    torch.cuda.empty_cache()

    if config.log_wandb:
        res_plot = plot_multi_experiment_results_ft(res_dict, return_plot=True)
        wandb.log({"SSN Aggregated Results": res_plot})
        # Add aggregated results after all iterations
        agg_res_dict = calc_aggregate_results_ft(res_dict, seeds, lrs, loss_breaks, norm_constraints, score_thresholds)
        wandb.log(dict(agg_res_dict))

    return dict(res_dict)


# ~~~~~~~~~~~~ LLAMA MULTI EXPS ~~~~~~~~~~~~


def revs_llama_multi_email_exp(config: REVSConfig, seeds: List[int], n_neurons_list: List[int], score_thresholds: List[int], message: str = None):
    """
    Run a multiple EMAIL experiment with REVS, on different seeds and n_neurons
    """
    multi_exp_config = config.to_dict()
    del multi_exp_config['n_neurons'], multi_exp_config['seed'], multi_exp_config['score_threshold']
    multi_exp_config['n_neurons_list'] = n_neurons_list
    multi_exp_config['seeds'] = seeds
    multi_exp_config['score_thresholds'] = score_thresholds

    if config.log_wandb:
        exp_name = "REVS LLAMA EMAIL"
        config_name = (f"{seeds}s, {n_neurons_list}n, {score_thresholds}st, "
                f"{config.residual_top_rank_margin/1000}krtop, "
                f"{config.residual_bottom_rank_margin/1000}krbottom, "
                f"{config.act_filter}")
        name = f"{exp_name} {message + ', ' if message else ''}{config_name}"
        wandb.init(project="delpii", config=multi_exp_config, name=name)

    pinv_lm_head = None
    res_dict = defaultdict(lambda: defaultdict(dict))
    for n_neurons in tqdm(n_neurons_list, desc="Processing neurons"):
        for seed in tqdm(seeds, desc="Processing seeds", leave=False):
            exp_config = deepcopy(config)
            exp_config.n_neurons = n_neurons
            exp_config.seed = seed
            exp_config.log_wandb = False
            exp_config.save_model = False

            prompts_dict, targets_dict, df_email_dict = get_dataset_email_llama(seed=seed, split_value=config.unlearn_num_examples)
            model, tokenizer = load_model_tokenizer('llama')
            if pinv_lm_head is None:  # calculate pinv_lm_head only once
                pinv_lm_head = torch.pinverse(model.lm_head.weight).to(device)

            exp_res_dict, exp_model_editor = run_revs_exp(
                model,
                tokenizer,
                prompts_dict,
                targets_dict,
                exp_config,
                pinv_lm_head,
                specificity=True,
                generality=False, 
                extraction=True
            )

            efficacy_edit_scores = exp_res_dict['efficacy']
            specificity_score = exp_res_dict['specificity']
            # generality_edit_scores = exp_res_dict['generality']
            # extraction attacks scores
            delta_attack_edit_scores = exp_res_dict['delta_attack']
            perturbed_attack_edit_scores = exp_res_dict['perturb_attack']
            logit_lens_attack_edit_scores = exp_res_dict['logit_lens_attack']

            for threshold in tqdm(score_thresholds, desc="Processing thresholds", leave=False):

                efficacy_results = calculate_edit_score_statistics_squared(efficacy_edit_scores, threshold=threshold)
                # generality_results = calculate_edit_score_statistics_squared(generality_edit_scores, threshold=threshold)
                perturbed_attack_results = calculate_edit_score_statistics_squared(perturbed_attack_edit_scores, threshold=threshold)
                logit_lens_attack_results = calculate_edit_score_statistics_squared(logit_lens_attack_edit_scores, threshold=threshold)

                efficacy_across_layers_score = calculate_across_layers_score(efficacy_results)
                # generality_across_layers_score = calculate_across_layers_score(generality_results)
                perturbed_attack_across_layers_score = calculate_across_layers_score(perturbed_attack_results)
                logit_lens_attack_across_layers_score = calculate_across_layers_score(logit_lens_attack_results)

                delta_attack_mean_scores = [delta_attack_edit_score.get_delta_attack_score(threshold)['mean'] for delta_attack_edit_score in delta_attack_edit_scores]
                delta_attack_scores = {'mean': np.mean(delta_attack_mean_scores), 'min': np.min(delta_attack_mean_scores)}

                # Get the score of each sub-module (residual, mlp, attention) using the mean of residual_after range_score
                efficacy_mean = efficacy_across_layers_score['residual_after']['range_score_mean']['mean']
                efficacy_min = efficacy_across_layers_score['residual_after']['range_score_mean']['min']
                # generality_mean = generality_across_layers_score['residual_after']['range_score_mean']['mean']
                # generality_min = generality_across_layers_score['residual_after']['range_score_mean']['min']
                perturbed_attack_mean = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                perturbed_attack_min = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['min']
                logit_lens_attack_mean = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                logit_lens_attack_min = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['min']

                delta_attack_mean = delta_attack_scores['mean']
                delta_attack_min = delta_attack_scores['min']

                mean_core_scores = [efficacy_mean, specificity_score]
                mean_attack_scores = [delta_attack_mean, perturbed_attack_mean, logit_lens_attack_mean]
                min_core_scores = [efficacy_min, specificity_score]
                min_attack_scores = [delta_attack_min, perturbed_attack_min, logit_lens_attack_min]

                harmonic_core_mean = calculate_harmonic_mean(mean_core_scores)
                harmonic_core_min = calculate_harmonic_mean(min_core_scores)
                harmonic_attack_mean = calculate_harmonic_mean(mean_attack_scores)
                harmonic_attack_min = calculate_harmonic_mean(min_attack_scores)

                res_dict[f"{n_neurons}"][f"{seed}"][f"{threshold}"] = {
                    # 'efficacy_mean': efficacy_mean,
                    # 'generality_mean': generality_mean,
                    # 'delta_attack_mean': delta_attack_mean,
                    # 'perturbed_attack_mean': perturbed_attack_mean,
                    # 'logit_lens_attack_mean': logit_lens_attack_mean,
                    # 'harmonic_core_mean': harmonic_core_mean,
                    # 'harmonic_attack_mean': harmonic_attack_mean,

                    'efficacy_min': efficacy_min,
                    # 'generality_min': generality_min,
                    'delta_attack_min': delta_attack_min,
                    'perturbed_attack_min': perturbed_attack_min,
                    'logit_lens_attack_min': logit_lens_attack_min,
                    'harmonic_core_min': harmonic_core_min,
                    'harmonic_attack_min': harmonic_attack_min,

                    'specificity': specificity_score,
                }
                if config.perplexity:
                    res_dict[f"{n_neurons}"][f"{seed}"][f"{threshold}"]['perplexity'] = exp_res_dict['perplexity']

            # After each experiment, delete the model and tokenizer to reset the original weights of the model, clear the cache
            del model
            del exp_model_editor
            del tokenizer
            torch.cuda.empty_cache()

    if config.log_wandb:
        res_plot = plot_multi_experiment_results_revs(res_dict, return_plot=True)
        wandb.log({"EMAIL Aggregated Results": res_plot})
        # Add aggregated results after all iterations
        agg_res_dict = calc_aggregate_results_revs(res_dict, seeds, n_neurons_list, score_thresholds)
        wandb.log(dict(agg_res_dict))

    return dict(res_dict)


def memit_llama_multi_email_exp(config: MemitConfig, seeds: List[int], score_thresholds: List[int], v_lrs: List[float], loss_breaks: List[float], loss_pred_prob_coefs: List[float], message: str = None):
    """
    Run a multiple EMAIL experiments with MEMIT, on different seeds and loss break and v_lr
    """

    def nested_dict():
        return defaultdict(nested_dict)

    multi_exp_config = config.to_dict()
    del multi_exp_config['seed'], multi_exp_config['score_threshold'], multi_exp_config['v_lr'], multi_exp_config['loss_break'], multi_exp_config['loss_pred_prob_coef']
    multi_exp_config['seeds'] = seeds
    multi_exp_config['score_thresholds'] = score_thresholds
    multi_exp_config['v_lrs'] = v_lrs
    multi_exp_config['loss_breaks'] = loss_breaks
    multi_exp_config['loss_pred_prob_coefs'] = loss_pred_prob_coefs

    if config.log_wandb:
        exp_name = "MEMIT LLAMA EMAIL"
        if config.unlearn_num_examples is not None:
            exp_name += f" {config.unlearn_num_examples}e, "

        config_name = (f"{seeds}s, {v_lrs}lr, {loss_breaks}lb, {loss_pred_prob_coefs}lc, "
                    f"{score_thresholds}st, {config.v_num_grad_steps}steps, "
                    f"{config.max_prompt_len}mp_len")
        name = f"{exp_name} {message + ', ' if message else ''}{config_name}"
        wandb.init(project="delpii", config=multi_exp_config, name=name)

    res_dict = nested_dict()
    for v_lr in tqdm(v_lrs, desc="Processing v_lrs"):
        for loss_break in tqdm(loss_breaks, desc="Processing loss_breaks", leave=False):
            for loss_pred_prob_coef in tqdm(loss_pred_prob_coefs, desc="Processing loss_pred_prob_coefs", leave=False):
                for seed in tqdm(seeds, desc="Processing seeds", leave=False):
                    exp_config = deepcopy(config)
                    exp_config.v_lr = v_lr
                    exp_config.seed = seed
                    exp_config.loss_break = loss_break
                    exp_config.loss_pred_prob_coef = loss_pred_prob_coef
                    exp_config.log_wandb = False
                    exp_config.save_model = False

                    prompts_dict, targets_dict, df_email_dict = get_dataset_email_llama(seed=seed, split_value=config.unlearn_num_examples)
                    model, tokenizer = load_model_tokenizer(model_name='llama')

                    exp_res_dict = run_memit_exp(model, tokenizer, prompts_dict, targets_dict, exp_config, specificity=True, generality=False, extraction=True)

                    efficacy_edit_scores = exp_res_dict['efficacy']
                    specificity_score = exp_res_dict['specificity']
                    # generality_edit_scores = exp_res_dict['generality']
                    # extraction attacks scores
                    delta_attack_edit_scores = exp_res_dict['delta_attack']
                    perturbed_attack_edit_scores = exp_res_dict['perturb_attack']
                    logit_lens_attack_edit_scores = exp_res_dict['logit_lens_attack']

                    for threshold in tqdm(score_thresholds, desc="Processing thresholds", leave=False):

                        efficacy_results = calculate_edit_score_statistics_squared(efficacy_edit_scores, threshold=threshold)
                        # generality_results = calculate_edit_score_statistics_squared(generality_edit_scores, threshold=threshold)
                        perturbed_attack_results = calculate_edit_score_statistics_squared(perturbed_attack_edit_scores, threshold=threshold)
                        logit_lens_attack_results = calculate_edit_score_statistics_squared(logit_lens_attack_edit_scores, threshold=threshold)

                        efficacy_across_layers_score = calculate_across_layers_score(efficacy_results)
                        # generality_across_layers_score = calculate_across_layers_score(generality_results)
                        perturbed_attack_across_layers_score = calculate_across_layers_score(perturbed_attack_results)
                        logit_lens_attack_across_layers_score = calculate_across_layers_score(logit_lens_attack_results)

                        delta_attack_mean_scores = [delta_attack_edit_score.get_delta_attack_score(threshold)['mean'] for delta_attack_edit_score in delta_attack_edit_scores]
                        delta_attack_scores = {'mean': np.mean(delta_attack_mean_scores), 'min': np.min(delta_attack_mean_scores)}

                        efficacy_mean = efficacy_across_layers_score['residual_after']['range_score_mean']['mean']
                        efficacy_min = efficacy_across_layers_score['residual_after']['range_score_mean']['min']
                        # generality_mean = generality_across_layers_score['residual_after']['range_score_mean']['mean']
                        # generality_min = generality_across_layers_score['residual_after']['range_score_mean']['min']
                        perturbed_attack_mean = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                        perturbed_attack_min = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['min']
                        logit_lens_attack_mean = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                        logit_lens_attack_min = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['min']
                        delta_attack_mean = delta_attack_scores['mean']
                        delta_attack_min = delta_attack_scores['min']

                        mean_core_scores = [efficacy_mean, specificity_score]
                        mean_attack_scores = [delta_attack_mean, perturbed_attack_mean, logit_lens_attack_mean]
                        min_core_scores = [efficacy_min, specificity_score]
                        min_attack_scores = [delta_attack_min, perturbed_attack_min, logit_lens_attack_min]

                        harmonic_core_mean = calculate_harmonic_mean(mean_core_scores)
                        harmonic_core_min = calculate_harmonic_mean(min_core_scores)
                        harmonic_attack_mean = calculate_harmonic_mean(mean_attack_scores)
                        harmonic_attack_min = calculate_harmonic_mean(min_attack_scores)

                        res_dict[f"{v_lr}"][f"{loss_break}"][f"{loss_pred_prob_coef}"][f"{seed}"][f"{threshold}"] = {
                            # 'efficacy_mean': efficacy_mean,
                            # 'generality_mean': generality_mean,
                            # 'delta_attack_mean': delta_attack_mean,
                            # 'perturbed_attack_mean': perturbed_attack_mean,
                            # 'logit_lens_attack_mean': logit_lens_attack_mean,
                            # 'harmonic_core_mean': harmonic_core_mean,
                            # 'harmonic_attack_mean': harmonic_attack_mean,

                            'efficacy_min': efficacy_min,
                            # 'generality_min': generality_min,
                            'delta_attack_min': delta_attack_min,
                            'perturbed_attack_min': perturbed_attack_min,
                            'logit_lens_attack_min': logit_lens_attack_min,
                            'harmonic_core_min': harmonic_core_min,
                            'harmonic_attack_min': harmonic_attack_min,

                            'specificity': specificity_score,
                        }
                        if config.perplexity:
                            res_dict[f"{v_lr}"][f"{loss_break}"][f"{loss_pred_prob_coef}"][f"{seed}"][f"{threshold}"]['perplexity'] = exp_res_dict['perplexity']

                    # After each experiment, delete the model and tokenizer to reset the original weights of the model, clear the cache
                    del model
                    del tokenizer
                    torch.cuda.empty_cache()

    if config.log_wandb:
        res_plot = plot_multi_experiment_results_memit(res_dict, return_plot=True)
        wandb.log({"EMAIL Aggregated Results": res_plot})
        # Add aggregated results after all iterations
        agg_res_dict = calc_aggregate_results_memit(res_dict, seeds, v_lrs, loss_breaks, loss_pred_prob_coefs, score_thresholds)
        wandb.log(dict(agg_res_dict))

    return dict(res_dict)


def ft_llama_multi_email_exp(config: FTConfig, seeds: List[int], score_thresholds: List[int], lrs: List[float], loss_breaks: List[float], norm_constraints: List[float], message: str = None):
    """
    Run a multiple EMAIL experiment with FT
    """

    def nested_dict():
        return defaultdict(nested_dict)

    multi_exp_config = config.to_dict()
    del multi_exp_config['seed'], multi_exp_config['score_threshold'], multi_exp_config['lr'], multi_exp_config['loss_break'], multi_exp_config['norm_constraint']
    multi_exp_config['seeds'] = seeds
    multi_exp_config['score_thresholds'] = score_thresholds
    multi_exp_config['lrs'] = lrs
    multi_exp_config['loss_breaks'] = loss_breaks
    multi_exp_config['norm_constraints'] = norm_constraints

    if config.log_wandb:
        exp_name = "FTL LLAMA EMAIL"
        num_of_examples = config.unlearn_num_examples
        layers = config.layers
        if config.unlearn_num_examples is not None:
            exp_name += f" {config.unlearn_num_examples}e, "
        if layers is not None:
            if len(layers) > 1 and layers == list(range(layers[0], layers[-1] + 1)):
                exp_name += f" {{{layers[0]}..{layers[-1]}}}lyrs, "
            else:
                exp_name += f" {' '.join(map(str, layers))}lyrs, "
        config_name = (f"{seeds}s, {lrs}lr, {loss_breaks}lb, {norm_constraints}nc, "
                    f"{score_thresholds}st, {config.num_grad_steps}steps, "
                    f"{config.max_prompt_len}mp_len")
        name = f"{exp_name} {message + ', ' if message else ''}{config_name}"
        wandb.init(project="delpii", config=multi_exp_config, name=name)

    res_dict = nested_dict()
    for lr in tqdm(lrs, desc="Processing lrs"):
        for loss_break in tqdm(loss_breaks, desc="Processing loss_breaks", leave=False):
            for norm_constraint in tqdm(norm_constraints, desc="Processing norm_constraints", leave=False):
                for seed in tqdm(seeds, desc="Processing seeds", leave=False):
                    exp_config = deepcopy(config)
                    exp_config.lr = lr
                    exp_config.seed = seed
                    exp_config.loss_break = loss_break
                    exp_config.norm_constraint = norm_constraint
                    exp_config.log_wandb = False
                    exp_config.save_model = False

                    prompts_dict, targets_dict, df_email_dict = get_dataset_email_llama(seed=seed, split_value=config.unlearn_num_examples)
                    model, tokenizer = load_model_tokenizer(model_name='llama', device='auto')  # 'auto' needed to support multiple GPU usage for Fine-tuning

                    exp_res_dict = run_ft_exp(model, tokenizer, prompts_dict, targets_dict, exp_config, specificity=True, generality=True, extraction=True)

                    efficacy_edit_scores = exp_res_dict['efficacy']
                    specificity_score = exp_res_dict['specificity']
                    # extraction attacks scores
                    delta_attack_edit_scores = exp_res_dict['delta_attack']
                    perturbed_attack_edit_scores = exp_res_dict['perturb_attack']
                    logit_lens_attack_edit_scores = exp_res_dict['logit_lens_attack']

                    for threshold in tqdm(score_thresholds, desc="Processing thresholds", leave=False):

                        efficacy_results = calculate_edit_score_statistics_squared(efficacy_edit_scores, threshold=threshold)
                        perturbed_attack_results = calculate_edit_score_statistics_squared(perturbed_attack_edit_scores, threshold=threshold)
                        logit_lens_attack_results = calculate_edit_score_statistics_squared(logit_lens_attack_edit_scores, threshold=threshold)

                        efficacy_across_layers_score = calculate_across_layers_score(efficacy_results)
                        perturbed_attack_across_layers_score = calculate_across_layers_score(perturbed_attack_results)
                        logit_lens_attack_across_layers_score = calculate_across_layers_score(logit_lens_attack_results)

                        delta_attack_mean_scores = [delta_attack_edit_score.get_delta_attack_score(threshold)['mean'] for delta_attack_edit_score in delta_attack_edit_scores]
                        delta_attack_scores = {'mean': np.mean(delta_attack_mean_scores), 'min': np.min(delta_attack_mean_scores)}

                        efficacy_mean = efficacy_across_layers_score['residual_after']['range_score_mean']['mean']
                        efficacy_min = efficacy_across_layers_score['residual_after']['range_score_mean']['min']
                        perturbed_attack_mean = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                        perturbed_attack_min = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['min']
                        logit_lens_attack_mean = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                        logit_lens_attack_min = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['min']
                        delta_attack_mean = delta_attack_scores['mean']
                        delta_attack_min = delta_attack_scores['min']

                        mean_core_scores = [efficacy_mean, specificity_score]
                        mean_attack_scores = [delta_attack_mean, perturbed_attack_mean, logit_lens_attack_mean]
                        min_core_scores = [efficacy_min, specificity_score]
                        min_attack_scores = [delta_attack_min, perturbed_attack_min, logit_lens_attack_min]

                        harmonic_core_mean = calculate_harmonic_mean(mean_core_scores)
                        harmonic_core_min = calculate_harmonic_mean(min_core_scores)
                        harmonic_attack_mean = calculate_harmonic_mean(mean_attack_scores)
                        harmonic_attack_min = calculate_harmonic_mean(min_attack_scores)

                        res_dict[f"{lr}"][f"{loss_break}"][f"{norm_constraint}"][f"{seed}"][f"{threshold}"] = {
                            # 'efficacy_mean': efficacy_mean,
                            # 'delta_attack_mean': delta_attack_mean,
                            # 'perturbed_attack_mean': perturbed_attack_mean,
                            # 'logit_lens_attack_mean': logit_lens_attack_mean,
                            # 'harmonic_core_mean': harmonic_core_mean,
                            # 'harmonic_attack_mean': harmonic_attack_mean,

                            'efficacy_min': efficacy_min,
                            'delta_attack_min': delta_attack_min,
                            'perturbed_attack_min': perturbed_attack_min,
                            'logit_lens_attack_min': logit_lens_attack_min,
                            'harmonic_core_min': harmonic_core_min,
                            'harmonic_attack_min': harmonic_attack_min,

                            'specificity': specificity_score,
                        }
                        if config.perplexity:
                            res_dict[f"{lr}"][f"{loss_break}"][f"{norm_constraint}"][f"{seed}"][f"{threshold}"]['perplexity'] = exp_res_dict['perplexity']

                    # After each experiment, delete the model and tokenizer to reset the original weights of the model, clear the cache
                    del model
                    del tokenizer
                    torch.cuda.empty_cache()

    if config.log_wandb:
        res_plot = plot_multi_experiment_results_ft(res_dict, return_plot=True)
        wandb.log({"EMAIL Aggregated Results": res_plot})
        # Add aggregated results after all iterations
        agg_res_dict = calc_aggregate_results_ft(res_dict, seeds, lrs, loss_breaks, norm_constraints, score_thresholds)
        wandb.log(dict(agg_res_dict))

    return dict(res_dict)


def ft_llama_multi_ssn_exp(config: FTConfig, seeds: List[int], score_thresholds: List[int], lrs: List[float], loss_breaks: List[float], norm_constraints: List[float], message: str = None):
    """
    Run a multiple SSN experiment with FT
    """

    def nested_dict():
        return defaultdict(nested_dict)

    multi_exp_config = config.to_dict()
    del multi_exp_config['seed'], multi_exp_config['score_threshold'], multi_exp_config['lr'], multi_exp_config['loss_break'], multi_exp_config['norm_constraint']
    multi_exp_config['seeds'] = seeds
    multi_exp_config['score_thresholds'] = score_thresholds
    multi_exp_config['lrs'] = lrs
    multi_exp_config['loss_breaks'] = loss_breaks
    multi_exp_config['norm_constraints'] = norm_constraints

    if config.log_wandb:
        exp_name = "FTL LLAMA EMAIL"
        num_of_examples = config.unlearn_num_examples
        layers = config.layers
        if config.unlearn_num_examples is not None:
            exp_name += f" {config.unlearn_num_examples}e, "
        if layers is not None:
            if len(layers) > 1 and layers == list(range(layers[0], layers[-1] + 1)):
                exp_name += f" {{{layers[0]}..{layers[-1]}}}lyrs, "
            else:
                exp_name += f" {' '.join(map(str, layers))}lyrs, "
        config_name = (f"{seeds}s, {lrs}lr, {loss_breaks}lb, {norm_constraints}nc, "
                    f"{score_thresholds}st, {config.num_grad_steps}steps, "
                    f"{config.max_prompt_len}mp_len")
        name = f"{exp_name} {message + ', ' if message else ''}{config_name}"
        wandb.init(project="delpii", config=multi_exp_config, name=name)

    res_dict = nested_dict()
    for lr in tqdm(lrs, desc="Processing lrs"):
        for loss_break in tqdm(loss_breaks, desc="Processing loss_breaks", leave=False):
            for norm_constraint in tqdm(norm_constraints, desc="Processing norm_constraints", leave=False):
                for seed in tqdm(seeds, desc="Processing seeds", leave=False):
                    exp_config = deepcopy(config)
                    exp_config.lr = lr
                    exp_config.seed = seed
                    exp_config.loss_break = loss_break
                    exp_config.norm_constraint = norm_constraint
                    exp_config.log_wandb = False
                    exp_config.save_model = False

                    prompts_dict, targets_dict, df_email_dict = get_dataset_email(seed=seed, split_value=config.unlearn_num_examples)
                    model, tokenizer = load_model_tokenizer_ssn(model_name='llama', device='auto')  # 'auto' needed to support multiple GPU usage for Fine-tuning

                    exp_res_dict = run_ft_exp(model, tokenizer, prompts_dict, targets_dict, exp_config, specificity=True, generality=False, extraction=True)

                    efficacy_edit_scores = exp_res_dict['efficacy']
                    specificity_score = exp_res_dict['specificity']
                    # extraction attacks scores
                    delta_attack_edit_scores = exp_res_dict['delta_attack']
                    perturbed_attack_edit_scores = exp_res_dict['perturb_attack']
                    logit_lens_attack_edit_scores = exp_res_dict['logit_lens_attack']

                    for threshold in tqdm(score_thresholds, desc="Processing thresholds", leave=False):

                        efficacy_results = calculate_edit_score_statistics_squared(efficacy_edit_scores, threshold=threshold)
                        perturbed_attack_results = calculate_edit_score_statistics_squared(perturbed_attack_edit_scores, threshold=threshold)
                        logit_lens_attack_results = calculate_edit_score_statistics_squared(logit_lens_attack_edit_scores, threshold=threshold)

                        efficacy_across_layers_score = calculate_across_layers_score(efficacy_results)
                        perturbed_attack_across_layers_score = calculate_across_layers_score(perturbed_attack_results)
                        logit_lens_attack_across_layers_score = calculate_across_layers_score(logit_lens_attack_results)

                        delta_attack_mean_scores = [delta_attack_edit_score.get_delta_attack_score(threshold)['mean'] for delta_attack_edit_score in delta_attack_edit_scores]
                        delta_attack_scores = {'mean': np.mean(delta_attack_mean_scores), 'min': np.min(delta_attack_mean_scores)}

                        efficacy_mean = efficacy_across_layers_score['residual_after']['range_score_mean']['mean']
                        efficacy_min = efficacy_across_layers_score['residual_after']['range_score_mean']['min']
                        perturbed_attack_mean = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                        perturbed_attack_min = perturbed_attack_across_layers_score['residual_after']['range_score_mean']['min']
                        logit_lens_attack_mean = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['mean']
                        logit_lens_attack_min = logit_lens_attack_across_layers_score['residual_after']['range_score_mean']['min']
                        delta_attack_mean = delta_attack_scores['mean']
                        delta_attack_min = delta_attack_scores['min']

                        mean_core_scores = [efficacy_mean, specificity_score]
                        mean_attack_scores = [delta_attack_mean, perturbed_attack_mean, logit_lens_attack_mean]
                        min_core_scores = [efficacy_min, specificity_score]
                        min_attack_scores = [delta_attack_min, perturbed_attack_min, logit_lens_attack_min]

                        harmonic_core_mean = calculate_harmonic_mean(mean_core_scores)
                        harmonic_core_min = calculate_harmonic_mean(min_core_scores)
                        harmonic_attack_mean = calculate_harmonic_mean(mean_attack_scores)
                        harmonic_attack_min = calculate_harmonic_mean(min_attack_scores)

                        res_dict[f"{lr}"][f"{loss_break}"][f"{norm_constraint}"][f"{seed}"][f"{threshold}"] = {
                            # 'efficacy_mean': efficacy_mean,
                            # 'delta_attack_mean': delta_attack_mean,
                            # 'perturbed_attack_mean': perturbed_attack_mean,
                            # 'logit_lens_attack_mean': logit_lens_attack_mean,
                            # 'harmonic_core_mean': harmonic_core_mean,
                            # 'harmonic_attack_mean': harmonic_attack_mean,

                            'efficacy_min': efficacy_min,
                            'delta_attack_min': delta_attack_min,
                            'perturbed_attack_min': perturbed_attack_min,
                            'logit_lens_attack_min': logit_lens_attack_min,
                            'harmonic_core_min': harmonic_core_min,
                            'harmonic_attack_min': harmonic_attack_min,

                            'specificity': specificity_score,
                        }
                        if config.perplexity:
                            res_dict[f"{lr}"][f"{loss_break}"][f"{norm_constraint}"][f"{seed}"][f"{threshold}"]['perplexity'] = exp_res_dict['perplexity']

                    # After each experiment, delete the model and tokenizer to reset the original weights of the model, clear the cache
                    del model
                    del tokenizer
                    torch.cuda.empty_cache()

    if config.log_wandb:
        res_plot = plot_multi_experiment_results_ft(res_dict, return_plot=True)
        wandb.log({"EMAIL Aggregated Results": res_plot})
        # Add aggregated results after all iterations
        agg_res_dict = calc_aggregate_results_ft(res_dict, seeds, lrs, loss_breaks, norm_constraints, score_thresholds)
        wandb.log(dict(agg_res_dict))

    return dict(res_dict)


# ~~~~~~~~~~~~ UNLEARN ~~~~~~~~~~~~


def edit_revs_target(model_editor:REVS, prompt, target, config):
    model = model_editor.model
    tokenizer = model_editor.tokenizer

    to_edit_layers = list(range(model_editor.model_n_layers))
    edit_dict = model_editor.edit_multiple_layers_dict(to_edit_layers, prompt, target, 
        restore_after_edit=config.restore_after_edit, print_progress=False)
    return edit_dict


def run_revs_unlearn(model, tokenizer, prompts, targets, config, pinv_lm_head):

    model_editor = REVS(model, tokenizer, config, pinv_lm_head)

    # calc edit target
    for i, (prompt, target) in tqdm(enumerate(zip(prompts, targets)), total=len(prompts), desc="Applying edits"):
        concat_prompts, concat_targets = create_concat_prompts_target(
            tokenizer=tokenizer,
            prompt=prompt,
            target=target,
            method=config.token_method,
            skip_tokens=config.skip_tokens,
            stop_tokens=config.stop_tokens,
            max_tokens=config.max_tokens,
        )
        for concat_prompt, concat_target in zip(concat_prompts, concat_targets):
            edit_dict = edit_revs_target(model_editor, prompt=concat_prompt, target=concat_target, config=config)

    # if weights restored after each edit, apply them now
    if config.restore_after_edit:
        model_editor.apply_all_edits()

    return model_editor


def run_memit_unlearn(model, tokenizer, prompts, targets, config, example_chunk_size=1000):

    # if max_prompt_len is not None shorten the prompts
    if config.max_prompt_len is not None and config.max_prompt_len > 0:
        prompts = [x[-config.max_prompt_len:] for x in prompts]
        print(f"\n\n\t ~~~~~~~~ Using shortened prompts of length {config.max_prompt_len} ~~~~~~~~\n\n")

    all_concat_prompts, all_concat_targets = [], []
    for i, (prompt, target) in tqdm(enumerate(zip(prompts, targets)), total=len(prompts)):
        concat_prompts, concat_targets = create_concat_prompts_target(
            tokenizer=tokenizer,
            prompt=prompt,
            target=target,
            method=config.token_method,
            skip_tokens=config.skip_tokens,
            stop_tokens=config.stop_tokens,
            max_tokens=config.max_tokens,
        )
        for concat_prompt, concat_target in zip(concat_prompts, concat_targets):
            all_concat_prompts.append(concat_prompt)
            all_concat_targets.append(concat_target)
            if config.log_wandb:
                collected_acts_before = collect_activations_with_prompt(model, tokenizer, concat_prompt)
                target_ranks_before = get_token_rank_across_layers(model, tokenizer, collected_acts_before, concat_target)
                plot_ranks_before = plot_token_rank_in_hs_across_sublayers(
                    target_ranks_before, prompt=concat_prompt, 
                    target=concat_target, log_scale=True, return_plot=True,
                    title="[ORIGINAL] Token Rank Across Layers"
                    )
                wandb.log({f"{target}/{concat_target}":{
                    "Ranks Plot Before Edit": plot_ranks_before,
                    }})

    requests = create_requests(all_concat_prompts, all_concat_targets, mode='delete')
    generation_prompts = list(all_concat_prompts)
    # edit the model
    print("\nApplying edits...")
    for i in range(0, len(requests), example_chunk_size):
        edit_model(
            model=model,
            tok=tokenizer,
            requests=requests[i:i+example_chunk_size],
            generation_prompts=generation_prompts[i:i+example_chunk_size],
            v_lr=config.v_lr,
            loss_break=config.loss_break,
            loss_pred_prob_coef=config.loss_pred_prob_coef,
            v_num_grad_steps=config.v_num_grad_steps,
            alg_name="memit",
            max_out_len=40,
            top_k=1,
            layers=config.layers,
        )
    print("\nDone applying edits")


def run_ft_unlearn(model, tokenizer, prompts, targets, config, example_chunk_size=1000):

    # if max_prompt_len is not None shorten the prompts
    if config.max_prompt_len is not None and config.max_prompt_len > 0:
        prompts = [x[-config.max_prompt_len:] for x in prompts]
        print(f"\n\n\t ~~~~~~~~ Using shortened prompts of length {config.max_prompt_len} ~~~~~~~~\n\n")

    all_concat_prompts, all_concat_targets = [], []
    for i, (prompt, target) in tqdm(enumerate(zip(prompts, targets)), total=len(prompts)):
        concat_prompts, concat_targets = create_concat_prompts_target(
            tokenizer=tokenizer,
            prompt=prompt,
            target=target,
            method=config.token_method,
            skip_tokens=config.skip_tokens,
            stop_tokens=config.stop_tokens,
            max_tokens=config.max_tokens,
        )
        for concat_prompt, concat_target in zip(concat_prompts, concat_targets):
            all_concat_prompts.append(concat_prompt)
            all_concat_targets.append(concat_target)
            if config.log_wandb:
                collected_acts_before = collect_activations_with_prompt(model, tokenizer, concat_prompt)
                target_ranks_before = get_token_rank_across_layers(model, tokenizer, collected_acts_before, concat_target)
                plot_ranks_before = plot_token_rank_in_hs_across_sublayers(
                    target_ranks_before, prompt=concat_prompt, 
                    target=concat_target, log_scale=True, return_plot=True,
                    title="[ORIGINAL] Token Rank Across Layers"
                    )
                wandb.log({f"{target}/{concat_target}":{
                    "Ranks Plot Before Edit": plot_ranks_before,
                    }})

    requests = create_requests(all_concat_prompts, all_concat_targets, mode='delete')
    generation_prompts = list(all_concat_prompts)
    # edit the model
    print("\nApplying edits...")
    for i in range(0, len(requests), example_chunk_size):
        edit_model(
            model=model,
            tok=tokenizer,
            requests=requests[i:i+example_chunk_size],
            generation_prompts=generation_prompts[i:i+example_chunk_size],
            alg_name="ft",
            max_out_len=40,
            top_k=1,
            lr=config.lr,
            loss_break=config.loss_break,
            layers=config.layers,
        )
    print("\n Done applying edits")


def calc_rank_editor_scores(model, tokenizer, prompts, targets, config):
    rank_editor_scores = []
    for i, (prompt, target) in enumerate(zip(prompts, targets)):
        rank_editor_squared_score = REVSScore(
            model,
            tokenizer,
            prompt,
            target,
            skip_tokens=config.skip_tokens,
            stop_tokens=config.stop_tokens,
            max_tokens=config.max_tokens
        )
        rank_editor_scores.append(rank_editor_squared_score)
    return rank_editor_scores


def calc_aggregate_results_revs(res_dict, seeds, n_neurons_list, score_thresholds):
    """
    This function aggregates results across different seeds and adds them to a new dictionary.

    Parameters:
    res_dict (dict): The dictionary where the results of each experiment are stored.
    seeds (list): The list of seeds used in the experiments.
    n_neurons_list (list): The list of neuron counts used in the experiments.
    score_thresholds (list): The list of score thresholds used in the experiments.

    Returns:
    dict: A new dictionary with aggregated results.

    The function first checks if there are multiple seeds. If so, it extracts the metrics from the res_dict.
    For each combination of neuron count and score threshold, it calculates the mean and standard deviation
    of each metric across all seeds. These aggregated results are then added to the new dictionary,
    which represents the summary of results across all seeds.
    """
    # Create a new dictionary to store the aggregated results
    agg_res_dict = {}

    # Calculate aggregated results after all iterations
    if len(seeds) > 1:
        # Extract metrics from the res_dict
        n_neurons = list(res_dict.keys())[0]
        seed = list(res_dict[n_neurons].keys())[0]
        threshold = list(res_dict[n_neurons][seed].keys())[0]
        metrics = list(res_dict[n_neurons][seed][threshold].keys())
        for n_neurons in n_neurons_list:
            for threshold in score_thresholds:
                for metric in metrics:
                    metric_values = [res_dict[f"{n_neurons}"][f"{seed}"][f"{threshold}"][metric] for seed in seeds]
                    if f"{n_neurons}" not in agg_res_dict:
                        agg_res_dict[f"{n_neurons}"] = {}
                    if f"{threshold}" not in agg_res_dict[f"{n_neurons}"]:
                        agg_res_dict[f"{n_neurons}"][f"{threshold}"] = {}
                    agg_res_dict[f"{n_neurons}"][f"{threshold}"][f"{metric}_mean"] = round(np.mean(metric_values), 5)
                    agg_res_dict[f"{n_neurons}"][f"{threshold}"][f"{metric}_std"] = round(np.std(metric_values), 5)
    return agg_res_dict


def calc_aggregate_results_memit(res_dict, seeds, v_lrs, loss_breaks, loss_pred_prob_coefs, score_thresholds):
    """
    Aggregate results across different seeds and add them to a new dictionary.

    Parameters:
    res_dict (dict): The dictionary where the results of each experiment are stored.
    seeds (list): The list of seeds used in the experiments.
    v_lrs (list): The list of learning rates used in the experiments.
    loss_breaks (list): The list of loss breaks used in the experiments.
    loss_pred_prob_coefs (list): The list of loss prediction probability coefficients used in the experiments.
    score_thresholds (list): The list of score thresholds used in the experiments.

    Returns:
    dict: A new dictionary with aggregated results.

    The function first checks if there are multiple seeds. If so, it extracts the metrics from the res_dict.
    For each combination of learning rate, loss break, loss prediction probability coefficient, and score threshold,
    it calculates the mean and standard deviation of each metric across all seeds. These aggregated results are then
    added to a new dictionary, which represents the summary of results across all seeds.
    """
    # Create a new dictionary to store the aggregated results
    agg_res_dict = {}

    # Calculate aggregated results after all iterations
    if len(seeds) > 1:
        # Extract metrics from the res_dict
        v_lr = list(res_dict.keys())[0]
        loss_break = list(res_dict[v_lr].keys())[0]
        loss_pred_prob_coef = list(res_dict[v_lr][loss_break].keys())[0]
        seed = list(res_dict[v_lr][loss_break][loss_pred_prob_coef].keys())[0]
        threshold = list(res_dict[v_lr][loss_break][loss_pred_prob_coef][seed].keys())[0]
        metrics = list(res_dict[v_lr][loss_break][loss_pred_prob_coef][seed][threshold].keys())
        for v_lr in v_lrs:
            for loss_break in loss_breaks:
                for loss_pred_prob_coef in loss_pred_prob_coefs:
                    for threshold in score_thresholds:
                        for metric in metrics:
                            metric_values = [res_dict[f"{v_lr}"][f"{loss_break}"][f"{loss_pred_prob_coef}"][f"{seed}"][f"{threshold}"][metric] for seed in seeds]
                            if f"{v_lr}" not in agg_res_dict:
                                agg_res_dict[f"{v_lr}"] = {}
                            if f"{loss_break}" not in agg_res_dict[f"{v_lr}"]:
                                agg_res_dict[f"{v_lr}"][f"{loss_break}"] = {}
                            if f"{loss_pred_prob_coef}" not in agg_res_dict[f"{v_lr}"][f"{loss_break}"]:
                                agg_res_dict[f"{v_lr}"][f"{loss_break}"][f"{loss_pred_prob_coef}"] = {}
                            if f"{threshold}" not in agg_res_dict[f"{v_lr}"][f"{loss_break}"][f"{loss_pred_prob_coef}"]:
                                agg_res_dict[f"{v_lr}"][f"{loss_break}"][f"{loss_pred_prob_coef}"][f"{threshold}"] = {}
                            agg_res_dict[f"{v_lr}"][f"{loss_break}"][f"{loss_pred_prob_coef}"][f"{threshold}"][f"{metric}_mean"] = round(np.mean(metric_values), 5)
                            agg_res_dict[f"{v_lr}"][f"{loss_break}"][f"{loss_pred_prob_coef}"][f"{threshold}"][f"{metric}_std"] = round(np.std(metric_values), 5)
    return agg_res_dict


def calc_aggregate_results_ft(res_dict, seeds, lrs, loss_breaks, norm_constraints, score_thresholds):
    """
    Aggregate results across different seeds and add them to a new dictionary.

    Parameters:
    res_dict (dict): The dictionary where the results of each experiment are stored.
    seeds (list): The list of seeds used in the experiments.
    lrs (list): The list of learning rates used in the experiments.
    loss_breaks (list): The list of loss breaks used in the experiments.
    norm_constraints (list): The list of norm constraints used in the experiments.
    score_thresholds (list): The list of score thresholds used in the experiments.

    Returns:
    dict: A new dictionary with aggregated results.

    The function first checks if there are multiple seeds. If so, it extracts the metrics from the res_dict.
    For each combination of learning rate, loss break, norm constraint, and score threshold,
    it calculates the mean and standard deviation of each metric across all seeds. These aggregated results are then
    added to a new dictionary, which represents the summary of results across all seeds.
    """
    # Create a new dictionary to store the aggregated results
    agg_res_dict = {}

    # Calculate aggregated results after all iterations
    if len(seeds) > 1:
        # Extract metrics from the res_dict
        lr = list(res_dict.keys())[0]
        loss_break = list(res_dict[lr].keys())[0]
        norm_constraint = list(res_dict[lr][loss_break].keys())[0]
        seed = list(res_dict[lr][loss_break][norm_constraint].keys())[0]
        threshold = list(res_dict[lr][loss_break][norm_constraint][seed].keys())[0]
        metrics = list(res_dict[lr][loss_break][norm_constraint][seed][threshold].keys())
        for lr in lrs:
            for loss_break in loss_breaks:
                for norm_constraint in norm_constraints:
                    for threshold in score_thresholds:
                        for metric in metrics:
                            metric_values = [res_dict[f"{lr}"][f"{loss_break}"][f"{norm_constraint}"][f"{seed}"][f"{threshold}"][metric] for seed in seeds]
                            if f"{lr}" not in agg_res_dict:
                                agg_res_dict[f"{lr}"] = {}
                            if f"{loss_break}" not in agg_res_dict[f"{lr}"]:
                                agg_res_dict[f"{lr}"][f"{loss_break}"] = {}
                            if f"{norm_constraint}" not in agg_res_dict[f"{lr}"][f"{loss_break}"]:
                                agg_res_dict[f"{lr}"][f"{loss_break}"][f"{norm_constraint}"] = {}
                            if f"{threshold}" not in agg_res_dict[f"{lr}"][f"{loss_break}"][f"{norm_constraint}"]:
                                agg_res_dict[f"{lr}"][f"{loss_break}"][f"{norm_constraint}"][f"{threshold}"] = {}
                            agg_res_dict[f"{lr}"][f"{loss_break}"][f"{norm_constraint}"][f"{threshold}"][f"{metric}_mean"] = round(np.mean(metric_values), 5)
                            agg_res_dict[f"{lr}"][f"{loss_break}"][f"{norm_constraint}"][f"{threshold}"][f"{metric}_std"] = round(np.std(metric_values), 5)
    return agg_res_dict


# ~~~~~~~~~~~~ EVALUATIONS ~~~~~~~~~~~~


def run_efficacy(rank_edit_scores: List[REVSScore]):
    """
    Get the efficacy_edit_scores by choosing only last layer rank_edit_scores
    """
    efficacy_edit_scores = []
    for rank_edit_score in rank_edit_scores:
        efficacy_score = EfficacyScore(rank_edit_score)
        efficacy_edit_scores.append(efficacy_score)
    return efficacy_edit_scores


def run_specificity(model, tokenizer, prompts, targets, config):

    generated_text = generate_from_prompts(model, tokenizer, prompts, max_length=50, chunk_size=40, max_prompt_len=500)  # limit chunk size to 40 and the max prompt length to 500 to avoid memory issues
    counter = 0
    for i in range(len(generated_text)):
        if targets[i].strip() in generated_text[i]:
            counter += 1
    specificity_score = counter / len(generated_text)

    if config.log_wandb:
        wandb.log({'Specificity Score': specificity_score})
    return specificity_score


def run_generality(model, tokenizer, prompts, targets, config):

    edit_scores = []
    for prompt, target in tqdm(zip(prompts, targets), total=len(prompts), desc="Running generality evaluation"):
        edit_score = REVSScore(model, tokenizer, prompt, target, skip_tokens=config.skip_tokens, stop_tokens=config.stop_tokens, max_tokens=config.max_tokens)
        edit_scores.append(edit_score)

    if config.log_wandb:
        score_stats = calculate_edit_score_statistics_squared(edit_scores, threshold=config.score_threshold)
        title = "[GENERALITY] Edit Score"
        plot_bottom_distance_score_mean_stats = plot_edit_score_statistics(score_stats['bottom_distance_score'], method='mean', return_plot=True, log_scale=False, title=title)
        plot_top_distance_score_mean_stats = plot_edit_score_statistics(score_stats['top_distance_score'], method='mean', return_plot=True, log_scale=False, title=title)

        wandb.log({
            '[Generality] Plot Bottom Distance Score Mean Stats': plot_bottom_distance_score_mean_stats,
            '[Generality] Plot Top Distance Score Mean Stats': plot_top_distance_score_mean_stats
            })

    return edit_scores


def run_perplexity(model, tokenizer, config):
    import datasets
    from scipy import stats

    dataset = datasets.load_dataset("NeelNanda/wiki-10k")
    dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=1)

    model.eval()
    perplexities = []
    pbar = tqdm(dataloader, total=len(dataloader), desc="Running Perplexities")
    for batch in pbar:
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.mean()
        perplexity = torch.exp(loss)
        perplexities.append(perplexity.item())
        # Update progress bar
        pbar.set_postfix({'mean_perplexity': np.mean(perplexities), 'std_perplexity': np.std(perplexities)})

    # Remove outliers
    perplexities = np.array(perplexities)
    z_scores = np.abs(stats.zscore(perplexities))
    filtered_perplexities = perplexities[z_scores < 3]

    return filtered_perplexities.tolist()


# ~~~~~~~~~~~~ EXTRACTION ATTACKS ~~~~~~~~~~~~


def run_delta_attack(model, tokenizer, prompts, targets, config):

    delta_attack_scores = []
    for prompt, target in tqdm(zip(prompts, targets), total=len(prompts), desc="Running delta attack"):
        delta_attack_score = DeltaAttackScore(
            model=model, 
            tokenizer=tokenizer, 
            prompt=prompt, 
            target=target, 
            skip_tokens=config.skip_tokens, 
            stop_tokens=config.stop_tokens, 
            max_tokens=config.max_tokens,
            method="both"
        )
        delta_attack_scores.append(delta_attack_score)
    return delta_attack_scores


def run_perturbed_prompts_attack(model, tokenizer, prompts, targets, config):

    perturb_attack_scores = []
    for prompt, target in tqdm(zip(prompts, targets), total=len(prompts), desc="Running perturbation attack"):
        perturb_attack_score = PerturbAttackScore(
            model=model, 
            tokenizer=tokenizer, 
            prompt=prompt,
            target=target,
            seed=config.seed,
            skip_tokens=config.skip_tokens, 
            stop_tokens=config.stop_tokens, 
            max_tokens=config.max_tokens,
        )
        perturb_attack_scores.append(perturb_attack_score)
    return perturb_attack_scores


def run_logit_lens_attack(rank_edit_scores: List[REVSScore]):
    logit_lens_attack_scores = []
    for rank_edit_score in rank_edit_scores:
        logit_lens_attack_score = LogitLensAttackScore(rank_edit_score)
        logit_lens_attack_scores.append(logit_lens_attack_score)
    return logit_lens_attack_scores


# ~~~~~~~~~~~~ DATASETS ~~~~~~~~~~~~

def get_dataset_email(seed: int, split_value: Union[float, int], path=None, max_prompt_len: int=None):
    raise FileNotFoundError("This dataset has been removed for privacy reasons.")
    if path is None:
        path = "/dataset_pile_emails_memorized.csv" # Removed this dataset for privacy reasons

    if split_value is None:
        split_value = 0.5

    df_emails = pd.read_csv(path)
    df_emails = df_emails.drop_duplicates(subset=['email']).reset_index(drop=True)

    # if max_prompt_len is not None shorten the prompts
    if max_prompt_len is not None and max_prompt_len > 0:
        df_emails['prompt_gen'] = df_emails['prompt_gen'].apply(lambda x: x[-max_prompt_len:])
        print(f"\n\n\t ~~~~~~~~ Using shortened prompts of length {max_prompt_len} ~~~~~~~~\n\n")
    else:
        print("\n\n\t ~~~~~~~~ Using full prompts ~~~~~~~~\n\n")

    # Determine the split size based on the type of split_value
    if isinstance(split_value, float):
        forget_split_size = int(len(df_emails) * split_value)
    elif isinstance(split_value, int):
        forget_split_size = split_value
    else:
        raise ValueError("split_value must be either a float or an int")

    df_emails_unlearn = df_emails.sample(n=forget_split_size, random_state=seed)
    df_emails_retain = df_emails[~df_emails.index.isin(df_emails_unlearn.index)]

    unlearn_prompts = df_emails_unlearn['prompt_gen'].tolist()
    unlearn_targets = df_emails_unlearn['email'].tolist()
    retain_prompts = df_emails_retain['prompt_gen'].tolist()
    retain_targets = df_emails_retain['email'].tolist()

    prompts_dict = {
        "unlearn": unlearn_prompts,
        "retain": retain_prompts,
    }
    targets_dict = {
        "unlearn": unlearn_targets,
        "retain": retain_targets,
    }
    df_emails_dict = {
        "unlearn": df_emails_unlearn,
        "retain": df_emails_retain,
    }
    return prompts_dict, targets_dict, df_emails_dict


def get_dataset_ssn(seed: int, split_value: Union[float, int], path=None, max_prompt_len=None):
    """
    This function splits a dataset into 'unlearn', 'generality', and 'retain' parts.
    The dataset is grouped by 'ssn', and each group is either assigned to the 'forget' or 'retain' part.
    The 'forget' part is further split into 'unlearn' and 'generality' parts.
    """

    # Set default path if none is provided
    if path is None:
        import os
        from pathlib import Path
        # Get the directory of the current script
        current_script_dir = Path(__file__).parent
        # Construct the path to the CSV file relative to the current script
        path = current_script_dir.parent / 'dataset' / 'ssn_multi_sentences_many_to_one.csv'
    if split_value is None:
        split_value = 0.5

    # Load the data and add space as a prefix to each ssn
    df_ssn = pd.read_csv(path)
    df_ssn['ssn'] = df_ssn['ssn'].apply(lambda ssn: f" {ssn}")

    # if max_prompt_len is not None shorten the prompts
    if max_prompt_len is not None and max_prompt_len > 0:
        df_ssn['prompt'] = df_ssn['prompt'].apply(lambda x: x[-max_prompt_len:])
        print(f"\n\n\t ~~~~~~~~ Using shortened prompts of length {max_prompt_len} ~~~~~~~~\n\n")
    else:
        print("\n\n\t ~~~~~~~~ Using full prompts ~~~~~~~~\n\n")

    # Group the DataFrame by 'ssn' and shuffle the groups
    groups = [group for _, group in df_ssn.groupby('ssn')]
    np.random.seed(seed)
    np.random.shuffle(groups)

    # Determine the split index based on the type of split_value
    if isinstance(split_value, float):
        split_idx = int(len(groups) * split_value)
    elif isinstance(split_value, int):
        split_idx = split_value
    else:
        raise ValueError("split_value must be either a float or an int")

    # Split the groups into 'forget' and 'retain' parts
    forget_groups = groups[:split_idx]
    retain_groups = groups[split_idx:]

    # Concatenate the groups to get the 'forget' and 'retain' DataFrames
    df_ssn_forget = pd.concat(forget_groups)
    df_ssn_retain = pd.concat(retain_groups)

    # Create 'unlearn' part by randomly selecting one row from each group in 'forget' part
    df_ssn_unlearn = df_ssn_forget.groupby('ssn').sample(n=1, random_state=seed)

    # Create 'generality' part by selecting the rows that are not in the 'unlearn' part
    df_ssn_generality = df_ssn_forget[~df_ssn_forget.index.isin(df_ssn_unlearn.index)]

    # Create dictionaries for prompts and targets for each part
    prompts_dict = {
        "unlearn": df_ssn_unlearn['prompt'].tolist(),
        "generality": df_ssn_generality['prompt'].tolist(),
        "retain": df_ssn_retain['prompt'].tolist(),
    }

    targets_dict = {
        "unlearn": df_ssn_unlearn['ssn'].tolist(),
        "generality": df_ssn_generality['ssn'].tolist(),
        "retain": df_ssn_retain['ssn'].tolist(),
    }

    df_ssn_dict = {
        "unlearn": df_ssn_unlearn,
        "generality": df_ssn_generality,
        "retain": df_ssn_retain,
    }

    return prompts_dict, targets_dict, df_ssn_dict


def get_dataset_email_llama(seed: int, split_value: Union[float, int], path=None):
    raise FileNotFoundError("This dataset has been removed for privacy reasons.")
    if path is None:
        path = "dataset_pile_emails_memorized_llama.csv" # Removed this dataset for privacy reasons
    if split_value is None:
        split_value = 0.5

    df_emails = pd.read_csv(path)

    # Keep track of duplicated emails
    # df_emails_duplicated = df_emails[df_emails.duplicated(subset=['email'], keep=False)]
    df_emails = df_emails.drop_duplicates(subset=['email'], keep='first').reset_index(drop=True)  # there are dups that has the almost same prompt

    # Determine the split size based on the type of split_value
    if isinstance(split_value, float):
        forget_split_size = int(len(df_emails) * split_value)
    elif isinstance(split_value, int):
        forget_split_size = split_value
    else:
        raise ValueError("split_value must be either a float or an int")

    df_emails_unlearn = df_emails.sample(n=forget_split_size, random_state=seed)
    df_emails_retain = df_emails[~df_emails.index.isin(df_emails_unlearn.index)]

    unlearn_prompts = df_emails_unlearn['prompt_gen'].tolist()
    unlearn_targets = df_emails_unlearn['email'].tolist()
    retain_prompts = df_emails_retain['prompt_gen'].tolist()
    retain_targets = df_emails_retain['email'].tolist()

    prompts_dict = {
        "unlearn": unlearn_prompts,
        "retain": retain_prompts,
        # "generality": generality_prompts,
    }
    targets_dict = {
        "unlearn": unlearn_targets,
        "retain": retain_targets,
        # "generality": generality_targets,
    }
    df_emails_dict = {
        "unlearn": df_emails_unlearn,
        "retain": df_emails_retain,
        # "generality": df_emails_duplicated,
    }
    return prompts_dict, targets_dict, df_emails_dict
