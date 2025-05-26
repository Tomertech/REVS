from typing import List, Union, Dict, Literal, Optional
from copy import deepcopy
from collections import defaultdict
from tqdm.notebook import tqdm
import numpy as np
import torch
import pandas as pd
import wandb
from datasets import load_dataset

# My utils:
from utils.generation import generate_from_prompts
from utils.model import load_model_tokenizer, edit_model, create_requests, MemitConfig, FTConfig
from revs.revs import REVSConfig, REVS, REVSScore
from utils.metrics import calculate_edit_score_statistics_squared, calculate_across_layers_score, calculate_harmonic_mean, \
    EfficacyScore, DeltaAttackScore, PerturbAttackScore, LogitLensAttackScore
from utils.hidden_state_ops import get_token_rank_across_layers
from utils.activations_collector import collect_activations_with_prompt
from utils.plot import plot_token_rank_in_hs_across_sublayers, plot_edit_score_statistics, \
    plot_multi_experiment_results_revs, plot_multi_experiment_results_memit, plot_multi_experiment_results_ft, \
    plot_multi_experiment_results_rmu, plot_multi_experiment_results_npo
from utils.globals import device, PROJECT_PATH, PROJECT_NAME
from utils.data import create_concat_prompts_target
from baselines.rmu.rmu_unlearn import run_rmu, RMUConfig
from baselines.npo.npo_unlearn import run_npo, NPOConfig


# ~~~~~~~~~~~~ EXPS ~~~~~~~~~~~~


def run_revs_exp(model, tokenizer, prompts, targets, config, pinv_lm_head, specificity=True, generality=True, extraction=True):

    res_dict = {}
    model_editor = None
    if not config.not_unlearn:
        model_editor = run_revs_unlearn(model, tokenizer, prompts['unlearn'], targets['unlearn'], config, pinv_lm_head)

    rank_editor_scores = calc_rank_editor_scores(model, tokenizer, prompts['unlearn'], targets['unlearn'], config)

    res_dict['efficacy'] = run_efficacy(rank_editor_scores)

    if specificity and 'retain' in prompts:
        res_dict['specificity'] = run_specificity(model, tokenizer, prompts['retain'], targets['retain'], config)

    if generality and 'generality' in prompts:
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

    if not config.not_unlearn:
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

    if not config.not_unlearn:
        run_ft_unlearn(model, tokenizer, prompts['unlearn'], targets['unlearn'], config)

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


def run_rmu_exp(updated_model, frozen_model, tokenizer, prompts, targets, config, specificity=False, generality=False, extraction=False):
    res_dict = {}

    if not config.not_unlearn:
        run_rmu_unlearn(updated_model, frozen_model, tokenizer, prompts['unlearn'], targets['unlearn'], config)

    rank_editor_scores = calc_rank_editor_scores(updated_model, tokenizer, prompts['unlearn'], targets['unlearn'], config)

    res_dict['efficacy'] = run_efficacy(rank_editor_scores)

    if specificity:
        res_dict['specificity'] = run_specificity(updated_model, tokenizer, prompts['retain'], targets['retain'], config)

    if generality:
        res_dict['generality'] = run_generality(updated_model, tokenizer, prompts['generality'], targets['generality'], config)

    if extraction:
        res_dict['delta_attack'] = run_delta_attack(updated_model, tokenizer, prompts['unlearn'], targets['unlearn'], config)
        res_dict['perturb_attack'] = run_perturbed_prompts_attack(updated_model, tokenizer, prompts['unlearn'], targets['unlearn'], config)
        res_dict['logit_lens_attack'] = run_logit_lens_attack(rank_editor_scores)

    if config.perplexity:
        perplexities = run_perplexity(updated_model, tokenizer, config)
        res_dict['perplexity'] = np.mean(perplexities)

    return res_dict


def run_npo_exp(updated_model, frozen_model, tokenizer, prompts, targets, config, specificity=False, generality=False, extraction=False):
    res_dict = {}

    if not config.not_unlearn:
        run_npo_unlearn(updated_model, frozen_model, tokenizer, prompts['unlearn'], targets['unlearn'], config)

    rank_editor_scores = calc_rank_editor_scores(updated_model, tokenizer, prompts['unlearn'], targets['unlearn'], config)

    res_dict['efficacy'] = run_efficacy(rank_editor_scores)

    if specificity:
        res_dict['specificity'] = run_specificity(updated_model, tokenizer, prompts['retain'], targets['retain'], config)

    if generality:
        res_dict['generality'] = run_generality(updated_model, tokenizer, prompts['generality'], targets['generality'], config)

    if extraction:
        res_dict['delta_attack'] = run_delta_attack(updated_model, tokenizer, prompts['unlearn'], targets['unlearn'], config)
        res_dict['perturb_attack'] = run_perturbed_prompts_attack(updated_model, tokenizer, prompts['unlearn'], targets['unlearn'], config)
        res_dict['logit_lens_attack'] = run_logit_lens_attack(rank_editor_scores)

    if config.perplexity:
        perplexities = run_perplexity(updated_model, tokenizer, config)
        res_dict['perplexity'] = np.mean(perplexities)

    return res_dict


# ~~~~~~~~~~~~ MULTI EXPS ~~~~~~~~~~~~

def revs_multi_exp(
    config: REVSConfig,
    seeds: List[int],
    n_neurons_list: List[int],
    score_thresholds: List[int],
    model_type: str,
    exp_type: str,
    message: str = None,
    wandb_tags: List[str] = None
):
    def nested_dict():
        return defaultdict(nested_dict)

    multi_exp_config = config.to_dict()
    del multi_exp_config['n_neurons'], multi_exp_config['seed'], multi_exp_config['score_threshold']
    multi_exp_config['n_neurons_list'] = n_neurons_list
    multi_exp_config['seeds'] = seeds
    multi_exp_config['score_thresholds'] = score_thresholds

    if config.log_wandb:
        exp_name = f"SIRE {model_type.upper()} {exp_type.upper()}"
        if config.unlearn_num_examples is not None:
            exp_name += f" {config.unlearn_num_examples}e, "
        config_name = (f"{seeds}s, {n_neurons_list}n, {score_thresholds}st, "
                f"{config.residual_top_rank_margin/1000}krtop, "
                f"{config.residual_bottom_rank_margin/1000}krbottom, "
                f"{config.act_filter}")
        name = f"{exp_name} {message + ', ' if message else ''}{config_name}"
        tags = ["SIRE", f"{exp_type.upper()}", f"{model_type.upper()}"] + (wandb_tags if wandb_tags else [])
        wandb.init(project=PROJECT_NAME, config=multi_exp_config, name=name, tags=tags)

    pinv_lm_head = None
    res_dict = nested_dict()

    for n_neurons in n_neurons_list:
        for seed in seeds:
            exp_config = deepcopy(config)
            exp_config.n_neurons = n_neurons
            exp_config.seed = seed
            exp_config.log_wandb = False
            exp_config.save_model = False

            model, tokenizer = load_model_tokenizer(model_type, exp_type)
            prompts_dict, targets_dict, df_dict = get_dataset(
                seed=seed,
                split_value=config.unlearn_num_examples,
                max_prompt_len=config.max_prompt_len,
                model_type=model_type,
                exp_type=exp_type
            )

            if pinv_lm_head is None:
                pinv_lm_head = torch.pinverse(model.lm_head.weight).to(device)

            exp_res_dict, exp_model_editor = run_revs_exp(
                model, tokenizer, prompts_dict, targets_dict,
                exp_config, pinv_lm_head
                )

            hyperparams = {
                'n_neurons': n_neurons,
                'seed': seed
            }

            exp_res_dict = process_experiment_results(
                exp_res_dict, score_thresholds, config, hyperparams, calc_type="min"
            )
            res_dict = deep_merge_dicts(res_dict, exp_res_dict)
            if config.log_wandb:
                res_plot = plot_multi_experiment_results_revs(res_dict, return_plot=True)
                wandb.log({f"{model_type.upper()} Aggregated Results": res_plot})

            del model, exp_model_editor, tokenizer
            torch.cuda.empty_cache()

    if config.log_wandb:
        agg_res_dict = calc_aggregate_results_revs(
            res_dict=res_dict,
            seeds=seeds,
            n_neurons_list=n_neurons_list,
            score_thresholds=score_thresholds
        )
        wandb.log(dict(agg_res_dict))

    return dict(res_dict)


def memit_multi_exp(
    config: MemitConfig,
    seeds: List[int],
    score_thresholds: List[int],
    v_lrs: List[float],
    loss_breaks: List[float],
    loss_pred_prob_coefs: List[float],
    model_type: str,
    exp_type: str,
    message: str = None
):
    """
    Run multiple experiments with MEMIT for different model types and experiment types
    Args:
        model_type: 'gptj' or 'llama'
        exp_type: 'email' or 'ssn'
    """
    def nested_dict():
        return defaultdict(nested_dict)

    multi_exp_config = config.to_dict()
    del multi_exp_config['seed'], multi_exp_config['score_threshold']
    del multi_exp_config['v_lr'], multi_exp_config['loss_break']
    del multi_exp_config['loss_pred_prob_coef']
    multi_exp_config['seeds'] = seeds
    multi_exp_config['score_thresholds'] = score_thresholds
    multi_exp_config['v_lrs'] = v_lrs
    multi_exp_config['loss_breaks'] = loss_breaks
    multi_exp_config['loss_pred_prob_coefs'] = loss_pred_prob_coefs

    if config.log_wandb:
        method_name = "MEMIT" if config.mode.lower() == "delete" else "PATIL"
        exp_name = f"{method_name} {model_type.upper()} {exp_type.upper()}"
        if config.unlearn_num_examples is not None:
            exp_name += f" {config.unlearn_num_examples}e"
        config_name = (f"{seeds}s, {v_lrs}lr, {loss_breaks}lb, {loss_pred_prob_coefs}lc, "
                    f"{score_thresholds}st, {config.v_num_grad_steps}steps, "
                    f"{config.max_prompt_len}mp_len")
        name = f"{exp_name} {message + ', ' if message else ''}{config_name}"
        tags = [method_name, f"{exp_type.upper()}", f"{model_type.upper()}", f"{config.mode.upper()}"]
        wandb.init(project=PROJECT_NAME, config=multi_exp_config, name=name, tags=tags)

    res_dict = nested_dict()

    for v_lr in v_lrs:
        for loss_break in loss_breaks:
            for loss_pred_prob_coef in loss_pred_prob_coefs:
                for seed in seeds:
                    exp_config = deepcopy(config)
                    exp_config.v_lr = v_lr
                    exp_config.seed = seed
                    exp_config.loss_break = loss_break
                    exp_config.loss_pred_prob_coef = loss_pred_prob_coef
                    exp_config.log_wandb = False
                    exp_config.save_model = False

                    model, tokenizer = load_model_tokenizer(model_type, exp_type)
                    prompts_dict, targets_dict, df_dict = get_dataset(
                        seed=seed,
                        split_value=config.unlearn_num_examples,
                        max_prompt_len=config.max_prompt_len,
                        model_type=model_type,
                        exp_type=exp_type
                    )

                    exp_res_dict = run_memit_exp(
                        model,
                        tokenizer,
                        prompts_dict,
                        targets_dict,
                        exp_config,
                        specificity=True,
                        generality=(exp_type=='ssn'),
                        extraction=True
                    )

                    hyperparams = {
                        'v_lr': v_lr,
                        'loss_break': loss_break,
                        'loss_pred_prob_coef': loss_pred_prob_coef,
                        'seed': seed,
                    }

                    exp_res_dict = process_experiment_results(
                        exp_res_dict, score_thresholds, config, hyperparams, calc_type="min"
                    )
                    res_dict = deep_merge_dicts(res_dict, exp_res_dict)
                    if config.log_wandb:
                        res_plot = plot_multi_experiment_results_memit(res_dict, return_plot=True)
                        wandb.log({f"{model_type.upper()} Aggregated Results": res_plot})
                    del model, tokenizer
                    torch.cuda.empty_cache()

    if config.log_wandb:
        agg_res_dict = calc_aggregate_results_memit(
            res_dict=res_dict,
            seeds=seeds,
            v_lrs=v_lrs,
            loss_breaks=loss_breaks,
            loss_pred_prob_coefs=loss_pred_prob_coefs,
            score_thresholds=score_thresholds
        )
        wandb.log(dict(agg_res_dict))

    return dict(res_dict)


def ft_multi_exp(
    config: FTConfig,
    seeds: List[int],
    score_thresholds: List[int],
    lrs: List[float],
    loss_breaks: List[float],
    norm_constraints: List[float],
    model_type: str,
    exp_type: str,
    message: str = None,
    wandb_tags: List[str] = None
):
    """
    Run multiple experiments with FT for different experiment types (EMAIL/SSN) and model types (GPT-J/LLAMA)
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
        exp_name = f"FTL {model_type.upper()} {exp_type.upper()}"
        if config.unlearn_num_examples is not None:
            exp_name += f" {config.unlearn_num_examples}e, "
        if config.layers is not None:
            layers = config.layers
            if len(layers) > 1 and layers == list(range(layers[0], layers[-1] + 1)):
                exp_name += f" {{{layers[0]}..{layers[-1]}}}lyrs, "
            else:
                exp_name += f" {' '.join(map(str, layers))}lyrs, "
        config_name = (f"{seeds}s, {lrs}lr, {loss_breaks}lb, {norm_constraints}nc, "
                    f"{score_thresholds}st, {config.num_grad_steps}steps, "
                    f"{config.max_prompt_len}mp_len")
        name = f"{exp_name} {message + ', ' if message else ''}{config_name}"
        tags = ["FT", f"{exp_type.upper()}", f"{model_type.upper()}"] + (wandb_tags if wandb_tags else [])
        wandb.init(project=PROJECT_NAME, config=multi_exp_config, name=name, tags=tags)

    res_dict = nested_dict()
    for lr in lrs:
        for loss_break in loss_breaks:
            for norm_constraint in norm_constraints:
                for seed in seeds:
                    exp_config = deepcopy(config)
                    exp_config.lr = lr
                    exp_config.seed = seed
                    exp_config.loss_break = loss_break
                    exp_config.norm_constraint = norm_constraint
                    exp_config.log_wandb = False
                    exp_config.save_model = False

                    model, tokenizer = load_model_tokenizer(model_type, exp_type)
                    prompts_dict, targets_dict, df_dict = get_dataset(
                        seed=seed,
                        split_value=config.unlearn_num_examples,
                        max_prompt_len=config.max_prompt_len,
                        model_type=model_type,
                        exp_type=exp_type
                    )

                    exp_res_dict = run_ft_exp(
                        model, tokenizer, prompts_dict, targets_dict,
                        exp_config,
                        specificity=True,
                        generality=(exp_type.lower() == 'ssn'),
                        extraction=True
                    )

                    hyperparams = {
                        'lr': lr,
                        'loss_break': loss_break,
                        'norm_constraint': norm_constraint,
                        'seed': seed
                    }

                    exp_res_dict = process_experiment_results(
                        exp_res_dict, score_thresholds, config, hyperparams, calc_type="min"
                    )
                    res_dict = deep_merge_dicts(res_dict, exp_res_dict)
                    if config.log_wandb:
                        res_plot = plot_multi_experiment_results_ft(res_dict, return_plot=True)
                        wandb.log({f"{model_type.upper()} Aggregated Results": res_plot})

                    del model, tokenizer
                    torch.cuda.empty_cache()

    if config.log_wandb:
        agg_res_dict = calc_aggregate_results_ft(
            res_dict=res_dict,
            seeds=seeds,
            lrs=lrs,
            loss_breaks=loss_breaks,
            norm_constraints=norm_constraints,
            score_thresholds=score_thresholds
        )
        wandb.log(dict(agg_res_dict))

    return dict(res_dict)


def rmu_multi_exp(
    config: RMUConfig,
    seeds: List[int],
    score_thresholds: List[int],
    num_epochs_list: List[int],
    lrs: List[float],
    alphas: List[float],
    steering_coeffs: List[float],
    model_type: str,
    exp_type: str,
    message: str = None
):
    """
    Run multiple experiments with RMU for different model types and experiment types
    Args:
        model_type: 'gptj' or 'llama'
        exp_type: 'email' or 'ssn'
    """
    def nested_dict():
        return defaultdict(nested_dict)

    multi_exp_config = config.to_dict()
    del multi_exp_config['seed'], multi_exp_config['score_threshold']
    multi_exp_config['seeds'] = seeds
    multi_exp_config['score_thresholds'] = score_thresholds
    multi_exp_config['num_epochs_list'] = num_epochs_list
    multi_exp_config['lrs'] = lrs
    multi_exp_config['alphas'] = alphas
    multi_exp_config['steering_coeffs'] = steering_coeffs

    if config.log_wandb:
        exp_name = f"RMU {model_type.upper()} {exp_type.upper()}"
        if config.unlearn_num_examples is not None:
            exp_name += f" {config.unlearn_num_examples}e"
        config_name = (f"{seeds}s, {score_thresholds}st, {num_epochs_list}epochs, "
                    f"{lrs}lr, {alphas}alphas, {steering_coeffs}steering_coeffs")
        name = f"{exp_name} {message + ', ' if message else ''}{config_name}"
        tags = ["RMU", f"{exp_type.upper()}", f"{model_type.upper()}"]
        wandb.init(project=PROJECT_NAME, config=multi_exp_config, name=name, tags=tags)

    frozen_model, tokenizer = load_model_tokenizer(model_type, exp_type)
    res_dict = nested_dict()
    for num_epochs in num_epochs_list:
        for lr in lrs:
            for alpha in alphas:
                for steering_coeff in steering_coeffs:
                    for seed in seeds:
                        exp_config = deepcopy(config)
                        exp_config.seed = seed
                        exp_config.lr = lr
                        exp_config.alpha = alpha
                        exp_config.steering_coeff = steering_coeff
                        exp_config.num_epochs = num_epochs
                        exp_config.log_wandb = False
                        exp_config.save_model = False

                        updated_model, _ = load_model_tokenizer(model_type, exp_type)

                        # Get dataset based on experiment type
                        prompts_dict, targets_dict, df_dict = get_dataset(
                            seed=seed,
                            split_value=config.unlearn_num_examples,
                            max_prompt_len=config.max_prompt_len,
                            model_type=model_type,
                            exp_type=exp_type
                        )

                        exp_res_dict = run_rmu_exp(
                            updated_model,
                            frozen_model,
                            tokenizer,
                            prompts_dict,
                            targets_dict,
                            exp_config,
                            specificity=True,
                            generality=(exp_type=='ssn'),
                            extraction=True
                        )

                        hyperparams = {
                            'num_epochs': num_epochs,
                            'lr': lr,
                            'alpha': alpha,
                            'steering_coeff': steering_coeff,
                            'seed': seed,
                        }

                        exp_res_dict = process_experiment_results(
                            exp_res_dict, score_thresholds, config, hyperparams, calc_type="min"
                        )
                        res_dict = deep_merge_dicts(res_dict, exp_res_dict)

                        if config.log_wandb:
                            res_plot = plot_multi_experiment_results_rmu(res_dict, return_plot=True)
                            wandb.log({f"{model_type.upper()} Aggregated Results": res_plot})

                        del updated_model
                        torch.cuda.empty_cache()

    if config.log_wandb:
        agg_res_dict = calc_aggregate_results_rmu(
            res_dict=res_dict,
            seeds=seeds,
            score_thresholds=score_thresholds,
            num_epochs_list=num_epochs_list,
            lrs=lrs,
            alphas=alphas,
            steering_coeffs=steering_coeffs
        )
        wandb.log(dict(agg_res_dict))

    return dict(res_dict)


def npo_multi_exp(
    config: NPOConfig,
    seeds: List[int],
    score_thresholds: List[int],
    num_epochs_list: List[int],
    lrs: List[float],
    betas: List[float],
    model_type: str,
    exp_type: str,
    message: str = None
):
    """
    Run multiple experiments with NPO for different model types and experiment types
    Args:
        model_type: 'gptj' or 'llama'
        exp_type: 'email' or 'ssn'
    """
    def nested_dict():
        return defaultdict(nested_dict)

    multi_exp_config = config.to_dict()
    del multi_exp_config['seed'], multi_exp_config['score_threshold']
    multi_exp_config['seeds'] = seeds
    multi_exp_config['score_thresholds'] = score_thresholds
    multi_exp_config['num_epochs_list'] = num_epochs_list
    multi_exp_config['lrs'] = lrs
    multi_exp_config['betas'] = betas

    if config.log_wandb:
        exp_name = f"NPO {model_type.upper()} {exp_type.upper()}"
        if config.unlearn_num_examples is not None:
            exp_name += f" {config.unlearn_num_examples}e"
        config_name = (f"{seeds}s, {score_thresholds}st, {num_epochs_list}epochs, "
                    f"{lrs}lr, {betas}beta")
        name = f"{exp_name} {message + ', ' if message else ''}{config_name}"
        tags = ["NPO", f"{exp_type.upper()}", f"{model_type.upper()}", f"{config.loss_type.upper()}"]
        wandb.init(project=PROJECT_NAME, config=multi_exp_config, name=name, tags=tags)

    frozen_model, tokenizer = load_model_tokenizer(model_type, exp_type)
    res_dict = nested_dict()
    for num_epochs in num_epochs_list:
        for lr in lrs:
            for beta in betas:
                for seed in seeds:
                    exp_config = deepcopy(config)
                    exp_config.seed = seed
                    exp_config.lr = lr
                    exp_config.beta = beta
                    exp_config.num_epochs = num_epochs
                    exp_config.log_wandb = False
                    exp_config.save_model = False

                    updated_model, _ = load_model_tokenizer(model_type, exp_type)

                    # Get dataset based on experiment type
                    prompts_dict, targets_dict, df_dict = get_dataset(
                        seed=seed,
                        split_value=config.unlearn_num_examples,
                        max_prompt_len=config.max_prompt_len,
                        model_type=model_type,
                        exp_type=exp_type
                    )

                    exp_res_dict = run_npo_exp(
                        updated_model,
                        frozen_model,
                        tokenizer,
                        prompts_dict,
                        targets_dict,
                        exp_config,
                        specificity=True,
                        generality=(exp_type=='ssn'),
                        extraction=True
                    )

                    hyperparams = {
                        'num_epochs': num_epochs,
                        'lr': lr,
                        'beta': beta,
                        'seed': seed
                    }

                    exp_res_dict = process_experiment_results(
                        exp_res_dict, score_thresholds, config, hyperparams, calc_type="min"
                    )
                    res_dict = deep_merge_dicts(res_dict, exp_res_dict)

                    if config.log_wandb:
                        res_plot = plot_multi_experiment_results_npo(res_dict, return_plot=True)
                        wandb.log({f"{model_type.upper()} Aggregated Results": res_plot})

                    del updated_model
                    torch.cuda.empty_cache()

    if config.log_wandb:
        agg_res_dict = calc_aggregate_results_npo(
            res_dict=res_dict,
            seeds=seeds,
            score_thresholds=score_thresholds,
            num_epochs_list=num_epochs_list,
            lrs=lrs,
            betas=betas
        )
        wandb.log(dict(agg_res_dict))

    return dict(res_dict)

# ~~~~~~~~~~~~ UNLEARN ~~~~~~~~~~~~


def run_revs_unlearn(model, tokenizer, prompts, targets, config, pinv_lm_head):

    def edit_revs_target(model_editor:REVS, prompt, target, config):
        model = model_editor.model
        tokenizer = model_editor.tokenizer

        to_edit_layers = list(range(model_editor.model_n_layers))
        edit_dict = model_editor.edit_multiple_layers_dict(to_edit_layers, prompt, target,
            restore_after_edit=config.restore_after_edit, print_progress=False)
        return edit_dict

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

    requests = create_requests(all_concat_prompts, all_concat_targets, mode=config.mode)
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
            layers_patil=config.layers_patil,
            score_threshold=config.score_threshold,
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

    requests = create_requests(all_concat_prompts, all_concat_targets, mode='delete')  # mode is always delete for FT
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
            num_grad_steps=config.num_grad_steps,
            norm_constraint=config.norm_constraint,
            layers=config.layers,
        )
    print("\n Done applying edits")


def run_rmu_unlearn(updated_model, frozen_model, tokenizer, prompts, targets, config):

    wiki_data = []
    raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    for x in raw_data:
        if len(x['text']) > config.min_len:
            wiki_data.append(str(x['text']))
    # split the data into batches
    wiki_data = [wiki_data[i:i + config.batch_size] for i in range(0, len(wiki_data), config.batch_size)]

    unlearn_data = []
    for prompt, target in (zip(prompts, targets)):
        concat_prompts, concat_targets = create_concat_prompts_target(
            tokenizer=tokenizer,
            prompt=prompt,
            target=target,
            method=config.token_method,
            skip_tokens=config.skip_tokens,
            stop_tokens=config.stop_tokens,
            max_tokens=config.max_tokens,
        )
        unlearn_data.extend(concat_prompts)
    # split the data into batches
    unlearn_data = [unlearn_data[i:i + config.batch_size] for i in range(0, len(unlearn_data), config.batch_size)]

    run_rmu(
        updated_model,
        frozen_model,
        tokenizer,
        unlearn_data,
        wiki_data,
        config,
    )


def run_npo_unlearn(updated_model, frozen_model, tokenizer, prompts, targets, config):

    wiki_data = []
    raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    for x in raw_data:
        if len(x['text']) > config.min_len:
            wiki_data.append(str(x['text']))

    unlearn_data, unlearn_labels = [], []
    for prompt, target in (zip(prompts, targets)):
        concat_prompts, concat_targets = create_concat_prompts_target(
            tokenizer=tokenizer,
            prompt=prompt,
            target=target,
            method=config.token_method,
            skip_tokens=config.skip_tokens,
            stop_tokens=config.stop_tokens,
            max_tokens=config.max_tokens,
        )
        unlearn_data.extend(concat_prompts)
        unlearn_labels.extend(concat_targets)

    run_npo(
        unlearn_model=updated_model,
        ref_model=frozen_model,
        tokenizer=tokenizer,
        unlearn_prompts=unlearn_data,
        unlearn_targets=unlearn_labels,
        retain_prompts=wiki_data,
        config=config
    )


# ~~~~~~~~~~~~ EVALUATIONS ~~~~~~~~~~~~


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


def calc_aggregate_results_rmu(res_dict, seeds, score_thresholds, num_epochs_list, lrs, alphas, steering_coeffs):
    """
    Aggregate results across different seeds and add them to a new dictionary.

    Parameters:
    res_dict (dict): The dictionary where the results of each experiment are stored.
    seeds (list): The list of seeds used in the experiments.
    score_thresholds (list): The list of score thresholds used in the experiments.
    num_epochs_list (list): The list of number of epochs used in the experiments.
    lrs (list): The list of learning rates used in the experiments.
    alphas (list): The list of alphas used in the experiments.
    steering_coeffs (list): The list of steering coefficients used in the experiments.

    Returns:
    dict: A new dictionary with aggregated results.

    The function first checks if there are multiple seeds. If so, it extracts the metrics from the res_dict.
    For each combination of number of epochs, learning rate, alpha, steering coefficient, and score threshold,
    it calculates the mean and standard deviation of each metric across all seeds. These aggregated results are then
    added to a new dictionary, which represents the summary of results across all seeds.
    """
    # Create a new dictionary to store the aggregated results
    agg_res_dict = {}

    # Extract metrics from the res_dict
    num_epochs = list(res_dict.keys())[0]
    lr = list(res_dict[num_epochs].keys())[0]
    alpha = list(res_dict[num_epochs][lr].keys())[0]
    steering_coeff = list(res_dict[num_epochs][lr][alpha].keys())[0]
    seed = list(res_dict[num_epochs][lr][alpha][steering_coeff].keys())[0]
    threshold = list(res_dict[num_epochs][lr][alpha][steering_coeff][seed].keys())[0]
    metrics = list(res_dict[num_epochs][lr][alpha][steering_coeff][seed][threshold].keys())
    for num_epochs in num_epochs_list:
            for lr in lrs:
                for alpha in alphas:
                    for steering_coeff in steering_coeffs:
                        for threshold in score_thresholds:
                            for metric in metrics:
                                metric_values = [res_dict[f"{num_epochs}"][f"{lr}"][str(alpha)][str(steering_coeff)][f"{seed}"][f"{threshold}"][metric] for seed in seeds]
                                if f"{num_epochs}" not in agg_res_dict:
                                    agg_res_dict[f"{num_epochs}"] = {}
                                if f"{lr}" not in agg_res_dict[f"{num_epochs}"]:
                                    agg_res_dict[f"{num_epochs}"][f"{lr}"] = {}
                                if str(alpha) not in agg_res_dict[f"{num_epochs}"][f"{lr}"]:
                                    agg_res_dict[f"{num_epochs}"][f"{lr}"][str(alpha)] = {}
                                if str(steering_coeff) not in agg_res_dict[f"{num_epochs}"][f"{lr}"][str(alpha)]:
                                    agg_res_dict[f"{num_epochs}"][f"{lr}"][str(alpha)][str(steering_coeff)] = {}
                                if f"{threshold}" not in agg_res_dict[f"{num_epochs}"][f"{lr}"][str(alpha)][str(steering_coeff)]:
                                    agg_res_dict[f"{num_epochs}"][f"{lr}"][str(alpha)][str(steering_coeff)][f"{threshold}"] = {}
                                agg_res_dict[f"{num_epochs}"][f"{lr}"][str(alpha)][str(steering_coeff)][f"{threshold}"][f"{metric}_mean"] = round(np.mean(metric_values), 5)
                                agg_res_dict[f"{num_epochs}"][f"{lr}"][str(alpha)][str(steering_coeff)][f"{threshold}"][f"{metric}_std"] = round(np.std(metric_values), 5)
    return agg_res_dict


def calc_aggregate_results_npo(res_dict, seeds: List[int], score_thresholds: List[int],
                             num_epochs_list: List[int], lrs: List[float], betas: List[float]):
    """
    Aggregate results across different seeds and add them to a new dictionary.

    Parameters:
    res_dict (dict): The dictionary where the results of each experiment are stored.
    seeds (list): The list of seeds used in the experiments.
    score_thresholds (list): The list of score thresholds used in the experiments.
    num_epochs_list (list): The list of number of epochs used in the experiments.
    lrs (list): The list of learning rates used in the experiments.
    betas (list): The list of beta values used in the experiments.

    Returns:
    dict: A new dictionary with aggregated results.
    """
    # Create a new dictionary to store the aggregated results
    agg_res_dict = {}

    # Extract metrics from the res_dict
    num_epochs = list(res_dict.keys())[0]
    lr = list(res_dict[num_epochs].keys())[0]
    beta = list(res_dict[num_epochs][lr].keys())[0]
    seed = list(res_dict[num_epochs][lr][beta].keys())[0]
    threshold = list(res_dict[num_epochs][lr][beta][seed].keys())[0]
    metrics = list(res_dict[num_epochs][lr][beta][seed][threshold].keys())

    for num_epochs in num_epochs_list:
            for lr in lrs:
                for beta in betas:
                    for threshold in score_thresholds:
                        for metric in metrics:
                            metric_values = [
                                res_dict[f"{num_epochs}"][f"{lr}"][f"{beta}"][f"{seed}"][f"{threshold}"][metric]
                                for seed in seeds
                            ]
                            if f"{num_epochs}" not in agg_res_dict:
                                agg_res_dict[f"{num_epochs}"] = {}
                            if f"{lr}" not in agg_res_dict[f"{num_epochs}"]:
                                agg_res_dict[f"{num_epochs}"][f"{lr}"] = {}
                            if f"{beta}" not in agg_res_dict[f"{num_epochs}"][f"{lr}"]:
                                agg_res_dict[f"{num_epochs}"][f"{lr}"][f"{beta}"] = {}
                            if f"{threshold}" not in agg_res_dict[f"{num_epochs}"][f"{lr}"][f"{beta}"]:
                                agg_res_dict[f"{num_epochs}"][f"{lr}"][f"{beta}"][f"{threshold}"] = {}

                            agg_res_dict[f"{num_epochs}"][f"{lr}"][f"{beta}"][f"{threshold}"][f"{metric}_mean"] = round(np.mean(metric_values), 5)
                            agg_res_dict[f"{num_epochs}"][f"{lr}"][f"{beta}"][f"{threshold}"][f"{metric}_std"] = round(np.std(metric_values), 5)

    return agg_res_dict


def deep_merge_dicts(d1, d2):
    """
    Recursively merge two nested defaultdicts

    Args:
        d1: First defaultdict
        d2: Second defaultdict to merge into d1

    Returns:
        Merged defaultdict preserving all nested values
    """
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
            deep_merge_dicts(d1[k], v)
        else:
            d1[k] = v
    return d1


def calculate_result_metrics(
    exp_res_dict: Dict,
    threshold: int = 100,
    calc_type: Literal["min", "mean"] = "min",
    perplexity: Optional[bool] = False
) -> Dict[str, float]:
    """Calculate metrics for a single threshold and return results dictionary."""

    # Extract scores from input dictionary
    efficacy_edit_scores = exp_res_dict['efficacy']
    specificity_score = exp_res_dict['specificity']
    delta_attack_edit_scores = exp_res_dict['delta_attack']
    perturbed_attack_edit_scores = exp_res_dict['perturb_attack']
    logit_lens_attack_edit_scores = exp_res_dict['logit_lens_attack']
    generality_edit_scores = exp_res_dict.get('generality')
    perplexity_score = exp_res_dict.get('perplexity') if perplexity else None

    # Calculate metrics across layers
    efficacy_results = calculate_edit_score_statistics_squared(efficacy_edit_scores, threshold=threshold)
    efficacy_across_layers_score = calculate_across_layers_score(efficacy_results)

    perturbed_attack_results = calculate_edit_score_statistics_squared(perturbed_attack_edit_scores, threshold=threshold)
    perturbed_attack_across_layers_score = calculate_across_layers_score(perturbed_attack_results)

    logit_lens_attack_results = calculate_edit_score_statistics_squared(logit_lens_attack_edit_scores, threshold=threshold)
    logit_lens_attack_across_layers_score = calculate_across_layers_score(logit_lens_attack_results)

    # Extract scores
    efficacy = efficacy_across_layers_score['residual_after']['range_score_mean'][calc_type]
    perturbed_attack = perturbed_attack_across_layers_score['residual_after']['range_score_mean'][calc_type]
    logit_lens_attack = logit_lens_attack_across_layers_score['residual_after']['range_score_mean'][calc_type]

    delta_attack_mean_scores = [score.get_delta_attack_score(threshold)['mean'] for score in delta_attack_edit_scores]
    delta_attack = np.min(delta_attack_mean_scores) if calc_type == "min" else np.mean(delta_attack_mean_scores)

    # Calculate generality if available
    generality = None
    if generality_edit_scores is not None:
        generality_results = calculate_edit_score_statistics_squared(generality_edit_scores, threshold=threshold)
        generality_across_layers_score = calculate_across_layers_score(generality_results)
        generality = generality_across_layers_score['residual_after']['range_score_mean'][calc_type]

    # Calculate harmonic means
    core_scores = [efficacy, specificity_score]
    if generality is not None:
        core_scores.append(generality)

    attack_scores = [delta_attack, perturbed_attack, logit_lens_attack]
    harmonic_core = calculate_harmonic_mean(core_scores)
    harmonic_attack = calculate_harmonic_mean(attack_scores)

    # Build result dictionary
    result_dict = {
        'harmonic_core': round(harmonic_core, 5),
        'harmonic_attack': round(harmonic_attack, 5),
        'efficacy': round(efficacy, 5),
        'specificity': round(specificity_score, 5),
        'delta_attack': round(delta_attack, 5),
        'perturbed_attack': round(perturbed_attack, 5),
        'logit_lens_attack': round(logit_lens_attack, 5),
    }

    if generality is not None:
        result_dict['generality'] = round(generality, 5)

    if perplexity_score is not None:
        result_dict['perplexity'] = round(perplexity_score, 5)

    return result_dict


def process_experiment_results(
    exp_res_dict: Dict,
    score_thresholds: List[int],
    config: object,
    hyperparams: Dict[str, Union[int, float]],
    calc_type: Literal["min", "mean"] = "min"
) -> Dict:
    """Process experiment results and return nested dictionary with results."""
    def nested_dict():
        return defaultdict(nested_dict)

    res_dict = nested_dict()

    for threshold in score_thresholds:
        # Calculate all metrics using the simplified helper function
        result_dict = calculate_result_metrics(
            exp_res_dict=exp_res_dict,
            threshold=threshold,
            calc_type=calc_type,
            perplexity=config.perplexity
        )

        # Build nested dictionary path
        nested_dict = res_dict
        for param in hyperparams.values():
            if f"{param}" not in nested_dict:
                nested_dict[f"{param}"] = defaultdict(dict)
            nested_dict = nested_dict[f"{param}"]

        # Store results at the appropriate nesting level
        nested_dict[f"{threshold}"] = result_dict

    return res_dict


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


def get_dataset_email(seed: int, split_value: Union[float, int], max_prompt_len: int=None, model_type: str='gptj'):

    if model_type == 'gptj':
        path = "dataset_pile_emails_memorized.csv"
    elif model_type == 'llama':
        path = "dataset_pile_emails_memorized_llama.csv"
    else:
        raise ValueError("model_type must be either 'gptj' or 'llama'")

    if split_value is None:
        split_value = 0.5

    df_emails = pd.read_csv(path)
    df_emails = df_emails.drop_duplicates(subset=['email']).reset_index(drop=True)

    # if max_prompt_len is not None shorten the prompts
    if max_prompt_len is not None and max_prompt_len > 0:
        df_emails['prompt_gen'] = df_emails['prompt_gen'].apply(lambda x: x[-max_prompt_len:])
        print(f"\n\n\t ~~~~~~~~ Using shortened prompts of length {max_prompt_len} [EMAIL] ~~~~~~~~\n\n")
    else:
        print("\n\n\t ~~~~~~~~ Using full prompts [EMAIL] ~~~~~~~~\n\n")

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
        path = "ssn_multi_sentences_many_to_one.csv"
    if split_value is None:
        split_value = 0.5

    # Load the data and add space as a prefix to each ssn
    df_ssn = pd.read_csv(path)
    df_ssn['ssn'] = df_ssn['ssn'].apply(lambda ssn: f" {ssn}")

    # if max_prompt_len is not None shorten the prompts
    if max_prompt_len is not None and max_prompt_len > 0:
        df_ssn['prompt'] = df_ssn['prompt'].apply(lambda x: x[-max_prompt_len:])
        print(f"\n\n\t ~~~~~~~~ Using shortened prompts of length {max_prompt_len} [SSN] ~~~~~~~~\n\n")
    else:
        print("\n\n\t ~~~~~~~~ Using full prompts [SSN] ~~~~~~~~\n\n")

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


def get_dataset(seed: int, split_value: Union[float, int], model_type: str, exp_type: str, max_prompt_len: int=None, ):
    if exp_type == 'email':
        return get_dataset_email(seed=seed, split_value=split_value, max_prompt_len=max_prompt_len, model_type=model_type)
    elif exp_type == 'ssn':
        return get_dataset_ssn(seed=seed, split_value=split_value, max_prompt_len=max_prompt_len)
    else:
        raise ValueError("exp_type must be either 'email', 'gmail', 'ssn', or 'email_llama'")


def get_dataset_email_new(seed: int, split_value: Union[float, int], max_prompt_len: int=None):
    path = "dataset_pile_emails_new_memorized_llama_205.csv"

    if split_value is None:
        split_value = 0.5

    df_emails = pd.read_csv(path)
    df_emails = df_emails.drop_duplicates(subset=['pii_gen']).reset_index(drop=True)

    if max_prompt_len is not None and max_prompt_len > 0:
        df_emails['prompt_gen'] = df_emails['prompt_gen'].apply(lambda x: x[-max_prompt_len:])
        print(f"\n\n\t ~~~~~~~~ Using shortened prompts of length {max_prompt_len} [EMAIL_NEW] ~~~~~~~~\n\n")
    else:
        print("\n\n\t ~~~~~~~~ Using full prompts [EMAIL_NEW] ~~~~~~~~\n\n")

    if isinstance(split_value, float):
        forget_split_size = int(len(df_emails) * split_value)
    elif isinstance(split_value, int):
        forget_split_size = split_value
    else:
        raise ValueError("split_value must be either a float or an int")

    df_emails_unlearn = df_emails.sample(n=forget_split_size, random_state=seed)
    df_emails_retain = df_emails[~df_emails.index.isin(df_emails_unlearn.index)]

    unlearn_prompts = df_emails_unlearn['prompt_gen'].tolist()
    unlearn_targets = df_emails_unlearn['pii_gen'].tolist()
    retain_prompts = df_emails_retain['prompt_gen'].tolist()
    retain_targets = df_emails_retain['pii_gen'].tolist()

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

def get_dataset_email_new_disjoint(seed: int, split_value: Union[float, int], max_prompt_len: int=None):
    path = "dataset_pile_emails_new_disjoint_memorized_llama_115_filtered.csv"

    if split_value is None:
        split_value = 0.5

    df_emails = pd.read_csv(path)
    df_emails = df_emails.drop_duplicates(subset=['pii_gen']).reset_index(drop=True)

    if max_prompt_len is not None and max_prompt_len > 0:
        df_emails['prompt_gen'] = df_emails['prompt_gen'].apply(lambda x: x[-max_prompt_len:])
        print(f"\n\n\t ~~~~~~~~ Using shortened prompts of length {max_prompt_len} [EMAIL_NEW] ~~~~~~~~\n\n")
    else:
        print("\n\n\t ~~~~~~~~ Using full prompts [EMAIL_NEW] ~~~~~~~~\n\n")

    if isinstance(split_value, float):
        forget_split_size = int(len(df_emails) * split_value)
    elif isinstance(split_value, int):
        forget_split_size = split_value
    else:
        raise ValueError("split_value must be either a float or an int")

    df_emails_unlearn = df_emails.sample(n=forget_split_size, random_state=seed)
    df_emails_retain = df_emails[~df_emails.index.isin(df_emails_unlearn.index)]

    unlearn_prompts = df_emails_unlearn['prompt_gen'].tolist()
    unlearn_targets = df_emails_unlearn['pii_gen'].tolist()
    retain_prompts = df_emails_retain['prompt_gen'].tolist()
    retain_targets = df_emails_retain['pii_gen'].tolist()

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

def get_dataset_url(seed: int, split_value: Union[float, int], max_prompt_len: int=None):
    path = "dataset_pile_url_memorized_llama_new_203.csv"

    if split_value is None:
        split_value = 0.5

    df_urls = pd.read_csv(path)
    df_urls = df_urls.drop_duplicates(subset=['pii_gen']).reset_index(drop=True)

    if max_prompt_len is not None and max_prompt_len > 0:
        df_urls['prompt_gen'] = df_urls['prompt_gen'].apply(lambda x: x[-max_prompt_len:])
        print(f"\n\n\t ~~~~~~~~ Using shortened prompts of length {max_prompt_len} [URL] ~~~~~~~~\n\n")
    else:
        print("\n\n\t ~~~~~~~~ Using full prompts [URL] ~~~~~~~~\n\n")

    if isinstance(split_value, float):
        forget_split_size = int(len(df_urls) * split_value)
    elif isinstance(split_value, int):
        forget_split_size = split_value
    else:
        raise ValueError("split_value must be either a float or an int")

    df_urls_unlearn = df_urls.sample(n=forget_split_size, random_state=seed)
    df_urls_retain = df_urls[~df_urls.index.isin(df_urls_unlearn.index)]

    unlearn_prompts = df_urls_unlearn['prompt_gen'].tolist()
    unlearn_targets = df_urls_unlearn['pii_gen'].tolist()
    retain_prompts = df_urls_retain['prompt_gen'].tolist()
    retain_targets = df_urls_retain['pii_gen'].tolist()

    prompts_dict = {
        "unlearn": unlearn_prompts,
        "retain": retain_prompts,
    }
    targets_dict = {
        "unlearn": unlearn_targets,
        "retain": retain_targets,
    }
    df_urls_dict = {
        "unlearn": df_urls_unlearn,
        "retain": df_urls_retain,
    }
    return prompts_dict, targets_dict, df_urls_dict

def get_dataset(seed: int, split_value: Union[float, int], model_type: str, exp_type: str, max_prompt_len: int=None):
    if exp_type == 'email':
        return get_dataset_email(seed=seed, split_value=split_value, max_prompt_len=max_prompt_len, model_type=model_type)
    elif exp_type == 'ssn':
        return get_dataset_ssn(seed=seed, split_value=split_value, max_prompt_len=max_prompt_len)
    elif exp_type == 'email_new' and model_type == 'llama':
        return get_dataset_email_new(seed=seed, split_value=split_value, max_prompt_len=max_prompt_len)
    elif exp_type == 'email_new_disjoint' and model_type == 'llama':
        return get_dataset_email_new_disjoint(seed=seed, split_value=split_value, max_prompt_len=max_prompt_len)
    elif exp_type == 'url' and model_type == 'llama':
        return get_dataset_url(seed=seed, split_value=split_value, max_prompt_len=max_prompt_len)
    else:
        raise ValueError("exp_type must be either 'email', 'ssn', 'email_new' or 'url' and model_type must be 'llama'")