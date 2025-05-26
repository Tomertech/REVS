import wandb
import argparse

from delpii.baselines.npo.npo_unlearn import NPOConfig
from delpii.utils.experiment import run_npo_exp, calculate_result_metrics, get_dataset
from delpii.utils.globals import PROJECT_NAME_SWEEP
from delpii.utils.model import load_model_tokenizer


def run_sweep_exp():
    with wandb.init() as run:
        # Access hyperparameters from wandb
        config = wandb.config

        # Create NPO config with sweep parameters
        npo_config = NPOConfig(
            exp_type=config.exp_type,
            model_type=config.model_type,
            seed=config.seed,

            num_epochs=config.num_epochs,
            lr=config.lr,
            beta=config.beta,
            loss_type=config.loss_type,
            kl_coeff=config.kl_coeff,

            skip_tokens=["@", ".", "gmail", "com"],
            stop_tokens=["@"],
            max_tokens=2,
            max_prompt_len=100,
            score_threshold=100,
            perplexity=False,
            unlearn_num_examples=None,
            not_unlearn=False,
            save_model=False,
            verbose=True,
            log_wandb=False,
        )

        # Run single experiment with current sweep parameters
        model_type, exp_type = npo_config.model_type, npo_config.exp_type

        frozen_model, tokenizer = load_model_tokenizer(model_type, exp_type)
        updated_model, _ = load_model_tokenizer(model_type, exp_type)

        # Get dataset based on experiment type
        prompts_dict, targets_dict, df_dict = get_dataset(
            seed=npo_config.seed,
            split_value=npo_config.unlearn_num_examples,
            max_prompt_len=npo_config.max_prompt_len,
            model_type=model_type,
            exp_type=exp_type
        )

        exp_res_dict = run_npo_exp(
            updated_model,
            frozen_model,
            tokenizer,
            prompts_dict,
            targets_dict,
            npo_config,
            specificity=True,
            generality=(exp_type=='ssn'),
            extraction=True
        )
        result_metrics = calculate_result_metrics(exp_res_dict)
        wandb.log(result_metrics)
        return result_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Configure NPOConfig')
    parser.add_argument('--exp_type', type=str, choices=['ssn', 'email', 'email_new', 'email_new_disjoint', 'url'], required=True, help='Type of experiment to run')
    parser.add_argument('--model_type', type=str, choices=['llama', 'gptj'], required=True, help='Type of model to use')
    parser.add_argument('--seed', type=int, default=0, help='Seed to run the experiment on')
    parser.add_argument('--loss_type', type=str, default='npo_kl', help='Type of loss to use')
    parser.add_argument('--count', type=int, default=100, help='Number of experiments to run')

    args = parser.parse_args()

    # Define sweep configuration
    sweep_config = {
        'name': f'NPO-{args.model_type.upper()}-{args.exp_type.upper()}',
        'method': 'bayes',
        'metric': {
            'name': 'harmonic_core',
            'goal': 'maximize'
        },
        'parameters': {
            # Fixed parameters from args
            'exp_type': {'value': args.exp_type},
            'model_type': {'value': args.model_type},
            'loss_type': {'value': args.loss_type},
            'seed': {'value': args.seed},

            # Optimization parameters
            'lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-7,
                'max': 5e-6
            },
            'num_epochs': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 3
            },
            'beta': {
                'values': [1, 10, 100, 1000]
            },
            'kl_coeff': {
                'values': [1, 10, 100, 1000]
            },
        }
    }

    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project=PROJECT_NAME_SWEEP,
    )

    # Start the sweep
    wandb.agent(sweep_id, function=run_sweep_exp, count=args.count)
