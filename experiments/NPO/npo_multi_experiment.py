import argparse
from delpii.baselines.npo.npo_unlearn import NPOConfig
from delpii.utils.experiment import npo_multi_exp

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Configure NPOConfig')
    parser.add_argument('--exp_type', type=str, choices=['ssn', 'email', 'email_new', 'email_new_disjoint', 'url'], required=True, help='Type of experiment to run')
    parser.add_argument('--model_type', type=str, choices=['llama', 'gptj'], required=True, help='Type of model to use')
    parser.add_argument('--skip_tokens', type=str, nargs='*', default=None, help='Skip the following tokens in the edit')
    parser.add_argument('--stop_tokens', type=str, nargs='*', default=None, help='Stop the edit at the following tokens')
    parser.add_argument('--max_tokens', type=int, default=None, help='Max tokens to unlearn in target')
    parser.add_argument('--score_thresholds', type=int, nargs='*', help='Score thresholds')
    parser.add_argument('--num_epochs_list', type=int, nargs='*', help='Number of epochs')
    parser.add_argument('--lrs', type=float, nargs='*', help='Learning rates')
    parser.add_argument('--npo_coeff', type=float, help='NPO coefficient')
    parser.add_argument('--retain_coeff', type=float, help='Retain coefficient')
    parser.add_argument('--kl_coeff', type=float, help='KL coefficient')
    parser.add_argument('--betas', type=float, nargs='*', help='Beta parameters')
    parser.add_argument('--max_prompt_len', type=int, default=100, help='Max prompt length')
    parser.add_argument('--seeds', type=int, nargs='*', help='Seeds to run the experiment on')
    parser.add_argument('--loss_type', type=str, default='npo_kl', help='Loss type to use')
    parser.add_argument('--perplexity', default=False, action='store_true', help='Calculate perplexity')
    parser.add_argument('--unlearn_num_examples', type=int, default=None, help='Number of examples to unlearn')
    parser.add_argument('--not_unlearn', default=False, action='store_true', help='Do not unlearn')
    parser.add_argument('--save_model', default=False, action='store_true', help='Save the model')
    parser.add_argument('--log_wandb', default=True, action='store_true', help='Log with Weights & Biases')
    parser.add_argument('--min_len', type=int, default=20, help='Minimum length')
    parser.add_argument('--verbose', default=False, action='store_true', help='Verbose output')
    parser.add_argument('--token_method', type=str, default='rarest', help='Token selection method (rarest, first, frequent, random)')
    parser.add_argument('--message', type=str, default=None, help='Add a message to the name of the experiment')

    args = parser.parse_args()

    config = NPOConfig(
        exp_type=args.exp_type,
        model_type=args.model_type,
        score_threshold=args.score_thresholds[0],
        skip_tokens=args.skip_tokens,
        stop_tokens=args.stop_tokens,
        max_tokens=args.max_tokens,
        max_prompt_len=args.max_prompt_len,
        seed=args.seeds[0],
        loss_type=args.loss_type,
        perplexity=args.perplexity,
        save_model=args.save_model,
        log_wandb=args.log_wandb,
        token_method=args.token_method,
        unlearn_num_examples=args.unlearn_num_examples,
        not_unlearn=args.not_unlearn,
        num_epochs=args.num_epochs_list[0],
        npo_coeff=args.npo_coeff,
        retain_coeff=args.retain_coeff,
        kl_coeff=args.kl_coeff,
        beta=args.betas[0],
        lr=args.lrs[0],
        min_len=args.min_len,
        verbose=args.verbose
    )

    print(config)

    npo_multi_exp(
        config=config,
        seeds=args.seeds,
        score_thresholds=args.score_thresholds,
        num_epochs_list=args.num_epochs_list,
        lrs=args.lrs,
        betas=args.betas,
        model_type=args.model_type,
        exp_type=args.exp_type,
        message=args.message
    )
