import argparse
from baselines.rmu.rmu_unlearn import RMUConfig
from utils.experiment_old import rmu_multi_exp


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Configure RMUConfig')
    parser.add_argument('--exp_type', type=str, choices=['ssn', 'email', 'email_new', 'email_new_disjoint', 'url'], required=True, help='Type of experiment to run')
    parser.add_argument('--model_type', type=str, choices=['llama', 'gptj'], required=True, help='Type of model to use')
    parser.add_argument('--skip_tokens', type=str, nargs='*', default=None, help='Skip the following tokens in the edit')
    parser.add_argument('--stop_tokens', type=str, nargs='*', default=None, help='Stop the edit at the following tokens')
    parser.add_argument('--max_tokens', type=int, default=None, help='Max tokens to unlearn in target')
    parser.add_argument('--score_thresholds', type=int, nargs='*', help='Score thresholds')
    parser.add_argument('--num_epochs_list', type=int, nargs='*', help='Number of epochs')
    parser.add_argument('--lrs', type=float, nargs='*', help='Learning rates')
    parser.add_argument('--alphas', type=float, nargs='*', help='Alphas')
    parser.add_argument('--steering_coeffs', type=float, nargs='*', help='Steering coefficients')
    parser.add_argument('--max_prompt_len', type=int, default=100, help='Max prompt length')
    parser.add_argument('--seeds', type=int, nargs='*', help='Seeds to run the experiment on')
    parser.add_argument('--perplexity', default=False, action='store_true', help='Calculate perplexity')
    parser.add_argument('--unlearn_num_examples', type=int, default=None, help='Number of examples to unlearn')
    parser.add_argument('--not_unlearn', default=False, action='store_true', help='Do not unlearn')
    parser.add_argument('--save_model', default=False, action='store_true', help='Save the model')
    parser.add_argument('--log_wandb', default=False, action='store_true', help='Log with Weights & Biases')
    parser.add_argument('--layer_id', type=int, default=7, help='Layer id to calculate loss on')
    parser.add_argument('--layer_ids', type=int, nargs='*', default=[5, 6, 7], help='Layer ids to edit weights on')
    parser.add_argument('--message', type=str, default=None, help='Message for the experiment')

    args = parser.parse_args()

    config = RMUConfig(
        exp_type=args.exp_type,
        model_type=args.model_type,
        score_threshold=args.score_thresholds[0],
        skip_tokens=args.skip_tokens,
        stop_tokens=args.stop_tokens,
        max_tokens=args.max_tokens,
        num_epochs=args.num_epochs_list[0],
        lr=args.lrs[0],
        alpha=args.alphas[0],
        steering_coeff=args.steering_coeffs[0],
        max_prompt_len=args.max_prompt_len,
        seed=args.seeds[0],
        perplexity=args.perplexity,
        unlearn_num_examples=args.unlearn_num_examples,
        not_unlearn=args.not_unlearn,
        save_model=args.save_model,
        log_wandb=args.log_wandb,
        layer_id=args.layer_id,
        layer_ids=args.layer_ids,
    )

    print(config)

    rmu_multi_exp(
        config=config,
        seeds=args.seeds,
        score_thresholds=args.score_thresholds,
        num_epochs_list=args.num_epochs_list,
        lrs=args.lrs,
        alphas=args.alphas,
        steering_coeffs=args.steering_coeffs,
        model_type=args.model_type,
        exp_type=args.exp_type,
        message=args.message,
    )
