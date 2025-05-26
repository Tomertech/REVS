import argparse
from utils.model import FTConfig
from utils.experiment_old import ft_multi_ssn_exp


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Configure MemitConfig')
    parser.add_argument('--skip_tokens', type=str, nargs='*', default=None, help='Skip the following tokens in the edit')
    parser.add_argument('--stop_tokens', type=str, nargs='*', default=None, help='Stop the edit at the following tokens')
    parser.add_argument('--max_tokens', type=int, default=None, help='Max tokens to unlearn in target')
    parser.add_argument('--score_thresholds', type=int, nargs='*', help='Score thresholds')
    parser.add_argument('--lrs', type=float, nargs='*', help='learning rates')
    parser.add_argument('--norm_constraints', type=float, nargs='*', help='Norm constraints')
    parser.add_argument('--loss_breaks', type=float, nargs='*', help='Loss breaks')
    parser.add_argument('--num_grad_steps', type=int, help='Number of gradient steps')
    parser.add_argument('--seeds', type=int, nargs='*', help='Seeds to run the experiment on')
    parser.add_argument('--perplexity', default=False, action='store_true', help='')
    parser.add_argument('--unlearn_num_examples', type=int, default=None, help='Number of examples to unlearn')
    parser.add_argument('--save_model', default=False, action='store_true', help='Save the model')
    parser.add_argument('--log_wandb', default=False, action='store_true',  help='Log with Weights & Biases')
    parser.add_argument('--layers', type=int, nargs='*', default=list(range(0,28)), help='Layers to apply fine tuning on')

    args = parser.parse_args()
    config = FTConfig(
        score_threshold=args.score_thresholds[0],
        skip_tokens=args.skip_tokens,
        stop_tokens=args.stop_tokens,
        max_tokens=args.max_tokens,
        lr=args.lrs[0],
        norm_constraint=args.norm_constraints[0],
        layers=args.layers,
        loss_break=args.loss_breaks[0],
        num_grad_steps=args.num_grad_steps,
        seed=args.seeds[0],
        save_model=args.save_model,
        perplexity=args.perplexity,
        unlearn_num_examples=args.unlearn_num_examples,
        log_wandb=args.log_wandb
    )
    ft_multi_ssn_exp(
        config=config,
        seeds=args.seeds,
        score_thresholds=args.score_thresholds,
        lrs=args.lrs,
        loss_breaks=args.loss_breaks,
        norm_constraints=args.norm_constraints,
    )