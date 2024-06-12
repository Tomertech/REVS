import argparse
from utils.model import MemitConfig
from utils.experiment import memit_multi_ssn_exp


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Configure MemitConfig')
    parser.add_argument('--skip_tokens', type=str, nargs='*', default=None, help='Skip the following tokens in the edit')
    parser.add_argument('--stop_tokens', type=str, nargs='*', default=None, help='Stop the edit at the following tokens')
    parser.add_argument('--max_tokens', type=int, default=None, help='Max tokens to unlearn in target')
    parser.add_argument('--score_thresholds', type=int, nargs='*', help='Score thresholds')
    parser.add_argument('--v_lrs', type=float, nargs='*', help='V learning rates')
    parser.add_argument('--loss_breaks', type=float, nargs='*', help='Loss breaks')
    parser.add_argument('--loss_pred_prob_coefs', type=float, nargs='*', help='Loss pred prob coef')
    parser.add_argument('--v_num_grad_steps', type=int, default=25, help='Number of gradient steps for V')
    parser.add_argument('--max_prompt_len', type=int, default=None, help='Max prompt length')
    parser.add_argument('--seeds', type=int, nargs='*', help='Seeds to run the experiment on')
    parser.add_argument('--perplexity', default=False, action='store_true', help='')
    parser.add_argument('--unlearn_num_examples', type=int, default=None, help='Number of examples to unlearn')
    parser.add_argument('--save_model', default=False, action='store_true', help='Save the model')
    parser.add_argument('--log_wandb', default=False, action='store_true',  help='Log with Weights & Biases')
    args = parser.parse_args()
    config = MemitConfig(
        score_threshold=args.score_thresholds[0],
        skip_tokens=args.skip_tokens,
        stop_tokens=args.stop_tokens,
        max_tokens=args.max_tokens,
        v_lr=args.v_lrs[0],
        loss_break=args.loss_breaks[0],
        loss_pred_prob_coef=args.loss_pred_prob_coefs[0],
        v_num_grad_steps=args.v_num_grad_steps,
        max_prompt_len=args.max_prompt_len,
        seed=args.seeds[0],
        perplexity=args.perplexity,
        unlearn_num_examples=args.unlearn_num_examples,
        save_model=args.save_model,
        log_wandb=args.log_wandb
    )
    memit_multi_ssn_exp(config, args.seeds, args.score_thresholds, args.v_lrs, args.loss_breaks, args.loss_pred_prob_coefs)