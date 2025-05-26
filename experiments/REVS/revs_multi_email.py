import argparse
from utils.experiment_old import revs_multi_email_exp
from revs.revs import REVSConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configure REVS')

    parser.add_argument('--n_neurons_list', type=int, nargs='*', help='Number of neurons to edit')

    parser.add_argument('--residual_bottom_rank_margin', type=int, default=10000, help='Residual bottom rank margin')
    parser.add_argument('--residual_top_rank_margin', type=int, default=20000, help='Residual top rank margin')
    parser.add_argument('--max_iter_mlp_rank', type=int, default=100, help='Max iterations for MLP rank')
    parser.add_argument('--mlp_bottom_rank_margin', type=int, default=10000, help='MLP bottom rank margin')
    parser.add_argument('--mlp_top_rank_margin', type=int, default=10000, help='MLP top rank margin')
    parser.add_argument('--max_iter_neuron_rank', type=int, default=100, help='Max iterations for neuron rank')
    parser.add_argument('--neuron_bottom_rank_margin', type=int, default=90000, help='Neuron bottom rank margin')
    parser.add_argument('--neuron_top_rank_margin', type=int, default=100000, help='Neuron top rank margin')

    parser.add_argument('--act_filter', type=str, default='top_100', help='Activation filter')
    parser.add_argument('--neurons_score_method', type=str, default='rank', help='Neurons score method')
    parser.add_argument('--score_thresholds', type=int, nargs='*', help='Score thresholds')
    parser.add_argument('--zero_neurons', default=False, action='store_true', help='Zero out neurons')

    parser.add_argument('--skip_tokens', type=str, nargs='*', default=None, help='Skip the following tokens in the edit')
    parser.add_argument('--stop_tokens', type=str, nargs='*', default=None, help='Stop the edit at the following tokens')
    parser.add_argument('--max_tokens', type=int, default=None, help='Max tokens to unlearn in target')
    parser.add_argument('--max_prompt_len', type=int, default=None, help='Max prompt length')
    parser.add_argument('--token_method', type=str, default='rarest', help='Token selection method (rarest, first, frequent, random)')

    parser.add_argument('--insert_new_token', type=str, default=None, help='Insert new token, if None no token will be inserted.')
    parser.add_argument('--restore_after_edit', default=False, action='store_true', help='Restore weights after each target edit and apply all edits at once at the end')

    parser.add_argument('--seeds', type=int, nargs='*', help='Seeds to run the experiment on')
    parser.add_argument('--perplexity', default=False, action='store_true', help='Calculate Perplexity after edit')
    parser.add_argument('--not_unlearn', default=False, action='store_true', help='Do not unlearn, for specific exps and debugging')
    parser.add_argument('--unlearn_num_examples', type=int, default=None, help='Number of examples to unlearn')
    parser.add_argument('--save_model',  default=False, action='store_true', help='Save model after editing')
    parser.add_argument('--log_wandb', default=False, action='store_true',  help='Log with Weights & Biases')
    parser.add_argument('--message', type=str, default=None, help='Add a message to the name of the experiment')

    args = parser.parse_args()

    config = REVSConfig(
        n_neurons=args.n_neurons_list[0],
        residual_bottom_rank_margin=args.residual_bottom_rank_margin,
        residual_top_rank_margin=args.residual_top_rank_margin,
        max_iter_mlp_rank=args.max_iter_mlp_rank,
        mlp_bottom_rank_margin=args.mlp_bottom_rank_margin,
        mlp_top_rank_margin=args.mlp_top_rank_margin,
        max_iter_neuron_rank=args.max_iter_neuron_rank,
        neuron_bottom_rank_margin=args.neuron_bottom_rank_margin,
        neuron_top_rank_margin=args.neuron_top_rank_margin,
        act_filter=args.act_filter,
        neurons_score_method=args.neurons_score_method,
        score_threshold=args.score_thresholds[0],
        zero_neurons=args.zero_neurons,
        skip_tokens=args.skip_tokens,
        stop_tokens=args.stop_tokens,
        max_tokens=args.max_tokens,
        max_prompt_len=args.max_prompt_len,
        token_method=args.token_method,
        insert_new_token=args.insert_new_token,
        restore_after_edit=args.restore_after_edit,
        seed=args.seeds[0],
        perplexity=args.perplexity,
        not_unlearn=args.not_unlearn,
        unlearn_num_examples=args.unlearn_num_examples,
        save_model=args.save_model,
        log_wandb=args.log_wandb,
    )

    revs_multi_email_exp(config, seeds=args.seeds, n_neurons_list=args.n_neurons_list, score_thresholds=args.score_thresholds, message=args.message)
