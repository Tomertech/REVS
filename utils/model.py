from typing import List, Tuple, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from MEMIT.util import nethook
from MEMIT.baselines.ft import FTHyperParams, apply_ft_to_model
from MEMIT.memit import MEMITHyperParams, apply_memit_to_model
from MEMIT.rome import ROMEHyperParams, apply_rome_to_model
from MEMIT.util.globals import *

from utils.globals import CACHE_PATH, HG_TOKEN


class FTConfig:
    def __init__(
        self, 
        score_threshold, 
        skip_tokens, 
        stop_tokens, 
        max_tokens, 
        lr, 
        loss_break, 
        norm_constraint, 
        num_grad_steps, 
        seed, 
        perplexity, 
        save_model, 
        log_wandb, 
        max_prompt_len=None, 
        token_method=None, 
        unlearn_num_examples=None, 
        layers=None
    ):
        """
        Initializes the FTConfig class with various configuration settings for fine-tuning models.

        Parameters:
            score_threshold (int): Threshold for score; defines the range for rank consideration.
            skip_tokens (list): Tokens to skip during editing.
            stop_tokens (list): Tokens to stop at during editing.
            max_tokens (int): Maximum tokens to consider for editing.
            lr (float): Learning rate for the optimization.
            loss_break (float): Loss value at which to break the training loop.
            norm_constraint (float): Norm constraint for gradient clipping.
            num_grad_steps (int): Number of gradient steps for fine-tuning.
            seed (int): Seed for random number generation to ensure reproducibility.
            perplexity (bool): Whether to consider perplexity during fine-tuning.
            save_model (bool): Whether to save the model post fine-tuning.
            log_wandb (bool): Whether to log with Weights & Biases.
            max_prompt_len (int, optional): Maximum prompt length; defaults to None.
            token_method (str, optional): Method to select target token; defaults to None.
            unlearn_num_examples (int, optional): Number of examples for unlearning; defaults to None.
            layers (list, optional): Specific layers to fine-tune; defaults to None.
        """
        self.score_threshold = score_threshold
        self.skip_tokens = skip_tokens
        self.stop_tokens = stop_tokens
        self.max_tokens = max_tokens
        self.lr = lr
        self.loss_break = loss_break
        self.norm_constraint = norm_constraint
        self.num_grad_steps = num_grad_steps
        self.seed = seed
        self.perplexity = perplexity
        self.save_model = save_model
        self.log_wandb = log_wandb
        self.max_prompt_len = max_prompt_len
        self.token_method = token_method
        self.unlearn_num_examples = unlearn_num_examples
        self.layers = layers

    def to_dict(self):
        """
        Converts the configuration settings into a dictionary.

        Returns:
            dict: A dictionary representation of the configuration settings.
        """
        return {
            'score_threshold': self.score_threshold,
            'skip_tokens': self.skip_tokens,
            'stop_tokens': self.stop_tokens,
            'max_tokens': self.max_tokens,
            'lr': self.lr,
            'loss_break': self.loss_break,
            'norm_constraint': self.norm_constraint,
            'num_grad_steps': self.num_grad_steps,
            'seed': self.seed,
            'perplexity': self.perplexity,
            'save_model': self.save_model,
            'log_wandb': self.log_wandb,
            'max_prompt_len': self.max_prompt_len,
            'token_method': self.token_method,
            'unlearn_num_examples': self.unlearn_num_examples,
            'layers': self.layers,
        }

class MemitConfig:
    def __init__(
        self, 
        score_threshold, 
        skip_tokens, 
        stop_tokens, 
        max_tokens, 
        v_lr, 
        loss_break, 
        loss_pred_prob_coef, 
        v_num_grad_steps, 
        seed, 
        perplexity, 
        save_model, 
        log_wandb, 
        max_prompt_len=None, 
        token_method=None, 
        unlearn_num_examples=None, 
        layers=None
    ):
        """
        Initializes the MemitConfig class with various configuration settings for the Memit model.

        Parameters:
            score_threshold (int): Threshold for score; defines the range for rank consideration.
            skip_tokens (list): Tokens to skip during editing.
            stop_tokens (list): Tokens to stop at during editing.
            max_tokens (int): Maximum tokens to consider for editing.
            v_lr (float): Learning rate for the optimization.
            loss_break (float): Loss value at which to break the training loop.
            loss_pred_prob_coef (float): Coefficient for the loss prediction probability.
            v_num_grad_steps (int): Number of gradient steps for fine-tuning.
            seed (int): Seed for random number generation to ensure reproducibility.
            perplexity (bool): Whether to consider perplexity during fine-tuning.
            save_model (bool): Whether to save the model post fine-tuning.
            log_wandb (bool): Whether to log with Weights & Biases.
            max_prompt_len (int, optional): Maximum prompt length; defaults to None.
            token_method (str, optional): Method to select target token; defaults to None.
            unlearn_num_examples (int, optional): Number of examples for unlearning; defaults to None.
            layers (list, optional): Specific layers to fine-tune; defaults to None.
        """
        self.score_threshold = score_threshold
        self.skip_tokens = skip_tokens
        self.stop_tokens = stop_tokens
        self.max_tokens = max_tokens
        self.v_lr = v_lr
        self.loss_break = loss_break
        self.loss_pred_prob_coef = loss_pred_prob_coef
        self.v_num_grad_steps = v_num_grad_steps
        self.seed = seed
        self.perplexity = perplexity
        self.save_model = save_model
        self.log_wandb = log_wandb
        self.max_prompt_len = max_prompt_len
        self.token_method = token_method
        self.unlearn_num_examples = unlearn_num_examples
        self.layers = layers

    def to_dict(self):
        """
        Converts the configuration settings into a dictionary.

        Returns:
            dict: A dictionary representation of the configuration settings.
        """
        return {
            'score_threshold': self.score_threshold,
            'skip_tokens': self.skip_tokens,
            'stop_tokens': self.stop_tokens,
            'max_tokens': self.max_tokens,
            'v_lr': self.v_lr,
            'loss_break': self.loss_break,
            'loss_pred_prob_coef': self.loss_pred_prob_coef,
            'v_num_grad_steps': self.v_num_grad_steps,
            'seed': self.seed,
            'perplexity': self.perplexity,
            'save_model': self.save_model,
            'log_wandb': self.log_wandb,
            'max_prompt_len': self.max_prompt_len,
            'token_method': self.token_method,
            'unlearn_num_examples': self.unlearn_num_examples,
            'layers': self.layers,
        }


def load_model_tokenizer(model_name='gptj', cache_dir=CACHE_PATH, device="auto"):
    if model_name == 'gptj':
        model_name = "EleutherAI/gpt-j-6B"
        tok = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_PATH)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_PATH, device_map=device)
        tok.pad_token = tok.eos_token
        return model, tok

    elif model_name == 'llama':
        model_name = "meta-llama/Meta-Llama-3-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HG_TOKEN)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, token=HG_TOKEN, cache_dir=CACHE_PATH, device_map=device)
        return model, tokenizer

    else:
        raise ValueError(f"Unknown model name: {model_name}, use 'gptj' or 'llama'")


def load_model_tokenizer_ssn(model_name='gptj', device="auto"):

    if model_name == 'gptj':
        model_name = "EleutherAI/gpt-j-6B"
        model_path = "/path/to/the/ft/model/model_ft_multi_ssn"  # TODO: fill in the path to the fintuned model on the ssn dataset

    elif model_name == 'llama':
        model_name = "meta-llama/Meta-Llama-3-8B"
    else:
        raise ValueError(f"Unknown model name: {model_name}, use 'gptj' or 'llama'")

    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_PATH)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
    tok.pad_token = tok.eos_token
    return model, tok


def create_requests(prompts: list, targets: list, mode: str):
    requests = []
    assert mode in ["delete", "insert", "replace"]
    for prompt, target in zip(prompts, targets):
        # Take the last max_prompt_length characters from the prompt if max_prompt_length is provided
        request = {
            "prompt": "{}",
            "subject": prompt,
            "target_new": {"str": target},
            "target_true": {"str": ""},
            "mode": mode,
        }
        requests.append(request)
    return requests


def edit_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    generation_prompts: List[str],
    alg_name: str,
    max_out_len: int = 100,
    top_k=1,
    **kwargs,
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    """
    Applies the selected model editing algorithm. Generates text both before and after
    for comparison of model behavior. Returns the updated model and the original values of
    weights that were changed.
    """
    nethook.set_requires_grad(True, model)

    RewritingParamsClass, apply_method, hparams_prefix, hparams_suffix = load_alg(alg_name.upper())
    params_dir = 'MEMIT/hprarams'

    if model.config.model_type == 'gptj':
        model_name = 'EleutherAI_gpt-j-6B'
    elif model.config.model_type == 'llama':
        model_name = 'llama_3_8b'
    else:
        raise ValueError(f"Model type not supported {model.config.model_type}, supported models are 'gptj' and 'llama'")

    print_loud(f"Retrieving {alg_name} hyperparameters")
    if alg_name.lower() == "memit":
        v_lr = kwargs.get('v_lr', 0.05)
        loss_break = kwargs.get('loss_break', 0.1)
        loss_pred_prob_coef = kwargs.get('loss_pred_prob_coef', 1)
        v_num_grad_steps = kwargs.get('v_num_grad_steps', 25)
        layers = kwargs.get('layers', [3,4,5,6,7,8]) # default as in memit paper
        hparams = RewritingParamsClass.from_json(params_dir + '/MEMIT/' + model_name + '.json')
        hparams.v_lr = v_lr
        hparams.loss_break = loss_break
        hparams.loss_pred_prob_coef = loss_pred_prob_coef
        hparams.v_num_grad_steps = v_num_grad_steps
        hparams.layers = layers
    elif alg_name.lower() == "ft":
        lr = kwargs.get('lr', 1e-5)
        loss_break = kwargs.get('loss_break', 1e-3)
        if model_name == 'EleutherAI_gpt-j-6B':
            layers = kwargs.get('layers', list(range(0, 28))) # default is all layers
        elif model_name == 'llama_3_8b':
            layers = kwargs.get('layers', list(range(0, 32))) # default is all layers
        norm_constraint = kwargs.get('norm_constraint', 5e-4)
        num_grad_steps = kwargs.get('num_grad_steps', 5)
        hparams = RewritingParamsClass.from_json(params_dir + '/FT/' + model_name + '_constr.json')  # for constraint FT-L
        hparams.layers = layers
        hparams.lr = lr
        hparams.norm_constraint = norm_constraint
        hparams.num_steps = num_grad_steps
        hparams.loss_break = loss_break
    else:
        raise ValueError(f"Unknown algorithm: {alg_name}, please use 'memit' or 'ft'")
    print(hparams)

    # print_loud("Generating pre-update text")
    # pre_update_text = generate_fast(model, tok, generation_prompts, max_out_len=max_out_len)

    print_loud(f"Applying {alg_name} to model")
    model, orig_weights = apply_method(
        model,
        tok,
        requests,
        hparams,
        return_orig_weights=False,
    )

    # print_loud("Generating post-update text")
    # post_update_text = generate_fast(model, tok, generation_prompts, max_out_len=max_out_len)

    # print_prompts(generation_prompts, pre_update_text, post_update_text, alg_name)
    return orig_weights


def load_alg(alg_name):
    """
    Loads dependencies for the desired algorithm.
    Implementation is slightly awkward to prevent unnecessary imports on Colab.

    The return value is a tuple of the following:
    1. Class for storing hyperparameters
    2. Method for applying rewrites
    3. Location of parameters
    4. Predefined suffix for the param file
    """
    assert alg_name in [
        "FT",
        "FT-L",
        "FT-AttnEdit",
        "MEND",
        "MEND-CF",
        "MEND-zsRE",
        "ROME",
        "MEMIT",
    ]

    if alg_name == "ROME":
        return ROMEHyperParams, apply_rome_to_model, "ROME", ""
    elif alg_name == "MEMIT":
        return MEMITHyperParams, apply_memit_to_model, "MEMIT", ""
    elif "FT" in alg_name:
        d = {
            "FT": (FTHyperParams, apply_ft_to_model, "FT", "_unconstr"),
            "FT-AttnEdit": (FTHyperParams, apply_ft_to_model, "FT", "_attn"),
            "FT-L": (FTHyperParams, apply_ft_to_model, "FT", "_constr"),
        }
        return d[alg_name]
    else:
        from baselines.mend import MENDHyperParams, MendRewriteExecutor

        d = {
            "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model, "MEND", ""),
            "MEND-CF": (
                MENDHyperParams,
                MendRewriteExecutor().apply_to_model,
                "MEND",
                "_CF",
            ),
            "MEND-zsRE": (
                MENDHyperParams,
                MendRewriteExecutor().apply_to_model,
                "MEND",
                "_zsRE",
            ),
        }
        return d[alg_name]


def print_loud(text):
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~", text, "~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
