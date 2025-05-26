from typing import List, Literal, Optional
import os
import datetime
import numpy as np
import torch
from transformers import AdamW
import tqdm as tqdm

from utils.configs import BaseConfig



class RMUConfig(BaseConfig):
    """RMU-specific configuration that extends BaseConfig.

    Additional Args:
        alpha (float): Coefficient for retain loss of frozen model
        steering_coeff (float): Coefficient for random vector steering
        lr (float): Learning rate
        num_epochs (int): Number of training epochs
        min_len (int): Minimum sequence length
        max_len (int): Maximum sequence length
        batch_size (int): Training batch size
        max_num_batches (int): Maximum number of batches
        layer_id (int): Layer to optimize loss on
        layer_ids (List[int]): Layers to edit
        param_ids (List[int]): Parameter indices to edit (default 6 is the MLP layer as in the original paper)
        verbose (bool): Whether to print verbose output
    """
    def __init__(self,
                 score_threshold: int,
                 skip_tokens: list,
                 stop_tokens: list,
                 max_tokens: int,
                 max_prompt_len: int,
                 seed: int,
                 exp_type: Literal['ssn', 'email'],
                 model_type: Literal['llama', 'gptj'],
                 token_method: Literal['rarest', 'first', 'frequent', 'random'] = 'rarest',
                 perplexity: bool = False,
                 save_model: bool = False,
                 log_wandb: bool = True,
                 unlearn_num_examples: Optional[int] = None,
                 not_unlearn: bool = False,

                 alpha: float = 100.0,
                 steering_coeff: float = 20.0,
                 lr: float = 5e-5,
                 num_epochs: int = 1,
                 min_len: int = 0,
                 max_len: int = 1000,
                 batch_size: int = 1,
                 max_num_batches: int = 80,
                 layer_id: int = 7,
                 layer_ids: List[int] = [5, 6, 7],
                 param_ids: List[int] = [6],
                 verbose: bool = True):

        super().__init__(
            method_name="RMU",
            score_threshold=score_threshold,
            skip_tokens=skip_tokens,
            stop_tokens=stop_tokens,
            max_tokens=max_tokens,
            max_prompt_len=max_prompt_len,
            seed=seed,
            exp_type=exp_type,
            model_type=model_type,
            token_method=token_method,
            perplexity=perplexity,
            save_model=save_model,
            log_wandb=log_wandb,
            unlearn_num_examples=unlearn_num_examples,
            not_unlearn=not_unlearn
        )

        self.alpha = alpha
        self.steering_coeff = steering_coeff
        self.lr = lr
        self.min_len = min_len
        self.max_len = max_len
        self.batch_size = batch_size
        self.max_num_batches = max_num_batches
        self.layer_id = layer_id
        self.layer_ids = layer_ids
        self.param_ids = param_ids
        self.num_epochs = num_epochs
        self.verbose = verbose

    def to_dict(self):
        base_dict = super().to_dict()
        rmu_dict = {
            'alpha': self.alpha,
            'steering_coeff': self.steering_coeff,
            'lr': self.lr,
            'min_len': self.min_len,
            'max_len': self.max_len,
            'batch_size': self.batch_size,
            'max_num_batches': self.max_num_batches,
            'layer_id': self.layer_id,
            'layer_ids': self.layer_ids,
            'param_ids': self.param_ids,
            'num_epochs': self.num_epochs,
            'verbose': self.verbose,
        }
        return {**base_dict, **rmu_dict}

def get_params(model, layer_ids, param_ids):
    params = []
    if model.config.model_type == 'gptj':
        for layer_id in layer_ids:
            for i, p in enumerate(model.transformer.h[layer_id].parameters()):
                if i in param_ids:
                    params.append(p)
    elif model.config.model_type == 'llama':
        for layer_id in layer_ids:
            for i, p in enumerate(model.model.layers[layer_id].parameters()):
                if i in param_ids:
                    params.append(p)
    else:
        raise ValueError(f"Model type not supported {model.config.model_type}, supported models are 'gptj' and 'llama'")
    return params

def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None

    hook_handle = module.register_forward_hook(hook)

    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)

    hook_handle.remove()

    return cache[0]

def run_rmu(
    updated_model,
    frozen_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    config: RMUConfig,
):

    if config.verbose:
        print("====rmu Config====")
        print("\n".join(f"{k}={v}" for k,v in config.__dict__.items()))
        print("=====")

    updated_model = updated_model.train()
    params = get_params(updated_model, config.layer_ids, config.param_ids)
    optimizer = AdamW(params, lr=config.lr)

    if updated_model.config.model_type == 'gptj':
        frozen_module = eval(
            f"frozen_model.transformer.h[{config.layer_id}]"
        )
        updated_module = eval(
            f"updated_model.transformer.h[{config.layer_id}]"
        )
    elif updated_model.config.model_type == 'llama':
        frozen_module = eval(
            f"frozen_model.model.layers[{config.layer_id}]"
        )
        updated_module = eval(
            f"updated_model.model.layers[{config.layer_id}]"
        )
    else:
        raise ValueError(f"Model type not supported {updated_model.config.model_type}, supported models are 'gptj' and 'llama'")

    random_vector = torch.rand(1,1, updated_model.config.hidden_size, dtype=updated_model.dtype, device=updated_model.device)
    control_vec = random_vector / torch.norm(random_vector) * config.steering_coeff

    num_batches = min(
        config.max_num_batches,
        min([len(f) for f in forget_data_list]),
        min([len(r) for r in retain_data_list]),
    )

    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side="right"

    for epoch in range(config.num_epochs):
        print(f"======= Epoch {epoch} =======")
        with tqdm.tqdm(total=num_batches) as pbar:
            for idx in range(num_batches):
                batch_idx = idx // len(forget_data_list)
                unlearn_batch = forget_data_list[batch_idx]
                retain_batch = retain_data_list[batch_idx]

                # Unlearning loss
                unlearn_inputs = tokenizer(
                    unlearn_batch, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(updated_model.device)
                updated_forget_activations = forward_with_cache(
                    updated_model, unlearn_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)

                # we want to unlearn the last token only
                updated_forget_activations_last_tok = updated_forget_activations[..., -1:, :].contiguous()
                unlearn_loss = torch.nn.functional.mse_loss(
                    updated_forget_activations_last_tok, control_vec
                )

                # Retain loss
                retain_inputs = tokenizer(
                    retain_batch, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(updated_model.device)
                updated_retain_activations = forward_with_cache(
                    updated_model, retain_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)
                frozen_retain_activations = forward_with_cache(
                    frozen_model, retain_inputs, module=frozen_module, no_grad=True
                ).to(updated_model.device)

                # we want to retain the activations of all tokens
                retain_loss = torch.nn.functional.mse_loss(
                    updated_retain_activations, frozen_retain_activations
                )
                retain_loss *= config.alpha

                # Update model
                loss = unlearn_loss + retain_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"loss: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g} | retain_loss: {retain_loss.item():.4g} | param_change: {params[0].grad.abs().mean().item():.4g}")

                # ======= Logging ======
                if config.verbose:
                    frozen_forget_activations = forward_with_cache(frozen_model, unlearn_inputs, module=frozen_module, no_grad=True).to(updated_model.device)
                    unlearn_cosine= torch.nn.functional.cosine_similarity(updated_forget_activations, frozen_forget_activations, dim=-1).mean()
                    retain_cosine = torch.nn.functional.cosine_similarity(updated_retain_activations, frozen_retain_activations, dim=-1).mean()
                    print(f"unlearn_cosine_sim={unlearn_cosine.item()}")
                    print(f"retain_cosine_sim={retain_cosine.item()}")
                    print(f"updated_forget_activations.norm=",torch.mean(updated_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                    print(f"frozen_forget_activations.norm=",torch.mean(frozen_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                    print(f"updated_retain_activations.norm=",torch.mean(updated_retain_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                    print(f"frozen_retain_activations.norm=",torch.mean(frozen_retain_activations.norm(dim=-1).mean(dim=1), dim=0).item())

                pbar.update(1)

    tokenizer.truncation_side = truncation_side
