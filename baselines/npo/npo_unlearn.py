from typing import Literal, Optional
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils.configs import BaseConfig


class NPOConfig(BaseConfig):
    """NPO-specific configuration that extends BaseConfig.

    Additional Args:
        loss_type (str): Type of loss function to use, one of ['npo', 'npo_kl', 'npo_retain']
        num_epochs (int): Number of training epochs
        npo_coeff (float): NPO loss coefficient
        retain_coeff (float): Retention loss coefficient for NPO Retain loss
        kl_coeff (float): KL divergence coefficient for NPO KL loss
        beta (float): Beta value for NPO loss
        lr (float): Learning rate
        min_len (int): Minimum sequence length
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
                 loss_type: Literal['npo', 'npo_kl', 'npo_retain'],
                 token_method: Literal['rarest', 'first', 'frequent', 'random'] = 'rarest',
                 perplexity: bool = False,
                 save_model: bool = False,
                 log_wandb: bool = True,
                 unlearn_num_examples: Optional[int] = None,
                 not_unlearn: bool = False,

                 num_epochs: int = 1,
                 npo_coeff: float = 1.0,
                 retain_coeff: float = 1.0,
                 kl_coeff: float = 1.0,
                 beta: float = 1.0,
                 lr: float = 5e-5,
                 min_len: int = 20,
                 verbose: bool = True):

        super().__init__(
            method_name="NPO",
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

        self.loss_type = loss_type
        self.npo_coeff = npo_coeff
        self.retain_coeff = retain_coeff
        self.kl_coeff = kl_coeff
        self.beta = beta
        self.lr = lr
        self.min_len = min_len
        self.batch_size = 1
        self.num_epochs = num_epochs
        self.verbose = verbose

    def to_dict(self):
        base_dict = super().to_dict()
        npo_dict = {
            'loss_type': self.loss_type,
            'npo_coeff': self.npo_coeff,
            'retain_coeff': self.retain_coeff,
            'kl_coeff': self.kl_coeff,
            'beta': self.beta,
            'lr': self.lr,
            'min_len': self.min_len,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'verbose': self.verbose
        }
        return {**base_dict, **npo_dict}


def run_npo(
    unlearn_model,
    ref_model,
    tokenizer,
    unlearn_prompts,
    unlearn_targets,
    retain_prompts,
    config: NPOConfig,
):
    if config.verbose:
        print("====NPO Config====")
        print("\n".join(f"{k}={v}" for k,v in config.__dict__.items()))
        print("=====")

    unlearn_model = unlearn_model.train()
    ref_model = ref_model.eval()
    optimizer = torch.optim.AdamW(unlearn_model.parameters(), lr=config.lr)

    def compute_loss_fn(unlearn_model, ref_model, unlearn_inputs, unlearn_targets_inputs, retain_inputs):

        def npo_loss(unlearn_logits, ref_logits, unlearn_labels):

            loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
            # Select the logits corresponding to the last token in each sequence
            logits_last_tok = unlearn_logits[..., -1:, :].contiguous()
            unlearn_loss_current = loss_function(logits_last_tok.transpose(-1, -2), unlearn_labels)
            with torch.no_grad():
                # Select the logits corresponding to the last token in each sequence
                ref_logits_last_tok = ref_logits[..., -1:, :].contiguous()
                unlearn_loss_ref = loss_function(ref_logits_last_tok.transpose(-1, -2), unlearn_labels)
            neg_log_ratios = unlearn_loss_current - unlearn_loss_ref
            loss = -F.logsigmoid(config.beta * neg_log_ratios).mean() * 2 / config.beta
            return loss

        def kl_loss(unlearn_logits, retain_logits):

            with torch.no_grad():
                retain_ref_probs = F.log_softmax(retain_logits, dim=-1).view(-1, retain_logits.shape[-1])
            current_probs = F.log_softmax(unlearn_logits, dim=-1).view(-1, unlearn_logits.shape[-1])
            return F.kl_div(current_probs, retain_ref_probs, reduction='batchmean', log_target=True)

        # Forward passes
        unlearn_outputs = unlearn_model(**unlearn_inputs)
        unlearn_logits=unlearn_outputs.logits
        unlearn_labels=unlearn_targets_inputs["input_ids"]

        with torch.no_grad():
            ref_outputs = ref_model(**unlearn_inputs)
        ref_logits=ref_outputs.logits

        if config.loss_type == "npo":
            loss = npo_loss(unlearn_logits, ref_logits, unlearn_labels)
            if config.verbose:
                print(f"Loss: {loss.item():.4g}")
            return loss

        elif config.loss_type == "npo_kl":
            unlearn_loss = npo_loss(unlearn_logits, ref_logits, unlearn_labels)

            # calc kl divergence
            with torch.no_grad():
                retain_outputs = ref_model(**retain_inputs)
            retain_log_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_log_probs = retain_log_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = unlearn_model(**retain_inputs)
            current_log_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_log_probs = current_log_probs.view(-1, current_outputs.logits.shape[-1])

            # minimum KL divergence
            retain_loss = nn.functional.kl_div(current_log_probs, retain_log_probs, reduction='batchmean', log_target=True)
            loss = config.npo_coeff * unlearn_loss + config.kl_coeff * retain_loss
            if config.verbose:
                print(f"Loss: {loss.item():.4g}, NPO Loss: {unlearn_loss.item():.4g}, KL Loss: {retain_loss.item():.4g}")
            return loss

        elif config.loss_type == "npo_retain":
            unlearn_loss = npo_loss(unlearn_logits, ref_logits, unlearn_labels)

            # compute retain loss
            retain_outputs = unlearn_model(**retain_inputs)
            input_ids = retain_inputs["input_ids"]

            # Shift labels right by 1 position
            shifted_labels = input_ids[..., 1:].contiguous()
            logits = retain_outputs.logits[..., :-1, :].contiguous()

            loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
            retain_loss = loss_function(logits, shifted_labels)
            loss = config.npo_coeff * unlearn_loss + config.retain_coeff * retain_loss

            if config.verbose:
                print(f"Loss: {loss.item():.4g}, NPO Loss: {unlearn_loss.item():.4g}, Retain Loss: {retain_loss.item():.4g}")
            return loss

        else:
            raise ValueError(f"Unsupported loss type: {config.loss_type}")

    num_examples = len(unlearn_prompts)

    for epoch in range(config.num_epochs):
        print(f"======= Epoch {epoch} =======")

        with tqdm(total=num_examples) as pbar:
            for idx in range(num_examples):
                # Get single examples
                unlearn_prompt = unlearn_prompts[idx]
                unlearn_target = unlearn_targets[idx]
                retain_prompt = retain_prompts[idx]

                # Process single unlearn example
                unlearn_inputs = tokenizer(
                    unlearn_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=False
                ).to(unlearn_model.device)

                unlearn_targets_inputs = tokenizer(
                    unlearn_target,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=False
                ).to(unlearn_model.device)

                # Process single retain example
                retain_inputs = tokenizer(
                    retain_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=False
                ).to(unlearn_model.device)

                # Calculate loss for single example
                loss = compute_loss_fn(
                    unlearn_model=unlearn_model,
                    ref_model=ref_model,
                    unlearn_inputs=unlearn_inputs,
                    unlearn_targets_inputs=unlearn_targets_inputs,
                    retain_inputs=retain_inputs
                )

                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(1)