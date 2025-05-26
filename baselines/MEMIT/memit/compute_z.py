from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.MEMIT.rome import repr_tools
from baselines.MEMIT.util import nethook

from .memit_hparams import MEMITHyperParams


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
    mode: str = "delete",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """
    if mode not in ["both", "delete", "obscure"]:
        raise ValueError(f"mode={mode} not one of the following: both, delete, obscure")

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt", add_special_tokens=False).to("cuda")["input_ids"][0]

    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts= [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ]

    kl_prompts = ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
    elif hasattr(model.config, 'hidden_size'):
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device="cuda")
    else:
        raise NotImplementedError

    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                try:
                    cur_out[0][i, idx, :] += delta
                except Exception as e:
                    print(f"Error occurred at index {i} with idx {idx}: {e}")

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][:len(rewriting_prompts)]

        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w + lm_b, dim=2)

        try:
            mode = request["mode"]
            assert mode in ["delete", "insert", "replace"], "not a valid option"
        except KeyError or AssertionError:
            mode = "insert"

        if mode == "delete":
            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
            ).squeeze(2)
            mask = (rewriting_targets != -100).float()

            nll_loss_each = +(loss * mask).sum(1) / target_ids.size(0)
            nll_loss = nll_loss_each.mean()

            kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
                kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
            )
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init) ** 2
            )
            pred_prob = torch.exp(nll_loss)
            loss_pred_prob = hparams.loss_pred_prob_coef*pred_prob
            loss = loss_pred_prob + kl_loss + weight_decay
            print(
                f"loss {loss.item():.3e} = {loss_pred_prob.item():.3e} + {kl_loss.item():.3e} + {weight_decay.item():.3e} "
                f"avg prob of [{request['target_new']['str']}] "
                f"{torch.exp(nll_loss_each).mean().item():.3e}"
            )
            if loss < hparams.loss_break:
                break

        elif mode == "insert":
            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
            ).squeeze(2)
            mask = (rewriting_targets != -100).float()

            nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
            nll_loss = nll_loss_each.mean()
            kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
                kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
            )
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init) ** 2
            )
            loss = nll_loss + kl_loss + weight_decay
            print(
                f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                f"avg prob of [{request['target_new']['str']}] "
                f"{torch.exp(-nll_loss_each).mean().item()}"
            )
            if loss < 5e-2:
                break

        elif mode == "replace":
            # Compute rewriting targets
            rewriting_targets = torch.tensor(-100, device="cuda").repeat(
                len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
            )
            for i in range(len(rewriting_prompts)):
                ex_len = input_tok["attention_mask"][i].sum()
                rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

            target_true_ids = tok(request["target_true"]["str"], return_tensors="pt", add_special_tokens=False).to("cuda")["input_ids"][0]
            rewriting_targets_true = torch.tensor(-100, device="cuda").repeat(
                len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
            )
            for i in range(len(rewriting_prompts)):
                ex_len = input_tok["attention_mask"][i].sum()
                rewriting_targets_true[i, ex_len - len(target_true_ids) : ex_len] = target_true_ids

            target_log_probs = torch.gather(log_probs, 2, torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2)).squeeze(2)
            target_true_log_probs = torch.gather(log_probs, 2, torch.where(rewriting_targets_true != -100, rewriting_targets_true, 0).unsqueeze(2)).squeeze(2)
            mask = (rewriting_targets != -100).float()
            target_log_probs = (target_log_probs * mask).sum(1)
            target_true_log_probs = (target_true_log_probs * mask).sum(1)

            if it == 0:  # record initial values
                probs_at_start = {"target_log_probs": target_log_probs.detach().clone(), "target_true_log_probs": target_true_log_probs.detach().clone()}

            target_true_str = request["target_true"]["str"]
            target_new_str = request["target_new"]["str"]
            print(f"\t\tTarget True \'{target_true_str}\' prob: {torch.exp(target_true_log_probs.sum().mean()).item()}")
            print(f"\t\tTarget New \'{target_new_str}\' prob: {torch.exp(target_log_probs.sum().mean()).item()}")
            loss_true = torch.abs(target_true_log_probs - (probs_at_start["target_log_probs"]+1)).mean()
            loss_new = torch.abs(target_log_probs - probs_at_start["target_true_log_probs"]).mean()

            nll_loss_each =  +(loss_new + loss_true) / target_ids.size(0)
            nll_loss = nll_loss_each.mean()
            kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
                kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
            )
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init) ** 2
            )
            loss = nll_loss + kl_loss + weight_decay
            print(
                f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                f"avg prob of [{request['target_new']['str']}] "
                f"{torch.exp(-nll_loss_each).mean().item()}"
            )
            if loss < 5e-2:
                break

        if it == hparams.v_num_grad_steps - 1:
                    break
        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
    track=None,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        if track == 'out' or track == 'in':
            return repr_tools.get_reprs_at_word_tokens(
                track=track, subtoken=subtoken, **context_info, **word_repr_args
            )
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        if track == 'out' or track == 'in':
            return repr_tools.get_reprs_at_word_tokens(
                track=track, subtoken=subtoken, **context_info, **word_repr_args
            )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last" or fact_token_strategy == "subject_last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence, add_special_tokens=False)["input_ids"][ret]),
        )

    return ret
