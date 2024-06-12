import torch
from utils.globals import device

# ~~~~~~~~ Transforms ~~~~~~~~

@torch.no_grad()
def hs_to_logits(model, hs):
    '''
    hs: tensor of shape (num_neurons, embed_dim)
    '''
    if model.config.model_type == 'gptj':
        layer_norm = model.transformer.ln_f
        lm_head = model.lm_head
    elif model.config.model_type == 'llama':
        layer_norm = model.model.norm
        lm_head = model.lm_head
    else:
        raise ValueError(f"Model type not supported {model.config.model_type}, supported models are 'gptj' and 'llama'")

    logits = lm_head(layer_norm(hs))
    return logits

@torch.no_grad()
def hs_to_probs(model, hs):
    logits = hs_to_logits(model, hs)
    probs = torch.softmax(logits, dim=-1)
    return probs

@torch.no_grad()
def invert_lm_head(logits, lm_head_weight, lm_head_bias, lm_head_weight_inv=None):
    if lm_head_weight_inv is None:
        lm_head_weight_inv = torch.pinverse(lm_head_weight)
    hs = (logits - lm_head_bias) @ lm_head_weight_inv.T
    return hs

@torch.no_grad()
def invert_gptj_layer_norm(normed_hs, mean, var, layer_norm):
    # Extract the scale (gamma) and shift (beta) parameters from the layer norm module
    gamma = layer_norm.weight
    beta = layer_norm.bias
    eps = layer_norm.eps
    # Compute the standard deviation
    std = torch.sqrt(var + eps)
    # Invert the layer normalization operation
    x = ((normed_hs - beta) / gamma) * std.unsqueeze(1) + mean.unsqueeze(1)
    return x

@torch.no_grad()
def invert_llama_layer_norm(normed_hs, mean, var, rmsnorm):
    # Extract the scale (gamma) parameter from the RMSNorm module
    gamma = rmsnorm.weight
    eps = rmsnorm.variance_epsilon
    # Compute the root mean square
    rms = torch.sqrt(var + eps)
    # Invert the RMSNorm operation
    x = (normed_hs  * rms.unsqueeze(1)) / gamma + mean.unsqueeze(1)
    return x

@torch.no_grad()
def logits_to_hs(model, logits, original_mean, original_var, lm_head_weight_inv=None):
    if model.config.model_type == 'gptj':
        layer_norm = model.transformer.ln_f
        lm_head_weight = model.lm_head.weight
        lm_head_bias = model.lm_head.bias
        invert_layer_norm = invert_gptj_layer_norm
    elif model.config.model_type == 'llama':
        layer_norm = model.model.norm
        lm_head_weight = model.lm_head.weight
        lm_head_bias = torch.zeros_like(lm_head_weight.T[0])  # llama doesn't have a bias
        invert_layer_norm = invert_llama_layer_norm
    else:
        raise ValueError(f"Model type not supported {model.config.model_type}, supported models are 'gptj' and 'llama'")

    normed_hs = invert_lm_head(logits, lm_head_weight=lm_head_weight, lm_head_bias=lm_head_bias, lm_head_weight_inv=lm_head_weight_inv)
    hs = invert_layer_norm(normed_hs, original_mean, original_var, layer_norm)
    return hs

@torch.no_grad()
def test_logits_to_hs(model):
    if model.config.model_type == 'gptj':
        hs = model.transformer.h[0].mlp.fc_out.weight[:, :10].detach().clone().to(device).T
    elif model.config.model_type == 'llama':
        hs = model.model.layers[0].mlp.down_proj.weight[:, :10].detach().clone().to(device).T
    else:
        raise ValueError(f"Model type not supported {model.config.model_type}, supported models are 'gptj' and 'llama'")

    mean, var = hs.mean(dim=-1), hs.var(dim=-1)
    logits = hs_to_logits(model, hs)
    hs_restored = logits_to_hs(model, logits, mean, var)
    result = torch.allclose(hs, hs_restored, atol=1e-3, rtol=1e-3)

    if result:
        print("\n\t~~~~~~~~~~~ Test PASSED ~~~~~~~~~~~")
    else:
        print("\n\t~~~~~~~~~~~ Test FAILED ~~~~~~~~~~~")
    return hs, hs_restored


# ~~~~~~~~ Token Ranks ~~~~~~~~

def get_token_rank_in_hs_sublayers(model, tokenizer, collected_acts, target:str, layer:int, token_index:int=-1):
    '''
    return the token rank in the vocab distibution of the target token in the hidden state of each sublayer
    token_index: the index of the token in the prompt, -1 means the last token in the prompt, meaning the hs after the last token in the prompt
    '''
    target_token_id = tokenizer.encode(target, add_special_tokens=False)[0]
    token_ranks_dict = {"target_token": tokenizer.decode([target_token_id])}
    hs_residual_after = collected_acts[layer]['residual']['output'][token_index].to(device)
    hs_attention = collected_acts[layer]['attn']['output'][token_index].to(device)
    hs_mlp_output = collected_acts[layer]['mlp']['output'][token_index].to(device)

    residual_after = hs_to_logits(model, hs_residual_after)
    attention = hs_to_logits(model, hs_attention)
    mlp_output = hs_to_logits(model, hs_mlp_output)
    token_ranks_dict = {
        "residual_after": get_rank_of_token_in_vocab(residual_after, target_token_id).item(),
        "attention_output": get_rank_of_token_in_vocab(attention, target_token_id).item(),
        "mlp_output": get_rank_of_token_in_vocab(mlp_output, target_token_id).item(),
    }
    return token_ranks_dict

def get_token_rank_across_layers(model, tokenizer, collected_acts, target, token_index=-1):
    '''
    return the token rank in the vocab distibution of the target token in the hidden state of each sublayer
    token_index: the index of the token in the prompt, -1 means the last token in the prompt, meaning the hs after the last token in the prompt
    '''
    layers_token_ranks_dict = {}

    if model.config.model_type == 'gptj':
        n_layers = model.config.n_layer
    elif model.config.model_type == 'llama':
        n_layers = model.config.num_hidden_layers
    else:
        raise ValueError(f"Model type not supported {model.config.model_type}, supported models are 'gptj' and 'llama'")

    for layer in range(n_layers):
        layers_token_ranks_dict[layer] = get_token_rank_in_hs_sublayers(model, tokenizer, collected_acts, target, layer, token_index=token_index)
    return layers_token_ranks_dict

def get_topk_token_ranks_in_vocab(vocab_distibution, tokenizer, top_k, largest=True):
    '''
    return the top k tokens ranks in the vocab distibution and their scores and tokens strings
    '''
    # if vocab_distibution is 2 dimentional, get the rank of each token in the vocab distibution
    if len(vocab_distibution.shape) == 2:
        topk_token_ranks = torch.topk(vocab_distibution, top_k, largest=largest).indices
    elif len(vocab_distibution.shape) == 1:
        topk_token_ranks = torch.topk(vocab_distibution, top_k, largest=largest).indices
    else:
        raise ValueError("vocab_distibution should be 1 or 2 dimentional")
    # get the top k tokens scores
    topk_token_scores = vocab_distibution[topk_token_ranks]
    # get the top k tokens
    topk_tokens = [tokenizer.decode(token_rank) for token_rank in topk_token_ranks]
    return topk_token_ranks, topk_token_scores, topk_tokens

@torch.no_grad()
def get_rank_of_token_in_vocab(vocab_distibution, target_token_id) -> int:
    '''
    return the rank of the target token id in the vocab distibution
    '''
    # if vocab_distibution is 2 dimentional, get the rank of each token in the vocab distibution
    if len(vocab_distibution.shape) == 2:
        target_token_rank = (vocab_distibution > vocab_distibution[:, target_token_id].unsqueeze(1)).sum(dim=1)
    elif len(vocab_distibution.shape) == 1:
        target_token_rank = (vocab_distibution > vocab_distibution[target_token_id]).sum()
    else:
        raise ValueError("vocab_distibution should be 1 or 2 dimentional")
    return target_token_rank
