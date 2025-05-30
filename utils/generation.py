import unicodedata
from typing import List
from tqdm.notebook import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



def generate_from_prompt(model, tok, prompt, max_length=20):
    return generate_fast(model, tok, [prompt], max_out_len=max_length, top_k=1)


def generate_from_prompts(model, tok, prompts, max_length=20, chunk_size=20, max_prompt_len=None):
    results = []
    for i in tqdm(range(0, len(prompts), chunk_size), desc="Generating sentences from prompts", total=len(prompts)//chunk_size):
        chunk = prompts[i:i+chunk_size]
        if max_prompt_len is not None:
            chunk = [prompt[-max_prompt_len:] for prompt in chunk]
        results.extend(generate_fast(model, tok, chunk, max_length=max_length, top_k=1))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 1,
    max_length: int = 100,
):
    """
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.
    """

    # Unroll prompts and tokenize
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt", add_special_tokens=False).to(
        next(model.parameters()).device
    )
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    batch_size = input_ids.size(0)
    max_out_len = input_ids.size(1) + max_length  # generate max_length on top of input_ids length 

    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:  # while not exceeding max output length
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask= None if 'llama' in model.name_or_path.lower() else attention_mask[:, cur_context],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits, past_key_values = model_out.logits, model_out.past_key_values

            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            # Top-k sampling : 1. Select top-k tokens based on probability. 
            #2. The create a normalized probaility distribution for only those top-k tokens. 
            #3. Sample from those top-k token based on their normazlied probabilities
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)

    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("<|end_of_text|>", "")
        .replace('<|begin_of_text|>', '')#this is for llama
        .strip()
        for x in txt
    ]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return txt
