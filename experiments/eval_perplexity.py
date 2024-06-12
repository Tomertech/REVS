import torch
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
import wandb
from tqdm.notebook import tqdm
import plotly.io as pio
import datasets
from torch.utils.data import DataLoader
import argparse

# My utils:
from utils.generation import generate_from_prompt, generate_from_prompts
from utils.model import load_model_tokenizer, load_model_editor_tokenizer, load_model_memit_tokenizer
from revs.revs import REVSConfig, REVS
from utils.hidden_state_ops import hs_to_logits, logits_to_hs, invert_layer_norm, get_rank_of_token_in_vocab, get_token_rank_across_layers
from utils.activations_collector import ActivationCollector, collect_activations_with_prompt
from utils.plot import plot_token_rank_in_hs_across_sublayers, plot_edit_score_statistics
from utils.globals import device


def evaluate_perplexity(model, tokenizer, dataloader):
    model.eval()
    perplexities = []
    for batch in tqdm(dataloader):
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
        perplexities.append(perplexity.item())
    return perplexities


def main(model_name, model_dir):
    if model_name == "memit":
        assert model_dir is not None, "model_dir must be provided"
        model, tokenizer = load_model_memit_tokenizer(model_dir)
        wandb.init(project="delpii", name="perplexity MEMIT edited model")
    elif model_name == "revs":
        assert model_dir is not None, "model_dir must be provided"
        model_editor, tokenizer = load_model_editor_tokenizer(model_dir)
        model = model_editor.model
        wandb.init(project="delpii", name="perplexity REVS edited model")
    elif model_name == "original":
        model, tokenizer = load_model_tokenizer()
        wandb.init(project="delpii", name="perplexity ORIGINAL model")
    else:
        raise ValueError("model_name must be one of the following: 'memit', 'revs', 'original'")

    dataset = datasets.load_dataset("NeelNanda/wiki-10k")
    dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=1)
    perplexities = evaluate_perplexity(model, tokenizer, dataloader)

    mean_perplexity = np.mean(perplexities)

    
    wandb.log({"perplexities": perplexities})
    wandb.log({"mean_perplexity": mean_perplexity})
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model perplexity")
    parser.add_argument("--model_name", type=str, help="must be one of the following: 'memit', 'revs', 'original'")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to model directory")
    args = parser.parse_args()
    main(args.model_name, args.model_dir)
