from typing import List
from collections import defaultdict
import numpy as np
import torch

from revs.revs import REVSScore
from utils.activations_collector import collect_activations_with_prompt
from utils.hidden_state_ops import hs_to_logits
from utils.data import create_concat_prompts_target
from utils.globals import device


def calculate_stats(values):
    # Calculate the mean, median, standard deviation, minimum, and maximum of the given values
    mean = round(np.mean(values), 3)
    median = round(np.median(values), 3)
    min_ = round(np.min(values), 3)
    max_ = round(np.max(values), 3)
    std = round(np.std(values), 3)
    binary = round(np.mean([0 if v < 1 else 1 for v in values]), 3)  # percentage of values that are lower than 1
    count = len(values)
    # Return the statistics as a dictionary
    return {'mean': mean, 'median': median, 'min': min_, 'max':max_, 'std': std, 'count': count, 'binary': binary}


def calculate_edit_score_statistics_squared(rank_edit_scores: List[REVSScore], threshold=100):
    """
    Calculates statistics for edit scores within a specified threshold across different layers and sub-modules.

    This function processes a list of REVSScore objects, each representing edit scores for different
    layers and sub-modules within a neural network model. It categorizes these scores into three categories based on
    their relation to a threshold: in_range, top_distance, and bottom_distance. For each category, it aggregates scores
    by layer and sub-module, then computes statistical measures for these scores.

    Parameters:
    - rank_edit_scores (List[REVSScore]): A list of REVSScore objects, each containing
      edit scores for different layers and sub-modules.
    - threshold (int, optional): The threshold value used to categorize edit scores. Defaults to 100.

    Returns:
    - dict: A dictionary containing the statistics of edit scores categorized into in_range, top_distance, and
      bottom_distance. Each category contains a nested dictionary structure where keys are layer names, and values
      are dictionaries with sub-module names as keys and statistical measures as values.

    The statistical measures are calculated by the `calculate_stats` function (not shown in this snippet), which
    typically includes measures such as mean, median, standard deviation, etc.

    Example of return structure:
    {
        "range_score": {
            "layer1": {
                "sub_module1": { "mean": 0.5, "median": 0.5, ... },
                "sub_module2": { "mean": 0.7, "median": 0.6, ... },
                ...
            },
            ...
        },
        "top_distance_score": { ... },
        "bottom_distance_score": { ... }
    }
    """
    edit_in_range_score_dict = defaultdict(lambda: defaultdict(list))
    edit_rank_top_distance_score_dict = defaultdict(lambda: defaultdict(list))
    edit_rank_bottom_distance_score_dict = defaultdict(lambda: defaultdict(list))

    # Loop over each RankEditScore object
    for res in rank_edit_scores:
        scores = res.get_edit_scores(threshold)
        # Loop over each layer and sub-module
        for layer, sub_modules in scores['in_range'].items():
            for sub_module, score in sub_modules.items():
                edit_in_range_score_dict[layer][sub_module].append(score)
        for layer, sub_modules in scores['top_distance'].items():
            for sub_module, score in sub_modules.items():
                edit_rank_top_distance_score_dict[layer][sub_module].append(score)
        for layer, sub_modules in scores['bottom_distance'].items():
            for sub_module, score in sub_modules.items():
                edit_rank_bottom_distance_score_dict[layer][sub_module].append(score)

    # Initialize dictionaries to store the statistics for each layer and sub-module
    edit_in_range_score_stats = defaultdict(dict)
    edit_rank_top_distance_score_stats = defaultdict(dict)
    edit_rank_bottom_distance_score_stats = defaultdict(dict)

    # Loop over each layer and sub-module
    for layer, sub_modules in edit_in_range_score_dict.items():
        for sub_module, scores in sub_modules.items():
            edit_in_range_score_stats[layer][sub_module] = calculate_stats(scores)
    for layer, sub_modules in edit_rank_top_distance_score_dict.items():
        for sub_module, scores in sub_modules.items():
            edit_rank_top_distance_score_stats[layer][sub_module] = calculate_stats(scores)
    for layer, sub_modules in edit_rank_bottom_distance_score_dict.items():
        for sub_module, scores in sub_modules.items():
            edit_rank_bottom_distance_score_stats[layer][sub_module] = calculate_stats(scores)

    # create a dictionary of the statistics
    edit_score_stats = {
        "range_score": edit_in_range_score_stats,
        "top_distance_score": edit_rank_top_distance_score_stats,
        "bottom_distance_score": edit_rank_bottom_distance_score_stats
    }
    return edit_score_stats


def calculate_across_layers_score(edit_score_stats, stat_type='mean'):
    """
    Aggregates and calculates statistics for edit scores across different layers and sub-modules.

    This function processes a nested dictionary of edit scores, aggregating them by sub-module and then calculating
    specified statistical measures (e.g., mean, median, std, minimum) for each aggregation. The aggregation is based
    on a specified statistical type (e.g., 'mean') for each type of score (e.g., 'range', 'top', 'bottom').

    Parameters:
    - edit_score_stats (dict): A nested dictionary containing edit scores. The structure is
      {score_type: {layer: {sub_module: {stat_type: value, ...}, ...}, ...}, ...}.
    - stat_type (str, optional): The type of statistic to aggregate and calculate for each sub-module across all
      layers and score types. Defaults to 'mean'.

    Returns:
    - dict: A dictionary where keys are sub-module names and values are dictionaries. Each inner dictionary contains
      aggregated statistics for the sub-module across all layers and score types, with keys formatted as
      "{score_type}_{stat_type}" and values being the result of the `calculate_stats` function applied to the
      aggregated values.

    The `calculate_stats` function (not shown in this snippet) is assumed to calculate various statistical measures
    such as mean, median, standard deviation, and minimum.

    Example of return structure:
    {
        "sub_module1": {
            "range_mean": 0.5,
            "top_mean": 0.7,
            "bottom_mean": 0.6,
            ...
        },
        ...
    }
    """
    score_stats = defaultdict(lambda: defaultdict(list))

    # Loop over each type of score (range, top, bottom)
    for score_type, layers in edit_score_stats.items():
        # Loop over each layer and sub-module
        for layer, sub_modules in layers.items():
            # Loop over each sub-module
            for sub_module, stats in sub_modules.items():
                score_stats[sub_module][f"{score_type}_{stat_type}"].append(stats[stat_type])

    # Calculate the mean, median, std, minimum, of the given values
    for sub_module, stats in score_stats.items():
        for stat_type, values in stats.items():
            score_stats[sub_module][stat_type] = calculate_stats(values)

    return dict(score_stats)


def calculate_harmonic_mean(values):
    # Calculate the harmonic mean of the given values
    return round(len(values) / sum(1 / (v + 1e-6) for v in values), 3)


class EfficacyScore:

    def __init__(self, rank_edit_score: REVSScore):
        self.rank_edit_score = rank_edit_score
    
    def get_edit_scores(self, threshold):
        '''
        calculate the score of unlearning a sequence target by getting the max score for each target token in LAST layer and sub-module
        as a measurement for the difficulty of extracting the target token from the model last layer (generation)
        '''
        return self.rank_edit_score.get_edit_scores(threshold, last_layer_only=True)


class DeltaAttackScore:
    def __init__(self, model, tokenizer, prompt, target, skip_tokens=None, stop_tokens=None, max_tokens=None, method="both"):
        self.prompt = prompt
        self.target = target
        self.topk = 500  # use high threshold then can get score of any lower threshold
        self.delta_attack_ranks = {}
        self.method = method

        if method not in ["both", "decrease"]:
            raise ValueError("Method must be either 'both' or 'decrease'")

        # Create concatenated prompts and targets
        concat_prompts, concat_targets = create_concat_prompts_target(tokenizer, prompt, target, skip_tokens=skip_tokens, stop_tokens=stop_tokens, max_tokens=max_tokens)

        for concat_prompt, concat_target in zip(concat_prompts, concat_targets):
            collected_acts = collect_activations_with_prompt(model, tokenizer, concat_prompt)
            self.delta_attack_ranks[concat_target] = self.calculate_delta_attack(model, tokenizer, collected_acts, concat_target)

    def get_delta_attack_score(self, threshold):
        """
        Calculate the mean, min, median, max, std, count, and binary delta_attack score of all tokens
        """
        # Initialize a list to store the delta_attack scores
        scores_list = []

        # Loop over each token
        for target, ranks in self.delta_attack_ranks.items():
            # Calculate the delta_attack score of the token
            score = [min(rank/threshold, 1) for layer in ranks for rank in ranks[layer].values()]
            # Append the score to the list
            scores_list.extend(score)

        # Use calculate_stats to calculate the statistics of the scores
        stats = calculate_stats(scores_list)

        # Return the statistics
        return stats

    def calculate_delta_attack(self, model, tokenizer, collected_acts, target):
        """
        The Delta attack finds the top_k tokens that have the highest logits {DECREASE} U {INCREASE} (i.e. rank increase, depromting) between the each consecutive layer.
        """
        # Define a dictionary to map sub module names
        sub_module_names_dict = {
            'residual': 'residual_after',
            'attn': 'attention_output',
            'mlp': 'mlp_output'
        }

        # Initialize delta_attack scores dictionary
        delta_attack_ranks = defaultdict(lambda: defaultdict(dict))

        # Set the token index to the last token in the prompt
        token_index = -1

        # Encode the target token
        target_token_id = tokenizer.encode(target, add_special_tokens=False)[0]

        # Iterate over each layer in the collected activations
        for layer in range(len(collected_acts) - 1):
            for sub_module in collected_acts[layer]:
                if sub_module not in sub_module_names_dict.keys():
                    continue

                # Get the output of the current and next layer
                hs_before = collected_acts[layer][sub_module]['output'][token_index].to(device)
                hs_after = collected_acts[layer + 1][sub_module]['output'][token_index].to(device)

                # Calculate the change in logits between the two layers
                logits_changes = self.get_logits_change(model, hs_before, hs_after)
                if self.method == "both":
                    # both means decrease and increase, looking for the topk tokens that have the highest logits decrease or increase between consecutive layers
                    logits_changes_values, logits_changes_indices = torch.topk(torch.abs(logits_changes), self.topk, largest=True)

                if self.method == "decrease":
                    # decrease means only looking for the topk tokens that have the highest logits decrease between consecutive layers
                    logits_changes_values, logits_changes_indices = torch.topk(-logits_changes, self.topk, largest=True)

                # Calculate the delta_attack score
                # if the target token is in the topk tokens that have the highest logits decrease, the score is it's rank in the topk tokens divided by the threshold, 1 otherwise
                if target_token_id in logits_changes_indices:
                    rank_in_topk = (logits_changes_indices == target_token_id).nonzero(as_tuple=True)[0].item()
                    rank = rank_in_topk
                else:
                    rank = 1e6  # for every token that is not in the topk, set the rank to a high value
                # Add the score to the delta_attack_scores dictionary
                delta_attack_ranks[layer][sub_module_names_dict[sub_module]] = rank
        return delta_attack_ranks

    def get_logits_change(self, model, hs_before, hs_after):
        """
        This function calculates the change in logits between two layers.
        """
        # Project the activations to the vocabulary space and get the logits
        logits_before = hs_to_logits(model, hs_before)
        logits_after = hs_to_logits(model, hs_after)

        # Calculate the difference in the ranks for each token
        logits_diff =  logits_after - logits_before
        return logits_diff


class PerturbAttackScore:

    def __init__(self, model, tokenizer, prompt, target, seed, skip_tokens=None, stop_tokens=None, max_tokens=None, num_perturbations=10):
        self.prompt = prompt
        self.perturbed_prompt = PerturbAttackScore.perturb_prompt(prompt, seed=seed, num_perturbations=num_perturbations, add_space_last=True)
        self.target = target
        self.num_perturbations = num_perturbations
        self.rank_editor_squared_score = REVSScore(
            model, 
            tokenizer, 
            self.perturbed_prompt, 
            target, 
            skip_tokens=skip_tokens, 
            stop_tokens=stop_tokens, 
            max_tokens=max_tokens
        )

    def get_edit_scores(self, threshold):
        return self.rank_editor_squared_score.get_edit_scores(threshold)

    @staticmethod
    def perturb_prompt(prompt: str, seed: int, num_perturbations: int, add_space_last: bool=True) -> str:
        """
        This function perturbs the given prompt by inserting a random character at a random position.
        The number of perturbations and the randomness are controlled by the provided arguments.

        Args:
            prompt (str): The prompt to be perturbed.
            num_perturbations (int): The number of random characters to insert into the prompt.
            seed (int): The seed for the random number generator, used to ensure reproducibility.

        Returns:
            str: The perturbed prompt.
        """
        np.random.seed(seed)
        # Include non-alphabetic characters
        perturbing_chars = ' '
        for _ in range(num_perturbations):
            position = np.random.randint(0, len(prompt))
            char = np.random.choice(list(perturbing_chars))
            prompt = prompt[:position] + char + prompt[position:]
        if add_space_last:
            prompt = prompt + ' '
        return prompt


class LogitLensAttackScore:
    '''
    Calculate the score of Logit Lens attack by getting the max score for each target token in EACH layer and sub-module
    '''
    def __init__(self, rank_editor_squared_score: REVSScore):
        self.rank_editor_squared_score = rank_editor_squared_score

    def get_edit_scores(self, threshold):
        return self.rank_editor_squared_score.get_edit_scores(threshold, last_layer_only=False)