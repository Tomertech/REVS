from collections import defaultdict
from tqdm.notebook import tqdm
import torch
from torch import Tensor

from utils.activations_collector import collect_activations_with_prompt
from utils.hidden_state_ops import hs_to_logits, logits_to_hs, get_rank_of_token_in_vocab, get_token_rank_across_layers
from utils.globals import device
from utils.data import create_concat_prompts_target
# from knowledge_neurons.knowledge_neurons import KnowledgeNeurons


class REVSConfig:
    def __init__(
        self, 
        n_neurons, 
        act_filter, 
        neurons_score_method, 
        score_threshold, 
        residual_bottom_rank_margin, 
        residual_top_rank_margin, 
        max_iter_mlp_rank, 
        mlp_bottom_rank_margin, 
        mlp_top_rank_margin, 
        max_iter_neuron_rank, 
        neuron_bottom_rank_margin, 
        neuron_top_rank_margin, 
        seed, 
        skip_tokens=None, 
        stop_tokens=None, 
        max_tokens=None,
        max_prompt_len=None, 
        insert_new_token=None, 
        restore_after_edit=False, 
        save_model=False, 
        zero_neurons=False, 
        token_method=None,
        unlearn_num_examples=None, 
        not_unlearn=False, 
        perplexity=False, 
        log_wandb=True
    ):
        """
        Initializes the REVSConfig class with various configuration settings for model rank editing.

        Parameters:
            n_neurons (int): Number of neurons to edit in each chosen layer.
            act_filter (str): Activation filter used when choosing neurons; options: "positive", "top_k", "no_filter".
            neurons_score_method (str): Method to score neurons; options: "rank", "act", "grad".
            score_threshold (int): Threshold for score; defines the range for rank consideration.
            residual_bottom_rank_margin (int): Bottom rank margin for the residual.
            residual_top_rank_margin (int): Top rank margin for the residual.
            max_iter_mlp_rank (int): Maximum iterations for MLP rank editing.
            mlp_bottom_rank_margin (int): Bottom rank margin for MLP.
            mlp_top_rank_margin (int): Top rank margin for MLP.
            max_iter_neuron_rank (int): Maximum iterations for neuron rank editing.
            neuron_bottom_rank_margin (int): Bottom rank margin for neurons.
            neuron_top_rank_margin (int): Top rank margin for neurons.
            seed (int): Seed for random number generation to ensure reproducibility.
            skip_tokens (list, optional): Tokens to skip during editing; defaults to None.
            stop_tokens (list, optional): Tokens to stop at during editing; defaults to None.
            max_tokens (int, optional): Maximum tokens to consider for editing; defaults to None.
            max_prompt_len (int, optional): Maximum prompt length; defaults to None.
            insert_new_token (str, optional): Token to insert during editing; defaults to None.
            restore_after_edit (bool, optional): Whether to restore model state post-edit; defaults to False.
            save_model (bool, optional): Whether to save the model post-edit; defaults to False.
            zero_neurons (bool, optional): Whether to zero neurons as edited value; defaults to False.
            token_method (str, optional): Method to select target token; defaults to None.
            unlearn_num_examples (int, optional): Number of examples for unlearning; defaults to None.
            not_unlearn (bool, optional): Whether to skip unlearning; defaults to False.
            perplexity (bool, optional): Whether to consider perplexity; defaults to False.
            log_wandb (bool, optional): Whether to log with Weights & Biases; defaults to True.
        """
        self.n_neurons = n_neurons
        self.act_filter = act_filter
        self.neurons_score_method = neurons_score_method
        self.score_threshold = score_threshold

        # Rank margins and iteration limits
        self.residual_bottom_rank_margin = residual_bottom_rank_margin
        self.residual_top_rank_margin = residual_top_rank_margin
        self.max_iter_mlp_rank = max_iter_mlp_rank
        self.mlp_bottom_rank_margin = mlp_bottom_rank_margin
        self.mlp_top_rank_margin = mlp_top_rank_margin
        self.max_iter_neuron_rank = max_iter_neuron_rank
        self.neuron_bottom_rank_margin = neuron_bottom_rank_margin
        self.neuron_top_rank_margin = neuron_top_rank_margin

        # Optional parameters for editing process
        self.skip_tokens = skip_tokens
        self.stop_tokens = stop_tokens
        self.max_tokens = max_tokens
        self.max_prompt_len = max_prompt_len
        self.insert_new_token = insert_new_token
        self.restore_after_edit = restore_after_edit
        self.save_model = save_model
        self.zero_neurons = zero_neurons
        self.token_method = token_method

        # Experiment and logging settings
        self.seed = seed
        self.unlearn_num_examples = unlearn_num_examples
        self.not_unlearn = not_unlearn
        self.perplexity = perplexity
        self.log_wandb = log_wandb

    def to_dict(self):
        return {
            'n_neurons': self.n_neurons,
            'act_filter': self.act_filter,
            'neurons_score_method': self.neurons_score_method,
            'score_threshold': self.score_threshold,
            'residual_bottom_rank_margin': self.residual_bottom_rank_margin,
            'residual_top_rank_margin': self.residual_top_rank_margin,
            'max_iter_mlp_rank': self.max_iter_mlp_rank,
            'mlp_bottom_rank_margin': self.mlp_bottom_rank_margin,
            'mlp_top_rank_margin': self.mlp_top_rank_margin,
            'max_iter_neuron_rank': self.max_iter_neuron_rank,
            'neuron_bottom_rank_margin': self.neuron_bottom_rank_margin,
            'neuron_top_rank_margin': self.neuron_top_rank_margin,
            'seed': self.seed,
            'skip_tokens': self.skip_tokens,
            'stop_tokens': self.stop_tokens,
            'max_tokens': self.max_tokens,
            'max_prompt_len': self.max_prompt_len,
            'insert_new_token': self.insert_new_token,
            'restore_after_edit': self.restore_after_edit,
            'save_model': self.save_model,
            'zero_neurons': self.zero_neurons,
            'token_method': self.token_method,
            'unlearn_num_examples': self.unlearn_num_examples,
            'not_unlearn': self.not_unlearn,
            'perplexity': self.perplexity,
            'log_wandb': self.log_wandb
        }

    def __str__(self):
        config_dict = self.to_dict()
        config_lines = [f"\t{k}={v}" for k, v in config_dict.items()]
        return "REVSConfig:\n" + "\n".join(config_lines)


class REVS:

    def __init__(self, model, tokenizer, config: REVSConfig, pinv_lm_head):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.edit_dicts = [] # [defaultdict(dict), target] -> [{layer: {token_index: {neurons edits info}}}, target]
        self.pinv_lm_head = pinv_lm_head
        self.knowledge_neoruns = None

        if model.config.model_type == 'gptj':
            self.model_n_layers = model.config.n_layer
        elif model.config.model_type == 'llama':
            self.model_n_layers = model.config.num_hidden_layers
        else:
            raise ValueError(f"Model type not supported {model.config.model_type}, supported models are 'gptj' and 'llama'")

        # unpack config values
        self.n_neurons = config.n_neurons
        self.act_filter = config.act_filter
        self.neurons_score_method = config.neurons_score_method

    # edits
    @torch.no_grad()
    def edit_layer_rank_iter(self, collected_acts, layer, prompt, target_token, mlp_bottom_rank_margin: int, mlp_top_rank_margin: int, 
                             residual_bottom_rank_margin: int, residual_top_rank_margin: int, insert_new_token:str=None):
        '''
        edit a layer by iteratively adjusting the neuron values until the rank of the target token in the MLP output is within the specified margins.
        '''
        assert mlp_bottom_rank_margin <= mlp_top_rank_margin, "mlp_top_rank_margin should be greater than mlp_bottom_rank_margin"

        # collect the relevant activations
        collected_acts = self.get_relevant_acts(layer, collected_acts)
        residual_before = collected_acts['residual_before']
        residual_after = collected_acts['residual_after']
        fc_out = collected_acts['fc_out']
        fc_out_act = collected_acts['fc_out_act']
        attn_out = collected_acts['attn_out']

        assert torch.allclose(residual_after, residual_before + fc_out + attn_out, atol=5e-5, rtol=5e-5), "residual_before and residual_after are not equal"

        target_token_id = self.tokenizer.encode(target_token, add_special_tokens=False)[0]
        mlp_logits = hs_to_logits(self.model, fc_out)
        mlp_target_token_rank = get_rank_of_token_in_vocab(mlp_logits, target_token_id)
        residual_logits = hs_to_logits(self.model, residual_after)
        residual_target_token_rank = get_rank_of_token_in_vocab(residual_logits, target_token_id)

        # if the rank of the target token is above the residual bottom margin, return without edit
        if residual_target_token_rank >= residual_bottom_rank_margin:
            return
        # if the rank of the target token is above the top MLP margin, return without edit
        if mlp_target_token_rank >= mlp_top_rank_margin:
            return

        fc_out_weights, fc_out_bias = self.get_model_fc_weights(layer)

        # get the top k neurons by the scores of the target token rank
        selected_neurons_indices, selected_neurons_scores = \
            self.get_selected_neurons_indices_scores(
                neurons=fc_out_weights,
                activations_values=fc_out_act,
                token_id=target_token_id,
                score_top_k=self.n_neurons,
                act_filter=self.act_filter,
                layer=layer,
                prompt=prompt,
                target_token=target_token,
                method=self.neurons_score_method
            )
        # get the top k neurons as hidden states - shape (num_neurons, embed_dim)
        original_neurons_value = fc_out_weights[:, selected_neurons_indices].detach().clone().to(device)
        selected_neurons_value = original_neurons_value.T.detach().clone().to(device)
        max_neurons_to_edit = self.config.n_neurons
        num_neurons_to_edit = 5
        neurons_to_edit_already_tried = set()  # {num_neurons_to_edit} - to prevent from trying the same number of neurons to edit

        # edit increasing number of neurons until the rank of the target token in the MLP output is within the specified margins
        for _ in range(self.config.max_iter_mlp_rank):
            # start editing num_neurons_to_edit neurons from the top k neurons
            num_neurons_to_edit = min(num_neurons_to_edit, max_neurons_to_edit)
            current_selected_neurons_value = selected_neurons_value[:num_neurons_to_edit]
            current_selected_neurons_indices = selected_neurons_indices[:num_neurons_to_edit]
            # calculate the deltas for each neuron value
            neurons_deltas = self.calc_deltas_delete(current_selected_neurons_value, target_token_id=target_token_id, 
                bottom_rank_margin=self.config.neuron_bottom_rank_margin, top_rank_margin=self.config.neuron_top_rank_margin, 
                max_iter=self.config.max_iter_neuron_rank)

            # apply the deltas to the neurons in the mlp layer
            fc_out_weights[:, current_selected_neurons_indices] += neurons_deltas.T

            # calculate the new MLP output
            edited_mlp_output = fc_out_act @ fc_out_weights.T + fc_out_bias
            edited_residual_output = residual_before + edited_mlp_output + attn_out

            # restore the original MLP weights
            fc_out_weights[:, current_selected_neurons_indices] -= neurons_deltas.T

            # calculate the rank of the target token in the MLP output
            edited_mlp_logits = hs_to_logits(self.model, edited_mlp_output)
            mlp_target_token_rank = get_rank_of_token_in_vocab(edited_mlp_logits, target_token_id)
            edited_residual_logits = hs_to_logits(self.model, edited_residual_output)
            residual_target_token_rank = get_rank_of_token_in_vocab(edited_residual_logits, target_token_id)

            # Conditions to break the loop
            mlp_rank_above_bottom_margin = mlp_target_token_rank >= mlp_bottom_rank_margin
            mlp_rank_below_top_margin = mlp_target_token_rank <= mlp_top_rank_margin
            mlp_rank_within_margins = mlp_rank_above_bottom_margin and mlp_rank_below_top_margin

            residual_rank_above_bottom_margin = residual_target_token_rank >= residual_bottom_rank_margin
            residual_rank_below_top_margin = residual_target_token_rank <= residual_top_rank_margin
            residual_rank_within_margins = residual_rank_above_bottom_margin and residual_rank_below_top_margin

            if mlp_rank_within_margins:
                # best case scenario, both MLP and residual ranks are within the margins
                if residual_rank_within_margins:
                    break

                # MLP rank is ok but residual is still below the bottom margin
                elif residual_target_token_rank < residual_bottom_rank_margin:
                    # in this case we want to INCREASE the number of neurons to edit by small amount
                    num_neurons_to_edit = int(min(num_neurons_to_edit + 1, max_neurons_to_edit))

                # MLP rank is ok but residual is above the top margin
                elif residual_target_token_rank > residual_top_rank_margin:
                    # in this case we want to DECREASE the number of neurons to edit by small amount
                    num_neurons_to_edit = int(max(num_neurons_to_edit - 1, 1))


            # if the rank of the target token is below the MLP specified margins, INCREASE the number of neurons to edit
            elif mlp_target_token_rank < mlp_bottom_rank_margin:
                num_neurons_to_edit = int(min(num_neurons_to_edit * 1.6, max_neurons_to_edit))

            # if the rank of the target token is above the MLP specified margins, DECREASE the number of neurons to edit
            elif mlp_target_token_rank > mlp_top_rank_margin:
                num_neurons_to_edit = int(max(num_neurons_to_edit * 0.8, 2))

            # if the number of neurons to edit was already tried, stop iterating
            if num_neurons_to_edit in neurons_to_edit_already_tried:
                break

            # add the number of neurons to edit to the set of already tried numbers
            neurons_to_edit_already_tried.add(num_neurons_to_edit)

        # store the edit information
        edit_dict = {
            "neurons_deltas": neurons_deltas.T,
            "neuron_indices": current_selected_neurons_indices,
            "original_neuron_values": current_selected_neurons_value.T,
            "edit_applied": False,
        }
        return edit_dict

    def edit_multiple_layers_dict(self, layers_to_edit, prompt, target, restore_after_edit=False, print_progress=True):
        '''
        edit multiple layers by edit_dict
        to_edit_dict: dict of {layer: [(token_idx, rank)]}
        '''
        total_iterations = len(layers_to_edit)
        progress_bar = tqdm(total=total_iterations, desc="Editing Layers") if print_progress else None
        edit_layers_dicts = {}  # {layer: [edit_dict]}

        mlp_bottom_rank_margin = self.config.mlp_bottom_rank_margin
        mlp_top_rank_margin = self.config.mlp_top_rank_margin
        residual_bottom_rank_margin = self.config.residual_bottom_rank_margin
        residual_top_rank_margin = self.config.residual_top_rank_margin
        insert_new_token = self.config.insert_new_token

        for layer in layers_to_edit:
            collected_acts = collect_activations_with_prompt(self.model, self.tokenizer, prompt)
            edit_dict = self.edit_layer_rank_iter(
                collected_acts=collected_acts, 
                layer=layer,
                prompt=prompt,
                target_token=target,
                mlp_bottom_rank_margin=mlp_bottom_rank_margin,
                mlp_top_rank_margin=mlp_top_rank_margin,
                residual_bottom_rank_margin=residual_bottom_rank_margin,
                residual_top_rank_margin=residual_top_rank_margin,
                insert_new_token=insert_new_token
                 )

            if edit_dict:
                edit_layers_dicts[layer] = edit_dict
                self.apply_layer_edit(layer, edit_dict)
            if progress_bar:
                progress_bar.update(1)
        if progress_bar:
            progress_bar.close()

        self.edit_dicts.append((edit_layers_dicts, target))

        if restore_after_edit:
            self.restore_all_edits()
        return edit_layers_dicts

    def calc_deltas_delete(self, neuron_values: Tensor, target_token_id: int,  bottom_rank_margin: int, top_rank_margin: int, max_iter: int = 100) -> Tensor:
        '''
        This function calculates the deltas for deleting from each neuron value by iteratively adjusting the neuron values 
        until the rank of the target token is within the specified margins.

        Parameters:
        neuron_values (tensor): A tensor of shape (embedding_dim, number_of_neurons) representing the values of the neurons.
        target_token_id (int): The ID of the target token.
        top_rank_margin (int): The upper limit for the target token rank. The rank of the target token should not exceed this value.
        bottom_rank_margin (int): The lower limit for the target token rank. The rank of the target token should not be less than this value.
        max_iter (int, optional): The maximum number of iterations for the loop. Default is 100.

        Returns:
        neurons_deltas (tensor): A tensor representing the final deltas for each neuron.

        Note:
        The function asserts that the top_rank_margin is greater than the bottom_rank_margin.
        '''
        assert top_rank_margin >= bottom_rank_margin, "top_rank_margin should be greater or equal bottom_rank_margin"

        if self.config.zero_neurons:  # for ablation experiments
            return -neuron_values  # in order to zero the neurons values

        # Convert hidden states to logits
        original_logits = hs_to_logits(self.model, neuron_values)

        # Get the rank of the target token in the vocabulary
        target_ranks = get_rank_of_token_in_vocab(original_logits, target_token_id)

        # Identify neurons to edit based on the bottom margin
        neurons_to_edit = target_ranks < bottom_rank_margin

        # Initialize a tensor of deltas for each neuron value
        num_neurons = neuron_values.shape[0]
        logits_deltas =  torch.zeros(num_neurons).to(device)
        logits_deltas[neurons_to_edit] = -10

        # Iterate over max_iter times
        for _ in range(max_iter):
            # Adjust the logits of the target token
            original_logits[:, target_token_id] += logits_deltas

            # Convert the adjusted logits back to hidden states
            edited_neuron_values = logits_to_hs(self.model, original_logits, neuron_values.mean(dim=-1), neuron_values.var(dim=-1), self.pinv_lm_head)

            # Restore the original logits
            original_logits[:, target_token_id] -= logits_deltas

            # Recalculate the rank of the target token
            edited_logits_restored = hs_to_logits(self.model, edited_neuron_values)
            target_ranks = get_rank_of_token_in_vocab(edited_logits_restored, target_token_id)

            # Identify neurons that are still outside the margins
            neurons_below_bottom_margin = target_ranks < bottom_rank_margin
            neurons_above_top_margin = target_ranks > top_rank_margin
            neurons_to_edit = neurons_below_bottom_margin | neurons_above_top_margin

            # Adjust the deltas for the next iteration
            if neurons_below_bottom_margin.any():
                logits_deltas[neurons_below_bottom_margin] *= 1.3  # Increase deltas for neurons below the bottom margin
            if neurons_above_top_margin.any():
                logits_deltas[neurons_above_top_margin] *= 0.8  # Decrease deltas for neurons above the top margin

            # If all neurons are within the margins, stop iterating
            if not neurons_to_edit.any():
                break

        # Calculate the final deltas for each neuron
        neurons_deltas = edited_neuron_values - neuron_values

        return neurons_deltas

    def calc_deltas_insert(self, neuron_values: Tensor, target_token_id: int, desired_rank: int, max_iter: int = 100) -> Tensor:
        '''
        This function calculates the deltas for each neuron value by iteratively adjusting the neuron values 
        until the rank of the target token is within the specified margins. In each step, it adds the delta, 
        which decreases the value of the rank.

        Parameters:
        neuron_values (Tensor): A tensor of shape (embedding_dim, number_of_neurons) representing the values of the neurons.
        target_token_id (int): The ID of the target token.
        desired_rank (int): The desired rank of the target token.
        max_iter (int, optional): The maximum number of iterations for the loop. Default is 100.

        Returns:
        neurons_deltas (Tensor): A tensor representing the final deltas for each neuron.

        Note:
        The function asserts that the top_rank_margin is greater than the bottom_rank_margin.
        '''
        assert desired_rank >=0 , "top_rank_margin should be greater than bottom_rank_margin"

        original_logits = hs_to_logits(self.model, neuron_values)
        target_ranks = get_rank_of_token_in_vocab(original_logits, target_token_id)

        neurons_to_edit = target_ranks > desired_rank

        num_neurons = neuron_values.shape[0]
        logits_deltas =  torch.zeros(num_neurons).to(device)
        logits_deltas[neurons_to_edit] = 10

        for _ in range(max_iter):
            original_logits[:, target_token_id] += logits_deltas

            edited_neuron_values = logits_to_hs(self.model, original_logits, neuron_values.mean(dim=-1), neuron_values.var(dim=-1), self.pinv_lm_head)
            original_logits[:, target_token_id] -= logits_deltas

            edited_logits_restored = hs_to_logits(self.model, edited_neuron_values)
            target_ranks = get_rank_of_token_in_vocab(edited_logits_restored, target_token_id)

            neurons_to_edit = target_ranks > desired_rank

            if neurons_to_edit.any():
                logits_deltas[neurons_to_edit] *= 1.3

            if not neurons_to_edit.any():
                break

        neurons_deltas = edited_neuron_values - neuron_values

        return neurons_deltas


    def get_selected_neurons_indices_scores(self, neurons, activations_values, token_id, score_top_k, act_filter, layer, prompt, target_token, method="rank"):
        '''
        return the top k by score selected neurons indices after filtering and their scores
        method options: "rank", "act", "knowledge_neurons"
        '''
        # assert orrect neuron values shape - (embedding_dim, number_of_neurons)
        assert neurons.shape[1] == activations_values.shape[0], "neuron_values shape is incorrect"

        # get neurons scores
        if method == "rank":
            # get the relevant neurons indices and values
            act_values, neuron_indices = self.filter_activations(activations_values, filter=act_filter)
            # get the relevant neuron values from the neuron_values
            filtered_neurons = neurons[:, neuron_indices].detach().clone().to(device).T
            neurons_scores = self.get_neurons_scores_by_token_rank(filtered_neurons, token_id)
            large_is_better = False  # the lower the rank the better

        elif method == "act":
            # get the relevant neurons indices and values
            act_values, neuron_indices = self.filter_activations(activations_values, filter=act_filter)
            neurons_scores = act_values
            large_is_better = True  # the higher the activation the better

        elif method == "grad":
            act_values, neuron_indices = self.filter_activations(activations_values, filter="no_filter")
            neurons_scores = self.get_neurons_scores_by_knowledge_neurons(prompt=prompt, target_token=target_token, layer_index=layer)
            large_is_better = True  # the higher the score the better

        elif method == "random":  # for ablation experiments
            # get the relevant neurons indices and values
            act_values, neuron_indices = self.filter_activations(activations_values, filter="no_filter")
            # generate random scores
            neurons_scores = torch.rand(neuron_indices.shape[0], device=device)
            large_is_better = True  # the higher the score the better

        else:
            raise ValueError("method for selecting neurons is invalid, try: rank, act, grad or random")

        topk_neurons_scores, topk_neurons_indices = torch.topk(neurons_scores, score_top_k, largest=large_is_better)
        topk_neurons_indices = neuron_indices[topk_neurons_indices]

        return topk_neurons_indices, topk_neurons_scores


    @torch.no_grad()
    def filter_activations(self, fc_out_act, filter):
        '''
        return the values and indices of the activations after filtering and sorting them by the activations values
        filtering options: "positive", "top_k"
        '''
        if filter == "positive":
            act_values, act_indices = fc_out_act[fc_out_act > 0], torch.nonzero(fc_out_act > 0, as_tuple=False).squeeze()
        elif "top_" in filter:
            top_k = int(filter.split("_")[-1])
            act_values, act_indices = torch.topk(fc_out_act, top_k)
        else:  # don't filter
            act_values, act_indices = fc_out_act, torch.arange(fc_out_act.shape[0], device=device)
        return act_values, act_indices

    # getters
    @torch.no_grad()
    def get_neurons_scores_by_token_rank(self, neurons, token_id):
        '''
        neurons: tensor of shape (number_of_neurons, embedding_dim)
        return the neurons scores and indices by token rank in the vocab distribution (logits) of the target token
        '''
        # get the neurons logits
        logits = hs_to_logits(self.model, neurons)
        # get the rank of the target token in the vocab distribution (logits)
        target_token_rank = get_rank_of_token_in_vocab(logits, token_id).to(device)
        return target_token_rank

    @torch.no_grad()
    def get_neurons_scores_by_knowledge_neurons(self, prompt, target_token, layer_index):
        if self.knowledge_neoruns is None:
            self.knowledge_neoruns = KnowledgeNeurons(self.model, self.tokenizer, model_type=model.config.model_type, device=device)
        self.model.eval()
        with torch.enable_grad():
            scores_for_layer = self.knowledge_neoruns.get_scores_for_layer(prompt, target_token, layer_idx=layer_index, batch_size=1, steps=20)
        return scores_for_layer


    @torch.no_grad()
    def get_relevant_acts(self, layer, collected_acts, token_index=-1):
        '''
        return the relevant hidden states for the editing process
        token_index: the index of the token in the prompt to collect the hidden states from, -1 is the last token
        '''
        relevant_hs = {
            'residual_after': collected_acts[layer]['residual']['output'][token_index].to(device),
            'fc_out': collected_acts[layer]['mlp_ff2']['output'][token_index].to(device),
            'fc_out_act': collected_acts[layer]['mlp_ff2']['input'][token_index].to(device),
            'attn_out': collected_acts[layer]['attn']['output'][token_index].to(device)
        }
        # if layer is 0, residual_before is the input embedding (the input of layer norm)
        # TODO: it might be OK always to use the else part
        if layer > 0:
            relevant_hs['residual_before'] = collected_acts[layer-1]['residual']['output'][token_index].to(device)
        else: # layer is 0
            relevant_hs['residual_before'] = collected_acts[layer]['ln1']['input'][token_index].to(device)
        return relevant_hs

    def get_model_fc_weights(self, layer):

        if self.model.config.model_type == 'gptj':
            fc_out_weights = self.model.transformer.h[layer].mlp.fc_out.weight.detach().clone().to(device)
            fc_out_bias = self.model.transformer.h[layer].mlp.fc_out.bias.detach().clone().to(device)

        elif self.model.config.model_type == 'llama':
            fc_out_weights = self.model.model.layers[layer].mlp.down_proj.weight.detach().clone().to(device)
            # no bias for llama, return zero tensor shaped as fc_out_weights
            fc_out_bias = torch.zeros_like(fc_out_weights.T[0])
        else:
            raise ValueError(f"Model type not supported {model.config.model_type}, supported models are 'gptj' and 'llama'")


        return fc_out_weights, fc_out_bias

    def get_edited_neurons_count(self):
        """
        return the number of neurons edited in each layer by calculating the union of the neurons indices in each layer for each edit dict
        """
        edited_neurons_sets = defaultdict(set)
        for edit_dict, target in self.edit_dicts:
            for layer, layer_edit_dict in edit_dict.items():
                neuron_indices = [nidx.item() for nidx in layer_edit_dict["neuron_indices"]]
                edited_neurons_sets[layer].update(neuron_indices)
        return {layer: len(neurons_set) for layer, neurons_set in edited_neurons_sets.items()}

    def get_total_number_of_edited_neurons(self):
        """
        return the total number of neurons edited in all layers
        """
        edited_neurons_count = self.get_edited_neurons_count()
        return sum(edited_neurons_count.values())

    def apply_layer_edit(self, layer, layer_edit_dict):
        '''
        apply the neurons deltas to the model
        '''
        if layer_edit_dict["edit_applied"]:
            print(f"edit already applied in layer {layer}")
            return
        neurons_indices = layer_edit_dict["neuron_indices"]
        neurons_deltas = layer_edit_dict["neurons_deltas"]
        
        if self.model.config.model_type == 'gptj':
            self.model.transformer.h[layer].mlp.fc_out.weight.data[:, neurons_indices] += neurons_deltas
        elif self.model.config.model_type == 'llama':
            self.model.model.layers[layer].mlp.down_proj.weight.data[:, neurons_indices] += neurons_deltas
        else:
            raise ValueError(f"Model type not supported {self.model.config.model_type}, supported models are 'gptj' and 'llama'")
        layer_edit_dict["edit_applied"] = True

    def restore_layer_edit(self, layer, layer_edit_dict):
        '''
        restore the model to the original neurons values
        '''
        if not layer_edit_dict["edit_applied"]:
            print(f"edit not applied in layer {layer}")
            return
        neurons_indices = layer_edit_dict["neuron_indices"]
        neurons_deltas = layer_edit_dict["neurons_deltas"]
        self.model.transformer.h[layer].mlp.fc_out.weight.data[:, neurons_indices] += -neurons_deltas
        layer_edit_dict["edit_applied"] = False

    def apply_all_edits(self):
        """
        apply all the new neurons values to the model
        """
        for target_edit_dict, target in self.edit_dicts:
            for layer, layer_edit_dict in target_edit_dict.items():
                self.apply_layer_edit(layer, layer_edit_dict)

    def restore_all_edits(self):
        """
        restore the model to the original neurons values
        """
        for target_edit_dict, target in self.edit_dicts:
            for layer, layer_edit_dict in target_edit_dict.items():
                self.restore_layer_edit(layer, layer_edit_dict)


class REVSTokenScore:
    """ 
    REVSTokenScore calculate the ranks of the target token for a given prompt in the vocab distribution for each layer and sub-module in the model.
    """
    def __init__(self, model, tok, prompt, target):
        self.vocab_size = model.config.vocab_size
        self.prompt = prompt
        self.target = target
        collected_acts = collect_activations_with_prompt(model, tok, prompt)
        self.ranks = get_token_rank_across_layers(model, tok, collected_acts, target)

    def get_ranks(self):
        # return a deep copy of the ranks
        return self.ranks

    def get_rank_bottom_distance(self):
        """
        Calculates and returns a dictionary of distances between a given threshold and the rank of each sub-module in each layer of a model.
        The distance is calculated as rank - threshold.
        Returns:
        dict: A dictionary where each key is a layer, and each value is another dictionary where each key is a sub-module and each value is the calculated distance for that sub-module.
        """
        rank_bottom_distance = {}
        for layer, ranks in self.ranks.items():
            rank_bottom_distance[layer] = {}
            for sub_module, rank in ranks.items():
                rank_bottom_distance[layer][sub_module] = rank
        return rank_bottom_distance

    def get_rank_top_distance(self):
        """
        If the distance is positive, the rank distance is safely within the threshold, else it is not.
        Calculates and returns a dictionary of distances between a given threshold and the rank of each sub-module in each layer of a model.
        The distance is calculated as the difference between the vocab size and the rank minus the threshold.
        Returns:
        dict: A dictionary where each key is a layer, and each value is another dictionary where each key is a sub-module and each value is the calculated distance for that sub-module.
        """
        rank_top_distance = {}
        for layer, ranks in self.ranks.items():
            rank_top_distance[layer] = {}
            for sub_module, rank in ranks.items():
                rank_top_distance[layer][sub_module] = self.vocab_size - rank 
        return rank_top_distance

    def get_edit_scores(self, threshold=100, last_layer_only=False):
        """
        Calculates and returns a dictionary of scores for each sub-module in each layer of a model.
        The score is calculated based on the distance of the rank from the top and bottom of the vocabulary distribution, 
        clipped by a threshold and normalized to the range [0, 1].

        If last_layer_only is True, the scores are only calculated for the last layer.

        The score is calculated as follows:
            - rank_top_distance_score: `1 - max(1, rank - threshold - self.vocab_size) / threshold`
            - rank_bottom_distance_score: `1 - max(1, threshold - rank) / threshold`
            - edit_in_range_score: min(rank_top_distance_score, rank_bottom_distance_score)

        The rank_top_distance_score and rank_bottom_distance_score represent the normalized distance of the rank from the 
        top and bottom of the vocabulary distribution, respectively. The edit_in_range_score represents the minimum of these 
        two scores, indicating whether the rank is within the range defined by the threshold (between threshold and vocab_size - threshold).

        Args:
            threshold (int, optional): The threshold for clipping the rank. Defaults to 100.
            last_layer_only (bool, optional): If True, only calculate scores for the last layer. Defaults to False.

        Returns:
            dict: A dictionary where each key is a layer, and each value is another dictionary where each key is a sub-module 
            and each value is a dictionary containing the calculated scores ('in_range', 'top_distance', 'bottom_distance') for that sub-module.
        """
        edit_in_range_score = {}
        edit_rank_top_distance_score = {}
        edit_rank_bottom_distance_score = {}

        if last_layer_only:
            last_layer = max(self.ranks.keys())
            ranks = {last_layer: self.ranks[last_layer]}
        else:
            ranks = self.ranks

        for layer, ranks in ranks.items():
            edit_in_range_score[layer] = {}
            edit_rank_top_distance_score[layer] = {}
            edit_rank_bottom_distance_score[layer] = {}
            for sub_module, rank in ranks.items():
                rank_bottom_distance_score = round(min(1, rank / threshold), 3)  # min(20/100, 1) -> 0.2, min(9030/100, 1) -> 1
                rank_top_distance_score = round(min(1, (self.vocab_size - rank) / threshold, 3))  # min((50400 - 50390)/100, 1) -> 0.1, min((50400 - 40000)/100, 1) -> 1
                edit_rank_bottom_distance_score[layer][sub_module] = rank_bottom_distance_score
                edit_rank_top_distance_score[layer][sub_module] = rank_top_distance_score
                edit_in_range_score[layer][sub_module] = min(rank_top_distance_score, rank_bottom_distance_score)
        scores_dict = {
            "in_range": edit_in_range_score,
            "top_distance": edit_rank_top_distance_score,
            "bottom_distance": edit_rank_bottom_distance_score
        }
        return scores_dict


class REVSScore:
    """ 
    REVSScore calculate the score of unlearning a sequence target.
    """
    def __init__(self, model, tokenizer, prompt, target, skip_tokens=None, stop_tokens=None, max_tokens=None):
        self.prompt = prompt
        self.target = target
        self.rank_edit_iter_tokens = {}
        self.edit_scores_tokens = {}
        concat_prompts, concat_targets = create_concat_prompts_target(tokenizer, prompt, target, skip_tokens=skip_tokens, stop_tokens=stop_tokens, max_tokens=max_tokens)

        for concat_prompt, concat_target in zip(concat_prompts, concat_targets):
            rank_edit_score_iter = REVSTokenScore(model, tokenizer, concat_prompt, concat_target)
            self.rank_edit_iter_tokens[concat_target] = rank_edit_score_iter

    def get_edit_scores(self, threshold=100, last_layer_only=False):
        '''
        Calculate the score of unlearning a sequence target by getting the max score for each target token in each layer and sub-module
        as a measurement for the difficulty of extracting the target token from the model.

        If last_layer_only is True, the scores are only calculated for the last layer.

        Args:
            threshold (int, optional): The threshold for clipping the rank. Defaults to 100.
            last_layer_only (bool, optional): If True, only calculate scores for the last layer. Defaults to False.

        Returns:
            dict: A dictionary where each key is a layer, and each value is another dictionary where each key is a sub-module 
            and each value is a dictionary containing the calculated scores ('in_range', 'top_distance', 'bottom_distance') for that sub-module.
        '''
        edit_rank_top_distance_score = {}
        edit_rank_bottom_distance_score = {}
        edit_in_range_score = {}

        # get the max score for each target token in each layer and sub-module
        for target_token, rank_edit_iter in self.rank_edit_iter_tokens.items():
            scores_dict = rank_edit_iter.get_edit_scores(threshold, last_layer_only)
            self.edit_scores_tokens[target_token] = scores_dict
            for layer, ranks in scores_dict["in_range"].items():
                for sub_module, score in ranks.items():
                    if layer not in edit_rank_top_distance_score:
                        edit_rank_top_distance_score[layer] = {}
                        edit_rank_bottom_distance_score[layer] = {}
                        edit_in_range_score[layer] = {}
                    if sub_module not in edit_rank_top_distance_score[layer]:
                        edit_rank_top_distance_score[layer][sub_module] = 0
                        edit_rank_bottom_distance_score[layer][sub_module] = 0
                        edit_in_range_score[layer][sub_module] = 0
                    edit_rank_top_distance_score[layer][sub_module] = max(edit_rank_top_distance_score[layer][sub_module], scores_dict["top_distance"][layer][sub_module])
                    edit_rank_bottom_distance_score[layer][sub_module] = max(edit_rank_bottom_distance_score[layer][sub_module], scores_dict["bottom_distance"][layer][sub_module])
                    # get the min score of top and bottom scores of the target token
                    edit_in_range_score[layer][sub_module] = min(edit_rank_top_distance_score[layer][sub_module], edit_rank_bottom_distance_score[layer][sub_module])
        scores_dict = {
            "in_range": edit_in_range_score,
            "top_distance": edit_rank_top_distance_score,
            "bottom_distance": edit_rank_bottom_distance_score,
        }
        return scores_dict
