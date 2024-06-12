from collections import defaultdict
import torch
from utils.globals import device


LAYER_NAMES_GPT_J_6B = {
    "residual": "transformer.h.{}",
    "mlp": "transformer.h.{}.mlp",
    "attn": "transformer.h.{}.attn",

    "ln1": "transformer.h.{}.ln_1",
    "attn_q": "transformer.h.{}.attn.q_proj",
    "attn_k": "transformer.h.{}.attn.k_proj",
    "attn_v": "transformer.h.{}.attn.v_proj",
    "attn_o": "transformer.h.{}.attn.out_proj",

    "mlp_ff1": "transformer.h.{}.mlp.fc_in",
    "mlp_ff2": "transformer.h.{}.mlp.fc_out",
}

LAYER_NAMES_LLAMA_3_8B = {
    "residual": "model.layers.{}",
    "mlp": "model.layers.{}.mlp",
    "attn": "model.layers.{}.self_attn",

    "ln1": "model.layers.{}.input_layernorm",
    "attn_q": "model.layers.{}.self_attn.q_proj",
    "attn_k": "model.layers.{}.self_attn.k_proj",
    "attn_v": "model.layers.{}.self_attn.v_proj",
    "attn_o": "model.layers.{}.self_attn.o_proj",

    "ln2": "model.layers.{}.post_attention_layernorm",
    "mlp_gate": "model.layers.{}.mlp.gate_proj",
    "mlp_ff1": "model.layers.{}.mlp.up_proj",
    "mlp_ff2": "model.layers.{}.mlp.down_proj",
}

# acts[layer]['mlp_ff1']['input'] == acts[layer]['mlp']['input']
# acts[layer]['mlp_ff2']['output'] == acts[layer]['mlp']['output']

class ActivationCollector:
    """
    This class is used to collect the activations from specified layers of a PyTorch model.
    
    Attributes:
        collected_acts (defaultdict): A nested dictionary to store the collected activations.
        handles (list): A list to store the handles of the registered hooks.
        layer_names (dict): A dictionary mapping layer indices to layer names.
    """
    def __init__(self, layer_names=LAYER_NAMES_GPT_J_6B):
        self.collected_acts = defaultdict(lambda: defaultdict(dict))
        self.handles = []
        self.layer_names = layer_names

    def hook(self, module, layer_idx, layer_name, input, output):
        # if the input or output are tuple, we take the first element
        try:
            if isinstance(input, tuple) and len(input) > 0:
                input = input[0].squeeze()
            if isinstance(output, tuple) and len(output) > 0:
                output = output[0].squeeze()
            if len(input) > 0 and input.shape[0] == 1:
                input = input.squeeze()
            if len(output) > 0 and output.shape[0] == 1:
                output = output.squeeze()
        except Exception as e:
            print(f"Error in hook: {e}")
        try:
            self.collected_acts[layer_idx][layer_name]['input'] = input
            self.collected_acts[layer_idx][layer_name]['output'] = output
        except Exception as e:
            print(f"Error in hook: {e}")


    def register_hooks(self, model):
        """
        Register a forward hook on each layer of the model. The hook is a lambda function that calls
        the `hook` method with the module, input, output, and current layer index and name.
        Default arguments are used to capture the current values of `layer_idx` and `layer_name_format`
        at the time the lambda function is created, avoiding issues with variable capture in the loop.
        
        Args:
            model (torch.nn.Module): The model on which to register the hooks.
        """
        if model.config.model_type == 'gptj':
            n_layers = model.config.n_layer
        elif model.config.model_type == 'llama':
            n_layers = model.config.num_hidden_layers
        else:
            raise ValueError(f"Model type not supported {model.config.model_type}, supported models are 'gptj' and 'llama'")

        for layer_idx in range(n_layers):
            for layer_key, layer_name_format in self.layer_names.items():
                layer_name = layer_name_format.format(layer_idx)
                layer = self._get_module_by_name(model, layer_name)
                if layer is not None:
                # Register a forward hook on the layer. The hook is a lambda function that calls
                # the `hook` method with the module, input, output, and current layer index and name.
                # Default arguments are used to capture the current values of `layer_idx` and `layer_name_format`
                # at the time the lambda function is created, avoiding issues with variable capture in the loop.
                    handle = layer.register_forward_hook(
                        lambda module, input, output, layer_idx=layer_idx, layer_key=layer_key: 
                        self.hook(module, layer_idx, layer_key, input, output))
                    self.handles.append(handle)
                else:
                    print(f"Warning: Layer {layer_name} not found in model")

    def _get_module_by_name(self, model, name):
        for module_name, module in model.named_modules():
            if module_name == name:
                return module
        return None

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def collect_activations_with_prompt(model, tokenizer, prompt):

    if model.config.model_type == 'gptj':
        layer_names = LAYER_NAMES_GPT_J_6B
    elif model.config.model_type == 'llama':
        layer_names = LAYER_NAMES_LLAMA_3_8B
    else:
        raise ValueError(f"Model type not supported {model.config.model_type}, supported models are 'gptj' and 'llama'")
    # Tokenize the prompt and convert it to a tensor
    inputs_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    # Create an instance of the ActivationCollector class
    collector = ActivationCollector(layer_names)

    # Register hooks to collect activations from the model
    collector.register_hooks(model)

    # Forward pass to collect activations
    with torch.no_grad():
        model(inputs_ids)

    # Remove the hooks
    collector.remove_hooks()

    return collector.collected_acts