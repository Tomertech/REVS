from typing import Literal, Optional

class BaseConfig:
    """Configuration class for model training and evaluation.

    This class handles configuration settings for language model training,
    unlearning, and evaluation experiments.

    Args:
        method_name (str): Method name for the unlearning experiment: ['SIRE', 'MEMIT', 'FT', 'PATIL', 'NPO', 'RMU']
        score_threshold (int): Threshold value for score calculation
        skip_tokens (list): List of tokens to skip during processing
        stop_tokens (list): List of tokens that signal when to stop generation
        max_tokens (int): Maximum number of tokens to generate
        max_prompt_len (int): Maximum length of input prompts
        seed (int): Random seed for reproducibility
        token_method (str): Strategy for selecting tokens to unlearn - ['rarest', 'first', 'frequent', 'random']
        exp_type (Literal['ssn', 'email']): Type of experiment - either SSN or email
            processing
        model_type (Literal['llama', 'gptj']): Type of model architecture to use
        perplexity (bool, optional): Whether to calculate perplexity. Defaults to False
        save_model (bool, optional): Whether to save model checkpoints. Defaults to False
        log_wandb (bool, optional): Whether to log metrics to Weights & Biases. 
            Defaults to True
        unlearn_num_examples (Optional[int], optional): Number of examples to use for
            unlearning. Defaults to None
        not_unlearn (bool, optional): Flag to skip unlearning process. Defaults to False

    Example:
        >>> config = BaseConfig(
        ...     method_name="SIRE",
        ...     score_threshold=100,
        ...     skip_tokens=["gmail", ".com"],
        ...     stop_tokens=["@"],
        ...     max_tokens=100,
        ...     max_prompt_len=100,
        ...     seed=0,
        ...     token_method="rarest",
        ...     exp_type="ssn",
        ...     model_type="llama"
        ... )
    """
    def __init__(self, 
                 method_name: str,
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
                 not_unlearn: bool = False):

        self.method_name = method_name
        self.score_threshold = score_threshold
        self.skip_tokens = skip_tokens  
        self.stop_tokens = stop_tokens
        self.max_tokens = max_tokens
        self.max_prompt_len = max_prompt_len
        self.token_method = token_method
        self.seed = seed
        self.perplexity = perplexity
        self.unlearn_num_examples = unlearn_num_examples
        self.save_model = save_model
        self.log_wandb = log_wandb
        self.not_unlearn = not_unlearn
        self.exp_type = exp_type
        self.model_type = model_type

    def to_dict(self):
        return {
            'method_name': self.method_name,
            'score_threshold': self.score_threshold,
            'skip_tokens': self.skip_tokens,
            'stop_tokens': self.stop_tokens, 
            'max_tokens': self.max_tokens,
            'max_prompt_len': self.max_prompt_len,
            'token_method': self.token_method,
            'seed': self.seed,
            'perplexity': self.perplexity,
            'unlearn_num_examples': self.unlearn_num_examples,
            'save_model': self.save_model,
            'log_wandb': self.log_wandb,
            'not_unlearn': self.not_unlearn,
            'exp_type': self.exp_type,
            'model_type': self.model_type
        }

    def __str__(self):
        config_dict = self.to_dict()
        config_lines = [f"\t{k}={v}" for k, v in config_dict.items()]
        return f"{self.__class__.__name__}:\n" + "\n".join(config_lines)