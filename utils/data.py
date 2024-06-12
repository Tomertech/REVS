from typing import List, Optional
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


# Synthetic data generation
def date_generator():
    ''' yield random date as a string '''
    while True:
        year = np.random.randint(2010, 2023)
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29)
        yield f"{day}-{month}-{year}"


def str_to_token_strs(tokenizer, s: str) -> List[str]:
    tokens = tokenizer.encode(s)
    tokens_strs = [tokenizer.decode([t]) for t in tokens]
    return tokens_strs


def create_concat_prompts_target(
    tokenizer, 
    prompt: str, 
    target: str, 
    method: Optional[str] = None, 
    skip_tokens: Optional[List[str]] = None, 
    stop_tokens: Optional[List[str]] = None, 
    max_tokens: Optional[int] = None
) -> List[str]:
    """
    Create concatenated prompts and targets based on a given method.

    Parameters:
    tokenizer: The tokenizer to use for encoding and decoding.
    prompt: The initial prompt.
    target: The target string.
    method: The token selection method. Can be 'rarest', 'random', 'first', or 'frequent'. Default is 'rarest'.
    skip_tokens: Tokens to skip.
    stop_tokens: Tokens to stop at.
    max_tokens: The maximum number of tokens to return.

    Returns:
    A list of concatenated prompts of length max_tokens and a list of concatenated targets of length max_tokens.
    """
    if isinstance(skip_tokens, str) or isinstance(stop_tokens, str):
        raise TypeError("skip_tokens and stop_tokens should be a list of strings, not a single string")

    # Encode the target string into token IDs
    token_ids = tokenizer.encode(target, add_special_tokens=False)
    # Decode the token IDs back into strings
    target_tokens_strs = [tokenizer.decode([t]) for t in token_ids]

    concat_prompts = []
    concat_targets = []

    # Iterate over the target tokens
    for target_token in target_tokens_strs:
        # If the token is a stop token, break the loop
        if stop_tokens and any(st in target_token for st in stop_tokens):
            break
        # If the token is not a skip token, add it to the prompts and targets
        if not (skip_tokens and any(sk in target_token for sk in skip_tokens)):
            concat_prompts.append(prompt)
            concat_targets.append(target_token)
        # Add the target token to the prompt
        prompt += target_token

    # If max tokens is specified, sort and truncate the prompts and targets
    if max_tokens:
        # Pair the prompts and targets together
        pairs = list(zip(concat_prompts, concat_targets))

        # Sort the pairs based on the method
        if method == 'rarest' or method is None:  # Default to 'rarest'
            pairs.sort(key=lambda x: tokenizer.encode(x[1]), reverse=True)
        elif method == 'random':
            np.random.seed(0)
            np.random.shuffle(pairs)
        elif method == 'first':
            pairs = pairs[:max_tokens]
        elif method == 'frequent':
            pairs.sort(key=lambda x: tokenizer.encode(x[1]))
        else:
            raise ValueError(f"Invalid method: {method}, must be 'rarest', 'random', 'first', or 'frequent'")

        # Unzip the sorted and truncated pairs back into prompts and targets
        concat_prompts, concat_targets = zip(*pairs)
        concat_prompts = list(concat_prompts)
        concat_targets = list(concat_targets)
        concat_prompts = concat_prompts[:max_tokens]
        concat_targets = concat_targets[:max_tokens]

    return concat_prompts, concat_targets


def ssn_generator():
    ''' yield random ssn in the correct format (AAA-GG-SSSS) as a string'''
    while True:
        ssn = f"{np.random.randint(100, 1000)}-{np.random.randint(10, 100)}-{np.random.randint(1000, 10000)}"
        yield ssn

def name_generator(names_list):
    ''' yield random name as a string '''
    while True:
        yield np.random.choice(names_list, replace=False)  # choose a random name from the list of names only once (no replacement)

def template_to_data(sentenses: List[str]) -> pd.DataFrame:
    '''
    This function preprocesses a list of sentences and returns a DataFrame.

    Each sentence in the list may contain special tags: [SSN], [DATE], and [NAME]. 
    These tags are replaced with randomly generated SSNs, dates, and names respectively.

    The function creates a DataFrame with the following columns:
    - 'prompt': The original sentence text that comes before the [SSN] tag.
    - 'sentence': The sentence after replacing the tags with generated values.
    - 'ssn': The randomly generated SSN that replaced the [SSN] tag.
    - 'date': The randomly generated date that replaced the [DATE] tag.
    - 'name': The randomly generated name that replaced the [NAME] tag.

    Parameters:
    sentences (List[str]): A list of sentences to preprocess.

    Returns:
    df (pd.DataFrame): A DataFrame containing the preprocessed data.
    '''
    ssn_gen = ssn_generator()
    date_gen = date_generator()
    name_gen = name_generator(names_list)
    ssn_list = []
    date_list = []
    name_list = []
    prompt_list = []
    new_sentence_list = []
    for sentence in sentenses:
        ssn = next(ssn_gen)
        date = next(date_gen)
        name = next(name_gen)
        new_sentence = sentence.replace("[SSN]", ssn).replace("[DATE]", date).replace("[NAME]", name)
        ssn_list.append(ssn)
        date_list.append(date)
        name_list.append(name)
        prompt = new_sentence.split(f" {ssn}")[0]  # get the text before the [SSN] tag
        prompt_list.append(prompt)
        new_sentence_list.append(new_sentence)
    df = pd.DataFrame({"prompt": prompt_list, "sentence": new_sentence_list, "ssn": ssn_list, "date": date_list, "name": name_list})
    return df

def template_to_data_many_to_one(sentences: List[str], names_list: List[str], prompts_target_ratio: int) -> pd.DataFrame:
    '''
    This function preprocesses a list of sentences and returns a DataFrame.

    Each sentence in the list may contain special tags: [SSN], [DATE], and [NAME]. 
    These tags are replaced with randomly generated SSNs, dates, and names respectively.
    However, the [NAME] and [SSN] tags are replaced only after prompts_target_ratio examples.

    The function creates a DataFrame with the following columns:
    - 'prompt': The original sentence text that comes before the [SSN] tag.
    - 'sentence': The sentence after replacing the tags with generated values.
    - 'ssn': The randomly generated SSN that replaced the [SSN] tag.
    - 'date': The randomly generated date that replaced the [DATE] tag.
    - 'name': The randomly generated name that replaced the [NAME] tag.

    Parameters:
    sentences (List[str]): A list of sentences to preprocess.
    prompts_target_ratio (int): The number of examples before replacing [NAME] and [SSN] tags.

    Returns:
    df (pd.DataFrame): A DataFrame containing the preprocessed data.
    '''
    ssn_gen = ssn_generator()
    date_gen = date_generator()
    name_gen = name_generator(names_list)
    ssn_list = []
    date_list = []
    name_list = []
    prompt_list = []
    new_sentence_list = []
    counter = 0
    ssn, name, date = None, None, None
    for sentence in sentences:
        if counter % prompts_target_ratio == 0:
            ssn = next(ssn_gen)
            name = next(name_gen)
        date = next(date_gen)
        new_sentence = sentence.replace("[SSN]", ssn).replace("[DATE]", date).replace("[NAME]", name)
        ssn_list.append(ssn)
        date_list.append(date)
        name_list.append(name)
        prompt = new_sentence.split(f" {ssn}")[0]  # get the text before the [SSN] tag
        prompt_list.append(prompt)
        new_sentence_list.append(new_sentence)
        counter += 1
    df = pd.DataFrame({'prompt': prompt_list, 'sentence': new_sentence_list, 'ssn': ssn_list, 'date': date_list, 'name': name_list})
    return df
