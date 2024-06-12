import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CACHE_PATH = ''  # TODO: to be filled in by the user
HG_TOKEN = '' # TODO: to be filled in by the user