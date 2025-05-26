import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_PATH = ""
PROJECT_NAME = ""
CACHE_PATH = ''  # TODO: to be filled in by the user
HG_TOKEN = '' # TODO: to be filled in by the user