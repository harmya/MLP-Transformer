import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import re
from transformers import BertTokenizer


# ----------------- HyperParameters -----------------
context_length = 128
batch_size = 8
d_embed = 64

# ----------------- Functions -----------------
def load_jokes(file_path):
    with open(file_path, 'r') as file:
        jokes = file.readlines()
    jokes = [re.sub('\n', '', joke) for joke in jokes]
    return jokes

def tokenize_jokes(jokes):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_jokes = tokenizer(jokes, return_tensors='pt', padding=True, truncation=True, max_length=context_length)
    return tokenized_jokes



