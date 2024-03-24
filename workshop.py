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


def get_batch(split_type=None):
    jokes = train_jokes if split_type == 'train'or split_type == None else val_jokes
    # get a offset between 0 and len(train_jokes) - batch_size - 1
    random_idx = np.random.randint(0, len(jokes) - batch_size - 1)
    x = torch.stack([jokes[random_idx + i] for i in range(0, batch_size)])
    y = torch.stack([torch.cat((jokes[random_idx + i][1:], torch.tensor([PAD_TOKEN]))) for i in range(0, batch_size)])
    return x, y

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# ----------------- Model Parts Classes -----------------
class InputEmbeddings(nn.Module):
    def __init__(self, d_embed : int, vocab_size : int):
        super().__init__()
        self.d_embed = d_embed
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_embed)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_embed)

class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, context_length):
        super().__init__()
        self.d_embed = d_embed
        self.context_length = context_length
        positional_encoding = torch.zeros(context_length, d_embed)
        position_index = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        denominator = torch.exp(torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed))
        positional_encoding[ : , 0::2] = torch.sin(position_index * denominator)
        positional_encoding[ : , 1::2] = torch.cos(position_index * denominator)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)
    
    def forward(self, x):
        return (x + self.positional_encoding[: , :x.shape[1], :])
    

