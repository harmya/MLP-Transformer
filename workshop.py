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
    

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(d_embed, head_size, bias=False)
        self.query = nn.Linear(d_embed, head_size, bias=False)
        self.value = nn.Linear(d_embed, head_size, bias=False)
        self.mask = torch.tril(torch.ones(context_length, context_length))
    
    def forward(self, x):
        batch_size, sequence_length, feature_dimension = x.shape
        K = self.key(x)
        Q = self.query(x)
        q_kt = Q @ K.transpose(-2, -1) / np.sqrt(feature_dimension) 
        q_kt = q_kt.masked_fill(self.mask == 0, float('-inf'))
        scaled_qkt = torch.nn.functional.softmax(q_kt, dim=-1)
        V = self.value(x)
        attention = scaled_qkt @ V
        return attention

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for i in range(num_heads)])
        self.linear_layer = nn.Linear(head_size * num_heads, d_embed) # head_size * num_heads = d_embed (usually)

    def forward(self, x):
        head_outputs = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.linear_layer(head_outputs)


class FeedForward(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(d_embed, 4 * d_embed),
            nn.ReLU(),
            nn.Linear(4 * d_embed, d_embed)
        )
    
    def forward(self, x):
        return self.linear_layer(x)


class Block(nn.Module):
    def __init__(self, d_embed, num_heads):
        super().__init__()
        head_size = d_embed // num_heads # head_size is "how" much of the embedding is "seen" by each head
        self.multi_head_attention = MultiHeadAttention(head_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(d_embed)
        self.feed_forward_layer = FeedForward(d_embed)
        self.layer_norm2 = nn.LayerNorm(d_embed)
    
    def forward(self, x):
        attention = self.multi_head_attention(x)
        x = self.layer_norm1(x + attention)
        feed_forward = self.feed_forward_layer(x)
        return self.layer_norm2(x + feed_forward)

class Transformer(nn.Module):
    def __init__(self, d_embed, num_heads, num_blocks):
        super().__init__()
        self.input_embeddings = InputEmbeddings(d_embed, vocab_size)
        self.positional_encoding = PositionalEncoding(d_embed, context_length)
        self.blocks = nn.Sequential(*[Block(d_embed, num_heads) for i in range(num_blocks)])
        self.final_layer_norm = nn.LayerNorm(d_embed)
        self.output_layer = nn.Linear(d_embed, vocab_size)
    
    def forward(self, x, y = None):
        batch_size, sequence_length = x.shape
        x_input_embeddings = self.input_embeddings(x)
        x_positional = self.positional_encoding(x_input_embeddings)
        block_out = self.blocks(x_positional)
        layer_norm_out = self.final_layer_norm(block_out)
        logits = self.output_layer(layer_norm_out)
        loss = None
        if y is None:
            return logits, loss
        else:
            loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), y.view(-1))
            return logits, loss
    

# ----------------- Model Run -----------------

# Load jokes
jokes = load_jokes('jokes.txt')

# Tokenize jokes
tokenized_jokes = tokenize_jokes(jokes)
vocab_size = tokenized_jokes['input_ids'].shape[1]
print(f'Vocab size: {vocab_size}')
