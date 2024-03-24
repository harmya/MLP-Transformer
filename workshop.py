import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math


context_length = 256
batch_size = 64
d_embed = 128
num_heads = 6
num_decoder_blocks = 8
max_iterations = 5000
learning_rate = 2e-4

def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


text = load_file('maze_runner.txt')

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.85*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in ix])
    y = torch.stack([data[i + 1 : i + context_length+1] for i in ix])
    return x, y


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
        self.mask = self.mask.to(q_kt.device)
        q_kt = q_kt.masked_fill(self.mask[:sequence_length, :sequence_length] == 0, float('-inf'))
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
    
    def generate(self, x, max_length=context_length):
        with torch.no_grad():
            for i in range(max_length):
                x = x[:, -context_length:]
                logits, _ = self.forward(x)
                logits = logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_token], dim=-1)
        return x
    
model = Transformer(d_embed, num_heads, num_decoder_blocks)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def calculate_loss():
    model.eval()
    for split in ['train', 'val']:
        total_loss = 0
        total_batches = 0
        for i in range(100):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            total_loss += loss
            total_batches += 1
        print(f'{split} loss: {total_loss / total_batches}')


model.train()
for epoch in range(max_iterations):
    x, y = get_batch('train')
    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'epoch {epoch} loss: {loss}')

# generate text
x = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(x)
print(decode(generated[0].tolist()))