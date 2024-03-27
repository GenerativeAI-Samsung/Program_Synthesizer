import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = torch.tensor(d_model)
        self.queries_linear = nn.Linear(d_model, d_model)
        self.keys_linear = nn.Linear(d_model, d_model)
        self.values_linear = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax()
    
    def forward(self, queries, keys, values):
        queries = self.queries_linear(queries)
        keys = self.keys_linear(keys)
        values = self.values_linear(values)

        x = torch.matmul(self.softmax(torch.matmul(queries, torch.transpose(keys, -2, -1))/torch.sqrt(self.d_model)), values)
        return x

class FeedForwad(nn.Module):
    def __init__(self, d_model, max_sequence_len):
        super().__init__()
        self.linear1 = nn.Linear(d_model*max_sequence_len, 128)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, d_model*max_sequence_len)

        self.d_model = d_model
        self.max_sequence_len = max_sequence_len
    
    def forward(self, x):
        x = x.view(-1, self.d_model*self.max_sequence_len)
        x = self.linear1(x)
        x = self.ReLU(x)
        x = self.linear2(x)
        x = self.ReLU(x)
        x = self.linear3(x)
        x = x.view(-1, 16, 128)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transform_condition(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(512 * 768, 128)
        self.layer2 = nn.Linear(128, 128 * 16)
        self.ReLU = nn.ReLU()
    
    def forward(self, condition_embed):
        x = condition_embed.view(-1, 512 * 768)
        x = self.layer1(x)
        x = self.ReLU(x)
        x = self.layer2(x)
        x = x.view(-1, 16, 128)
        return x

class GeneratorBodyLayer(nn.Module):
    def __init__(self, d_model=128, max_sequence_len=16):
        super().__init__()
        self.self_dattention = SelfAttention(d_model=d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.feedforwad = FeedForwad(d_model=d_model, max_sequence_len=max_sequence_len)
    
    def forward(self, input_embed, condition_embed):
        x_temp = self.self_dattention.forward(keys=condition_embed, queries=input_embed, values=condition_embed)
        x = input_embed + condition_embed + x_temp
        x = self.layernorm.forward(x)

        x_temp = self.feedforwad.forward(x)
        x = x + condition_embed + x_temp
        x = self.layernorm(x)
        return x

class Generator(nn.Module):
    def __init__(self, device, d_model=128, max_sequence_len=16):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.max_sequence_len = max_sequence_len

        self.embedding = nn.Embedding(6, d_model, padding_idx=5)
        self.transform_condi = Transform_condition() 
        # 0 -> [START]
        # 1 -> [FEEDFORWARDLAYER]
        # 2 -> [CONVOLUTIONAL2DLAYER], 
        # 3 -> [MAXPOOLING2DLAYER]
        # 4 -> [END]
        # 5 -> [PAD]
        # embedding_dim: d_model
        self.positional_encoder = PositionalEncoding(d_model=d_model, max_len=max_sequence_len)
        self.layers = nn.ModuleList([GeneratorBodyLayer(d_model=d_model, 
                                                        max_sequence_len=max_sequence_len)
                                    for _ in range(5)])
        self.linear = nn.Linear(d_model * max_sequence_len, 4)
    
    def forward(self, x, condition_embed):
        x_embed = self.embedding(x).to(self.device)
        x_pos = self.positional_encoder.forward(x=x).to(self.device)
        x = x_pos + x_embed
        condition_embed = self.transform_condi.forward(condition_embed).to(self.device)

        for layer in self.layers:
            x = layer.forward(x, condition_embed)

        x = x.view(-1, self.d_model * self.max_sequence_len)
        x = self.linear(x)
        return x