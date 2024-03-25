import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.queries_linear = nn.Linear(d_model, d_model)
        self.keys_linear = nn.Linear(d_model, d_model)
        self.values_linear = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax()
    
    def forward(self, queries, keys, values):
        queries = self.queries_linear(queries)
        keys = self.keys_linear(keys)
        values = self.values_linear(values)

        x = torch.matmul(self.softmax(torch.matmul(queries, torch.transpose(keys, 0, 1))/torch.sqrt(self.d_model)), values)
        return x

class Add(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(1)
        self.beta = nn.Parameter(1)
        self.alpha = nn.Parameter(1)

    def forward(self, input_embed, condition_embed, result):
        x = self.gamma * input_embed + self.beta * condition_embed + self.alpha * result 

class Add_pos_embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(1)
        self.beta = nn.Parameter(1)

    def forward(self, pos, inputs_embed):
        x = self.gamma * pos + self.beta * inputs_embed 

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class FeedForwad(nn.Module):
    def __init__(self, d_model, max_sequence_len):
        self.linear1 = nn.Linear(d_model*max_sequence_len, d_model*max_sequence_len)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(d_model*max_sequence_len, d_model*max_sequence_len)
        self.linear3 = nn.Linear(d_model*max_sequence_len, d_model*max_sequence_len)

        self.d_model = d_model
        self.max_sequence_len = max_sequence_len
    
    def forward(self, x):
        x = x.view(-1, self.d_model*self.max_sequence_len)
        x = self.linear1(x)
        x = self.ReLU(x)
        x = self.linear2(x)
        x = self.ReLU(x)
        x = self.linear3(x)
        return x

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_sequence_len):
        super().__init__()
        self.encoding = torch.zeros(max_sequence_len, d_model)
        self.encoding.require_grad = False

        pos = torch.arange(0, max_sequence_len)
        pos = pos.float.unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
    
    def forward(self, x):

        batch_size, seq_len = x.size()

        return self.encoding[:seq_len, :]

class GeneratorBodyLayer(nn.Module):
    def __init__(self, d_model=768, max_sequence_len=100):
        super().___init__()
        self.self_dattention = SelfAttention(d_model=d_model)
        self.add = Add()
        self.layernorm = LayerNorm(d_model=d_model)
        self.feedforwad = FeedForwad(d_model=d_model, max_sequence_len=max_sequence_len)
    
    def forward(self, input_embed, condition_embed):
        x_temp = self.self_dattention.forward(key=condition_embed, queries=input_embed, values=condition_embed)
        x = self.add.forward(input_embed=x, condition_embed=condition_embed, result=x_temp)
        x = self.layernorm.forward(x)

        x_temp = self.feedforwad.forward(x)
        x = self.add.forward(input_embed=x, condition_embed=condition_embed, result=x_temp)
        x = self.layernorm.forward(x)
        return x

class Generator(nn.Module):
    def __init__(self, d_model=768, max_sequence_len=100):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_len = max_sequence_len

        self.embedding = nn.Embedding(5, d_model)
        # 0 -> [START]
        # 1 -> [FEEDFORWARDLAYER]
        # 2 -> [CONVOLUTIONAL2DLAYER], 
        # 3 -> [MAXPOOLING2DLAYER]
        # 4 -> [END]
        # embedding_dim: d_model
        self.positional_encoder = PositionalEncoder(d_model=d_model, max_sequence_len=max_sequence_len)
        self.add_pos_embed = Add_pos_embed()
        self.layers = nn.ModuleList([GeneratorBodyLayer(d_model=d_model, 
                                                        max_sequence_len=max_sequence_len)
                                    for _ in range(5)])
        self.linear = nn.Linear(d_model * max_sequence_len, 4)
        self.softmax = nn.Softmax()
    
    def forward(self, x, condition_embed):
        x_embed = self.embedding(x)
        x_pos = self.positional_encoder.forward(x=x)
        x = self.add_pos_embed.forward(pos=x_pos, inputs_embed=x_embed)

        for layer in self.layers:
            x = layer.forward(x, condition_embed)

        x = x.view(-1, self.d_model * self.max_sequence_len)
        x = self.linear(x)
        x = self.softmax(x)
        return x