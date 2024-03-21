import torch
import torch.nn as nn

class WeirdAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.queries_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.values_linear = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax()
    
    def forward(self, queries, key, values):
        queries = self.queries_linear(queries)
        key = self.key_linear(key)
        values = self.values_linear(values)

        x = torch.matmul(self.softmax(torch.matmul(queries, torch.transpose(key, 0, 1))/torch.sqrt(self.d_model)), values)
        return x

class Add(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(1)
        self.beta = nn.Parameter(1)
        self.alpha = nn.Parameter(1)

    def forward(self, input_embed, condition_embed, result):
        x = self.gamma * input_embed + self.beta * condition_embed + self.alpha * result 

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

class GeneratorBodyLayer(nn.Module):
    def __init__(self, d_model=768, max_sequence_len=100):
        super().___init__()
        self.weirdattention = WeirdAttention(d_model=d_model)
        self.add = Add()
        self.layernorm = LayerNorm(d_model=d_model)
        self.feedforwad = FeedForwad(d_model=d_model, max_sequence_len=max_sequence_len)
    
    def forward(self, input_embed, condition_embed, last_token_predicted):
        x_temp = self.weirdattention.forward(key=last_token_predicted, queries=input_embed, values=input_embed)
        x = self.add.forward(input_embed=x, condition_embed=condition_embed, result=x_temp)
        x = self.layernorm.forward(x)

        x_temp = self.feedforwad.forward(x)
        x = self.add.forward(input_embed=x, condition_embed=condition_embed, result=x_temp)
        x = self.layernorm.forward(x)
        return x

# self.embedding = nn.Embedding(5, 768) 
# # 0 -> [START], 1 -> [FEEDFORWARDLAYER], 2 -> [CONVOLUTIONAL2DLAYER], 
# # 3 -> [MAXPOOLING2DLAYER], 4 -> [END]
# # num_embeddings: 5
# # embedding_dim: 768