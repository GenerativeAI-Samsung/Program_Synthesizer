import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel 

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
    
    def forward(self, inputs, device):
        x = self.tokenizer(inputs, return_tensors="pt", padding='max_length', max_length=512, truncation=True).to(device)
        x = self.model(**x).last_hidden_state
        return x

class Conditioner(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.encoder = Encoder()    # Encoder: bert-base-uncased
    
    def forward(self, inputs):
        x = self.encoder.forward(inputs, self.device)
        return x