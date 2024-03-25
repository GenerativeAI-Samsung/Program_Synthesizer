import torch
import torch.nn as nn

from Generator import Generator
from Conditioner import Conditioner

class Model(nn.Module):
    def __init__(self):
        self.generator = Generator()
        self.conditioner = Conditioner()
    
    def __forward__(self, text, prev_func_list):
        condition_embed = self.conditioner.forward(text)
        prediction = self.generator.forward(prev_func_list, condition_embed)
        return prediction