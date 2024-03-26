import torch
import torch.nn as nn

from Generator import Generator
from Conditioner import Conditioner

class Model(nn.Module):
    def __init__(self, device, freeze=False):
        super().__init__()
        self.generator = Generator(device=device)
        self.conditioner = Conditioner(device=device, freeze=True)
        self.device = device
    
    def forward(self, text, prev_func_list):
        condition_embed = self.conditioner.forward(text).to(self.device)
        prev_func_list = torch.tensor(prev_func_list).to(self.device)
        prediction = self.generator.forward(prev_func_list, condition_embed)
        return prediction

    def save_checkpoint(self, director="/content/drive/MyDrive/program_synthesizer/model.pt"):
        torch.save(self.state_dict(), director)

    def load_checkpoint(self, director="/content/drive/MyDrive/program_synthesizer/model.pt"):
        self.load_state_dict(torch.load(director)) 