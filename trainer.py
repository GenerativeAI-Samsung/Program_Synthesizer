import json

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datapreprocess import dataprocess, CustomDataset, custom_collate_func
from Model import Model

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setting = input("Chose setting (freeze/continue/new):")
    if (setting=="freeze"):
        model = Model(device=device, freeze=True).to(device)
        num_eps = 3
    if (setting=="continue"):
        model = Model(device=device).to(device)
        model.load_checkpoint()
        num_eps = 5
    if (setting=="new"):
        model = Model(device=device).to(device)
        num_eps = 10
    print(f"training with setting: {setting}")
    print(f"loading data...")
    with open("/content/Program_Synthesizer/data_manimML.json") as f:
        train_data = json.load(f)

    # Hyperparameter
    batch_size = 8
    lr = 5e-5

    print(f"number of parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    loss = nn.CrossEntropyLoss()

    optim = AdamW(model.parameters(), lr=lr)
    processed_data = dataprocess(train_data)
    processed_data = CustomDataset(processed_data) 
    
    dataloader = DataLoader(processed_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_func, drop_last=True)
    print("start training")
    for epoch in range(num_eps):
        for i, batch in enumerate(dataloader):
            optim.zero_grad()
            logits = model.forward(text=batch["text"], prev_func_list=batch["prev_func_list"])

            label = torch.tensor(batch["label"]).to(device)
            loss_value = loss(logits.float(), label.float())
            loss_value.backward()

            print(f"Epoch: {epoch}, Batch: {i}, loss: {loss_value}")
            optim.step()
    print("save checkpoint")
    model.save_checkpoint()