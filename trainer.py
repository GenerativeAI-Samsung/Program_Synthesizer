import json

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataloader

from datapreprocess import dataprocess, CustomDataset, custom_collate_func
from Model import Model

if __name__ == '__main__':

    with open("/content/data.json") as f:
        train_data = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameter
    batch_size = 8
    lr = 5e-5
    num_eps = 5

    model = Model().to(device)

    loss = nn.CrossEntropyLoss()

    optim = AdamW(model.parameters(), lr=lr)
    processed_data = dataprocess(train_data)

    dataloader = Dataloader(processed_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_func, drop_last=True)

    for epoch in range(num_eps):
        for i, batch in enumerate(dataloader):
            optim.zero_grad()
            logits = model.forward(text=batch["text"], prev_func_list=batch["prev_func_list"])

            label = torch.tensor(batch["label"]).to(device)
            loss_value = loss(logits.float(), label.float())
            loss_value.backward()

            if (i % 10 == 0):
                print(f"Epoch: {epoch}, Batch: {i}, loss: {loss_value}")
            optim.step()