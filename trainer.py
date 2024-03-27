import json

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datapreprocess import dataprocess, CustomDataset, custom_collate_func
from Model import Model

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup = input("Already has checkpoint and history_file? (yes/no)")
    buf = input()
    setting = input("Chose setting (freeze/continue):")
    if (setting=="freeze"):
        model = Model(device=device, freeze=True).to(device)

        if (setup=="yes"):
            model.load_checkpoint()
            for param in model.conditioner.encoder.parameters():
                param.requires_grad = False

        num_eps = 3
    if (setting=="continue"):
        model = Model(device=device).to(device)

        if (setup=="yes"):
            model.load_checkpoint()

        for param in model.conditioner.encoder.parameters():
            param.requires_grad = True

        num_eps = 5
    print(f"training with setting: {setting}")
    print(f"loading data...")
    with open("/content/Program_Synthesizer/data_manimML.json") as f:
        data = json.load(f)

    print(f"loading history file...")
    if (setup=="yes"):
        with open("/content/drive/MyDrive/program_synthesizer/history.json") as f:
            history = json.load(f)
        
        train_adapter_epochs = history["train_adapter_epochs"]
        train_adapter_losses = history["train_adapter_losses"]
        val_adapter_losses = history["val_adapter_losses"]
        train_epochs = history["train_epochs"]
        train_losses = history["train_losses"]
        val_losses = history["val_losses"]
    else:
        train_adapter_epochs = 0
        train_adapter_losses = []
        val_adapter_losses = []
        train_epochs = 0
        train_losses = []
        val_losses = []

    # Hyperparameter
    batch_size = 8
    lr = 5e-5

    print(f"number of parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    loss = nn.CrossEntropyLoss()

    optim = AdamW(model.parameters(), lr=lr)
    processed_data = dataprocess(data)

    training_len = round(len(processed_data) * 0.9)

    train_data = processed_data[:training_len]
    val_data = processed_data[training_len:]

    print(f"total data: {len(processed_data)}")
    print(f"train data: {len(train_data)}")
    print(f"validation data: {len(val_data)}")

    train_data = CustomDataset(train_data) 
    val_data = CustomDataset(val_data)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_func, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=1, collate_fn=custom_collate_func, drop_last=True)
    print("start training")
    for epoch in range(num_eps):
        num_batch_train = 0
        num_batch_val = 0
        train_loss = 0
        val_loss = 0
        for i, batch in enumerate(train_dataloader):
            optim.zero_grad()
            logits = model.forward(text=batch["text"], prev_func_list=batch["prev_func_list"])

            label = torch.tensor(batch["label"]).to(device)
            loss_value = loss(logits.float(), label.float())
            loss_value.backward()

            train_loss += loss_value
            num_batch_train += 1

            print(f"Epoch: {epoch}, step: {i}, loss: {loss_value}")
            optim.step()

        mean_loss_train = train_loss/num_batch_train

        print(f"starting validation for epoch {epoch}...")
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                optim.zero_grad()
                logits = model.forward(text=batch["text"], prev_func_list=batch["prev_func_list"])

                label = torch.tensor(batch["label"]).to(device)
                loss_value = loss(logits.float(), label.float())

                val_loss += loss_value
                num_batch_val += 1

                print(f"Epoch validation: {epoch}, step: {i}, loss: {loss_value}, prediction: {logits}")
            
        mean_loss_val = val_loss/num_batch_val

        if (setting=="freeze"):
            train_adapter_epochs += 1
            train_adapter_losses.append(mean_loss_train.item())
            val_adapter_losses.append(mean_loss_val.item())
        if (setting=="continue"):
            train_epochs += 1
            train_losses.append(mean_loss_train.item())
            val_losses.append(mean_loss_val.item())
    print("saving checkpoint...")
    model.save_checkpoint()

    print("saving history training...")
    history = {"train_adapter_epochs": train_adapter_epochs, "train_adapter_losses": train_adapter_losses, "val_adapter_losses": val_adapter_losses, "train_epochs": train_epochs, "train_losses": train_losses, "val_losses": val_losses}
    json_object = json.dumps(history)
    with open("/content/drive/MyDrive/program_synthesizer/history.json", "w") as outfile:
        outfile.write(json_object)