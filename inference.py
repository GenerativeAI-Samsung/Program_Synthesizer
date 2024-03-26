import torch
from Model import Model

if __name__ == '__main__':

    text = input("text: ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(device=device).to(device)
    model.load_checkpoint()

    prediction_list = [0] # 0 -> [START]
    last_value = 0

    while (last_value != 4):
        prev_func_list = torch.tensor(prediction_list)
        prediction = model.forward(text=text, prev_func_list=prev_func_list)
        last_value = torch.argmax(prediction)
        prediction_list.append(last_value + 1)
    
    for i, func in enumerate(prediction_list):
        if (func==0):
            print(f"{i}: [START]")
        if (func==1):
            print(f"{i}: [FeedForwardLayer]")
        if (func==2):
            print(f"{i}: [Convolutional2DLayer]")
        if (func==3):
            print(f"{i}: [MaxPooling2DLayer]")
        if (func==4):
            print(f"{i}: [END]")