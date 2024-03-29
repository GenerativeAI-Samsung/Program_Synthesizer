import torch
from Model import Model

if __name__ == '__main__':

    device_name = input("What device do you have? (cpu/gpu): ")
    buf = input()
    text = input("Text input: ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(device=device).to(device)
    print("loading checkpoint...")
    model.load_checkpoint()
    print("done!")
    prediction_list = [0] # 0 -> [START]
    last_value = 0
    print("running...")
    with torch.no_grad():
        while ((last_value + 1) != 4):
            prev_func_list = torch.tensor(prediction_list + [5 for _ in range(16 - len(prediction_list))]).unsqueeze(0).to(device)
            prediction = model.forward(text=text, prev_func_list=prev_func_list)
            last_value = torch.argmax(prediction).item()
            prediction_list.append(last_value + 1)
    print("finishing...")
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