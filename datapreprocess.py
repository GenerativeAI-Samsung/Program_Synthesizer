from torch.utils.data import Dataset

def dataprocess(raw_data):
    processed_data = []
    for sample in raw_data:
        text = sample["text"]

        functions = sample["functions_list"]
        functions.append("[END]")
        for i, value in enumerate(functions):
            prev_func_list = [0] # 0 -> [START]
            for func in functions[:i]:
                if (func=="Convolutional2DLayer"):
                    prev_func_list.append(2) # 2 -> [CONVOLUTIONAL2DLAYER]
                if (func=="MaxPooling2DLayer"):
                    prev_func_list.append(3) # 3 -> [MAXPOOLING2DLAYER]
                if (func=="FeedForwardLayer"):
                    prev_func_list.append(1) # 1 -> [FEEDFORWARDLAYER]
                if (func=="[END]"):
                    prev_func_list.append(4) # 4 -> [END]
            
            label = []
            if (value=="Convolutional2DLayer"):
                label = [0, 1, 0, 0]
            if (value=="FeedForwardLayer"):
                label = [1, 0, 0, 0]
            if (value=="MaxPooling2DLayer"):
                label = [0, 0, 1, 0]
            if (value=="[END]"):
                label = [0, 0, 0, 1]
            processed_sample = {"text": text, "prev_func_list": prev_func_list, "label": label}
            processed_data.append(processed_sample)
    return processed_data

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate_func(batch):
    text = []
    prev_func_list = []
    label = []

    for sample in batch:
        text.append(sample["text"])
        prev_func_list.append(sample["prev_func_list"])
        label.append(sample["label"])
    return {"text": text, "prev_func_list": prev_func_list, "label": label}