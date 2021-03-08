import torch
from torch.utils.data import Dataset

class DatasetYelp(Dataset):
    def __init__(self ,data, tokenizer):

        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index, 0]
        label = self.data.iloc[index, 1]
        print(text)
        print(label)

        encoding  = self.tokenizer(str(text), padding = 'max_length', truncation=True, max_length = 512)
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long),
            'text': text,
        }