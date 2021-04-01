import pandas as pd
import numpy as np

import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import nltk
from sec.util import LinearRegression,BertForClassification
from sec.EdgarDataset import EdgarDataset
import torch
import torch.nn as nn
import gc
from tqdm import tqdm
nltk.download('punkt')
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 5, 4

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

train_data = pd.read_csv('DowJones_10K_012.csv')
train_data.drop('Unnamed: 0', axis=1, inplace=True)
train_data = train_data.replace(np.nan, '', regex=True)
data_legal = train_data[['Legal Proceedings','label']]
data_legal = data_legal[data_legal['Legal Proceedings'].notna()]

train_data_legal, valid_data_legal= train_test_split(data_legal, test_size=0.20, random_state=RANDOM_SEED, shuffle=True)
#valid_data_legal, test_data_legal= train_test_split(valid_data_legal, test_size=0.50, random_state=RANDOM_SEED, shuffle=True)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


model = LinearRegression()
learning_rate = 0.001
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
bertLayer = BertForClassification()

edgarDataset = EdgarDataset(train_data_legal,tokenizer=tokenizer)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def train():
    for epoch in range(1):
        for j in range(2):#train_data_legal.shape[0]):
            # print(j)
            legali = edgarDataset.__getitem__(j)
            bert_out_cls = []
            Y = torch.tensor([legali['targets'][0].item()]).to(device)
            # print(legali)
            for i in range(legali['len'][0].item()):
                input_ids = legali['ids'][i].reshape(-1, 500)
                attention_mask = legali['mask'][i].reshape(-1, 500)
                token_type_ids = legali['token_type_ids'][i].reshape(-1, 500)
                # print(token_type_ids.reshape(-1,500))

                bert_out = bertLayer.forward(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             token_type_ids=token_type_ids)

                bert_out_cls.append(bert_out.last_hidden_state[0][0])

            mean_cls = torch.mean(torch.stack(bert_out_cls), 0).to(device)

            y_pred = model(mean_cls)
            # print(y_pred)
            # print(Y)

            l = loss(y_pred.reshape(1, -1), Y)

            l.backward()

            optimizer.step()

            optimizer.zero_grad()
            gc.collect()
            # 1/0
            if j % 10 == 0:
                print(f'Training example : {j + 1}  and loss: {l:.8f}')
    torch.save(model.state_dict(), 'checkpoint.pth')

def eval():
    model = torch.load('model')
    model.to(device)
    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_total = 0
        for j in tqdm(range(valid_data_legal.shape[0])):
            legali = edgarDataset.__getitem__(j)
            bert_out_cls = []
            Y = torch.tensor([legali['targets'][0].item()])
            for i in range(legali['len'][0].item()):
                input_ids = legali['ids'][i].reshape(-1, 500)
                attention_mask = legali['mask'][i].reshape(-1, 500)
                token_type_ids = legali['token_type_ids'][i].reshape(-1, 500)

                bert_out = bertLayer.forward(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             token_type_ids=token_type_ids)

                bert_out_cls.append(bert_out.last_hidden_state[0][0])

            mean_cls = torch.mean(torch.stack(bert_out_cls), 0)  # .to(device)

            output = model(mean_cls)
            # value , label
            _, predictions = torch.max(output.reshape(1, -1), 1)
            n_total += Y.shape[0]
            n_correct += (predictions == Y).sum().item()

        acc = 100 * n_correct / n_total
        return acc

print(eval())