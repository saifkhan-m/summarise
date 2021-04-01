import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BertForClassification():
    def __init__(self):
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return outputs


class LinearRegression(nn.Module):
  def __init__(self,):
    super(LinearRegression,self).__init__()
    self.lin = nn.Linear(768, 3)

  def forward(self, x):
    return self.lin(x)