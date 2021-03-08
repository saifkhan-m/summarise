import os
import math
import datetime

from tqdm import tqdm

import pandas as pd
import numpy as np

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from transformers import  Trainer, TrainingArguments
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import os
import copy
from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig, RobertaTokenizerFast, RobertaForSequenceClassification
import re
import nltk
from nltk.tokenize import sent_tokenize

import util_files

nltk.download('punkt')
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def clean_content(content):
  #print(content[:90])
  content = re.sub(r"&\S*;", r"", content)
  content = re.sub(r"[©®™]", r"",content)
  content = re.sub(r"""((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|(([^\s()<>]+|(([^\s()<>]+)))*))+(?:(([^\s()<>]+|(([^\s()<>]+)))*)|[^\s`!()[]{};:'".,<>?«»“”‘’]))""", '', content, flags=re.MULTILINE)
  content = content.replace("..........",'').replace("\n", " ").replace('ADVERTISEMENT','').strip()
  return content


def get_dataset():
    train_data = load_dataset("yelp_polarity", split='train').shuffle(seed=42).select(range(100000))
    test_data = load_dataset("yelp_polarity", split='test').shuffle(seed=42).select(range(10000))
def get_tokenizer():
    return RobertaTokenizerFast.from_pretrained('roberta-base', max_length=512)

def get_model(type='trained', path='../models/yelp'):
    if type=='trained':
        if os.path.exists(path):
            model = RobertaForSequenceClassification.from_pretrained(path)
        else:
            util_files.util.download_model('yelp')
            model = RobertaForSequenceClassification.from_pretrained(path)
    else:
        RobertaForSequenceClassification.from_pretrained('roberta-base')
    return model


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def get_prediction_args():
    return TrainingArguments(
        output_dir='results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        disable_tqdm=False,
        load_best_model_at_end=True,
        warmup_steps=200,
        weight_decay=0.01,
        logging_steps=4,
        fp16=True,
        logging_dir='logs',
        dataloader_num_workers=0,
        run_name='longformer-classification'
    )

def get_trainer():
    trainer2 = Trainer(
        model=get_model('yelp'),
        args=get_prediction_args(),
        compute_metrics=compute_metrics
    )
