import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_md(content):
    notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": content.split("\n")})

def add_code(content):
    notebook["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": content.split("\n")})

# ===== CELLS =====
add_md("""# 🚀 IndoBERT Sentiment Analysis - High Accuracy & No Overfitting

**Dataset**: gojek_reviews_relabelled_text_based.csv  
**Model**: IndoBERT (indobenchmark/indobert-base-p1)  
**Target**: Akurasi tinggi tanpa overfitting

### Anti-Overfitting Techniques:
1. Data balancing (undersampling)
2. Dropout regularization (0.3)
3. Label smoothing (0.1)
4. Early stopping (patience=3)
5. Weight decay (L2 regularization)
6. Learning rate warmup
7. Gradient clipping""")

add_code("""# Install dependencies
!pip install transformers torch pandas numpy scikit-learn matplotlib seaborn tqdm -q""")

add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.utils import resample
import random
import os
import copy

warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')""")

add_md("## 1. Load & Clean Data")

add_code("""# Load data
DATA_PATH = 'data/gojek_reviews_relabelled_text_based.csv'
df = pd.read_csv(DATA_PATH)

print('='*60)
print('ORIGINAL DATA')
print('='*60)
print(f'Total: {len(df):,}')
print(df['sentiment'].value_counts())

# === DATA CLEANING ===
# 1. Remove ambiguous: score 3 dengan label positive
df_clean = df[~((df['score'] == 3) & (df['sentiment'] == 'positive'))].copy()

# 2. Remove duplicates
df_clean = df_clean.drop_duplicates(subset=['content'])

# 3. Remove very short texts (< 4 words)
df_clean['word_count'] = df_clean['content'].astype(str).apply(lambda x: len(x.split()))
df_clean = df_clean[df_clean['word_count'] >= 4]

print('\\n' + '='*60)
print('AFTER CLEANING')
print('='*60)
print(f'Total: {len(df_clean):,}')
print(df_clean['sentiment'].value_counts())""")

add_md("## 2. Balance Data (Undersampling)")

add_code("""# Balance dengan undersampling ke kelas terkecil
min_count = df_clean['sentiment'].value_counts().min()
print(f'Min class count: {min_count}')

df_balanced = pd.DataFrame()
for sentiment in ['negative', 'neutral', 'positive']:
    df_class = df_clean[df_clean['sentiment'] == sentiment]
    df_sampled = resample(df_class, replace=False, n_samples=min_count, random_state=42)
    df_balanced = pd.concat([df_balanced, df_sampled])

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print('\\n' + '='*60)
print('BALANCED DATA')
print('='*60)
print(f'Total: {len(df_balanced):,}')
print(df_balanced['sentiment'].value_counts())

# Visualize
plt.figure(figsize=(8, 4))
df_balanced['sentiment'].value_counts().plot(kind='bar', color=['red', 'gray', 'green'])
plt.title('Balanced Sentiment Distribution')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()""")

add_md("## 3. Prepare Labels & Split Data")

add_code("""# Label mapping
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
label_names = ['negative', 'neutral', 'positive']
df_balanced['label'] = df_balanced['sentiment'].map(sentiment_mapping)

# Split: 70% train, 15% val, 15% test (stratified)
train_df, temp_df = train_test_split(df_balanced, test_size=0.3, random_state=42, stratify=df_balanced['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

print(f'Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}')
print(f'\\nTrain distribution: {dict(train_df["label"].value_counts().sort_index())}')""")

add_md("## 4. Tokenizer & Dataset")

add_code("""MODEL_NAME = 'indobenchmark/indobert-base-p1'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
print(f'Tokenizer: {MODEL_NAME}')

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }""")

with open('sentiment_training_indobert.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print("Part 1 done!")
