
import json
import os
import uuid

def create_notebook(filename, num_classes, data_path):
    
    # Define Label Mappings
    if num_classes == 3:
        label_map = "{'negative': 0, 'neutral': 1, 'positive': 2}"
        id2label = "{0: 'negative', 1: 'neutral', 2: 'positive'}"
        label2id = "{'negative': 0, 'neutral': 1, 'positive': 2}"
        class_names = "['negative', 'neutral', 'positive']"
    else:
        label_map = "{'very_negative': 0, 'negative': 1, 'neutral': 2, 'positive': 3, 'very_positive': 4}"
        id2label = "{0: 'very_negative', 1: 'negative', 2: 'neutral', 3: 'positive', 4: 'very_positive'}"
        label2id = "{'very_negative': 0, 'negative': 1, 'neutral': 2, 'positive': 3, 'very_positive': 4}"
        class_names = "['very_negative', 'negative', 'neutral', 'positive', 'very_positive']"

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# Training IndoBERT Sentiment Analysis ({num_classes} Classes)\n",
                "\n",
                "## Objectives\n",
                "- **Model:** `indobenchmark/indobert-base-p1`\n",
                "- **Task:** Sentiment Classification\n",
                "- **Optimization:** GPU (Mixed Precision fp16), Early Stopping, Best Model Loading\n",
                "- **Target:** High Accuracy without Overfitting"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# 1. Setup & Imports\n",
                "!pip install transformers datasets scikit-learn accelerate torch pandas seaborn matplotlib\n",
                "\n",
                "import torch\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report\n",
                "import seaborn as sns\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Check GPU\n",
                "device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'\n",
                "print(f\"Using Device: {device_name}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# 2. Load Data & Preprocessing\n",
                f"file_path = r'{data_path}'\n",
                "df = pd.read_csv(file_path)\n",
                "\n",
                "# Label Mapping\n",
                f"label_map = {label_map}\n",
                "df['label'] = df['sentiment'].map(label_map)\n",
                "\n",
                "# Split Data (Stratified to keep balance)\n",
                "# 80% Train, 10% Validation, 10% Test\n",
                "train_texts, temp_texts, train_labels, temp_labels = train_test_split(\n",
                "    df['text'].tolist(), \n",
                "    df['label'].tolist(), \n",
                "    test_size=0.2, \n",
                "    stratify=df['label'], \n",
                "    random_state=42\n",
                ")\n",
                "\n",
                "val_texts, test_texts, val_labels, test_labels = train_test_split(\n",
                "    temp_texts, \n",
                "    temp_labels, \n",
                "    test_size=0.5, \n",
                "    stratify=temp_labels, \n",
                "    random_state=42\n",
                ")\n",
                "\n",
                "print(f\"Train Size: {len(train_texts)}\")\n",
                "print(f\"Val Size:   {len(val_texts)}\")\n",
                "print(f\"Test Size:  {len(test_texts)}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# 3. Tokenization\n",
                "model_name = 'indobenchmark/indobert-base-p1'\n",
                "tokenizer = BertTokenizer.from_pretrained(model_name)\n",
                "\n",
                "def tokenize_data(texts, labels):\n",
                "    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)\n",
                "    dataset = []\n",
                "    for i in range(len(texts)):\n",
                "        item = {key: torch.tensor(val[i]) for key, val in encodings.items()}\n",
                "        item['labels'] = torch.tensor(labels[i])\n",
                "        dataset.append(item)\n",
                "    return dataset\n",
                "\n",
                "train_dataset = tokenize_data(train_texts, train_labels)\n",
                "val_dataset = tokenize_data(val_texts, val_labels)\n",
                "test_dataset = tokenize_data(test_texts, test_labels)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# 4. Metrics Function\n",
                "def compute_metrics(pred):\n",
                "    labels = pred.label_ids\n",
                "    preds = pred.predictions.argmax(-1)\n",
                "    acc = accuracy_score(labels, preds)\n",
                "    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')\n",
                "    return {\n",
                "        'accuracy': acc,\n",
                "        'f1': f1,\n",
                "        'precision': precision,\n",
                "        'recall': recall\n",
                "    }"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# 5. Training Configuration\n",
                f"id2label = {id2label}\n",
                f"label2id = {label2id}\n",
                "\n",
                f"model = BertForSequenceClassification.from_pretrained(\n",
                "    model_name, \n",
                f"    num_labels={num_classes}, \n",
                "    id2label=id2label, \n",
                "    label2id=label2id\n",
                ")\n",
                "\n",
                "# TRAINING ARGUMENTS (Optimized)\n",
                "training_args = TrainingArguments(\n",
                f"    output_dir='./results_{num_classes}class',\n",
                "    num_train_epochs=5,              # Max epochs (Early stopping will likely stop earlier)\n",
                "    per_device_train_batch_size=16,  # 16 is standard for 6-8GB VRAM\n",
                "    per_device_eval_batch_size=32,\n",
                "    gradient_accumulation_steps=2,   # Virtual batch size = 32\n",
                "    learning_rate=2e-5,              # Low LR to prevent overfitting\n",
                "    weight_decay=0.01,               # Regularization\n",
                "    warmup_ratio=0.1,\n",
                "    eval_strategy=\"epoch\",         # Check val every epoch\n",
                "    save_strategy=\"epoch\",           # Save model every epoch\n",
                "    load_best_model_at_end=True,     # ALWAYS load the best model found\n",
                "    metric_for_best_model=\"accuracy\",\n",
                "    fp16=True,                       # GPU Acceleration (Mixed Precision)\n",
                "    logging_dir='./logs',\n",
                "    logging_steps=50,\n",
                "    dataloader_num_workers=0         # Set >0 if on Linux, 0 on Windows usually safer\n",
                ")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# 6. Initialize Trainer\n",
                "trainer = Trainer(\n",
                "    model=model,\n",
                "    args=training_args,\n",
                "    train_dataset=train_dataset,\n",
                "    eval_dataset=val_dataset,\n",
                "    compute_metrics=compute_metrics,\n",
                "    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # Stop if no improve for 2 epochs\n",
                ")\n",
                "\n",
                "# START TRAINING\n",
                "trainer.train()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# 7. Evaluation on Test Set\n",
                "print(\"Evaluating on Test Set...\")\n",
                "test_result = trainer.predict(test_dataset)\n",
                "print(test_result.metrics)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# 8. Confusion Matrix & Report\n",
                "y_preds = np.argmax(test_result.predictions, axis=1)\n",
                "y_true = test_result.label_ids\n",
                "\n",
                f"print(classification_report(y_true, y_preds, target_names={class_names}))\n",
                "\n",
                "# Plot CM\n",
                "cm = confusion_matrix(y_true, y_preds)\n",
                "plt.figure(figsize=(8,6))\n",
                "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \n",
                f"            xticklabels={class_names}, \n",
                f"            yticklabels={class_names})\n",
                "plt.xlabel('Predicted')\n",
                "plt.ylabel('Actual')\n",
                "plt.title('Confusion Matrix')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# 9. Save Final Model\n",
                f"save_path = './saved_model_indobert_{num_classes}class'\n",
                "model.save_pretrained(save_path)\n",
                "tokenizer.save_pretrained(save_path)\n",
                "print(f\"Model saved to {save_path}\")"
            ]
        }
    ]

    # Add IDs to cells for nbformat 4.5+
    for cell in cells:
        cell["id"] = str(uuid.uuid4())

    notebook_content = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=1) # Minimal indent to look like native ipynb
    print(f"Created {filename}")

if __name__ == "__main__":
    base = r"d:\Skripsi\sentiment-analyst-ojol-review\data"
    
    # 3 Class
    create_notebook(
        filename="Training_IndoBERT_3Class.ipynb", 
        num_classes=3, 
        data_path=os.path.join(base, "gojek_scraped_3class_20251206_130028_FINAL_READY.csv").replace("\\", "/")
    )
    
    # 5 Class
    create_notebook(
        filename="Training_IndoBERT_5Class.ipynb", 
        num_classes=5, 
        data_path=os.path.join(base, "gojek_scraped_5class_20251206_130028_FINAL_READY.csv").replace("\\", "/")
    )
