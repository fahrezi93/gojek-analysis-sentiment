import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

def relabel_5class_with_bert():
    # 1. Load Model
    model_path = "saved_model_indobert_3class"
    print(f"Loading model from {model_path}...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 2. Load Data
    input_file = "data/gojek_scraped_5class_20251206_130028_FINAL_READY.csv"
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Map string labels to numeric for easier comparison
    label_map = {
        'very_negative': 1,
        'negative': 2,
        'neutral': 3,
        'positive': 4,
        'very_positive': 5
    }
    reverse_map = {v: k for k, v in label_map.items()}
    
    if 'score' not in df.columns:
        df['score'] = df['sentiment'].map(label_map)

    # 3. Predict
    print("Predicting with 3-class model...")
    
    batch_size = 32
    texts = df['text'].tolist() if 'text' in df.columns else df['clean_text'].tolist()
    
    predictions = []
    probs = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        preds = torch.argmax(probabilities, dim=-1).cpu().numpy()
        batch_probs = probabilities.cpu().numpy()
        
        predictions.extend(preds)
        probs.extend(batch_probs)

    # 3-class mapping: 0=negative, 1=neutral, 2=positive
    
    # 4. Apply Correction Logic
    new_scores = []
    changes_count = 0
    
    for idx, row in df.iterrows():
        original_score = row['score']
        pred_class = predictions[idx] # 0, 1, 2
        pred_prob = probs[idx][pred_class]
        
        new_score = original_score
        
        # Threshold for overriding
        CONFIDENCE_THRESHOLD = 0.95 
        
        if pred_prob > CONFIDENCE_THRESHOLD:
            # Model is VERY confident
            
            if original_score <= 2: # Originally Negative (1, 2)
                if pred_class == 2: # Model says Positive
                    new_score = 4 # Flip to Positive (conservative 4)
                elif pred_class == 1: # Model says Neutral
                    new_score = 3
                    
            elif original_score >= 4: # Originally Positive (4, 5)
                if pred_class == 0: # Model says Negative
                    new_score = 2 # Flip to Negative (conservative 2)
                elif pred_class == 1: # Model says Neutral
                    new_score = 3
                    
            elif original_score == 3: # Originally Neutral
                if pred_class == 0:
                    new_score = 2
                elif pred_class == 2:
                    new_score = 4
        
        if new_score != original_score:
            changes_count += 1
            
        new_scores.append(new_score)

    df['new_score'] = new_scores
    df['new_sentiment'] = df['new_score'].map(reverse_map)
    
    print(f"\nTotal rows: {len(df)}")
    print(f"Rows changed: {changes_count} ({changes_count/len(df)*100:.2f}%)")
    
    # Show samples
    changes = df[df['score'] != df['new_score']]
    if not changes.empty:
        print("\n--- Sample Changes (BERT-based) ---")
        print(changes[['text', 'sentiment', 'new_sentiment']].head(10).to_string())
    
    # Save
    output_file = "data/gojek_scraped_5class_RELABELED_BERT.csv"
    
    # Finalize dataframe
    df_final = df.copy()
    df_final['sentiment'] = df_final['new_sentiment']
    df_final = df_final.drop(columns=['new_score', 'new_sentiment', 'score'])
    
    df_final.to_csv(output_file, index=False)
    print(f"\nSaved BERT-relabeled data to {output_file}")

if __name__ == "__main__":
    relabel_5class_with_bert()
