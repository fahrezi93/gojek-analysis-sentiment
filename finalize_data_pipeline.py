import pandas as pd
import re
import numpy as np

class IndoBERTPreprocessor:
    def __init__(self):
        pass
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
            
        # 1. Lowercase (Already mostly done, but ensure)
        text = text.lower()
        
        # 2. Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 3. Remove User Mentions (@user) and Hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # 4. Remove Numbers (Subjective, but usually good for sentiment unless rating specific)
        text = re.sub(r'\d+', '', text)
        
        # 5. Remove Repeated Characters (e.g. "bangeeet" -> "banget") - Limit to 2 chars max
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # 6. Remove Punctuation/Special Chars (Keep basic ones if needed, but BERT usually handles clean text better)
        # Leaving spaces and basic letters. Removing emojis usually recommended unless using emoji-aware model.
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 7. Remove single letters (surplus cleaning, like 'yg k', remove 'k') - Skip for now, risky.
        
        # 8. Extra Whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

def balance_dataset(df, target_col='sentiment'):
    """
    Balances the dataset by undersampling majority classes to match the minority class count.
    """
    # Get count of minimum class
    min_count = df[target_col].value_counts().min()
    print(f"   Target per class: {min_count}")
    
    balanced_dfs = []
    for label in df[target_col].unique():
        df_class = df[df[target_col] == label]
        
        # Sample min_count
        if len(df_class) >= min_count:
            df_resampled = df_class.sample(n=min_count, random_state=42)
        else:
            # Should not happen if min_count is truly min, but safe fallback
            df_resampled = df_class 
            
        balanced_dfs.append(df_resampled)
        
    df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_balanced

def process_pipeline():
    base_dir = r"d:\Skripsi\sentiment-analyst-ojol-review\data"
    files = [
        "gojek_scraped_3class_20251206_130028_cleaned.csv",
        "gojek_scraped_5class_20251206_130028_cleaned.csv"
    ]
    
    preprocessor = IndoBERTPreprocessor()
    
    for filename in files:
        file_path = os.path.join(base_dir, filename)
        if not os.path.exists(file_path):
            continue
            
        print(f"\nProcessing {filename}...")
        df = pd.read_csv(file_path)
        
        # 1. CLEANING
        print("   Status: Cleaning Text (IndoBERT Standard)...")
        # Ensure we use the corrected sentiment column from previous step
        # If 'sentiment' is accurate, use it.
        
        df['text_clean'] = df['text'].apply(preprocessor.clean_text)
        
        # Filter empty strings after cleaning
        df = df[df['text_clean'].str.len() > 3] # Remove very short rubbish
        
        # 2. BALANCING
        print("   Status: Balancing Classes...")
        df_balanced = balance_dataset(df, target_col='sentiment')
        
        print("   Final Distribution:")
        print(df_balanced['sentiment'].value_counts())
        
        # Save Final Ready-to-Train Data
        final_filename = filename.replace('_cleaned.csv', '_FINAL_READY.csv')
        output_path = os.path.join(base_dir, final_filename)
        
        # Keep only text_clean (renamed to text) and label
        final_df = df_balanced[['text_clean', 'sentiment']].rename(columns={'text_clean': 'text'})
        final_df.to_csv(output_path, index=False)
        print(f"   Saved: {final_filename}")

if __name__ == "__main__":
    import os
    process_pipeline()
