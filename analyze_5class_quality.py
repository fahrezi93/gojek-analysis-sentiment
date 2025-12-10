import pandas as pd
import numpy as np

def analyze_5class_quality(file_path):
    print(f"Analyzing file: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("File not found.")
        return

    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Map sentiment to score if 'score' doesn't exist
    if 'score' not in df.columns and 'sentiment' in df.columns:
        label_map = {
            'very_negative': 1,
            'negative': 2,
            'neutral': 3,
            'positive': 4,
            'very_positive': 5
        }
        df['score'] = df['sentiment'].map(label_map)
    
    # Use 'text' if 'clean_text' doesn't exist
    text_col = 'clean_text' if 'clean_text' in df.columns else 'text'
    print(f"Using text column: {text_col}")

    # Define keywords
    positive_keywords = ['bagus', 'mantap', 'keren', 'suka', 'puas', 'ok', 'oke', 'good', 'best', 'cepat', 'membantu']
    negative_keywords = ['jelek', 'buruk', 'kecewa', 'parah', 'lambat', 'mahal', 'susah', 'gagal', 'error', 'lelet', 'hapus', 'uninstal']

    # Check for mismatches
    
    # Case 1: Low Rating (1-2) but Positive Text
    low_rating_positive_text = df[
        (df['score'].isin([1, 2])) & 
        (df[text_col].str.contains('|'.join(positive_keywords), case=False, na=False))
    ]
    
    # Case 2: High Rating (4-5) but Negative Text
    high_rating_negative_text = df[
        (df['score'].isin([4, 5])) & 
        (df[text_col].str.contains('|'.join(negative_keywords), case=False, na=False))
    ]

    # Case 3: Neutral Rating (3) with strong sentiment
    neutral_positive = df[
        (df['score'] == 3) & 
        (df[text_col].str.contains('|'.join(positive_keywords), case=False, na=False))
    ]
    neutral_negative = df[
        (df['score'] == 3) & 
        (df[text_col].str.contains('|'.join(negative_keywords), case=False, na=False))
    ]

    print("\n--- POTENTIAL MISMATCHES ---")
    print(f"1. Low Rating (1-2) with Positive Words: {len(low_rating_positive_text)} rows ({len(low_rating_positive_text)/len(df)*100:.2f}%)")
    print(f"2. High Rating (4-5) with Negative Words: {len(high_rating_negative_text)} rows ({len(high_rating_negative_text)/len(df)*100:.2f}%)")
    print(f"3. Neutral Rating (3) with Positive Words: {len(neutral_positive)} rows")
    print(f"4. Neutral Rating (3) with Negative Words: {len(neutral_negative)} rows")

    print("\n--- SAMPLES (Low Rating + Positive Text) ---")
    print(low_rating_positive_text[[text_col, 'sentiment', 'score']].head(10).to_string())

    print("\n--- SAMPLES (High Rating + Negative Text) ---")
    print(high_rating_negative_text[[text_col, 'sentiment', 'score']].head(10).to_string())

if __name__ == "__main__":
    file_path = "data/gojek_scraped_5class_20251206_130028_FINAL_READY.csv"
    analyze_5class_quality(file_path)
