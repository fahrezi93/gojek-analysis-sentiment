"""
MASTER DATA PROCESSOR
Proses lengkap: scraping -> cleaning -> labeling -> splitting (3 class & 5 class)
Output: Data siap training untuk IndoBERT dengan akurasi tinggi
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from text_cleaner_indobert import (
    clean_text_indobert, 
    validate_text_quality,
    correct_sentiment_label,
    calculate_sentiment_score
)

# ============================================
# CONFIGURATION
# ============================================
INPUT_FILE = 'data/gojek_reviews_75k_FIXED_BALANCED.csv'
OUTPUT_DIR = 'data/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# SENTIMENT MAPPING
# ============================================
def map_rating_to_sentiment_3class(rating: int, text: str = "") -> str:
    """
    Map rating to 3-class sentiment (negative, neutral, positive)
    With text analysis for better accuracy
    """
    if text:
        scores = calculate_sentiment_score(text)
        
        # Rating 3 is ambiguous - use text analysis
        if rating == 3:
            if scores['positive'] > 0.5 and scores['negative'] < 0.2:
                return 'positive'
            elif scores['negative'] > 0.5 and scores['positive'] < 0.2:
                return 'negative'
            else:
                return 'neutral'
    
    # Standard mapping
    if rating <= 2:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    else:  # 4-5
        return 'positive'

def map_rating_to_sentiment_5class(rating: int, text: str = "") -> str:
    """
    Map rating to 5-class sentiment (very_negative, negative, neutral, positive, very_positive)
    More granular classification
    """
    if text:
        scores = calculate_sentiment_score(text)
        
        # Rating 1: Check if VERY negative
        if rating == 1:
            if scores['negative'] > 0.7:
                return 'very_negative'
            else:
                return 'negative'
        
        # Rating 2: Negative (but check context)
        if rating == 2:
            if scores['negative'] > 0.6:
                return 'negative'
            elif scores['positive'] > 0.4:
                return 'neutral'  # Mixed sentiment
            else:
                return 'negative'
        
        # Rating 3: Most ambiguous
        if rating == 3:
            if scores['positive'] > 0.5:
                return 'positive'
            elif scores['negative'] > 0.5:
                return 'negative'
            else:
                return 'neutral'
        
        # Rating 4: Positive
        if rating == 4:
            if scores['positive'] > 0.6:
                return 'positive'
            elif scores['negative'] > 0.4:
                return 'neutral'
            else:
                return 'positive'
        
        # Rating 5: Check if VERY positive
        if rating == 5:
            if scores['positive'] > 0.7:
                return 'very_positive'
            else:
                return 'positive'
    
    # Default mapping based on rating only
    mapping = {
        1: 'very_negative',
        2: 'negative',
        3: 'neutral',
        4: 'positive',
        5: 'very_positive'
    }
    return mapping.get(rating, 'neutral')

# ============================================
# DATA BALANCING
# ============================================
def balance_dataset(df: pd.DataFrame, target_per_class: int = None) -> pd.DataFrame:
    """
    Balance dataset by sampling equal number from each class
    """
    if target_per_class is None:
        # Use minimum class count
        class_counts = df['sentiment'].value_counts()
        target_per_class = class_counts.min()
    
    balanced_dfs = []
    for sentiment in df['sentiment'].unique():
        df_class = df[df['sentiment'] == sentiment]
        
        if len(df_class) > target_per_class:
            # Downsample
            df_class = df_class.sample(n=target_per_class, random_state=42)
        elif len(df_class) < target_per_class:
            # Upsample (with replacement if needed)
            df_class = df_class.sample(n=target_per_class, replace=True, random_state=42)
        
        balanced_dfs.append(df_class)
    
    return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# ============================================
# MAIN PROCESSING
# ============================================
def process_complete_pipeline(input_file: str):
    """
    Complete pipeline:
    1. Load data
    2. Clean text for IndoBERT
    3. Validate quality (separate dirty data)
    4. Correct labels with text analysis
    5. Create 3-class dataset
    6. Create 5-class dataset
    7. Balance both datasets
    8. Save all outputs
    """
    print("=" * 80)
    print("🚀 MASTER DATA PROCESSOR - INDOBERT READY")
    print("=" * 80)
    
    # 1. Load data
    print(f"\n📂 Loading data from: {input_file}")
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return
    
    df = pd.read_csv(input_file)
    print(f"   Loaded: {len(df):,} rows")
    print(f"   Columns: {list(df.columns)}")
    
    # Ensure required columns
    if 'content' not in df.columns:
        if 'content_clean' in df.columns:
            df['content'] = df['content_clean']
        elif 'text' in df.columns:
            df['content'] = df['text']
        else:
            print("❌ No content column found!")
            return
    
    if 'rating' not in df.columns:
        if 'score' in df.columns:
            df['rating'] = df['score']
        else:
            print("❌ No rating column found!")
            return
    
    # 2. Clean text for IndoBERT
    print(f"\n🧹 Cleaning text for IndoBERT...")
    df['content_original'] = df['content']
    df['content_clean'] = df['content'].apply(clean_text_indobert)
    
    # Remove empty after cleaning
    df = df[df['content_clean'].str.len() > 0].copy()
    print(f"   After cleaning: {len(df):,} rows")
    
    # 3. Validate quality
    print(f"\n✅ Validating text quality...")
    validation_results = df['content_clean'].apply(validate_text_quality)
    df['is_valid'] = validation_results.apply(lambda x: x[0])
    df['invalid_reason'] = validation_results.apply(lambda x: x[1])
    
    df_clean = df[df['is_valid']].copy()
    df_dirty = df[~df['is_valid']].copy()
    
    print(f"   ✅ Clean data: {len(df_clean):,} ({len(df_clean)/len(df)*100:.1f}%)")
    print(f"   ❌ Dirty data: {len(df_dirty):,} ({len(df_dirty)/len(df)*100:.1f}%)")
    
    if not df_clean.empty:
        print(f"\n   Dirty data reasons:")
        for reason, count in df_dirty['invalid_reason'].value_counts().items():
            print(f"      - {reason}: {count:,}")
    
    # Save dirty data
    if not df_dirty.empty:
        dirty_file = os.path.join(OUTPUT_DIR, 'data_dirty_rejected.csv')
        df_dirty.to_csv(dirty_file, index=False, encoding='utf-8')
        print(f"\n   💾 Dirty data saved: {dirty_file}")
    
    # Continue with clean data only
    df = df_clean.copy()
    
    # 4. Calculate sentiment scores
    print(f"\n🔍 Analyzing sentiment from text content...")
    sentiment_scores = df['content_clean'].apply(calculate_sentiment_score)
    df['pos_score'] = sentiment_scores.apply(lambda x: x['positive'])
    df['neg_score'] = sentiment_scores.apply(lambda x: x['negative'])
    df['neu_score'] = sentiment_scores.apply(lambda x: x['neutral'])
    
    # 5. Create 3-class dataset
    print(f"\n📊 Creating 3-CLASS dataset (negative, neutral, positive)...")
    df_3class = df.copy()
    
    # Map to 3-class with text analysis
    df_3class['sentiment'] = df_3class.apply(
        lambda row: map_rating_to_sentiment_3class(row['rating'], row['content_clean']),
        axis=1
    )
    
    print(f"\n   Distribution before balancing:")
    for sent, count in df_3class['sentiment'].value_counts().items():
        print(f"      {sent:10s}: {count:,} ({count/len(df_3class)*100:.1f}%)")
    
    # Balance 3-class
    print(f"\n   Balancing 3-class dataset...")
    df_3class_balanced = balance_dataset(df_3class)
    
    print(f"   Distribution after balancing:")
    for sent, count in df_3class_balanced['sentiment'].value_counts().items():
        print(f"      {sent:10s}: {count:,} ({count/len(df_3class_balanced)*100:.1f}%)")
    
    # 6. Create 5-class dataset
    print(f"\n📊 Creating 5-CLASS dataset (very_negative, negative, neutral, positive, very_positive)...")
    df_5class = df.copy()
    
    # Map to 5-class with text analysis
    df_5class['sentiment'] = df_5class.apply(
        lambda row: map_rating_to_sentiment_5class(row['rating'], row['content_clean']),
        axis=1
    )
    
    print(f"\n   Distribution before balancing:")
    for sent, count in df_5class['sentiment'].value_counts().items():
        print(f"      {sent:15s}: {count:,} ({count/len(df_5class)*100:.1f}%)")
    
    # Balance 5-class
    print(f"\n   Balancing 5-class dataset...")
    df_5class_balanced = balance_dataset(df_5class)
    
    print(f"   Distribution after balancing:")
    for sent, count in df_5class_balanced['sentiment'].value_counts().items():
        print(f"      {sent:15s}: {count:,} ({count/len(df_5class_balanced)*100:.1f}%)")
    
    # 7. Prepare final dataframes
    print(f"\n💾 Preparing final datasets...")
    
    # 3-class unbalanced (clean)
    df_3class_final = df_3class[['content_clean', 'sentiment', 'rating', 'pos_score', 'neg_score', 'neu_score']].copy()
    df_3class_final.rename(columns={'content_clean': 'text'}, inplace=True)
    
    # 3-class balanced
    df_3class_balanced_final = df_3class_balanced[['content_clean', 'sentiment', 'rating', 'pos_score', 'neg_score', 'neu_score']].copy()
    df_3class_balanced_final.rename(columns={'content_clean': 'text'}, inplace=True)
    
    # 5-class unbalanced (clean)
    df_5class_final = df_5class[['content_clean', 'sentiment', 'rating', 'pos_score', 'neg_score', 'neu_score']].copy()
    df_5class_final.rename(columns={'content_clean': 'text'}, inplace=True)
    
    # 5-class balanced
    df_5class_balanced_final = df_5class_balanced[['content_clean', 'sentiment', 'rating', 'pos_score', 'neg_score', 'neu_score']].copy()
    df_5class_balanced_final.rename(columns={'content_clean': 'text'}, inplace=True)
    
    # 8. Save all outputs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    files = {
        '3class_clean': df_3class_final,
        '3class_balanced': df_3class_balanced_final,
        '5class_clean': df_5class_final,
        '5class_balanced': df_5class_balanced_final,
    }
    
    print(f"\n📁 Saving all datasets to: {OUTPUT_DIR}/")
    saved_files = []
    
    for name, data in files.items():
        filename = f"gojek_indobert_{name}_{timestamp}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        data.to_csv(filepath, index=False, encoding='utf-8')
        saved_files.append(filepath)
        print(f"   ✅ {filename} ({len(data):,} rows)")
    
    # 9. Generate summary report
    print(f"\n" + "=" * 80)
    print("✅ PROCESSING COMPLETE - SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 3-CLASS DATASET:")
    print(f"   Clean (unbalanced): {len(df_3class_final):,} rows")
    print(f"   Balanced:           {len(df_3class_balanced_final):,} rows")
    print(f"   Classes: {sorted(df_3class_final['sentiment'].unique())}")
    
    print(f"\n📊 5-CLASS DATASET:")
    print(f"   Clean (unbalanced): {len(df_5class_final):,} rows")
    print(f"   Balanced:           {len(df_5class_balanced_final):,} rows")
    print(f"   Classes: {sorted(df_5class_final['sentiment'].unique())}")
    
    print(f"\n📁 SAVED FILES:")
    for filepath in saved_files:
        print(f"   ✅ {os.path.basename(filepath)}")
    
    # Show samples
    print(f"\n📋 SAMPLE DATA (3-class balanced):")
    for idx, row in df_3class_balanced_final.head(5).iterrows():
        print(f"\n   [{row['sentiment'].upper():8s} | ⭐{row['rating']}]")
        print(f"   \"{row['text'][:80]}...\"")
    
    print(f"\n" + "=" * 80)
    print("🎉 READY FOR INDOBERT TRAINING!")
    print("=" * 80)
    print(f"\n📌 RECOMMENDED FOR TRAINING:")
    print(f"   - 3-class balanced: Best for general sentiment analysis")
    print(f"   - 5-class balanced: Better for nuanced sentiment")
    print(f"   - All texts are cleaned, normalized, and quality-checked")
    print(f"   - Labels are corrected based on text content analysis")
    print("=" * 80)
    
    return saved_files

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    try:
        saved_files = process_complete_pipeline(INPUT_FILE)
        
        if saved_files:
            print(f"\n✅ Success! {len(saved_files)} files created.")
            print(f"\n💡 Next steps:")
            print(f"   1. Review data quality in: {OUTPUT_DIR}/")
            print(f"   2. Use 3class_balanced for training")
            print(f"   3. Compare with 5class_balanced for better accuracy")
        else:
            print(f"\n❌ Processing failed!")
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
