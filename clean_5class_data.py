"""
Script untuk membersihkan dan menyiapkan data 5 kelas sentiment analysis.
Memperbaiki inkonsistensi antara rating, sentiment, dan konten review.
"""

import pandas as pd
import numpy as np
import re
from collections import Counter

def clean_text(text):
    """Clean review text"""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep Indonesian words
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def get_sentiment_from_rating(rating):
    """Map rating to sentiment label (5 classes)"""
    rating = int(rating)
    mapping = {
        1: (1, 'sangat_negatif'),
        2: (2, 'negatif'),
        3: (3, 'netral'),
        4: (4, 'positif'),
        5: (5, 'sangat_positif')
    }
    return mapping.get(rating, (3, 'netral'))

# Kata-kata untuk deteksi sentiment
VERY_NEGATIVE_WORDS = [
    'sampah', 'kacau', 'parah', 'bangsat', 'anjing', 'taik', 'tai', 'bego', 'bodoh',
    'goblok', 'tolol', 'busuk', 'jelek banget', 'buruk banget', 'sangat kecewa',
    'tidak berguna', 'gak guna', 'ga guna', 'bohong', 'nipu', 'penipuan', 'tipu',
    'kntl', 'kontol', 'memek', 'ngentot', 'bajingan', 'brengsek', 'sialan', 'kampret',
    'worst', 'terrible', 'horrible', 'sangat buruk', 'paling buruk', 'kapok',
    'gak akan', 'ga akan', 'tidak akan', 'never', 'rugi banget', 'buang waktu',
    'menyesal', 'nyesel', 'kecewa berat', 'sangat mengecewakan', 'uninstall'
]

NEGATIVE_WORDS = [
    'kecewa', 'lama', 'lambat', 'susah', 'mahal', 'jelek', 'buruk', 'error', 
    'gagal', 'batal', 'cancel', 'ribet', 'rumit', 'sulit', 'tidak bisa',
    'ga bisa', 'gak bisa', 'lemot', 'eror', 'bug', 'masalah', 'problem',
    'kesal', 'jengkel', 'sebel', 'sebal', 'cape', 'capek', 'males', 'malas',
    'komplain', 'keluhan', 'protes', 'zonk', 'rugi', 'telat', 'terlambat',
    'not good', 'bad', 'worse', 'belagu', 'judes', 'sombong', 'kurang'
]

POSITIVE_WORDS = [
    'bagus', 'baik', 'cepat', 'cepet', 'ramah', 'murah', 'ok', 'oke', 'okay',
    'mantap', 'mantab', 'mantul', 'sip', 'siip', 'membantu', 'memudahkan',
    'mudah', 'praktis', 'nyaman', 'aman', 'puas', 'senang', 'suka', 'like',
    'helpful', 'good', 'nice', 'great', 'best', 'terbaik', 'rekomendasi',
    'rekomen', 'recommended', 'lancar', 'sukses', 'berhasil', 'worth',
    'tepat', 'tepat waktu', 'on time', 'ontime', 'cukup', 'lumayan'
]

VERY_POSITIVE_WORDS = [
    'sangat bagus', 'sangat baik', 'luar biasa', 'amazing', 'excellent',
    'perfect', 'sempurna', 'terbaik', 'the best', 'mantap banget', 'top',
    'jempol', 'keren banget', 'hebat', 'istimewa', 'memuaskan', 'puas banget',
    'sangat puas', 'sangat membantu', 'sangat memuaskan', 'sukses selalu',
    'terima kasih', 'thanks', 'thank you', 'love', 'suka banget', 'super',
    'awesome', 'fantastic', 'wonderful', 'best app', 'aplikasi terbaik'
]

def analyze_text_sentiment(text):
    """Analyze text to detect sentiment based on keywords"""
    text_lower = str(text).lower()
    
    # Count sentiment words
    very_neg_count = sum(1 for word in VERY_NEGATIVE_WORDS if word in text_lower)
    neg_count = sum(1 for word in NEGATIVE_WORDS if word in text_lower)
    pos_count = sum(1 for word in POSITIVE_WORDS if word in text_lower)
    very_pos_count = sum(1 for word in VERY_POSITIVE_WORDS if word in text_lower)
    
    # Return detected sentiment
    scores = {
        'very_negative': very_neg_count * 2,
        'negative': neg_count,
        'positive': pos_count,
        'very_positive': very_pos_count * 2
    }
    
    return scores

def check_consistency(row):
    """Check if rating is consistent with text content"""
    text = str(row['review']).lower()
    rating = int(row['rating'])
    
    scores = analyze_text_sentiment(text)
    
    # Calculate overall sentiment score
    # Negative scores are negative, positive are positive
    overall_score = (scores['very_positive'] * 2 + scores['positive']) - \
                    (scores['very_negative'] * 2 + scores['negative'])
    
    # Check for major inconsistencies
    # Case 1: Very negative text but high rating (4-5)
    if scores['very_negative'] >= 2 and rating >= 4:
        return False, 'negative_text_high_rating'
    
    # Case 2: Very positive text but low rating (1-2)
    if scores['very_positive'] >= 2 and rating <= 2:
        return False, 'positive_text_low_rating'
    
    # Case 3: Strong negative sentiment but positive rating
    if overall_score <= -3 and rating >= 4:
        return False, 'negative_sentiment_high_rating'
    
    # Case 4: Strong positive sentiment but negative rating
    if overall_score >= 3 and rating <= 2:
        return False, 'positive_sentiment_low_rating'
    
    return True, 'consistent'

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/gojek_reviews_5class_augmented.csv')
    print(f"Original data: {len(df)} rows")
    
    # Show initial distribution
    print("\n=== Initial Distribution ===")
    print(df['sentiment_label'].value_counts())
    
    # Clean text
    print("\nCleaning text...")
    df['review'] = df['review'].apply(clean_text)
    
    # Remove empty reviews
    df = df[df['review'].str.len() > 5]
    print(f"After removing short reviews: {len(df)} rows")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['review'], keep='first')
    print(f"After removing duplicates: {len(df)} rows")
    
    # Check consistency
    print("\nChecking consistency...")
    consistency_results = df.apply(check_consistency, axis=1)
    df['is_consistent'] = consistency_results.apply(lambda x: x[0])
    df['inconsistency_type'] = consistency_results.apply(lambda x: x[1])
    
    # Show inconsistency stats
    inconsistent = df[~df['is_consistent']]
    print(f"\nInconsistent records: {len(inconsistent)}")
    print(inconsistent['inconsistency_type'].value_counts())
    
    # Show some examples of inconsistent data
    print("\n=== Examples of Inconsistent Data ===")
    for inc_type in inconsistent['inconsistency_type'].unique():
        print(f"\n--- {inc_type} ---")
        samples = inconsistent[inconsistent['inconsistency_type'] == inc_type].head(3)
        for _, row in samples.iterrows():
            print(f"Rating: {row['rating']}, Label: {row['sentiment_label']}")
            print(f"Review: {row['review'][:100]}...")
            print()
    
    # Filter to keep only consistent data
    df_clean = df[df['is_consistent']].copy()
    print(f"\nAfter removing inconsistent data: {len(df_clean)} rows")
    
    # Re-map sentiment based on rating (to ensure correct labels)
    df_clean['sentiment'] = df_clean['rating'].astype(int)
    df_clean['sentiment_label'] = df_clean['rating'].apply(lambda x: get_sentiment_from_rating(x)[1])
    
    # Remove helper columns
    df_clean = df_clean.drop(columns=['is_consistent', 'inconsistency_type'])
    
    # Balance dataset (undersample to minority class, but keep reasonable size)
    print("\n=== Balancing Dataset ===")
    print("Distribution before balancing:")
    print(df_clean['sentiment_label'].value_counts())
    
    # Find target count for each class (use min of all classes or max 3000)
    class_counts = df_clean['sentiment_label'].value_counts()
    min_count = class_counts.min()
    target_count = min(min_count, 3000)  # Cap at 3000 per class
    
    print(f"\nTarget per class: {target_count}")
    
    # Balance by undersampling
    df_balanced = pd.DataFrame()
    for label in ['sangat_negatif', 'negatif', 'netral', 'positif', 'sangat_positif']:
        df_class = df_clean[df_clean['sentiment_label'] == label]
        if len(df_class) > target_count:
            df_class = df_class.sample(n=target_count, random_state=42)
        df_balanced = pd.concat([df_balanced, df_class])
    
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print("\nFinal distribution:")
    print(df_balanced['sentiment_label'].value_counts())
    print(f"\nTotal samples: {len(df_balanced)}")
    
    # Save cleaned data
    output_path = 'data/gojek_reviews_5class_clean.csv'
    df_balanced.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nSaved to {output_path}")
    
    # Also save the mapping for reference
    label_mapping = {
        1: 'sangat_negatif',
        2: 'negatif', 
        3: 'netral',
        4: 'positif',
        5: 'sangat_positif'
    }
    print("\n=== Label Mapping ===")
    for k, v in label_mapping.items():
        print(f"Rating {k} -> {v}")

if __name__ == "__main__":
    main()
