"""
Script untuk membersihkan data sentiment 3 kelas
"""

import pandas as pd
import re
import os

def clean_text(text):
    """Bersihkan teks dari karakter tidak perlu"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Hapus URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Hapus mentions dan hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Hapus emoji (basic)
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    
    # Hapus multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Hapus leading/trailing punctuation
    text = re.sub(r'^[.,!?\s]+|[.,!?\s]+$', '', text)
    
    return text

def main():
    # Load data
    input_path = 'data/gojek_reviews_relabelled_text_based.csv'
    output_path = 'data/gojek_reviews_3class_clean.csv'
    
    print("=" * 60)
    print("PEMBERSIHAN DATA 3 KELAS SENTIMENT")
    print("=" * 60)
    
    df = pd.read_csv(input_path)
    print(f"\n1. Data original: {len(df):,} rows")
    print(f"   Distribusi sentiment:\n{df['sentiment'].value_counts()}")
    
    # Check score vs sentiment
    print(f"\n2. Score vs Sentiment (crosstab):")
    print(pd.crosstab(df['score'], df['sentiment']))
    
    # === CLEANING STEPS ===
    
    # Step 1: Hapus data ambigu - score 3 dengan label positive (seharusnya neutral)
    # Karena score 3 secara natural adalah netral, bukan positif
    ambigu_count = len(df[(df['score'] == 3) & (df['sentiment'] == 'positive')])
    print(f"\n3. Menghapus {ambigu_count} review ambigu (score=3, sentiment=positive)")
    df_clean = df[~((df['score'] == 3) & (df['sentiment'] == 'positive'))].copy()
    
    # Step 2: Bersihkan teks
    print("\n4. Membersihkan teks...")
    df_clean['content_clean'] = df_clean['content'].apply(clean_text)
    
    # Step 3: Hapus teks yang terlalu pendek (< 3 kata)
    df_clean['word_count'] = df_clean['content_clean'].apply(lambda x: len(str(x).split()))
    short_count = len(df_clean[df_clean['word_count'] < 3])
    print(f"5. Menghapus {short_count} review terlalu pendek (< 3 kata)")
    df_clean = df_clean[df_clean['word_count'] >= 3]
    
    # Step 4: Hapus duplikat
    dup_count = df_clean['content_clean'].duplicated().sum()
    print(f"6. Menghapus {dup_count} duplikat")
    df_clean = df_clean.drop_duplicates(subset=['content_clean'])
    
    # Step 5: Hapus teks yang terlalu panjang (> 500 kata) - mungkin bukan review asli
    too_long = len(df_clean[df_clean['word_count'] > 500])
    print(f"7. Menghapus {too_long} review terlalu panjang (> 500 kata)")
    df_clean = df_clean[df_clean['word_count'] <= 500]
    
    # Final cleanup
    df_final = df_clean[['userName', 'content', 'content_clean', 'score', 'at', 'sentiment']].copy()
    df_final = df_final.reset_index(drop=True)
    
    print(f"\n" + "=" * 60)
    print("HASIL PEMBERSIHAN")
    print("=" * 60)
    print(f"Total: {len(df_final):,} rows (berkurang {len(df) - len(df_final):,})")
    print(f"\nDistribusi sentiment:")
    print(df_final['sentiment'].value_counts())
    print(f"\nScore vs Sentiment:")
    print(pd.crosstab(df_final['score'], df_final['sentiment']))
    
    # Statistik panjang teks
    df_final['word_count'] = df_final['content_clean'].apply(lambda x: len(str(x).split()))
    print(f"\nStatistik panjang teks (jumlah kata):")
    print(df_final['word_count'].describe())
    
    # Save
    df_final.drop(columns=['word_count'], inplace=True)
    df_final.to_csv(output_path, index=False)
    print(f"\n✓ Data tersimpan di: {output_path}")
    
    return df_final

if __name__ == "__main__":
    main()
