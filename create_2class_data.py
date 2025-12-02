"""
Script untuk membuat dataset 2 kelas (Positive/Negative) dari data raw
- Positive: rating 4-5
- Negative: rating 1-2
- Rating 3 (neutral) TIDAK digunakan karena ambigu
"""

import pandas as pd
import os

def create_2class_dataset():
    print("=" * 60)
    print("📊 MEMBUAT DATASET 2 KELAS (POSITIVE/NEGATIVE)")
    print("=" * 60)
    
    # Cari file raw
    raw_files = [
        'data/gojek_reviews_raw_balanced.csv',
        'data/gojek_reviews_5class_raw_balanced.csv',
        'data/gojek_reviews_3class_raw_balanced.csv',
    ]
    
    raw_path = None
    for f in raw_files:
        if os.path.exists(f):
            raw_path = f
            break
    
    if not raw_path:
        print("❌ File raw tidak ditemukan!")
        print("Files yang ada di folder data:")
        for f in os.listdir('data'):
            print(f"  - {f}")
        return
    
    print(f"✓ Menggunakan: {raw_path}")
    df = pd.read_csv(raw_path)
    
    print(f"\n📁 Data Asli: {len(df):,} samples")
    
    # Cek kolom yang ada
    print(f"Columns: {df.columns.tolist()}")
    
    # Tentukan kolom score
    score_col = 'score' if 'score' in df.columns else 'rating'
    text_col = 'content' if 'content' in df.columns else 'text'
    
    if score_col not in df.columns:
        print(f"❌ Kolom score/rating tidak ditemukan!")
        return
    
    print(f"\n📊 Distribusi Rating Asli:")
    print(df[score_col].value_counts().sort_index())
    
    # Filter hanya rating 1-2 (negative) dan 4-5 (positive)
    # SKIP rating 3 karena ambigu!
    df_negative = df[df[score_col].isin([1, 2])].copy()
    df_positive = df[df[score_col].isin([4, 5])].copy()
    
    df_negative['sentiment'] = 'negative'
    df_positive['sentiment'] = 'positive'
    
    print(f"\n📈 Setelah filter (tanpa rating 3):")
    print(f"  Negative (rating 1-2): {len(df_negative):,}")
    print(f"  Positive (rating 4-5): {len(df_positive):,}")
    
    # Balance data
    min_count = min(len(df_negative), len(df_positive))
    print(f"\n⚖️ Balancing ke {min_count:,} per kelas...")
    
    df_negative_balanced = df_negative.sample(n=min_count, random_state=42)
    df_positive_balanced = df_positive.sample(n=min_count, random_state=42)
    
    # Gabung dan shuffle
    df_2class = pd.concat([df_negative_balanced, df_positive_balanced])
    df_2class = df_2class.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✓ Dataset 2 Kelas: {len(df_2class):,} samples")
    print(df_2class['sentiment'].value_counts())
    
    # Cleaning text
    print("\n🧹 Cleaning text...")
    
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower().strip()
        # Hapus karakter aneh tapi keep emoji
        text = ' '.join(text.split())  # Normalize whitespace
        return text
    
    df_2class['content_clean'] = df_2class[text_col].apply(clean_text)
    
    # Hapus yang kosong
    df_2class = df_2class[df_2class['content_clean'].str.len() > 10]
    
    # Pilih kolom yang diperlukan
    df_final = df_2class[[text_col, 'content_clean', score_col, 'sentiment']].copy()
    df_final.columns = ['content', 'content_clean', 'score', 'sentiment']
    
    print(f"\n✓ Setelah cleaning: {len(df_final):,} samples")
    print(df_final['sentiment'].value_counts())
    
    # Balance ulang setelah cleaning
    min_count_final = df_final['sentiment'].value_counts().min()
    print(f"\n⚖️ Re-balancing ke {min_count_final:,} per kelas...")
    
    df_neg = df_final[df_final['sentiment'] == 'negative'].sample(n=min_count_final, random_state=42)
    df_pos = df_final[df_final['sentiment'] == 'positive'].sample(n=min_count_final, random_state=42)
    df_final = pd.concat([df_neg, df_pos]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✓ Dataset balanced: {len(df_final):,} samples")
    print(df_final['sentiment'].value_counts())
    
    # Sample contoh data
    print("\n" + "=" * 60)
    print("📝 CONTOH DATA")
    print("=" * 60)
    
    for sentiment in ['negative', 'positive']:
        print(f"\n--- {sentiment.upper()} ---")
        samples = df_final[df_final['sentiment'] == sentiment]['content_clean'].head(3).tolist()
        for i, s in enumerate(samples, 1):
            print(f"  {i}. {s[:80]}...")
    
    # Save
    output_path = 'data/gojek_reviews_2class_clean.csv'
    df_final.to_csv(output_path, index=False)
    print(f"\n✅ Dataset saved: {output_path}")
    print(f"   Total: {len(df_final):,} samples")
    print(f"   Negative: {len(df_final[df_final['sentiment']=='negative']):,}")
    print(f"   Positive: {len(df_final[df_final['sentiment']=='positive']):,}")
    
    return df_final

if __name__ == "__main__":
    create_2class_dataset()
