"""
FINAL CLEANING - Analisis ulang SEMUA data dengan rule SANGAT KETAT
"""

import pandas as pd
import re

# Load data
df = pd.read_csv('data/gojek_scraped_3class_RELABELED.csv')
print(f"📊 Total data: {len(df)}")
print(f"Distribusi awal:")
print(df['sentiment'].value_counts())

def final_strict_analysis(text):
    """
    Rule SANGAT KETAT:
    - Positive: Murni pujian tanpa komplain/saran apapun
    - Negative: Ada komplain/keluhan
    - Neutral: Mixed, pertanyaan, saran, atau tidak jelas
    """
    text_lower = text.lower()
    
    # Kata yang PASTI membuat TIDAK positif murni
    not_pure_positive = [
        # Konjungsi penolakan
        'tapi', 'namun', 'cuma', 'cuman', 'hanya', 'sayang', 'sayangnya',
        'terkadang', 'kadang', 'kadang2', 'kadang-kadang', 'sometimes',
        # Komplain harga
        'mahal', 'kemahalan', 'naik', 'makin mahal',
        # Komplain waktu
        'lambat', 'lama', 'lamban', 'telat', 'terlambat',
        # Komplain teknis
        'error', 'lag', 'lemot', 'loading', 'stuck', 'hang', 'crash',
        'tidak bisa', 'gabisa', 'gak bisa', 'belum bisa',
        'susah', 'sulit', 'ribet', 'repot', 'bingung',
        # Kata negatif emosi
        'kecewa', 'jelek', 'buruk', 'parah', 'payah', 'gagal',
        'kesal', 'marah', 'bete', 'dongkol',
        # Kata kasar
        'goblok', 'tolol', 'bodo', 'idiot', 'anj', 'anjing', 'anjir',
        'kampret', 'kontol', 'memek', 'babi',
        # Pertanyaan/permintaan (neutral)
        'kenapa', 'mengapa', 'gimana', 'bagaimana', 'kapan', 'dimana',
        'tolong', 'mohon', 'minta', 'harap', 'sebaiknya',
        'saran', 'usul', 'kritik',
        # Negatif lain
        'tidak', 'bukan', 'belum', 'jangan', 'gak', 'ga', 'enggak',
        'kurang', 'minim', 'sedikit', 'jarang',
        'batal', 'cancel', 'decline', 'tolak',
        'aneh', 'lucu', 'gaje', 'garing',
        'upgrade', 'perbaiki', 'benerin', 'update',
        'komplain', 'lapor', 'report'
    ]
    
    # Cek apakah ada kata yang membuat tidak pure positive
    has_negative_signal = any(word in text_lower for word in not_pure_positive)
    
    # Kata POSITIF KUAT
    positive_strong = [
        'sangat bagus', 'sangat baik', 'sangat puas', 'sangat membantu',
        'terbaik', 'the best', 'bagus sekali', 'mantap', 'keren',
        'sempurna', 'perfect', 'memuaskan', 'puas',
        'terima kasih', 'thanks', 'makasih', 'thank you',
        'recommended', 'sukses', 'jaya',
        'cepat', 'lancar', 'mudah', 'praktis', 'aman', 'nyaman',
        'ramah', 'sopan', 'baik', 'bagus'
    ]
    
    # Hitung kata positif
    pos_count = sum(1 for word in positive_strong if word in text_lower)
    
    # DECISION TREE
    
    # 1. Jika ada sinyal negatif = TIDAK PURE POSITIVE
    if has_negative_signal:
        # Cek apakah lebih banyak positif atau negatif
        if pos_count >= 2:
            return 'neutral'  # Mixed sentiment
        else:
            return 'negative'
    
    # 2. Jika pure positive (tanpa negatif) dan ada kata positif
    if pos_count >= 1:
        return 'positive'
    
    # 3. Default: neutral (tidak jelas)
    return 'neutral'

# Analisis ulang semua data
print("\n🔧 Menganalisis ulang SEMUA data dengan rule KETAT...")
df['new_sentiment'] = df['text'].apply(final_strict_analysis)

# Cek perubahan
changed = df[df['sentiment'] != df['new_sentiment']]
print(f"\n📊 Perubahan: {len(changed)} dari {len(df)} ({len(changed)/len(df)*100:.1f}%)")

if len(changed) > 0:
    print("\n🔍 Contoh perubahan:")
    for idx, row in changed.head(20).iterrows():
        print(f"\n{row['text'][:100]}...")
        print(f"LAMA: {row['sentiment']} → BARU: {row['new_sentiment']}")

# Update
df['sentiment'] = df['new_sentiment']
df = df.drop('new_sentiment', axis=1)

print(f"\n📊 Distribusi SETELAH cleaning:")
print(df['sentiment'].value_counts())

# Balance dataset
min_count = df['sentiment'].value_counts().min()
print(f"\n⚖️ Balancing ke {min_count} per kelas...")

df_balanced = pd.DataFrame()
for sentiment in ['negative', 'neutral', 'positive']:
    df_class = df[df['sentiment'] == sentiment]
    if len(df_class) >= min_count:
        df_sampled = df_class.sample(n=min_count, random_state=42)
    else:
        df_sampled = df_class
    df_balanced = pd.concat([df_balanced, df_sampled])

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✅ Distribusi BALANCED FINAL:")
print(df_balanced['sentiment'].value_counts())
print(f"Total: {len(df_balanced)}")

# Save
output_path = 'data/gojek_scraped_3class_RELABELED.csv'
df_balanced.to_csv(output_path, index=False)
print(f"\n💾 File berhasil disimpan: {output_path}")

# VERIFIKASI KETAT
print("\n" + "="*80)
print("✅ VERIFIKASI POSITIVE (Harus MURNI tanpa komplain/saran):")
print("="*80)
pos_samples = df_balanced[df_balanced['sentiment'] == 'positive'].sample(min(20, len(df_balanced[df_balanced['sentiment'] == 'positive'])))
for idx, row in pos_samples.iterrows():
    text = row['text'][:150]
    print(f"• {text}...")

print("\n" + "="*80)
print("➖ VERIFIKASI NEUTRAL (Mixed/pertanyaan/saran):")
print("="*80)
neu_samples = df_balanced[df_balanced['sentiment'] == 'neutral'].sample(min(15, len(df_balanced[df_balanced['sentiment'] == 'neutral'])))
for idx, row in neu_samples.iterrows():
    text = row['text'][:150]
    print(f"• {text}...")

print("\n" + "="*80)
print("❌ VERIFIKASI NEGATIVE (Komplain/marah):")
print("="*80)
neg_samples = df_balanced[df_balanced['sentiment'] == 'negative'].sample(min(15, len(df_balanced[df_balanced['sentiment'] == 'negative'])))
for idx, row in neg_samples.iterrows():
    text = row['text'][:150]
    print(f"• {text}...")
