"""
Script perbaikan label KETAT - Mixed sentiment = Neutral
Komplain walau sedikit = Negative
"""

import pandas as pd
import re

# Load data
df = pd.read_csv('data/gojek_scraped_3class_RELABELED.csv')
print(f"Total data: {len(df)}")
print(f"\nDistribusi awal:")
print(df['sentiment'].value_counts())

def analyze_strict_sentiment(text):
    """
    Analisis KETAT:
    - Ada komplain sedikit pun = negative/neutral
    - Mixed sentiment = neutral
    - Murni positif tanpa komplain = positive
    """
    text = text.lower()
    
    # KATA NEGATIF/KOMPLAIN (walau ringan)
    negative_words = [
        'kecewa', 'jelek', 'buruk', 'parah', 'lambat', 'lama', 'error', 
        'tidak bisa', 'goblok', 'tolol', 'bodo', 'anj', 'kesel', 'marah',
        'komplain', 'keluh', 'rugi', 'mending', 'uninstall', 'hapus',
        'susah', 'sulit', 'tidak jelas', 'tidak akurat', 'tidak sesuai',
        'mengecewakan', 'kapok', 'males', 'bosan', 'ribet', 'repot',
        'mahal', 'kemahalan', 'naik', 'nambah', 'tambah harga',
        'gagal', 'salah', 'cancel', 'batal', 'tolak', 'decline',
        'lemot', 'lag', 'loading', 'stuck', 'hang', 'freeze',
        'ngaco', 'aneh', 'lucu', 'gaje', 'garing', 'payah',
        'kenceng', 'makin', 'semakin buruk', 'menurun'
    ]
    
    # KATA POSITIF KUAT
    positive_words = [
        'bagus', 'puas', 'senang', 'mantap', 'keren', 'recommended',
        'membantu', 'mudah', 'cepat', 'lancar', 'terima kasih', 'thanks',
        'good job', 'sangat baik', 'terbaik', 'sempurna', 'praktis',
        'bermanfaat', 'nyaman', 'aman', 'memuaskan'
    ]
    
    # KATA NETRAL (pertanyaan, saran)
    neutral_words = [
        'bagaimana', 'kenapa', 'mengapa', 'bisa tidak', 'tolong',
        'mohon', 'saran', 'harap', 'sebaiknya', 'minta', 'info',
        'tanya', 'kapan', 'dimana', 'update', 'perbaiki'
    ]
    
    # Deteksi negasi
    negation_patterns = [
        r'tidak\s+\w+ugal', r'tidak\s+kasar', r'tidak\s+lambat',
        r'tidak\s+buruk', r'tidak\s+jelek', r'bukan\s+\w+jelek'
    ]
    has_positive_negation = any(re.search(pattern, text) for pattern in negation_patterns)
    
    # Hitung kemunculan
    neg_count = sum(1 for word in negative_words if word in text)
    pos_count = sum(1 for word in positive_words if word in text)
    neu_count = sum(1 for word in neutral_words if word in text)
    
    # Bonus untuk negasi positif
    if has_positive_negation:
        pos_count += 2
        neg_count = max(0, neg_count - 1)
    
    # LOGIKA KETAT:
    
    # 1. MIXED SENTIMENT (positif + negatif) = NEUTRAL
    if pos_count >= 1 and neg_count >= 1:
        return 'neutral'
    
    # 2. NEGATIVE jika ada komplain (walau sedikit)
    if neg_count >= 2:
        return 'negative'
    
    if neg_count == 1 and pos_count == 0:
        return 'negative'
    
    # 3. NEUTRAL untuk pertanyaan/saran
    if neu_count >= 2:
        return 'neutral'
    
    if neu_count >= 1 and pos_count <= 1 and neg_count == 0:
        return 'neutral'
    
    # 4. POSITIVE hanya jika murni positif tanpa komplain
    if pos_count >= 2 and neg_count == 0:
        return 'positive'
    
    if pos_count >= 1 and neg_count == 0 and neu_count == 0:
        return 'positive'
    
    # 5. Default: lihat mayoritas atau neutral
    if pos_count > neg_count and neg_count == 0:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    else:
        return 'neutral'

# Perbaiki label
print("\n🔧 Memperbaiki label dengan logika KETAT...")
df['new_sentiment'] = df['text'].apply(analyze_strict_sentiment)

# Bandingkan perubahan
changed = df[df['sentiment'] != df['new_sentiment']]
print(f"\n📊 Perubahan label: {len(changed)} dari {len(df)} ({len(changed)/len(df)*100:.1f}%)")

if len(changed) > 0:
    print("\n🔍 Contoh perubahan:")
    for idx, row in changed.head(30).iterrows():
        print(f"\nTeks: {row['text'][:120]}...")
        print(f"Label LAMA: {row['sentiment']} → BARU: {row['new_sentiment']}")

# Ganti kolom sentiment
df['sentiment'] = df['new_sentiment']
df = df.drop('new_sentiment', axis=1)

# Distribusi akhir
print(f"\n📊 Distribusi SETELAH perbaikan KETAT:")
print(df['sentiment'].value_counts())

# Timpa file
output_path = 'data/gojek_scraped_3class_RELABELED.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ File berhasil ditimpa: {output_path}")

# Sample check untuk POSITIVE (harus murni positif!)
print("\n" + "="*80)
print("🔍 VERIFIKASI KELAS POSITIVE (harus MURNI tanpa komplain):")
print("="*80)
pos_samples = df[df['sentiment'] == 'positive'].sample(min(10, len(df[df['sentiment'] == 'positive'])))
for idx, row in pos_samples.iterrows():
    text = row['text']
    # Cek apakah ada kata negatif
    has_negative = any(word in text.lower() for word in ['mahal', 'lambat', 'lama', 'tapi', 'namun', 'cuma', 'hanya', 'sayang'])
    status = "⚠️ MIXED" if has_negative else "✅ PURE"
    print(f"\n{status} → {text[:150]}...")

print("\n" + "="*80)
print("🔍 VERIFIKASI KELAS NEUTRAL (pertanyaan/mixed/saran):")
print("="*80)
neu_samples = df[df['sentiment'] == 'neutral'].sample(min(10, len(df[df['sentiment'] == 'neutral'])))
for idx, row in neu_samples.iterrows():
    print(f"• {row['text'][:120]}...")

print("\n" + "="*80)
print("🔍 VERIFIKASI KELAS NEGATIVE (komplain/marah):")
print("="*80)
neg_samples = df[df['sentiment'] == 'negative'].sample(min(10, len(df[df['sentiment'] == 'negative'])))
for idx, row in neg_samples.iterrows():
    print(f"• {row['text'][:120]}...")
