"""
Script untuk memperbaiki label sentiment berdasarkan KONTEKS KALIMAT LENGKAP
Bukan hanya keyword, tapi mempertimbangkan negasi dan konteks
"""

import pandas as pd
import re
from transformers import pipeline

# Load data
df = pd.read_csv('data/gojek_scraped_3class_RELABELED.csv')
print(f"Total data: {len(df)}")
print(f"\nDistribusi awal:")
print(df['sentiment'].value_counts())

# Load sentiment analyzer untuk double check
print("\n⏳ Loading IndoBERT sentiment analyzer...")
sentiment_analyzer = pipeline("sentiment-analysis", 
                             model="mdhugol/indonesia-bert-sentiment-classification",
                             device=0 if __import__('torch').cuda.is_available() else -1)

def analyze_contextual_sentiment(text):
    """
    Analisis sentiment dengan mempertimbangkan konteks lengkap
    """
    text = text.lower()
    
    # POLA NEGATIF KUAT (tanpa negasi)
    strong_negative = [
        'kecewa', 'jelek', 'buruk', 'parah', 'lambat', 'error', 'tidak bisa',
        'goblok', 'tolol', 'bodo', 'anj', 'kampret', 'kesel', 'marah',
        'komplain', 'keluh', 'rugi', 'mending', 'uninstall', 'hapus',
        'susah', 'sulit', 'tidak jelas', 'tidak akurat', 'tidak sesuai',
        'mengecewakan', 'kapok', 'males', 'bosan'
    ]
    
    # POLA POSITIF KUAT
    strong_positive = [
        'bagus', 'puas', 'senang', 'mantap', 'keren', 'recommended',
        'membantu', 'mudah', 'cepat', 'lancar', 'terima kasih', 'thanks',
        'good job', 'sangat baik', 'terbaik', 'sempurna', 'praktis',
        'bermanfaat', 'nyaman'
    ]
    
    # POLA NETRAL
    neutral_patterns = [
        'bagaimana', 'kenapa', 'mengapa', 'bisa tidak', 'tolong',
        'mohon', 'saran', 'harap', 'sebaiknya', 'minta', 'info'
    ]
    
    # Cek negasi (kata yang membalikkan makna)
    negation_words = ['tidak', 'bukan', 'tanpa', 'jangan', 'belum', 'gak', 'ga', 'enggak']
    has_negation = any(neg in text for neg in negation_words)
    
    # Hitung skor
    neg_count = sum(1 for word in strong_negative if word in text)
    pos_count = sum(1 for word in strong_positive if word in text)
    neu_count = sum(1 for word in neutral_patterns if word in text)
    
    # CEK KONTEKS KHUSUS: "tidak + kata negatif" = POSITIF
    # Contoh: "driver tidak ugal-ugalan" = positif
    if has_negation:
        # Pattern: "tidak/bukan + kata_negatif" 
        for neg_word in negation_words:
            for bad_word in ['ugal', 'kasar', 'jelek', 'lambat', 'buruk', 'parah']:
                if f"{neg_word} {bad_word}" in text or f"{neg_word}{bad_word}" in text:
                    # Negasi dari negatif = positif
                    pos_count += 2
                    neg_count = max(0, neg_count - 2)
    
    # Logika keputusan
    if neg_count >= 3 or (neg_count >= 2 and pos_count == 0):
        return 'negative'
    elif pos_count >= 2 and neg_count == 0:
        return 'positive'
    elif pos_count > neg_count and pos_count >= 1:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    elif neu_count >= 2 or (neu_count >= 1 and pos_count == 0 and neg_count == 0):
        return 'neutral'
    else:
        # Use IndoBERT as fallback
        try:
            result = sentiment_analyzer(text[:512])[0]
            label = result['label'].lower()
            if 'positive' in label:
                return 'positive'
            elif 'negative' in label:
                return 'negative'
            else:
                return 'neutral'
        except:
            return 'neutral'

# Perbaiki label
print("\n🔧 Memperbaiki label berdasarkan konteks...")
df['new_sentiment'] = df['text'].apply(analyze_contextual_sentiment)

# Bandingkan perubahan
changed = df[df['sentiment'] != df['new_sentiment']]
print(f"\n📊 Perubahan label: {len(changed)} dari {len(df)} ({len(changed)/len(df)*100:.1f}%)")

if len(changed) > 0:
    print("\n🔍 Contoh perubahan:")
    for idx, row in changed.head(20).iterrows():
        print(f"\nTeks: {row['text'][:100]}...")
        print(f"Label LAMA: {row['sentiment']} → BARU: {row['new_sentiment']}")

# Ganti kolom sentiment dengan yang baru
df['sentiment'] = df['new_sentiment']
df = df.drop('new_sentiment', axis=1)

# Tampilkan distribusi akhir
print(f"\n📊 Distribusi SETELAH perbaikan:")
print(df['sentiment'].value_counts())

# TIMPA file lama (overwrite)
output_path = 'data/gojek_scraped_3class_RELABELED.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ File berhasil ditimpa: {output_path}")

# Cek kualitas dengan sample
print("\n🔍 Cek kualitas label (20 sample tiap kelas):")
for sentiment in ['negative', 'neutral', 'positive']:
    print(f"\n{'='*80}")
    print(f"KELAS: {sentiment.upper()}")
    print('='*80)
    samples = df[df['sentiment'] == sentiment].sample(min(5, len(df[df['sentiment'] == sentiment])))
    for idx, row in samples.iterrows():
        print(f"• {row['text'][:150]}...")
        print()
