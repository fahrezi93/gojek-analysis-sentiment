"""
Script untuk membersihkan dan menyeimbangkan dataset Gojek review
- Memperbaiki label sentiment berdasarkan konten teks dengan penanganan negasi yang lebih baik
- Menghapus spam/data tidak jelas
- Menyeimbangkan dataset
- Output hanya kolom: text, sentiment
"""

import pandas as pd
import re
import numpy as np
import os
from collections import Counter

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

# Load dataset
print("Loading dataset...")
input_path = os.path.join(base_dir, 'data', 'gojek_3class_PREPROCESSED.csv')
if not os.path.exists(input_path):
    # Fallback if PREPROCESSED doesn't exist, try BALANCED
    input_path = os.path.join(base_dir, 'data', 'gojek_3class_BALANCED.csv')

print(f"Reading from: {input_path}")
df = pd.read_csv(input_path)
print(f"Total data awal: {len(df)}")

# =============================================
# STEP 1: Ambil hanya kolom yang diperlukan
# =============================================
# Gunakan text_preprocessed sebagai text utama jika ada, jika tidak gunakan text
if 'text_preprocessed' in df.columns:
    df_clean = df[['text_preprocessed', 'sentiment']].copy()
    df_clean.columns = ['text', 'sentiment']
else:
    df_clean = df[['text', 'sentiment']].copy()

# =============================================
# STEP 2: Hapus data yang tidak valid
# =============================================
# Hapus row dengan text null atau kosong
df_clean = df_clean.dropna(subset=['text'])
df_clean['text'] = df_clean['text'].astype(str)
df_clean = df_clean[df_clean['text'].str.strip() != '']

# Hapus text yang terlalu pendek (kurang dari 10 karakter) - kemungkinan spam
df_clean = df_clean[df_clean['text'].str.len() >= 10]

# Hapus text yang terlalu banyak karakter berulang (spam)
def is_spam(text):
    if pd.isna(text):
        return True
    text = str(text).lower()
    
    # Check if mostly same character repeated
    if len(text) > 0:
        char_counts = Counter(text.replace(' ', ''))
        if char_counts:
            most_common_char, most_common_count = char_counts.most_common(1)[0]
            if most_common_count / len(text.replace(' ', '')) > 0.5:
                return True
    
    # Check for excessive non-Indonesian characters (simple heuristic)
    non_alpha = sum(1 for c in text if not c.isalpha() and c != ' ')
    if len(text) > 0 and non_alpha / len(text) > 0.4: # Increased threshold slightly
        return True
    
    # Check for too short after removing spaces
    clean_text = text.replace(' ', '')
    if len(clean_text) < 5:
        return True
    
    return False

print("\nMenghapus spam...")
df_clean = df_clean[~df_clean['text'].apply(is_spam)]

# =============================================
# STEP 3: Identifikasi dan perbaiki label sentiment berdasarkan konten
# =============================================

# Kata-kata kunci
positive_base = [
    'bagus', 'baik', 'mantap', 'puas', 'senang', 'suka', 'nyaman', 'mudah', 
    'cepat', 'ramah', 'membantu', 'terbantu', 'bermanfaat', 'rekomendasi',
    'terbaik', 'lancar', 'aman', 'enak', 'hebat', 'keren', 'top', 'mantul',
    'luar biasa', 'memuaskan', 'oke', 'ok', 'sukses', 'terimakasih', 'terima kasih',
    'thanks', 'thank', 'good', 'great', 'nice', 'love', 'awesome', 'excellent',
    'murah', 'gampang', 'praktis', 'gercep', 'lengkap', 'rapi', 'sopan', 'jujur'
]

negative_base = [
    'buruk', 'jelek', 'kecewa', 'mahal', 'lambat', 'lemot', 'susah', 'sulit',
    'gagal', 'error', 'eror', 'ribet', 'kesal', 'malas', 'benci', 'menyebalkan', 
    'menjengkelkan', 'rugi', 'sampah', 'parah', 'ancur', 'hancur', 'payah', 'zonk', 
    'bohong', 'tipu', 'penipu', 'blokir', 'diblokir', 'tidak jelas', 'gajelas', 'kapok',
    'sombong', 'kasar', 'jutek', 'marah', 'batal', 'cancel', 'hilang', 'kehilangan', 
    'kacau', 'mengecewakan', 'bad', 'worst', 'terrible', 'horrible', 'sucks', 'hate',
    'lelet', 'lola', 'ngelag', 'lag', 'bug', 'bermasalah', 'gangguan', 'berisik', 
    'bau', 'kotor', 'ugal', 'tidak sopan', 'kurang ajar'
]

neutral_base = [
    'biasa', 'lumayan', 'cukup', 'sedang', 'standar', 'normal', 'agak',
    'kadang', 'terkadang', 'mungkin', 'saran', 'masukan', 'tanya', 'bertanya'
]

negation_words = ['tidak', 'gak', 'ga', 'bukan', 'jangan', 'enggak', 'tak', 'bkn', 'kurang']

def calculate_sentiment_score(text):
    """
    Calculate sentiment based on keywords with negation handling.
    """
    if pd.isna(text):
        return 'neutral', 0
    
    text_lower = str(text).lower()
    words = text_lower.split()
    
    pos_score = 0
    neg_score = 0
    
    # Iterate through words to check for keywords and negations
    i = 0
    while i < len(words):
        word = words[i]
        
        # Check for negation window (look back 1 or 2 words)
        is_negated = False
        if i > 0 and words[i-1] in negation_words:
            is_negated = True
        elif i > 1 and words[i-2] in negation_words:
            # Allows for "tidak terlalu mahal" -> negated "mahal"
             is_negated = True
        
        # Check phrase matches first (2-gram)
        if i < len(words) - 1:
            bigram = f"{words[i]} {words[i+1]}"
            if bigram == 'tidak bisa' or bigram == 'gak bisa' or bigram == 'gabisa':
                neg_score += 2
                i += 1 # Skip next word
                continue
            if bigram == 'tidak ada' or bigram == 'gak ada':
                neg_score += 1
                i += 1
                continue
                
        # Keyword matching
        if word in positive_base:
            if is_negated:
                # "tidak bagus" -> negative
                neg_score += 1 
            else:
                # "bagus" -> positive
                pos_score += 1
        
        elif word in negative_base:
            if is_negated:
                # "tidak mahal" -> positive
                # "tidak ugal ugalan" falls here if 'ugal' is in negative_base
                pos_score += 1
            else:
                # "mahal" -> negative
                neg_score += 1
                
        i += 1
        
    # Additional specific phrase checks using regex for robustness
    
    # Priority check for Strong Negative Patterns (Critical failures often override positives)
    # Example: "Aplikasi bagus tapi tidak bisa dibuka" -> Should be Negative
    critical_neg_patterns = [
        r'tidak\s+bisa\s+buka', r'tidak\s+bisa\s+login', r'gak\s+bisa\s+masuk',
        r'saldo\s+hilang', r'uang\s+hilang', r'penipu', r'maling',
        r'kecewa\s+banget', r'sangat\s+kecewa', r'kapok', r'uninstal'
    ]
    
    is_critical_neg = False
    for pattern in critical_neg_patterns:
        if re.search(pattern, text_lower):
            is_critical_neg = True
            neg_score += 3 # Boost negative score significantly
            break

    # Specific negative patterns
    strong_neg_patterns = [
        r'tidak\s+bisa', r'gak\s+bisa', r'ga\s+bisa', r'gabisa', 
        r'susah\s+login', r'gagal\s+login', r'aplikasi\s+error',
        r'sangat\s+buruk', r'tolong\s+diperbaiki', 
        r'parah\s+banget', r'rugi\s+banget', r'makan\s+duit',
        r'tai', r'anjing', r'babi', 'bangsat', 'tolol', 'bodoh'
    ]
    for pattern in strong_neg_patterns:
        if re.search(pattern, text_lower):
            neg_score += 2
            
    # Specific positive patterns
    strong_pos_patterns = [
        r'sangat\s+bagus', r'sangat\s+membantu', r'sangat\s+puas',
        r'luar\s+biasa', r'mantap\s+banget', r'bagus\s+banget',
        r'terima\s+kasih', r'terimakasih', r'recommended', r'sukses\s+terus',
        r'tidak\s+mahal', r'murah\s+meriah', r'tidak\s+ugal'
    ]
    for pattern in strong_pos_patterns:
        if re.search(pattern, text_lower):
            pos_score += 2
            
    # Calculate difference
    difference = pos_score - neg_score
    
    # Priority Logic
    if is_critical_neg and difference >= 0:
        # If critical negative exists, force negative unless positive is overwhelming (diff >= 3)
        if difference < 3:
            return 'negative', difference

    if difference >= 1:
        return 'positive', difference
    elif difference <= -1:
        return 'negative', difference
    else:
        # Check for neutral markers if no strong sentiment
        if any(w in text_lower for w in neutral_base):
            return 'neutral', 0
        
        # If no sentiment words found, default to neutral
        return 'neutral', 0

print("\nMemperbaiki label sentiment berdasarkan konten teks...")

# Apply sentiment analysis
results = df_clean['text'].apply(calculate_sentiment_score)
df_clean['calculated_sentiment'] = [res[0] for res in results]
df_clean['score_diff'] = [res[1] for res in results]

# Update sentiment column entirely based on text calculation
# User instruction: "label itu tidak ikutin rating seharusnya ikut isi teks"
df_clean['sentiment'] = df_clean['calculated_sentiment']

# Drop helper columns
df_clean = df_clean.drop(columns=['calculated_sentiment', 'score_diff'])

# =============================================
# STEP 4: Hapus duplikat
# =============================================
print("\nMenghapus duplikat...")
df_clean = df_clean.drop_duplicates(subset=['text'])

print(f"\nDistribusi sentiment setelah dikoreksi:\n{df_clean['sentiment'].value_counts()}")

# =============================================
# STEP 5: Balance dataset dengan undersampling
# =============================================
print("\nMenyeimbangkan dataset (Undersampling)...")

# Cari jumlah minimum dari semua class
label_counts = df_clean['sentiment'].value_counts()
min_count = label_counts.min()
print(f"Jumlah per class saat ini:\n{label_counts}")
print(f"Target jumlah per class (minimum): {min_count}")

# Undersample setiap class ke jumlah minimum
df_balanced = df_clean.groupby('sentiment').apply(
    lambda x: x.sample(n=min_count, random_state=42)
).reset_index(drop=True)

# Shuffle dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nDistribusi sentiment setelah balancing:\n{df_balanced['sentiment'].value_counts()}")
print(f"\nTotal data setelah balancing: {len(df_balanced)}")

# =============================================
# STEP 6: Simpan hasil
# =============================================
# Simpan versi clean (belum balanced) untuk referensi (optional)
# output_clean_path = os.path.join(base_dir, 'data', 'gojek_3class_CLEANED_NEW.csv')
# df_clean.to_csv(output_clean_path, index=False)

# Simpan versi balanced (overwrite file yang diminta)
output_balanced_path = os.path.join(base_dir, 'data', 'gojek_3class_BALANCED.csv')
df_balanced.to_csv(output_balanced_path, index=False)
print(f"Data balanced disimpan ke: {output_balanced_path}")

# Tampilkan beberapa contoh check manual
print("\n" + "="*80)
print("CHECK CONTOH DATA (Negation Check):")
print("="*80)

check_phrases = ['tidak mahal', 'tidak puas', 'tidak bisa', 'tidak ugal']
for phrase in check_phrases:
    print(f"\nPhrase: '{phrase}'")
    samples = df_balanced[df_balanced['text'].str.contains(phrase, case=False, na=False)].head(2)
    if not samples.empty:
        for idx, row in samples.iterrows():
            print(f"  [{row['sentiment']}] {row['text'][:100]}...")
    else:
        print("  (Tidak ditemukan sampel di dataset balanced)")

print("\n" + "="*80)
print("PROSES SELESAI!")
print("="*80)
