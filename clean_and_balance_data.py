import pandas as pd
import re
import numpy as np
import os
from collections import Counter

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

print("Loading dataset...")
input_path = os.path.join(base_dir, 'data', 'gojek_3class_PREPROCESSED.csv')
if not os.path.exists(input_path):
    input_path = os.path.join(base_dir, 'data', 'gojek_3class_BALANCED.csv')

print(f"Reading from: {input_path}")
df = pd.read_csv(input_path)
print(f"Total data awal: {len(df)}")

# STEP 1: Ambil hanya kolom yang diperlukan
if 'text_preprocessed' in df.columns:
    df_clean = df[['text_preprocessed', 'sentiment']].copy()
    df_clean.columns = ['text', 'sentiment']
else:
    df_clean = df[['text', 'sentiment']].copy()

# STEP 2: Hapus data yang tidak valid
df_clean = df_clean.dropna(subset=['text'])
df_clean['text'] = df_clean['text'].astype(str)
df_clean = df_clean[df_clean['text'].str.strip() != '']

df_clean = df_clean[df_clean['text'].str.len() >= 10]

def is_spam(text):
    if pd.isna(text):
        return True
    text = str(text).lower()
    
    if len(text) > 0:
        char_counts = Counter(text.replace(' ', ''))
        if char_counts:
            most_common_char, most_common_count = char_counts.most_common(1)[0]
            if most_common_count / len(text.replace(' ', '')) > 0.5:
                return True
    
    non_alpha = sum(1 for c in text if not c.isalpha() and c != ' ')
    if len(text) > 0 and non_alpha / len(text) > 0.4:
        return True
    
    clean_text = text.replace(' ', '')
    if len(clean_text) < 5:
        return True
    
    return False

print("\nMenghapus spam...")
df_clean = df_clean[~df_clean['text'].apply(is_spam)]

# STEP 3: Identifikasi dan perbaiki label sentiment berdasarkan konten
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
    
    i = 0
    while i < len(words):
        word = words[i]
        
        is_negated = False
        if i > 0 and words[i-1] in negation_words:
            is_negated = True
        elif i > 1 and words[i-2] in negation_words:
             is_negated = True
        
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
                
        if word in positive_base:
            if is_negated:
                neg_score += 1 
            else:
                pos_score += 1
        
        elif word in negative_base:
            if is_negated:
                pos_score += 1
            else:
                neg_score += 1
                
        i += 1
        
    critical_neg_patterns = [
        r'tidak\s+bisa\s+buka', r'tidak\s+bisa\s+login', r'gak\s+bisa\s+masuk',
        r'saldo\s+hilang', r'uang\s+hilang', r'penipu', r'maling',
        r'kecewa\s+banget', r'sangat\s+kecewa', r'kapok', r'uninstal'
    ]
    
    is_critical_neg = False
    for pattern in critical_neg_patterns:
        if re.search(pattern, text_lower):
            is_critical_neg = True
            neg_score += 3
            break

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
            
    strong_pos_patterns = [
        r'sangat\s+bagus', r'sangat\s+membantu', r'sangat\s+puas',
        r'luar\s+biasa', r'mantap\s+banget', r'bagus\s+banget',
        r'terima\s+kasih', r'terimakasih', r'recommended', r'sukses\s+terus',
        r'tidak\s+mahal', r'murah\s+meriah', r'tidak\s+ugal'
    ]
    for pattern in strong_pos_patterns:
        if re.search(pattern, text_lower):
            pos_score += 2
            
    difference = pos_score - neg_score
    
    if is_critical_neg and difference >= 0:
        if difference < 3:
            return 'negative', difference

    if difference >= 1:
        return 'positive', difference
    elif difference <= -1:
        return 'negative', difference
    else:
        if any(w in text_lower for w in neutral_base):
            return 'neutral', 0
        
        return 'neutral', 0

print("\nMemperbaiki label sentiment berdasarkan konten teks...")

results = df_clean['text'].apply(calculate_sentiment_score)
df_clean['calculated_sentiment'] = [res[0] for res in results]
df_clean['score_diff'] = [res[1] for res in results]

df_clean['sentiment'] = df_clean['calculated_sentiment']

df_clean = df_clean.drop(columns=['calculated_sentiment', 'score_diff'])

# STEP 4: Hapus duplikat
print("\nMenghapus duplikat...")
df_clean = df_clean.drop_duplicates(subset=['text'])

print(f"\nDistribusi sentiment setelah dikoreksi:\n{df_clean['sentiment'].value_counts()}")

# STEP 5: Balance dataset dengan undersampling
print("\nMenyeimbangkan dataset (Undersampling)...")

label_counts = df_clean['sentiment'].value_counts()
min_count = label_counts.min()
print(f"Jumlah per class saat ini:\n{label_counts}")
print(f"Target jumlah per class (minimum): {min_count}")

df_balanced = df_clean.groupby('sentiment').apply(
    lambda x: x.sample(n=min_count, random_state=42)
).reset_index(drop=True)

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nDistribusi sentiment setelah balancing:\n{df_balanced['sentiment'].value_counts()}")
print(f"\nTotal data setelah balancing: {len(df_balanced)}")

# STEP 6: Simpan hasil

output_balanced_path = os.path.join(base_dir, 'data', 'gojek_3class_BALANCED.csv')
df_balanced.to_csv(output_balanced_path, index=False)
print(f"Data balanced disimpan ke: {output_balanced_path}")

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
