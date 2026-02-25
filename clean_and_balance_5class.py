
import pandas as pd
import re
import numpy as np
import os
from collections import Counter

# Configuration
INPUT_FILE = r'd:\Skripsi\sentiment-analyst-ojol-review\data\gojek_scraped_5class_RELABELED.csv'
OUTPUT_FILE = r'd:\Skripsi\sentiment-analyst-ojol-review\data\gojek_5class_BALANCED_FIXED.csv'

# ==========================================
# 1. KEYWORD LISTS (Expanded for 5 Classes)
# ==========================================

# Strong Positive (Indicates Very Positive)
strong_pos_patterns = [
    r'sangat\s+bagus', r'sangat\s+membantu', r'sangat\s+puas', r'puas\s+banget',
    r'luar\s+biasa', r'mantap\s+banget', r'bagus\s+banget', r'keren\s+habis',
    r'terima\s+kasih', r'terimakasih', r'recommended', r'sukses\s+terus', r'the\s+best',
    r'cinta\s+banget', r'top\s+markotop', r'sangat\s+mudah', r'sangat\s+cepat',
    r'tidak\s+mahal', r'murah\s+meriah', r'tidak\s+ugal', r'bintang\s+5', r'bintang\s+lima'
]

# Positive Base
positive_base = [
    'bagus', 'baik', 'mantap', 'puas', 'senang', 'suka', 'nyaman', 'mudah', 
    'cepat', 'ramah', 'membantu', 'terbantu', 'bermanfaat', 'rekomendasi',
    'terbaik', 'lancar', 'aman', 'enak', 'hebat', 'keren', 'top', 
    'memuaskan', 'oke', 'ok', 'sukses', 'good', 'great', 'nice', 'love', 
    'murah', 'gampang', 'praktis', 'gercep', 'lengkap', 'rapi', 'sopan', 'jujur'
]

# Neutral Base
neutral_base = [
    'biasa', 'lumayan', 'cukup', 'sedang', 'standar', 'normal', 'agak',
    'kadang', 'terkadang', 'mungkin', 'saran', 'masukan', 'tanya', 'bertanya',
    'perlu', 'butuh', 'harap', 'semoga', 'tolong'
]

# Negative Base
negative_base = [
    'buruk', 'jelek', 'kecewa', 'mahal', 'lambat', 'lemot', 'susah', 'sulit',
    'gagal', 'error', 'eror', 'ribet', 'kesal', 'malas', 'benci', 'menyebalkan', 
    'menjengkelkan', 'rugi', 'payah', 'zonk', 'bohong', 'tipu', 'penipu', 
    'tidak jelas', 'gajelas', 'kapok', 'sombong', 'kasar', 'jutek', 'marah', 
    'batal', 'cancel', 'hilang', 'kehilangan', 'kacau', 'mengecewakan', 
    'bad', 'worst', 'lelet', 'lola', 'ngelag', 'lag', 'bug', 'bermasalah', 
    'gangguan', 'berisik', 'ba', 'bau', 'kotor', 'ugal'
]

# Strong Negative / Critical (Indicates Very Negative)
strong_neg_patterns = [
    r'sangat\s+kecewa', r'kecewa\s+banget', r'parah\s+banget', r'rugi\s+banget',
    r'sangat\s+buruk', r'ancur', r'hancur', r'tidak\s+bisa', r'gak\s+bisa', r'gabisa',
    r'tidak\s+mau', r'gak\s+mau', r'susah\s+banget', r'ribet\s+banget',
    r'tolong\s+diperbaiki', r'makan\s+duit', r'aplikasi\s+sampah', r'aplikasi\s+bodoh',
    r'tai', r'anjing', r'babi', 'bangsat', 'tolol', 'bodoh', 'goblok', 'setan', 'biadab'
]

# Critical Patterns (Almost always Very Negative)
critical_patterns = [
    r'saldo\s+hilang', r'uang\s+hilang', r'saldo\s+kepotong', r'uang\s+kepotong',
    r'akun\s+diblokir', r'akun\s+di\s+suspend', r'tidak\s+bisa\s+login', r'gabisa\s+login',
    r'gagal\s+login', r'gagal\s+masuk', r'tidak\s+bisa\s+dibuka', r'uninstal', 
    r'hapus\s+aplikasi', r'nyesel', r'menyesal', r'penipuan', r'maling'
]

negation_words = ['tidak', 'gak', 'ga', 'bukan', 'jangan', 'enggak', 'tak', 'bkn', 'kurang']

# ==========================================
# 1b. SLANG NORMALIZATION DICTIONARY
# ==========================================
slang_dict = {
    # Negasi
    'gak': 'tidak', 'ga': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak',
    'gk': 'tidak', 'tdk': 'tidak', 'gx': 'tidak', 'ndak': 'tidak',
    
    # Kata umum
    'yg': 'yang', 'dgn': 'dengan', 'dg': 'dengan', 'utk': 'untuk',
    'sm': 'sama', 'sma': 'sama', 'dr': 'dari', 'ke': 'ke',
    'tp': 'tapi', 'sdh': 'sudah', 'udh': 'sudah', 'udah': 'sudah',
    'dh': 'sudah', 'blm': 'belum', 'blom': 'belum',
    'lg': 'lagi', 'lgi': 'lagi', 'lgu': 'lagi',
    'jd': 'jadi', 'jdi': 'jadi',
    'krn': 'karena', 'krna': 'karena', 'karna': 'karena',
    'mgkn': 'mungkin', 'mungkn': 'mungkin',
    'hrs': 'harus', 'pdhl': 'padahal', 'pdahal': 'padahal',
    'bgt': 'banget', 'bngtt': 'banget', 'bgtt': 'banget', 'bngt': 'banget',
    'aja': 'saja', 'aj': 'saja', 'doang': 'saja',
    'bkn': 'bukan', 'emg': 'memang', 'emang': 'memang',
    'jg': 'juga', 'jga': 'juga',
    'bs': 'bisa', 'dpt': 'dapat',
    'skrg': 'sekarang', 'skrng': 'sekarang', 'skg': 'sekarang',
    'kmrn': 'kemarin', 'kmren': 'kemarin',
    'bsk': 'besok', 'td': 'tadi', 'tdi': 'tadi',
    'brp': 'berapa', 'brapa': 'berapa',
    'knp': 'kenapa', 'knapa': 'kenapa',
    'gmn': 'bagaimana', 'gimana': 'bagaimana', 'gmana': 'bagaimana', 'bgmn': 'bagaimana',
    'klo': 'kalau', 'kalo': 'kalau',
    'nih': 'ini', 'tuh': 'itu',
    'bener': 'benar', 'bner': 'benar', 'bnr': 'benar',
    'org': 'orang', 'jgn': 'jangan', 'jng': 'jangan',
    'msh': 'masih', 'masi': 'masih',
    'pesen': 'pesan', 'nyampe': 'sampai', 'nyampai': 'sampai',
    'telat': 'terlambat', 'males': 'malas',
    
    # Kata terkait Gojek
    'apps': 'aplikasi', 'app': 'aplikasi',
    
    # Slang positif
    'mantul': 'mantap', 'mantab': 'mantap',
    'okee': 'oke', 'okeee': 'oke', 'okeh': 'oke',
    'makasi': 'terima kasih', 'makasih': 'terima kasih',
    'thx': 'terima kasih', 'thanks': 'terima kasih', 'tengkyu': 'terima kasih',
    'rekomend': 'rekomendasi', 'rekomen': 'rekomendasi',
    
    # Slang negatif
    'lelet': 'lambat', 'lemot': 'lambat',
    'zonk': 'buruk',
    
    # Pronoun
    'sy': 'saya', 'sya': 'saya', 'aq': 'saya', 'ak': 'saya',
    'gw': 'saya', 'gue': 'saya', 'ane': 'saya',
    'lu': 'kamu', 'lo': 'kamu',
}

def normalize_text(text):
    """Normalisasi slang/singkatan ke bentuk baku menggunakan regex word boundary"""
    if pd.isna(text):
        return text
    text = str(text).lower()
    for slang, baku in slang_dict.items():
        text = re.sub(r'\b' + re.escape(slang) + r'\b', baku, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 2. FUNCTIONS
# ==========================================

def is_spam(text):
    if pd.isna(text): return True
    text = str(text).lower()
    if len(text) < 5: return True # Too short
    
    # Repetitive chars
    if len(text) > 0:
        char_counts = Counter(text.replace(' ', ''))
        if char_counts:
            most_common, count = char_counts.most_common(1)[0]
            if count / len(text.replace(' ', '')) > 0.5: return True
    
    # Non-alpha ratio
    non_alpha = sum(1 for c in text if not c.isalpha() and c != ' ')
    if len(text) > 0 and non_alpha / len(text) > 0.4: return True
    
    return False

def calculate_5class_sentiment(text):
    if pd.isna(text): return 'neutral'
    text_lower = str(text).lower()
    
    # 1. Critical Checks (Force Very Negative)
    for pattern in critical_patterns:
        if re.search(pattern, text_lower):
            return 'very_negative'
            
    # 2. Score Calculation
    score = 0
    words = text_lower.split()
    
    # Regex scoring (Phrases)
    for pattern in strong_pos_patterns:
        if re.search(pattern, text_lower):
            score += 3 # Big boost for strong phrases
            
    for pattern in strong_neg_patterns:
        if re.search(pattern, text_lower):
            score -= 3 # Big penalty for strong negative phrases
            
    # Word scoring with negation
    i = 0
    while i < len(words):
        word = words[i]
        
        # Negation check
        is_negated = False
        if i > 0 and words[i-1] in negation_words: is_negated = True
        elif i > 1 and words[i-2] in negation_words: is_negated = True
        
        # Bigram checks for negation
        if i < len(words) - 1:
            bigram = f"{words[i]} {words[i+1]}"
            if bigram in ['tidak bisa', 'gak bisa', 'gabisa', 'ga bisa']:
                score -= 2
                i += 1
                continue
            if bigram in ['tidak ada', 'gak ada']:
                score -= 1
                i += 1
                continue
        
        # Keyword Scoring
        if word in positive_base:
            if is_negated: score -= 2 # "tidak bagus" -> negative
            else: score += 1
            
        elif word in negative_base:
            if is_negated: score += 1 # "tidak mahal" -> positive (mild)
            else: score -= 1
            
        i += 1
        
    # 3. Determing Label from Score
    if score >= 4:
        return 'very_positive'
    elif score >= 1:
        return 'positive'
    elif score == 0:
        # If text contains "lumayan" or "cukup", force neutral even if 0
        if any(w in text_lower for w in neutral_base):
            return 'neutral'
        return 'neutral' # Default
    elif score >= -2:
        return 'negative'
    else: # score <= -3
        return 'very_negative'

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

print("Loading dataset...")
if not os.path.exists(INPUT_FILE):
    print(f"Error: File not found {INPUT_FILE}")
    exit()

df = pd.read_csv(INPUT_FILE)
print(f"Total data awal: {len(df)}")

# Standardize columns
if 'text' not in df.columns and 'text_preprocessed' in df.columns:
    df.rename(columns={'text_preprocessed': 'text'}, inplace=True)

# Clean Data
print("Cleaning data (removing spam)...")
df = df.dropna(subset=['text'])
df['text'] = df['text'].astype(str)
df = df[~df['text'].apply(is_spam)]
df = df.drop_duplicates(subset=['text'])
print(f"Sisa data setelah cleaning: {len(df)}")

# Normalize slang
print("Normalizing slang/singkatan...")
df['text'] = df['text'].apply(normalize_text)
df = df.drop_duplicates(subset=['text'])
print(f"Sisa data setelah normalisasi: {len(df)}")

# Apply Logic-based Sentiment
print("Recalculating 5-Class Sentiment Labels...")
df['new_sentiment'] = df['text'].apply(calculate_5class_sentiment)

print("\nPerubahan Distribusi:")
print("OLD Sentiment:\n", df['sentiment'].value_counts())
print("\nNEW Label (Logic-Based):\n", df['new_sentiment'].value_counts())

# Use the new label
df['sentiment'] = df['new_sentiment']
df_clean = df[['text', 'sentiment']]

# Balancing (Undersampling)
print("\nBalancing dataset (Undersampling to Minority Class)...")
label_counts = df_clean['sentiment'].value_counts()
min_count = label_counts.min()
print(f"Target count per class: {min_count}")

df_balanced = df_clean.groupby('sentiment').apply(
    lambda x: x.sample(n=min_count, random_state=42)
).reset_index(drop=True)

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal Distribution:\n{df_balanced['sentiment'].value_counts()}")

# Save
df_balanced.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved FIXED & BALANCED data to: {OUTPUT_FILE}")
