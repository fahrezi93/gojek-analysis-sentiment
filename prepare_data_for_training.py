"""
Script untuk Mempersiapkan Data Training IndoBERT
==================================================
1. Hapus data duplikat
2. Minimal cleaning (pertahankan struktur kalimat untuk IndoBERT)
3. Balance data jika perlu

Author: Fahrezi
Date: November 2025
"""

import pandas as pd
import numpy as np
import re
import os
from sklearn.utils import resample

# =====================================================
# KONFIGURASI
# =====================================================
INPUT_FILE = 'data/gojek_reviews_final_augmented.csv'
OUTPUT_FILE = 'data/gojek_reviews_3class_clean.csv'

# =====================================================
# 1. LOAD DATA
# =====================================================
print('=' * 60)
print('📂 LOADING DATA')
print('=' * 60)

df = pd.read_csv(INPUT_FILE)
print(f'Total data awal: {len(df):,}')
print(f'Kolom: {df.columns.tolist()}')

# =====================================================
# 2. CEK & HAPUS DUPLIKAT
# =====================================================
print('\n' + '=' * 60)
print('🔍 CEK DUPLIKAT')
print('=' * 60)

# Cek duplikat berdasarkan content
duplicates = df.duplicated(subset=['content'], keep='first')
num_duplicates = duplicates.sum()
print(f'Data duplikat ditemukan: {num_duplicates}')

# Hapus duplikat
df_clean = df.drop_duplicates(subset=['content'], keep='first').reset_index(drop=True)
print(f'Data setelah hapus duplikat: {len(df_clean):,}')

# =====================================================
# 3. CEK DISTRIBUSI SENTIMEN
# =====================================================
print('\n' + '=' * 60)
print('📊 DISTRIBUSI SENTIMEN (Sebelum Balance)')
print('=' * 60)
print(df_clean['sentiment'].value_counts())

# =====================================================
# 4. BALANCE DATA
# =====================================================
print('\n' + '=' * 60)
print('⚖️ BALANCE DATA')
print('=' * 60)

counts = df_clean['sentiment'].value_counts()
min_count = counts.min()

# Undersample ke jumlah kelas terkecil
df_balanced = pd.DataFrame()
for sentiment in ['negative', 'neutral', 'positive']:
    df_class = df_clean[df_clean['sentiment'] == sentiment]
    df_sampled = resample(df_class, replace=False, n_samples=min_count, random_state=42)
    df_balanced = pd.concat([df_balanced, df_sampled])

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f'Data setelah balancing: {len(df_balanced):,}')
print(df_balanced['sentiment'].value_counts())

# =====================================================
# 5. CLEANING UNTUK INDOBERT
# =====================================================
print('\n' + '=' * 60)
print('🧹 CLEANING DATA UNTUK INDOBERT')
print('=' * 60)

def clean_for_indobert(text):
    """
    Minimal cleaning untuk IndoBERT:
    - Hapus HTML tags
    - Hapus URL
    - Normalize whitespace
    - PERTAHANKAN: tanda baca, huruf besar/kecil, struktur kalimat
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Hapus HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Hapus URL
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply cleaning ke kolom content (teks asli)
df_balanced['content_clean'] = df_balanced['content'].apply(clean_for_indobert)

# Hapus baris dengan teks kosong atau terlalu pendek
df_balanced = df_balanced[df_balanced['content_clean'].str.len() > 5].reset_index(drop=True)

print(f'Data final: {len(df_balanced):,}')
print('\nContoh data:')
for i in range(3):
    print(f'  [{i+1}] {df_balanced["content_clean"].iloc[i][:70]}...')
    print(f'      Sentiment: {df_balanced["sentiment"].iloc[i]}')

# =====================================================
# 6. SAVE DATA
# =====================================================
print('\n' + '=' * 60)
print('💾 SAVE DATA')
print('=' * 60)

# Simpan dengan kolom yang diperlukan
df_save = df_balanced[['content', 'content_clean', 'sentiment']].copy()
df_save.to_csv(OUTPUT_FILE, index=False)
print(f'✓ Data saved: {OUTPUT_FILE}')
print(f'  Rows: {len(df_save):,}')

# =====================================================
# 7. SUMMARY
# =====================================================
print('\n' + '=' * 60)
print('📊 SUMMARY')
print('=' * 60)

print(f'''
┌─────────────────────────────────────────────────────────┐
│              DATA PREPARATION SUMMARY                   │
├─────────────────────────────────────────────────────────┤
│ Data Awal            : {len(df):>6,} baris                 │
│ Duplikat Dihapus     : {num_duplicates:>6,} baris                 │
│ Setelah Hapus Duplikat: {len(df_clean):>5,} baris                 │
│ Setelah Balancing    : {len(df_balanced):>6,} baris                 │
├─────────────────────────────────────────────────────────┤
│ DISTRIBUSI AKHIR:                                       │
│   Positive : {df_balanced['sentiment'].value_counts().get('positive', 0):>6,}                                  │
│   Neutral  : {df_balanced['sentiment'].value_counts().get('neutral', 0):>6,}                                  │
│   Negative : {df_balanced['sentiment'].value_counts().get('negative', 0):>6,}                                  │
├─────────────────────────────────────────────────────────┤
│ OUTPUT: {OUTPUT_FILE:<43} │
└─────────────────────────────────────────────────────────┘

✅ Data siap untuk training IndoBERT!

📌 CATATAN PENTING:
   - Data sudah bersih dari duplikat
   - Data sudah seimbang (balanced)
   - Teks asli dipertahankan untuk IndoBERT
   - IndoBERT butuh struktur kalimat asli untuk konteks
''')
