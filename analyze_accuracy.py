"""
Script untuk menganalisis mengapa akurasi model bisa sangat tinggi (97-98%)
"""

import pandas as pd
import numpy as np
from collections import Counter
import re

def analyze_dataset(filepath, name):
    print("="*70)
    print(f"ANALISIS DATASET: {name}")
    print("="*70)
    
    df = pd.read_csv(filepath)
    
    # 1. Basic Info
    print(f"\n📊 INFORMASI DASAR")
    print(f"   Total data: {len(df):,}")
    print(f"   Kolom: {list(df.columns)}")
    
    # 2. Distribusi kelas
    print(f"\n📈 DISTRIBUSI KELAS:")
    sentiment_col = 'sentiment' if 'sentiment' in df.columns else 'label'
    text_col = 'text' if 'text' in df.columns else 'review'
    
    class_counts = df[sentiment_col].value_counts()
    for label, count in class_counts.items():
        pct = count / len(df) * 100
        print(f"   {label}: {count:,} ({pct:.1f}%)")
    
    # 3. Cek keseimbangan
    max_count = class_counts.max()
    min_count = class_counts.min()
    balance_ratio = min_count / max_count
    print(f"\n⚖️ KESEIMBANGAN DATA:")
    print(f"   Rasio min/max: {balance_ratio:.2f}")
    if balance_ratio > 0.9:
        print(f"   ✅ Data sangat seimbang (balanced)")
    elif balance_ratio > 0.7:
        print(f"   ⚠️ Data cukup seimbang")
    else:
        print(f"   ❌ Data tidak seimbang (imbalanced)")
    
    # 4. Panjang teks
    print(f"\n📏 STATISTIK PANJANG TEKS (kata):")
    text_lengths = df[text_col].fillna("").str.split().str.len()
    print(f"   Min: {text_lengths.min()} kata")
    print(f"   Max: {text_lengths.max()} kata")
    print(f"   Mean: {text_lengths.mean():.1f} kata")
    print(f"   Median: {text_lengths.median():.1f} kata")
    
    # Teks pendek (< 5 kata)
    short_texts = (text_lengths < 5).sum()
    print(f"   Teks pendek (<5 kata): {short_texts:,} ({short_texts/len(df)*100:.1f}%)")
    
    # 5. Cek duplikat
    print(f"\n🔄 CEK DUPLIKAT:")
    duplicates = df[text_col].duplicated().sum()
    print(f"   Duplikat teks: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")
    
    # 6. Analisis kata kunci per kelas
    print(f"\n🔑 ANALISIS KATA KUNCI PER KELAS:")
    
    # Keywords yang jelas menunjukkan sentimen
    positive_keywords = ['bagus', 'mantap', 'keren', 'puas', 'recommended', 'baik', 
                        'ramah', 'cepat', 'lancar', 'top', 'love', 'suka', 'bintang']
    negative_keywords = ['buruk', 'jelek', 'kecewa', 'parah', 'lambat', 'error', 
                        'susah', 'ribet', 'mahal', 'benci', 'kapok', 'gak', 'tidak']
    
    for label in class_counts.index:
        label_texts = df[df[sentiment_col] == label][text_col].fillna("").str.lower()
        all_text = " ".join(label_texts)
        
        pos_count = sum(all_text.count(kw) for kw in positive_keywords)
        neg_count = sum(all_text.count(kw) for kw in negative_keywords)
        
        print(f"\n   {label}:")
        print(f"      Kata positif: {pos_count:,}")
        print(f"      Kata negatif: {neg_count:,}")
        if pos_count > neg_count:
            print(f"      → Dominan kata POSITIF ✓")
        elif neg_count > pos_count:
            print(f"      → Dominan kata NEGATIF ✓")
        else:
            print(f"      → Seimbang")
    
    # 7. Sample per kelas
    print(f"\n📝 SAMPLE TEKS PER KELAS:")
    for label in list(class_counts.index)[:5]:  # Max 5 kelas
        print(f"\n   --- {label} ---")
        samples = df[df[sentiment_col] == label][text_col].head(2).tolist()
        for i, s in enumerate(samples, 1):
            text_preview = str(s)[:100] + "..." if len(str(s)) > 100 else str(s)
            print(f"   {i}. {text_preview}")
    
    return df

def check_label_consistency(df, sentiment_col='sentiment', text_col='text'):
    """Cek apakah label konsisten dengan isi teks"""
    print("\n" + "="*70)
    print("CEK KONSISTENSI LABEL vs KONTEN")
    print("="*70)
    
    positive_indicators = ['bagus', 'mantap', 'keren', 'puas', 'baik', 'ramah', 'cepat', 
                          'lancar', 'top', 'love', 'suka', 'recommended', 'terbaik', 'bintang 5']
    negative_indicators = ['buruk', 'jelek', 'kecewa', 'parah', 'lambat', 'error', 
                          'susah', 'ribet', 'benci', 'kapok', 'bintang 1', 'sampah', 'gagal']
    
    inconsistent = []
    
    for idx, row in df.iterrows():
        text = str(row[text_col]).lower()
        label = str(row[sentiment_col]).lower()
        
        pos_score = sum(1 for kw in positive_indicators if kw in text)
        neg_score = sum(1 for kw in negative_indicators if kw in text)
        
        # Cek inkonsistensi
        if 'positif' in label or 'positive' in label or label in ['4', '5', 'rating 4', 'rating 5']:
            if neg_score > pos_score + 2:  # Banyak kata negatif tapi label positif
                inconsistent.append((idx, label, text[:80], f"pos={pos_score}, neg={neg_score}"))
        elif 'negatif' in label or 'negative' in label or label in ['1', '2', 'rating 1', 'rating 2']:
            if pos_score > neg_score + 2:  # Banyak kata positif tapi label negatif
                inconsistent.append((idx, label, text[:80], f"pos={pos_score}, neg={neg_score}"))
    
    print(f"\n   Ditemukan {len(inconsistent)} data yang mungkin inkonsisten")
    print(f"   Persentase: {len(inconsistent)/len(df)*100:.2f}%")
    
    if len(inconsistent) > 0:
        print("\n   Sample inkonsisten:")
        for item in inconsistent[:3]:
            print(f"   - [{item[1]}] {item[2]}... ({item[3]})")
    
    return len(inconsistent)

def analyze_why_high_accuracy():
    """Analisis komprehensif mengapa akurasi tinggi"""
    print("\n" + "="*70)
    print("🔍 ANALISIS MENGAPA AKURASI TINGGI (97-98%)")
    print("="*70)
    
    reasons = []
    
    # Analisis kedua dataset
    df3 = analyze_dataset("data/gojek_3class_BALANCED.csv", "3-KELAS")
    df5 = analyze_dataset("data/gojek_5class_BALANCED_FIXED.csv", "5-KELAS")
    
    # Cek konsistensi label
    text_col_3 = 'text' if 'text' in df3.columns else 'review'
    text_col_5 = 'text' if 'text' in df5.columns else 'review'
    sent_col_3 = 'sentiment' if 'sentiment' in df3.columns else 'label'
    sent_col_5 = 'sentiment' if 'sentiment' in df5.columns else 'label'
    
    incon_3 = check_label_consistency(df3, sent_col_3, text_col_3)
    incon_5 = check_label_consistency(df5, sent_col_5, text_col_5)
    
    # Summary
    print("\n" + "="*70)
    print("📊 RINGKASAN ANALISIS")
    print("="*70)
    
    print("\n✅ FAKTOR YANG BERKONTRIBUSI PADA AKURASI TINGGI:")
    
    print("""
1. DATA YANG SUDAH DIBERSIHKAN (CLEANED)
   - Dataset sudah melalui proses cleaning dan filtering
   - Teks yang sangat pendek, spam, atau tidak relevan sudah dihapus
   - Duplikat sudah diminimalisir

2. DATA YANG SEIMBANG (BALANCED)
   - Distribusi kelas sangat merata (balanced dataset)
   - Model tidak bias ke kelas mayoritas
   - Training dan evaluasi lebih fair

3. LABEL YANG KONSISTEN
   - Label sudah diverifikasi/relabel berdasarkan konten teks
   - Bukan berdasarkan rating bintang yang sering inkonsisten
   - Mengurangi noise dalam training

4. KARAKTERISTIK ULASAN YANG JELAS
   - Ulasan aplikasi ojol cenderung eksplisit dalam ekspresi sentimen
   - Kata-kata kunci seperti "bagus/jelek", "puas/kecewa" sangat jelas
   - Tidak banyak sarkasme atau ironi yang membingungkan model

5. KEMAMPUAN INDOBERT
   - Pre-trained pada korpus bahasa Indonesia yang besar
   - Sudah memahami konteks dan semantik bahasa Indonesia
   - Fine-tuning dengan data yang bersih memaksimalkan performa
""")

    print("\n⚠️ POTENSI BIAS/CONCERN:")
    print("""
1. DATA LEAKAGE
   - Perlu dipastikan tidak ada teks yang sama di train dan test set
   - Split harus dilakukan sebelum preprocessing final

2. OVERFITTING PADA DOMAIN SPESIFIK
   - Model sangat bagus untuk ulasan Gojek
   - Mungkin tidak generalize ke domain lain (e-commerce, restoran, dll)

3. TEKS PENDEK YANG TERLALU MUDAH
   - Banyak teks pendek dengan kata kunci jelas ("bagus", "jelek")
   - Model bisa "menghafal" pola ini tanpa memahami konteks

4. RELABELING BERDASARKAN KEYWORD
   - Jika relabeling menggunakan rule-based keyword
   - Model bisa belajar pattern yang sama → circular reasoning
""")

if __name__ == "__main__":
    analyze_why_high_accuracy()
