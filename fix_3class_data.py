"""
Script untuk memperbaiki labeling data 3 class
1. Load dari raw balanced
2. Clean text
3. Re-label berdasarkan kombinasi rating + analisis konten
4. Balance data
5. Simpan ke clean.csv
"""

import pandas as pd
import re

print("="*60)
print("STEP 1: LOAD DATA RAW")
print("="*60)

df = pd.read_csv('data/gojek_reviews_3class_raw_balanced.csv')
print(f"Total data: {len(df)}")
print(f"Distribusi awal:")
print(df['sentiment'].value_counts())

print("\n" + "="*60)
print("STEP 2: CLEAN TEXT")
print("="*60)

def clean_text(text):
    """Clean dan normalize text Indonesia"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Hapus emoji dan karakter khusus
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize slang Indonesia
    slang_dict = {
        'gak': 'tidak', 'ga': 'tidak', 'gk': 'tidak', 'g': 'tidak',
        'ngga': 'tidak', 'nggak': 'tidak', 'enggak': 'tidak',
        'tdk': 'tidak', 'gx': 'tidak', 'kagak': 'tidak',
        'bgt': 'banget', 'bngt': 'banget', 'bngtt': 'banget',
        'bkn': 'bukan', 'blm': 'belum', 'udh': 'sudah', 'udah': 'sudah',
        'sdh': 'sudah', 'dah': 'sudah',
        'yg': 'yang', 'dgn': 'dengan', 'dg': 'dengan',
        'utk': 'untuk', 'u': 'untuk', 'bwt': 'buat',
        'krn': 'karena', 'karna': 'karena', 'krna': 'karena',
        'tp': 'tapi', 'tpi': 'tapi',
        'sm': 'sama', 'sma': 'sama',
        'lg': 'lagi', 'lgi': 'lagi',
        'jg': 'juga', 'jga': 'juga',
        'aja': 'saja', 'aj': 'saja',
        'bs': 'bisa', 'bsa': 'bisa',
        'klo': 'kalau', 'kl': 'kalau', 'klu': 'kalau',
        'gmn': 'bagaimana', 'gimana': 'bagaimana',
        'knp': 'kenapa', 'knpa': 'kenapa',
        'org': 'orang', 'orng': 'orang',
        'dpt': 'dapat', 'dpet': 'dapat', 'dapet': 'dapat',
        'hrs': 'harus', 'hrus': 'harus',
        'msh': 'masih', 'msih': 'masih',
        'skrg': 'sekarang', 'skrng': 'sekarang',
        'trs': 'terus', 'trus': 'terus',
        'bnyk': 'banyak', 'byk': 'banyak',
        'sgt': 'sangat', 'sngat': 'sangat',
        'bgus': 'bagus', 'bgs': 'bagus',
        'jln': 'jalan', 'jlnn': 'jalan',
        'mslh': 'masalah', 'mslah': 'masalah',
        'aplksi': 'aplikasi', 'apk': 'aplikasi', 'app': 'aplikasi',
        'nyebelin': 'menyebalkan', 'sebel': 'menyebalkan',
        'lemot': 'lambat', 'lelet': 'lambat',
    }
    
    words = text.split()
    words = [slang_dict.get(w, w) for w in words]
    text = ' '.join(words)
    
    # Hapus multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

df['content_clean'] = df['content'].apply(clean_text)
print(f"Text cleaning selesai")
print(f"Contoh: {df['content_clean'].iloc[0][:100]}...")

print("\n" + "="*60)
print("STEP 3: RE-LABEL BERDASARKAN KONTEN")
print("="*60)

# Keywords untuk deteksi sentiment
NEGATIVE_KEYWORDS = [
    # Kata negatif kuat
    'kecewa', 'parah', 'buruk', 'jelek', 'sampah', 'maling', 'tolol', 'bodoh',
    'menyebalkan', 'kesal', 'marah', 'emosi', 'kapok', 'rugi', 'benci',
    'tidak profesional', 'tidak sopan', 'kurang ajar', 'belagu', 'sombong',
    # Masalah layanan
    'susah dapat', 'susah banget', 'lama banget', 'tidak bisa', 'gagal',
    'error', 'bug', 'lambat', 'cancel', 'dibatalkan', 'tidak respon',
    'tidak jalan', 'macet', 'delay', 'nunggu lama',
    # Keluhan harga
    'mahal', 'kemahalan', 'tidak worth',
    # Uninstall/pindah
    'uninstall', 'hapus', 'pindah', 'aplikasi sebelah', 'mending pakai',
    # Kata negatif umum
    'tidak ada', 'tidak dapat', 'tidak mau', 'tidak responsif',
    'jarang', 'susah', 'ribet', 'repot',
]

POSITIVE_KEYWORDS = [
    # Kata positif kuat
    'bagus', 'mantap', 'keren', 'hebat', 'luar biasa', 'terbaik', 'recommended',
    'puas', 'senang', 'suka', 'love', 'cinta', 'terima kasih', 'terimakasih',
    'sukses', 'lancar', 'aman', 'nyaman', 'enak',
    # Pujian layanan
    'ramah', 'sopan', 'cepat', 'tepat waktu', 'profesional',
    'membantu', 'memuaskan', 'worth it', 'worth',
    # Kata positif umum
    'sangat baik', 'sangat bagus', 'sangat membantu', 'sangat puas',
    'selalu', 'setia', 'favorit', 'andalan',
]

def count_keywords(text, keywords):
    """Hitung jumlah keyword yang muncul dalam teks"""
    text_lower = text.lower()
    count = 0
    for kw in keywords:
        if kw in text_lower:
            count += 1
    return count

def analyze_sentiment(row):
    """
    Analisis sentiment berdasarkan kombinasi rating dan konten
    """
    content = str(row['content_clean']).lower()
    rating = row['rating']
    
    neg_count = count_keywords(content, NEGATIVE_KEYWORDS)
    pos_count = count_keywords(content, POSITIVE_KEYWORDS)
    
    # Rule-based sentiment dengan prioritas konten
    
    # 1. Rating 1-2: Hampir pasti negative
    if rating <= 2:
        return 'negative'
    
    # 2. Rating 5: Cenderung positive, kecuali konten sangat negatif
    if rating == 5:
        if neg_count > pos_count + 3:
            return 'neutral'  # Anomali - rating tinggi tapi konten negatif
        return 'positive'
    
    # 3. Rating 3: Analisis konten
    if rating == 3:
        if neg_count >= 2 and neg_count > pos_count:
            return 'negative'
        if pos_count >= 2 and pos_count > neg_count:
            return 'positive'
        return 'neutral'
    
    # 4. Rating 4: Cenderung positive, tapi cek konten
    if rating == 4:
        if neg_count >= 3 and neg_count > pos_count:
            return 'negative'  # Banyak keluhan meski rating 4
        if neg_count >= 2 and neg_count > pos_count:
            return 'neutral'  # Ada keluhan
        return 'positive'
    
    return 'neutral'

df['sentiment_new'] = df.apply(analyze_sentiment, axis=1)

# Bandingkan perubahan
changes = df[df['sentiment'] != df['sentiment_new']]
print(f"Jumlah data yang berubah label: {len(changes)} ({len(changes)/len(df)*100:.1f}%)")

print("\nDistribusi setelah re-label:")
print(df['sentiment_new'].value_counts())

# Tampilkan contoh perubahan
print("\nContoh perubahan label:")
for old_sent in ['positive', 'neutral']:
    for new_sent in ['negative', 'neutral', 'positive']:
        if old_sent != new_sent:
            subset = changes[(changes['sentiment'] == old_sent) & (changes['sentiment_new'] == new_sent)]
            if len(subset) > 0:
                print(f"\n{old_sent.upper()} → {new_sent.upper()} ({len(subset)} data):")
                for _, row in subset.head(2).iterrows():
                    content_preview = row['content'][:70] + "..." if len(str(row['content'])) > 70 else row['content']
                    print(f"  R{row['rating']}: {content_preview}")

# Update kolom sentiment
df['sentiment'] = df['sentiment_new']
df = df.drop(columns=['sentiment_new'])

print("\n" + "="*60)
print("STEP 4: BALANCE DATA")
print("="*60)

min_count = df['sentiment'].value_counts().min()
print(f"Jumlah minimum per kelas: {min_count}")

# Undersample setiap kelas
balanced_dfs = []
for sentiment in ['negative', 'neutral', 'positive']:
    sentiment_df = df[df['sentiment'] == sentiment]
    if len(sentiment_df) > min_count:
        sentiment_df = sentiment_df.sample(n=min_count, random_state=42)
    balanced_dfs.append(sentiment_df)

df_balanced = pd.concat(balanced_dfs, ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nDistribusi akhir (balanced):")
print(df_balanced['sentiment'].value_counts())
print(f"Total: {len(df_balanced)} data")

print("\n" + "="*60)
print("STEP 5: SIMPAN DATA")
print("="*60)

df_balanced.to_csv('data/gojek_reviews_3class_clean.csv', index=False)
print(f"✅ Data disimpan ke data/gojek_reviews_3class_clean.csv")

# Verifikasi
print("\nVerifikasi contoh data per kelas:")
for sent in ['negative', 'neutral', 'positive']:
    sample = df_balanced[df_balanced['sentiment'] == sent].iloc[0]
    print(f"\n{sent.upper()} (Rating {sample['rating']}):")
    print(f"  {sample['content'][:80]}...")
