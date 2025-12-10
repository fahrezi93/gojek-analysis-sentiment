"""
Balance dataset dengan FILTER KETAT untuk positive
"""

import pandas as pd
import re

# Load file yang sudah diperbaiki
df_cleaned = pd.read_csv('data/gojek_scraped_3class_RELABELED.csv')
print("📊 Distribusi data yang sudah dibersihkan:")
print(df_cleaned['sentiment'].value_counts())

# Load file original
df_original = pd.read_csv('data/gojek_scraped_3class_20251206_130028.csv')
print(f"\n📂 Data original: {len(df_original)} rows")

# Filter positive dari original dengan rating 4-5
df_positive_original = df_original[
    (df_original['sentiment'] == 'positive') & 
    (df_original['rating'].isin([4, 5]))
][['text', 'sentiment']].copy()

print(f"✅ Data positive dari original (rating 4-5): {len(df_positive_original)}")

# Hapus duplikat dengan data yang sudah ada
existing_texts = set(df_cleaned['text'].tolist())
df_positive_new = df_positive_original[~df_positive_original['text'].isin(existing_texts)]
print(f"📌 Data positive BARU: {len(df_positive_new)}")

# Fungsi KETAT untuk cek positive murni
def is_truly_positive_strict(text):
    text = text.lower()
    
    # BLACKLIST: kata yang membuat tidak murni positive
    blacklist = [
        'tapi', 'namun', 'cuma', 'hanya', 'sayang', 'sayangnya',
        'mahal', 'kemahalan', 'lambat', 'lama', 'lamban',
        'kecewa', 'jelek', 'buruk', 'parah', 'payah',
        'tidak', 'belum', 'gagal', 'batal', 'cancel',
        'error', 'lag', 'lemot', 'susah', 'sulit', 'ribet', 'repot',
        'kenapa', 'mengapa', 'minta', 'tolong', 'mohon',
        'kalo bisa', 'kalau bisa', 'saran', 'harap', 'sebaiknya',
        'aneh', 'lucu', 'gaje', 'males', 'bosan',
        'kurang', 'minim', 'sedikit', 'jarang',
        'gada otak', 'goblok', 'tolol', 'bodo', 'anj'
    ]
    
    # Cek setiap kata blacklist
    for word in blacklist:
        if word in text:
            return False
    
    # WHITELIST: harus ada minimal 1 kata positif
    whitelist = [
        'bagus', 'baik', 'mantap', 'keren', 'oke', 'ok',
        'puas', 'senang', 'suka', 'cinta',
        'membantu', 'mudah', 'praktis', 'simple', 'simpel',
        'cepat', 'lancar', 'smooth', 'aman',
        'recommended', 'recommend', 'terbaik', 'the best',
        'terima kasih', 'thanks', 'makasih', 'thank you',
        'sempurna', 'perfect', 'lengkap', 'memuaskan',
        'nyaman', 'enak', 'asik', 'seru',
        'efisien', 'ekonomis', 'hemat', 'murah',
        'ramah', 'sopan', 'profesional'
    ]
    
    has_positive = any(word in text for word in whitelist)
    
    # Minimal 15 karakter (filter terlalu pendek)
    if len(text.strip()) < 15:
        return False
    
    return has_positive

# Filter positive yang benar-benar murni
df_positive_pure = df_positive_new[df_positive_new['text'].apply(is_truly_positive_strict)]
print(f"✨ Data positive MURNI (filter KETAT): {len(df_positive_pure)}")

# Hitung kebutuhan
neg_count = len(df_cleaned[df_cleaned['sentiment'] == 'negative'])
neu_count = len(df_cleaned[df_cleaned['sentiment'] == 'neutral'])
pos_count = len(df_cleaned[df_cleaned['sentiment'] == 'positive'])

target_count = min(neg_count, neu_count)
needed_positive = target_count - pos_count

print(f"\n🎯 Target balance: {target_count} per kelas")
print(f"🔢 Positive yang dibutuhkan: {needed_positive}")

# Ambil sejumlah yang dibutuhkan
if len(df_positive_pure) >= needed_positive:
    df_additional_positive = df_positive_pure.sample(n=needed_positive, random_state=42)
    print(f"✅ Mengambil {needed_positive} data positive MURNI")
else:
    df_additional_positive = df_positive_pure.copy()
    print(f"⚠️ Hanya ada {len(df_positive_pure)} data positive murni")
    print(f"   Masih kurang {needed_positive - len(df_positive_pure)} data")

# Gabungkan
df_final = pd.concat([df_cleaned, df_additional_positive], ignore_index=True)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n📊 Distribusi setelah penambahan:")
print(df_final['sentiment'].value_counts())

# Balance: ambil jumlah minimal dari ketiga kelas
min_count = df_final['sentiment'].value_counts().min()
print(f"\n⚖️ Balancing ke {min_count} per kelas...")

df_balanced = pd.DataFrame()
for sentiment in ['negative', 'neutral', 'positive']:
    df_class = df_final[df_final['sentiment'] == sentiment]
    df_sampled = df_class.sample(n=min_count, random_state=42)
    df_balanced = pd.concat([df_balanced, df_sampled])

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✅ Distribusi BALANCED FINAL:")
print(df_balanced['sentiment'].value_counts())
print(f"Total: {len(df_balanced)}")

# Save
output_path = 'data/gojek_scraped_3class_RELABELED.csv'
df_balanced.to_csv(output_path, index=False)
print(f"\n💾 File berhasil disimpan: {output_path}")

# Verifikasi sample dari ketiga kelas
print("\n" + "="*80)
print("🔍 VERIFIKASI SAMPLE POSITIVE (harus MURNI tanpa komplain):")
print("="*80)
pos_samples = df_balanced[df_balanced['sentiment'] == 'positive'].sample(15)
for idx, row in pos_samples.iterrows():
    print(f"✅ {row['text'][:130]}...")

print("\n" + "="*80)
print("🔍 VERIFIKASI SAMPLE NEUTRAL:")
print("="*80)
neu_samples = df_balanced[df_balanced['sentiment'] == 'neutral'].sample(10)
for idx, row in neu_samples.iterrows():
    print(f"➖ {row['text'][:130]}...")

print("\n" + "="*80)
print("🔍 VERIFIKASI SAMPLE NEGATIVE:")
print("="*80)
neg_samples = df_balanced[df_balanced['sentiment'] == 'negative'].sample(10)
for idx, row in neg_samples.iterrows():
    print(f"❌ {row['text'][:130]}...")
