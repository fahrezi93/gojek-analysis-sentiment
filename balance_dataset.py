"""
Balance dataset dengan mengambil data positive dari file original (yang punya rating)
"""

import pandas as pd

# Load file yang sudah diperbaiki
df_cleaned = pd.read_csv('data/gojek_scraped_3class_RELABELED.csv')
print("📊 Distribusi data yang sudah dibersihkan:")
print(df_cleaned['sentiment'].value_counts())
print(f"Total: {len(df_cleaned)}")

# Load file original yang masih punya rating
df_original = pd.read_csv('data/gojek_scraped_3class_20251206_130028.csv')
print(f"\n📂 Data original: {len(df_original)} rows")
print(df_original['sentiment'].value_counts())

# Filter positive dari original (rating 4-5)
if 'rating' in df_original.columns:
    df_positive_original = df_original[
        (df_original['sentiment'] == 'positive') & 
        (df_original['rating'].isin([4, 5]))
    ].copy()
    print(f"\n✅ Data positive dari original (rating 4-5): {len(df_positive_original)}")
else:
    df_positive_original = df_original[df_original['sentiment'] == 'positive'].copy()
    print(f"\n✅ Data positive dari original: {len(df_positive_original)}")

# Ambil hanya kolom text dan sentiment
df_positive_original = df_positive_original[['text', 'sentiment']]

# Hapus duplikat dengan data yang sudah ada
existing_texts = set(df_cleaned['text'].tolist())
df_positive_new = df_positive_original[~df_positive_original['text'].isin(existing_texts)]
print(f"📌 Data positive BARU (belum ada di cleaned): {len(df_positive_new)}")

# Hitung berapa yang dibutuhkan untuk balance
neg_count = len(df_cleaned[df_cleaned['sentiment'] == 'negative'])
neu_count = len(df_cleaned[df_cleaned['sentiment'] == 'neutral'])
pos_count = len(df_cleaned[df_cleaned['sentiment'] == 'positive'])

print(f"\n📊 Distribusi saat ini:")
print(f"  Negative: {neg_count}")
print(f"  Neutral:  {neu_count}")
print(f"  Positive: {pos_count}")

# Target: balance ke jumlah terkecil antara neg dan neu
target_count = min(neg_count, neu_count)
needed_positive = target_count - pos_count

print(f"\n🎯 Target balance: {target_count} per kelas")
print(f"🔢 Positive yang dibutuhkan: {needed_positive}")

# Fungsi untuk cek apakah text benar-benar positive (tanpa komplain)
def is_truly_positive(text):
    text = text.lower()
    negative_keywords = [
        'tapi', 'namun', 'cuma', 'hanya', 'sayang', 'mahal', 'lambat',
        'lama', 'kecewa', 'jelek', 'buruk', 'tidak', 'belum', 'gagal',
        'error', 'lag', 'lemot', 'susah', 'sulit', 'ribet'
    ]
    return not any(word in text for word in negative_keywords)

# Filter positive yang benar-benar murni
df_positive_pure = df_positive_new[df_positive_new['text'].apply(is_truly_positive)]
print(f"✨ Data positive MURNI (tanpa komplain): {len(df_positive_pure)}")

# Ambil sejumlah yang dibutuhkan
if len(df_positive_pure) >= needed_positive:
    df_additional_positive = df_positive_pure.sample(n=needed_positive, random_state=42)
    print(f"✅ Mengambil {needed_positive} data positive MURNI")
else:
    # Jika kurang, ambil semua yang murni + sisanya dari yang biasa
    df_additional_positive = df_positive_pure.copy()
    remaining_needed = needed_positive - len(df_positive_pure)
    
    if remaining_needed > 0 and len(df_positive_new) > len(df_positive_pure):
        df_positive_mixed = df_positive_new[~df_positive_new['text'].apply(is_truly_positive)]
        if len(df_positive_mixed) >= remaining_needed:
            df_additional = df_positive_mixed.sample(n=remaining_needed, random_state=42)
            df_additional_positive = pd.concat([df_additional_positive, df_additional])
        else:
            df_additional_positive = pd.concat([df_additional_positive, df_positive_mixed])
    
    print(f"✅ Mengambil {len(df_additional_positive)} data positive (murni + mixed)")

# Gabungkan dengan data yang sudah dibersihkan
df_final = pd.concat([df_cleaned, df_additional_positive], ignore_index=True)

# Shuffle data
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n📊 Distribusi FINAL:")
print(df_final['sentiment'].value_counts())
print(f"Total: {len(df_final)}")

# Balance final: ambil jumlah sama untuk setiap kelas
min_class_count = df_final['sentiment'].value_counts().min()
print(f"\n⚖️ Balancing ke {min_class_count} per kelas...")

df_balanced = pd.DataFrame()
for sentiment in ['negative', 'neutral', 'positive']:
    df_class = df_final[df_final['sentiment'] == sentiment]
    df_sampled = df_class.sample(n=min_class_count, random_state=42)
    df_balanced = pd.concat([df_balanced, df_sampled])

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✅ Distribusi BALANCED FINAL:")
print(df_balanced['sentiment'].value_counts())
print(f"Total: {len(df_balanced)}")

# Save
output_path = 'data/gojek_scraped_3class_RELABELED.csv'
df_balanced.to_csv(output_path, index=False)
print(f"\n💾 File berhasil disimpan: {output_path}")

# Sample check
print("\n" + "="*80)
print("🔍 SAMPLE POSITIVE YANG DITAMBAHKAN:")
print("="*80)
for idx, row in df_additional_positive.head(10).iterrows():
    print(f"✅ {row['text'][:120]}...")
