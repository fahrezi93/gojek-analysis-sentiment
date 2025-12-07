"""
Analisis Kritis - Cek apakah data Neutral masih mengandung sentiment kuat
"""

import pandas as pd
import re

# Load data
df = pd.read_csv('data/processed/gojek_scraped_3class_BALANCED_20251206_130122.csv')

print("=" * 80)
print("🔍 ANALISIS KRITIS - CEK KEBOCORAN LABEL")
print("=" * 80)

# Keyword definitions
STRONG_NEGATIVE = [
    'buruk', 'jelek', 'parah', 'kecewa', 'kesal', 'marah', 'lambat', 'lama',
    'error', 'eror', 'rusak', 'tidak bisa', 'tidak ramah', 'kasar', 'zonk',
    'payah', 'mengecewakan', 'menyebalkan', 'gaje', 'ngaco', 'kapok', 'jera',
    'menyesal', 'susah', 'ribet', 'mahal', 'cancel', 'batal', 'ditolak',
    'tidak puas', 'tidak profesional', 'sial', 'sampah', 'bodoh', 'tolol',
    'ancur', 'hancur', 'gagal', 'bermasalah'
]

STRONG_POSITIVE = [
    'bagus', 'baik', 'puas', 'senang', 'suka', 'cepat', 'ramah', 'nyaman',
    'enak', 'murah', 'mantap', 'recommended', 'terbaik', 'sempurna', 'excellent',
    'hebat', 'luar biasa', 'profesional', 'sopan', 'tepat waktu', 'lancar',
    'mudah', 'membantu', 'praktis', 'efisien', 'terima kasih', 'makasih',
    'sukses', 'keren', 'top', 'jos', 'memuaskan'
]

# Analisis per kelas
print("\n📊 DISTRIBUSI DASAR:")
print(df['sentiment'].value_counts())
print(f"\nTotal: {len(df):,} rows")

# Check Neutral class
df_neutral = df[df['sentiment'] == 'neutral'].copy()
print(f"\n🔍 ANALISIS KELAS NEUTRAL ({len(df_neutral):,} rows):")

# Count strong negative words in neutral
def count_keywords(text, keywords):
    if pd.isna(text):
        return 0
    text_lower = str(text).lower()
    return sum(1 for kw in keywords if kw in text_lower)

df_neutral['neg_keywords'] = df_neutral['text'].apply(lambda x: count_keywords(x, STRONG_NEGATIVE))
df_neutral['pos_keywords'] = df_neutral['text'].apply(lambda x: count_keywords(x, STRONG_POSITIVE))

# Find neutral with strong sentiment
neutral_with_strong_neg = df_neutral[df_neutral['neg_keywords'] >= 2]
neutral_with_strong_pos = df_neutral[df_neutral['pos_keywords'] >= 2]

print(f"\n⚠️ KEBOCORAN DITEMUKAN:")
print(f"   Neutral dengan 2+ kata NEGATIF kuat: {len(neutral_with_strong_neg):,} ({len(neutral_with_strong_neg)/len(df_neutral)*100:.1f}%)")
print(f"   Neutral dengan 2+ kata POSITIF kuat:  {len(neutral_with_strong_pos):,} ({len(neutral_with_strong_pos)/len(df_neutral)*100:.1f}%)")

# Show examples
print(f"\n📋 CONTOH 'NEUTRAL' yang sebenarnya NEGATIF:")
for idx, row in neutral_with_strong_neg.head(10).iterrows():
    print(f"\n   Rating: {row['rating']} | Neg Keywords: {row['neg_keywords']}")
    print(f"   Text: \"{row['text'][:100]}...\"")

print(f"\n📋 CONTOH 'NEUTRAL' yang sebenarnya POSITIF:")
for idx, row in neutral_with_strong_pos.head(10).iterrows():
    print(f"\n   Rating: {row['rating']} | Pos Keywords: {row['pos_keywords']}")
    print(f"   Text: \"{row['text'][:100]}...\"")

# Check other classes
print(f"\n" + "=" * 80)
print("🔍 CEK KELAS LAIN (Negative & Positive):")

df_negative = df[df['sentiment'] == 'negative'].copy()
df_negative['pos_keywords'] = df_negative['text'].apply(lambda x: count_keywords(x, STRONG_POSITIVE))
neg_with_pos = df_negative[df_negative['pos_keywords'] >= 2]

df_positive = df[df['sentiment'] == 'positive'].copy()
df_positive['neg_keywords'] = df_positive['text'].apply(lambda x: count_keywords(x, STRONG_NEGATIVE))
pos_with_neg = df_positive[df_positive['neg_keywords'] >= 2]

print(f"\n   Negative dengan 2+ kata POSITIF: {len(neg_with_pos):,} ({len(neg_with_pos)/len(df_negative)*100:.1f}%)")
print(f"   Positive dengan 2+ kata NEGATIF: {len(pos_with_neg):,} ({len(pos_with_neg)/len(df_positive)*100:.1f}%)")

# Rating distribution per sentiment
print(f"\n" + "=" * 80)
print("📊 RATING DISTRIBUTION PER SENTIMENT:")

for sentiment in ['negative', 'neutral', 'positive']:
    df_sent = df[df['sentiment'] == sentiment]
    print(f"\n{sentiment.upper()}:")
    rating_dist = df_sent['rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        print(f"   Rating {rating}: {count:,} ({count/len(df_sent)*100:.1f}%)")

# Calculate severity
print(f"\n" + "=" * 80)
print("⚠️ KESIMPULAN:")
print("=" * 80)

total_problematic = len(neutral_with_strong_neg) + len(neutral_with_strong_pos)
print(f"\n📉 Data 'Neutral' yang bermasalah: {total_problematic:,} / {len(df_neutral):,} ({total_problematic/len(df_neutral)*100:.1f}%)")

if total_problematic > len(df_neutral) * 0.15:  # > 15%
    print(f"\n🚨 CRITICAL: Kebocoran >15%!")
    print(f"   Ini AKAN menurunkan akurasi model significantly!")
    print(f"   REKOMENDASI: Lakukan re-labeling sebelum training!")
elif total_problematic > len(df_neutral) * 0.10:  # > 10%
    print(f"\n⚠️ WARNING: Kebocoran 10-15%")
    print(f"   Model masih bisa belajar, tapi akurasi class Neutral akan rendah")
    print(f"   REKOMENDASI: Re-labeling untuk hasil optimal")
else:
    print(f"\n✅ ACCEPTABLE: Kebocoran <10%")
    print(f"   Data masih bisa dipakai, tapi ada room for improvement")

print(f"\n💡 AKSI YANG DISARANKAN:")
if total_problematic > len(df_neutral) * 0.15:
    print(f"   1. ❌ JANGAN PAKAI data ini langsung")
    print(f"   2. ✅ Jalankan re-labeling script dulu")
    print(f"   3. ✅ Balance ulang setelah re-labeling")
else:
    print(f"   1. ⚠️ Bisa dipakai, tapi tidak optimal")
    print(f"   2. ✅ Re-labeling disarankan untuk akurasi maksimal")
    print(f"   3. 📊 Expected accuracy: 75-82% (dengan noise)")
    print(f"   4. 📊 Expected accuracy: 88-95% (setelah re-labeling)")

print("=" * 80)
