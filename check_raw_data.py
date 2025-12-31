"""
Script untuk mengecek apakah data scraped_all masih raw/kotor atau sudah cleaned
"""
import pandas as pd
import re

print('='*70)
print('ANALISIS DATA SCRAPED_ALL (Apakah Sudah Cleaned atau Masih Raw?)')
print('='*70)

# Load data scraped_all
df = pd.read_csv('data/gojek_scraped_3class_all.csv')

print(f'\nTotal data: {len(df):,}')
print(f'Kolom: {list(df.columns)}')

# Sample lihat isi
print('\n' + '='*70)
print('SAMPLE DATA (5 baris pertama):')
print('='*70)
for i, row in df.head(5).iterrows():
    text = str(row['text'])[:100]
    sent = row.get('sentiment', '?')
    print(f'{i+1}. [{sent}] {text}...')

# Cek apakah ada emoji
print('\n' + '='*70)
print('CEK EMOJI:')
print('='*70)
emoji_pattern = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
has_emoji = df['text'].apply(lambda x: bool(emoji_pattern.search(str(x))))
print(f'Data dengan emoji: {has_emoji.sum()} ({has_emoji.sum()/len(df)*100:.1f}%)')

# Sample data dengan emoji
if has_emoji.sum() > 0:
    sample_emoji = df[has_emoji]['text'].head(3).tolist()
    print('Sample:')
    for s in sample_emoji:
        print(f'  "{s[:80]}..."')

# Cek URL
print('\n' + '='*70)
print('CEK URL:')
print('='*70)
url_pattern = df['text'].apply(lambda x: bool(re.search(r'http[s]?://', str(x))))
print(f'Data dengan URL: {url_pattern.sum()}')

# Cek slang yang belum dinormalisasi
print('\n' + '='*70)
print('CEK SLANG (Apakah Sudah Dinormalisasi?):')
print('='*70)
slang_words = ['gw', 'gue', 'lu', 'lo', 'bgt', 'bngt', 'gk', 'ga', 'gak', 'yg', 'skrg', 'dgn', 'utk']
for slang in slang_words:
    count = df['text'].str.lower().str.contains(r'\b' + slang + r'\b', regex=True).sum()
    if count > 0:
        print(f'  "{slang}": {count:,} data')

# Cek karakter berulang (baguuuus, driverrrr)
print('\n' + '='*70)
print('CEK KARAKTER BERULANG (3x atau lebih):')
print('='*70)
repeated = df['text'].apply(lambda x: bool(re.search(r'(.)\1{2,}', str(x))))
print(f'Data dengan karakter berulang: {repeated.sum()} ({repeated.sum()/len(df)*100:.1f}%)')

# Cek duplikat
print('\n' + '='*70)
print('CEK DUPLIKAT:')
print('='*70)
duplicates = df['text'].duplicated().sum()
print(f'Duplikat: {duplicates:,} ({duplicates/len(df)*100:.2f}%)')

# Distribusi sentiment
print('\n' + '='*70)
print('DISTRIBUSI SENTIMENT:')
print('='*70)
for sent, count in df['sentiment'].value_counts().items():
    pct = count/len(df)*100
    print(f'  {sent}: {count:,} ({pct:.1f}%)')

# ============================================
# BANDINGKAN DENGAN DATA BALANCED (CLEANED)
# ============================================
print('\n' + '='*70)
print('PERBANDINGAN DENGAN DATA BALANCED (CLEANED):')
print('='*70)

df_clean = pd.read_csv('data/gojek_3class_BALANCED.csv')

# Cek slang di data clean
slang_clean = 0
for slang in slang_words:
    slang_clean += df_clean['text'].str.lower().str.contains(r'\b' + slang + r'\b', regex=True).sum()

slang_raw = 0
for slang in slang_words:
    slang_raw += df['text'].str.lower().str.contains(r'\b' + slang + r'\b', regex=True).sum()

emoji_clean = df_clean['text'].apply(lambda x: bool(emoji_pattern.search(str(x)))).sum()
emoji_raw = has_emoji.sum()

repeated_clean = df_clean['text'].apply(lambda x: bool(re.search(r'(.)\1{2,}', str(x)))).sum()
repeated_raw = repeated.sum()

dup_clean = df_clean['text'].duplicated().sum()
dup_raw = duplicates

print(f'\n{"Aspek":<25} {"Raw (scraped_all)":<20} {"Cleaned (balanced)":<20}')
print('-'*65)
print(f'{"Total Data":<25} {len(df):<20,} {len(df_clean):<20,}')
print(f'{"Emoji":<25} {emoji_raw:<20} {emoji_clean:<20}')
print(f'{"Slang Words":<25} {slang_raw:<20,} {slang_clean:<20,}')
print(f'{"Karakter Berulang":<25} {repeated_raw:<20} {repeated_clean:<20}')
print(f'{"Duplikat":<25} {dup_raw:<20} {dup_clean:<20}')

# Kesimpulan
print('\n' + '='*70)
print('KESIMPULAN:')
print('='*70)
if slang_raw > slang_clean or emoji_raw > emoji_clean:
    print('⚠️  Data scraped_all SUDAH melalui SEBAGIAN proses cleaning')
    print('    (emoji mungkin sudah dihapus, tapi slang belum dinormalisasi)')
else:
    print('✅ Data scraped_all sudah melalui proses cleaning yang sama')
