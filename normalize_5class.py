"""
Final normalization for 5-class balanced data - handles punctuation-attached slang
"""
import pandas as pd
import re

slang_dict = {
    'gak': 'tidak', 'ga': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak',
    'gk': 'tidak', 'tdk': 'tidak', 'gx': 'tidak', 'ndak': 'tidak',
    'yg': 'yang', 'dgn': 'dengan', 'dg': 'dengan', 'utk': 'untuk',
    'sm': 'sama', 'sma': 'sama', 'dr': 'dari',
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
    'apps': 'aplikasi', 'app': 'aplikasi',
    'mantul': 'mantap', 'mantab': 'mantap',
    'okee': 'oke', 'okeee': 'oke', 'okeh': 'oke',
    'makasi': 'terima kasih', 'makasih': 'terima kasih',
    'thx': 'terima kasih', 'thanks': 'terima kasih', 'tengkyu': 'terima kasih',
    'rekomend': 'rekomendasi', 'rekomen': 'rekomendasi',
    'lelet': 'lambat', 'lemot': 'lambat',
    'zonk': 'buruk',
    'sy': 'saya', 'sya': 'saya', 'aq': 'saya', 'ak': 'saya',
    'gw': 'saya', 'gue': 'saya', 'ane': 'saya',
    'lu': 'kamu', 'lo': 'kamu',
}

def normalize_text_advanced(text):
    """Normalize slang with punctuation handling"""
    if pd.isna(text):
        return text
    text = str(text).lower()
    
    # Use regex word boundaries for each slang word
    for slang, baku in slang_dict.items():
        # Match the slang word as a whole word (with word boundaries)
        text = re.sub(r'\b' + re.escape(slang) + r'\b', baku, text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load
path = 'data/gojek_5class_BALANCED_FIXED.csv'
df = pd.read_csv(path)
print(f'Loaded: {len(df)} rows')

# Count slang BEFORE
slang_check = ['bgt', 'gk', 'gak', 'ga', 'tp', 'yg', 'sdh', 'krn', 'dgn', 'utk', 
               'jg', 'pdhl', 'blm', 'lg', 'bngt', 'sm', 'hrs', 'emg', 'skrg', 'bs',
               'aja', 'udh', 'klo', 'gmn', 'knp', 'gw', 'gue', 'lu', 'lo', 'sy',
               'msh', 'dr', 'jd', 'org', 'brp', 'td', 'bkn', 'jgn', 'mgkn']

before_total = 0
for s in slang_check:
    c = df['text'].str.contains(r'\b' + s + r'\b', regex=True, na=False).sum()
    if c > 0:
        before_total += c

print(f'Slang BEFORE: {before_total}')

# Apply regex-based normalization
print('Applying regex-based normalization...')
df['text'] = df['text'].apply(normalize_text_advanced)
df = df.drop_duplicates(subset=['text'])

# Count slang AFTER
after_total = 0
remaining = []
for s in slang_check:
    c = df['text'].str.contains(r'\b' + s + r'\b', regex=True, na=False).sum()
    if c > 0:
        after_total += c
        remaining.append(f'  {s}: {c}')

print(f'Slang AFTER: {after_total}')
if remaining:
    print('Remaining:')
    for r in remaining:
        print(r)

# Save
df.to_csv(path, index=False)
print(f'\nSaved: {len(df)} rows to {path}')
print('DONE!')
