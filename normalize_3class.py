"""Apply same normalization to 3-class BALANCED data for full consistency"""
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
    'lelet': 'lambat', 'lemot': 'lambat', 'zonk': 'buruk',
    'sy': 'saya', 'sya': 'saya', 'aq': 'saya', 'ak': 'saya',
    'gw': 'saya', 'gue': 'saya', 'ane': 'saya',
    'lu': 'kamu', 'lo': 'kamu',
}

def normalize(text):
    if pd.isna(text): return text
    text = str(text).lower()
    for slang, baku in slang_dict.items():
        text = re.sub(r'\b' + re.escape(slang) + r'\b', baku, text)
    return re.sub(r'\s+', ' ', text).strip()

path = 'data/gojek_3class_BALANCED.csv'
df = pd.read_csv(path)
print(f'Loaded: {len(df)} rows')
df['text'] = df['text'].apply(normalize)
df = df.drop_duplicates(subset=['text'])
df.to_csv(path, index=False)
print(f'Saved: {len(df)} rows')

# Verify
slang_check = ['bgt','gk','gak','ga','tp','yg','sdh','krn','dgn','utk','jg','pdhl','blm','lg','bngt','sm','hrs','emg','skrg','bs','aja','udh','klo','gmn','knp','gw','gue','lu','lo','sy','msh','dr','jd','org','brp','td','bkn','jgn','mgkn']
total = sum(df['text'].str.contains(r'\b' + s + r'\b', regex=True, na=False).sum() for s in slang_check)
print(f'Remaining slang: {total}')
