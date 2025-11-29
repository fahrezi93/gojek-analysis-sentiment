"""
Script untuk Cleaning Data Review Gojek
Untuk persiapan training IndoBERT

Input:
- data/gojek_reviews_3class_raw_balanced.csv
- data/gojek_reviews_5class_raw_balanced.csv

Output:
- data/gojek_reviews_3class_clean.csv
- data/gojek_reviews_5class_clean.csv
"""

import pandas as pd
import re
import string
from datetime import datetime

# =====================================================
# CLEANING FUNCTIONS
# =====================================================

def remove_urls(text):
    """Hapus URL"""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def remove_emails(text):
    """Hapus email"""
    email_pattern = re.compile(r'\S+@\S+')
    return email_pattern.sub('', text)

def remove_phone_numbers(text):
    """Hapus nomor telepon"""
    phone_pattern = re.compile(r'(\+62|62|0)[\s-]?\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}')
    return phone_pattern.sub('', text)

def remove_emojis(text):
    """Hapus emoji"""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"
        u"\u3030"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub('', text)

def remove_html_tags(text):
    """Hapus HTML tags"""
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub('', text)

def remove_special_characters(text):
    """Hapus karakter khusus, simpan huruf, angka, dan spasi"""
    # Keep Indonesian characters
    text = re.sub(r'[^\w\s]', ' ', text)
    return text

def remove_extra_whitespace(text):
    """Hapus spasi berlebih"""
    return ' '.join(text.split())

def remove_single_characters(text):
    """Hapus karakter tunggal (kecuali 'a' dan 'i' yang bermakna dalam bahasa Indonesia)"""
    words = text.split()
    words = [w for w in words if len(w) > 1 or w.lower() in ['a', 'i']]
    return ' '.join(words)

def remove_numbers_only(text):
    """Hapus kata yang hanya berisi angka"""
    words = text.split()
    words = [w for w in words if not w.isdigit()]
    return ' '.join(words)

def normalize_slang(text):
    """Normalisasi kata-kata slang/singkatan umum Indonesia"""
    slang_dict = {
        # Negasi
        'gk': 'tidak',
        'ga': 'tidak',
        'gak': 'tidak',
        'tdk': 'tidak',
        'gx': 'tidak',
        'ngga': 'tidak',
        'nggak': 'tidak',
        'enggak': 'tidak',
        'kagak': 'tidak',
        'kaga': 'tidak',
        # Kata penghubung
        'yg': 'yang',
        'dgn': 'dengan',
        'dg': 'dengan',
        'utk': 'untuk',
        'krn': 'karena',
        'karna': 'karena',
        'krna': 'karena',
        # Waktu
        'sdh': 'sudah',
        'udh': 'sudah',
        'udah': 'sudah',
        'blm': 'belum',
        'blum': 'belum',
        'blom': 'belum',
        'skrg': 'sekarang',
        'skrng': 'sekarang',
        'lg': 'lagi',
        'lgi': 'lagi',
        # Kata umum
        'jg': 'juga',
        'jga': 'juga',
        'jd': 'jadi',
        'jdi': 'jadi',
        'klo': 'kalau',
        'kalo': 'kalau',
        'klau': 'kalau',
        'bs': 'bisa',
        'bsa': 'bisa',
        'dr': 'dari',
        'dri': 'dari',
        'sm': 'sama',
        'sma': 'sama',
        'spt': 'seperti',
        'sprti': 'seperti',
        'spy': 'supaya',
        'biar': 'supaya',
        'tp': 'tapi',
        'tpi': 'tapi',
        # Pronouns
        'sy': 'saya',
        'gw': 'saya',
        'gue': 'saya',
        'gua': 'saya',
        'ane': 'saya',
        'kmu': 'kamu',
        'lu': 'kamu',
        'lo': 'kamu',
        'elu': 'kamu',
        'ente': 'kamu',
        # Kata lain
        'org': 'orang',
        'orng': 'orang',
        'ornag': 'orang',
        'hrs': 'harus',
        'hrus': 'harus',
        'dpt': 'dapat',
        'dpat': 'dapat',
        'dapet': 'dapat',
        'msh': 'masih',
        'msih': 'masih',
        'emg': 'memang',
        'emang': 'memang',
        'mmg': 'memang',
        'knp': 'kenapa',
        'knapa': 'kenapa',
        'gmn': 'bagaimana',
        'gmna': 'bagaimana',
        'gimana': 'bagaimana',
        'dmn': 'dimana',
        'dmna': 'dimana',
        'kpn': 'kapan',
        # Ucapan
        'thx': 'terima kasih',
        'tks': 'terima kasih',
        'thanks': 'terima kasih',
        'makasih': 'terima kasih',
        'mksh': 'terima kasih',
        'makasi': 'terima kasih',
        'trims': 'terima kasih',
        'trmksh': 'terima kasih',
        'ok': 'oke',
        'okey': 'oke',
        'okay': 'oke',
        'oks': 'oke',
        # Intensifier
        'bgt': 'banget',
        'bngt': 'banget',
        'bngtt': 'banget',
        'bgtt': 'banget',
        # Positif expressions
        'mantap': 'mantap',
        'mantab': 'mantap',
        'mantul': 'mantap',
        'mantep': 'mantap',
        'keren': 'keren',
        'josss': 'jos',
        'joss': 'jos',
        # Negatif expressions
        'jelek': 'jelek',
        'jlek': 'jelek',
        'parah': 'parah',
        'ancur': 'hancur',
        'payah': 'payah',
        'nyebelin': 'menyebalkan',
        'sebel': 'kesal',
        'kesel': 'kesal',
        # Kata partikel
        'bkn': 'bukan',
        'bukn': 'bukan',
        'aja': 'saja',
        'doang': 'saja',
        'doank': 'saja',
        'nih': 'ini',
        'tuh': 'itu',
        'bener': 'benar',
        'bnr': 'benar',
        'salh': 'salah',
        'slh': 'salah',
        # Aplikasi Gojek - keep as is
        'aplikasinya': 'aplikasi',
        'appnya': 'aplikasi',
        'app': 'aplikasi',
        'apps': 'aplikasi',
        'apk': 'aplikasi',
        'drivernya': 'driver',
        'drver': 'driver',
        'drvr': 'driver',
        'ojol': 'ojek online',
        # Speed
        'lelet': 'lambat',
        'lemot': 'lambat',
        'cpt': 'cepat',
        'cpet': 'cepat',
        'cepet': 'cepat',
        'lma': 'lama',
    }
    
    words = text.lower().split()
    normalized = []
    for word in words:
        if word in slang_dict:
            if slang_dict[word]:  # Skip empty replacements
                normalized.append(slang_dict[word])
        else:
            normalized.append(word)
    
    return ' '.join(normalized)

def clean_text(text):
    """
    Main cleaning function - gabungan semua tahap cleaning
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Remove URLs
    text = remove_urls(text)
    
    # Step 3: Remove emails
    text = remove_emails(text)
    
    # Step 4: Remove phone numbers
    text = remove_phone_numbers(text)
    
    # Step 5: Remove HTML tags
    text = remove_html_tags(text)
    
    # Step 6: Remove emojis
    text = remove_emojis(text)
    
    # Step 7: Normalize slang/abbreviations
    text = normalize_slang(text)
    
    # Step 8: Remove special characters
    text = remove_special_characters(text)
    
    # Step 9: Remove numbers-only words
    text = remove_numbers_only(text)
    
    # Step 10: Remove single characters
    text = remove_single_characters(text)
    
    # Step 11: Remove extra whitespace
    text = remove_extra_whitespace(text)
    
    return text.strip()

def is_valid_review(text, min_words=3, min_chars=10):
    """
    Check apakah review valid untuk training
    """
    if not text or pd.isna(text):
        return False
    
    text = str(text).strip()
    
    # Minimal karakter
    if len(text) < min_chars:
        return False
    
    # Minimal kata
    if len(text.split()) < min_words:
        return False
    
    # Bukan hanya angka
    if text.replace(' ', '').isdigit():
        return False
    
    # Bukan hanya repeated characters
    if len(set(text.replace(' ', ''))) < 3:
        return False
    
    return True

# =====================================================
# MAIN CLEANING PROCESS
# =====================================================

def clean_dataset(input_path, output_path, dataset_name):
    """
    Clean dataset dan simpan hasilnya
    """
    print(f'\n{"="*60}')
    print(f'🧹 CLEANING: {dataset_name}')
    print(f'{"="*60}')
    
    # Load data
    print(f'\n📂 Loading: {input_path}')
    df = pd.read_csv(input_path)
    original_count = len(df)
    print(f'   Original rows: {original_count:,}')
    
    # Show sample before cleaning
    print(f'\n📝 Sample BEFORE cleaning:')
    for i, row in df.head(3).iterrows():
        print(f'   [{row["sentiment"]}] {row["content"][:80]}...')
    
    # Step 1: Clean text
    print(f'\n🔄 Step 1: Cleaning text...')
    df['content_clean'] = df['content'].apply(clean_text)
    
    # Step 2: Remove invalid reviews
    print(f'🔄 Step 2: Removing invalid reviews...')
    df['is_valid'] = df['content_clean'].apply(is_valid_review)
    invalid_count = len(df[~df['is_valid']])
    df = df[df['is_valid']].drop(columns=['is_valid'])
    print(f'   Removed {invalid_count:,} invalid reviews')
    
    # Step 3: Remove duplicates
    print(f'🔄 Step 3: Removing duplicates...')
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['content_clean'])
    dup_count = before_dedup - len(df)
    print(f'   Removed {dup_count:,} duplicates')
    
    # Step 4: Remove empty content_clean
    print(f'🔄 Step 4: Removing empty content...')
    before_empty = len(df)
    df = df[df['content_clean'].str.len() > 0]
    empty_count = before_empty - len(df)
    print(f'   Removed {empty_count:,} empty reviews')
    
    # Show sample after cleaning
    print(f'\n📝 Sample AFTER cleaning:')
    for i, row in df.head(3).iterrows():
        print(f'   [{row["sentiment"]}] {row["content_clean"][:80]}...')
    
    # Distribution check
    print(f'\n📊 Distribution after cleaning:')
    dist = df['sentiment'].value_counts()
    for sentiment, count in dist.items():
        pct = count / len(df) * 100
        print(f'   {sentiment}: {count:,} ({pct:.1f}%)')
    
    # Balance data (undersample to min class)
    print(f'\n⚖️ Balancing data...')
    min_count = dist.min()
    balanced_dfs = []
    for sentiment in dist.index:
        df_class = df[df['sentiment'] == sentiment]
        df_sampled = df_class.sample(n=min_count, random_state=42)
        balanced_dfs.append(df_sampled)
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f'\n📊 Final balanced distribution:')
    for sentiment, count in df_balanced['sentiment'].value_counts().items():
        pct = count / len(df_balanced) * 100
        print(f'   {sentiment}: {count:,} ({pct:.1f}%)')
    
    # Save
    print(f'\n💾 Saving: {output_path}')
    df_balanced.to_csv(output_path, index=False)
    
    # Summary
    print(f'\n📊 SUMMARY:')
    print(f'   Original: {original_count:,}')
    print(f'   After cleaning: {len(df):,}')
    print(f'   After balancing: {len(df_balanced):,}')
    print(f'   Reduction: {(1 - len(df_balanced)/original_count)*100:.1f}%')
    
    return df_balanced

def main():
    print('\n' + '='*60)
    print('🧹 DATA CLEANING FOR INDOBERT TRAINING')
    print('='*60)
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    # Clean 3-class data
    df_3class = clean_dataset(
        input_path='data/gojek_reviews_3class_raw_balanced.csv',
        output_path='data/gojek_reviews_3class_clean.csv',
        dataset_name='3-Class Dataset'
    )
    
    # Clean 5-class data
    df_5class = clean_dataset(
        input_path='data/gojek_reviews_5class_raw_balanced.csv',
        output_path='data/gojek_reviews_5class_clean.csv',
        dataset_name='5-Class Dataset'
    )
    
    # Final summary
    print('\n' + '='*60)
    print('✅ CLEANING COMPLETED!')
    print('='*60)
    print(f'\n📁 Output files:')
    print(f'   1. data/gojek_reviews_3class_clean.csv ({len(df_3class):,} rows)')
    print(f'   2. data/gojek_reviews_5class_clean.csv ({len(df_5class):,} rows)')
    print(f'\n✅ Data siap untuk training IndoBERT!')
    print('='*60)

if __name__ == '__main__':
    main()
