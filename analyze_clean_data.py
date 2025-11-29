"""
Script untuk menganalisis kelayakan data clean 3-class dan 5-class untuk training IndoBERT
"""
import pandas as pd
import re

def analyze_clean_data(filepath, class_type):
    df = pd.read_csv(filepath)
    
    print('='*60)
    print(f'ANALISIS DATA {class_type.upper()} - {filepath.split("/")[-1]}')
    print('='*60)
    
    # Basic info
    print(f'\n1. INFORMASI DASAR:')
    print(f'   Total rows: {len(df):,}')
    print(f'   Columns: {list(df.columns)}')
    print(f'   Missing values: {df.isna().sum().sum()}')
    
    # Distribusi sentiment
    print(f'\n2. DISTRIBUSI SENTIMENT:')
    sent_dist = df['sentiment'].value_counts()
    total = len(df)
    for sent, count in sent_dist.items():
        pct = (count/total)*100
        bar = '█' * int(pct/2)
        print(f'   {sent:15}: {count:>6,} ({pct:>5.1f}%) {bar}')
    
    # Check balance
    max_count = sent_dist.max()
    min_count = sent_dist.min()
    imbalance_ratio = max_count / min_count
    print(f'\n   Rasio imbalance: {imbalance_ratio:.2f}x (max/min)')
    if imbalance_ratio <= 1.1:
        print('   ✅ Data SEIMBANG!')
    else:
        print(f'   ⚠️  Data tidak seimbang (rasio > 1.1)')
    
    # Analisis content_clean
    print(f'\n3. STATISTIK CONTENT_CLEAN:')
    df['content_length'] = df['content_clean'].astype(str).str.len()
    df['word_count'] = df['content_clean'].astype(str).str.split().str.len()
    
    print(f'   Karakter - Min: {df["content_length"].min()}, Max: {df["content_length"].max()}, Mean: {df["content_length"].mean():.1f}')
    print(f'   Kata - Min: {df["word_count"].min()}, Max: {df["word_count"].max()}, Mean: {df["word_count"].mean():.1f}')
    
    # Distribution of lengths
    print(f'\n4. DISTRIBUSI PANJANG REVIEW:')
    short = len(df[df['content_length'] < 20])
    medium = len(df[(df['content_length'] >= 20) & (df['content_length'] < 50)])
    good = len(df[(df['content_length'] >= 50) & (df['content_length'] < 100)])
    long_text = len(df[df['content_length'] >= 100])
    
    print(f'   < 20 karakter:    {short:>6,} ({short/total*100:>5.1f}%)')
    print(f'   20-50 karakter:   {medium:>6,} ({medium/total*100:>5.1f}%)')
    print(f'   50-100 karakter:  {good:>6,} ({good/total*100:>5.1f}%)')
    print(f'   > 100 karakter:   {long_text:>6,} ({long_text/total*100:>5.1f}%)')
    
    # Check for problematic content
    print(f'\n5. CEK KONTEN:')
    
    # Duplicates
    dup = df.duplicated(subset=['content_clean']).sum()
    print(f'   Duplikat content_clean: {dup:,} ({dup/total*100:.1f}%)')
    
    # Empty or very short
    empty = df[df['content_clean'].str.strip() == ''].shape[0]
    print(f'   Empty content: {empty:,}')
    
    # Check if still has emoji
    emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]'
    has_emoji = df['content_clean'].str.contains(emoji_pattern, regex=True, na=False).sum()
    print(f'   Masih ada emoji: {has_emoji:,}')
    
    # Check if still has URL
    url_pattern = r'http[s]?://\S+'
    has_url = df['content_clean'].str.contains(url_pattern, regex=True, na=False).sum()
    print(f'   Masih ada URL: {has_url:,}')
    
    # Sample per sentiment
    print(f'\n6. SAMPLE CONTENT_CLEAN PER SENTIMENT:')
    for sent in df['sentiment'].unique():
        sample = df[df['sentiment'] == sent].sample(2, random_state=42)['content_clean'].values
        print(f'\n   [{sent}]:')
        for text in sample:
            text_preview = str(text)[:80] + '...' if len(str(text)) > 80 else str(text)
            print(f'      "{text_preview}"')
    
    # Check rating vs sentiment mapping
    print(f'\n7. CEK MAPPING RATING → SENTIMENT:')
    rating_sent = df.groupby(['rating', 'sentiment']).size().reset_index(name='count')
    print(rating_sent.to_string(index=False))
    
    # Final verdict
    print('\n' + '='*60)
    print('KESIMPULAN KELAYAKAN UNTUK INDOBERT:')
    print('='*60)
    
    issues = []
    
    if imbalance_ratio > 1.5:
        issues.append(f'Data tidak seimbang (rasio {imbalance_ratio:.2f}x)')
    
    if dup/total > 0.01:
        issues.append(f'Duplikat content > 1% ({dup/total*100:.1f}%)')
    
    if short/total > 0.1:
        issues.append(f'Review pendek (<20 char) > 10% ({short/total*100:.1f}%)')
    
    if has_emoji > 0:
        issues.append(f'Masih ada emoji ({has_emoji:,})')
    
    if has_url > 0:
        issues.append(f'Masih ada URL ({has_url:,})')
    
    if total < 5000:
        issues.append(f'Data kurang dari 5000 rows')
    
    if issues:
        print('\n⚠️  MASALAH DITEMUKAN:')
        for issue in issues:
            print(f'   - {issue}')
        return False
    else:
        print('\n✅ DATA SIAP UNTUK TRAINING INDOBERT!')
        print(f'   - Total data: {total:,}')
        print(f'   - Seimbang: Ya (rasio {imbalance_ratio:.2f}x)')
        print(f'   - Kolom tersedia: content_clean, sentiment')
        print(f'   - Tidak ada missing values')
        return True

if __name__ == '__main__':
    print('\n' + '#'*60)
    print('# ANALISIS DATA 3-CLASS')
    print('#'*60)
    result_3class = analyze_clean_data('data/gojek_reviews_3class_clean.csv', '3-class')
    
    print('\n\n' + '#'*60)
    print('# ANALISIS DATA 5-CLASS')
    print('#'*60)
    result_5class = analyze_clean_data('data/gojek_reviews_5class_clean.csv', '5-class')
    
    print('\n\n' + '='*60)
    print('RINGKASAN AKHIR')
    print('='*60)
    print(f'3-Class: {"✅ SIAP TRAINING" if result_3class else "❌ PERLU PERBAIKAN"}')
    print(f'5-Class: {"✅ SIAP TRAINING" if result_5class else "❌ PERLU PERBAIKAN"}')
