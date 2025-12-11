import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data/gojek_scraped_3class_RELABELED.csv')

print("="*80)
print("ANALISIS DATASET RELABELED 3-CLASS")
print("="*80)

# 1. Distribusi Label
print("\n1. DISTRIBUSI LABEL")
print("-"*80)
print(f"Total data: {len(df):,}")
print("\nJumlah per label:")
dist = df['sentiment'].value_counts()
for sentiment in ['negative', 'neutral', 'positive']:
    count = dist.get(sentiment, 0)
    pct = count / len(df) * 100
    print(f"  {sentiment.capitalize():10s}: {count:5,} ({pct:5.2f}%)")

# Check balance
max_count = dist.max()
min_count = dist.min()
balance_ratio = min_count / max_count
print(f"\nBalance Ratio (min/max): {balance_ratio:.3f}")
if balance_ratio >= 0.8:
    print("✓ Dataset SEIMBANG (balance ratio >= 0.8)")
elif balance_ratio >= 0.5:
    print("⚠ Dataset CUKUP SEIMBANG (balance ratio 0.5-0.8)")
else:
    print("✗ Dataset TIDAK SEIMBANG (balance ratio < 0.5)")

# 2. Sample Review untuk Quality Check
print("\n\n2. SAMPLE REVIEW - QUALITY CHECK")
print("="*80)

for sentiment in ['negative', 'neutral', 'positive']:
    print(f"\n{'='*80}")
    print(f"SENTIMENT: {sentiment.upper()}")
    print(f"{'='*80}")
    
    samples = df[df['sentiment'] == sentiment].sample(n=min(15, len(df[df['sentiment'] == sentiment])), random_state=42)
    
    for idx, (i, row) in enumerate(samples.iterrows(), 1):
        review = row['text']
        # Truncate jika terlalu panjang
        if len(review) > 150:
            review_display = review[:150] + "..."
        else:
            review_display = review
        print(f"\n{idx}. {review_display}")

# 3. Deteksi Potential Mislabeling
print("\n\n3. DETEKSI POTENTIAL MISLABELING")
print("="*80)

# Keywords untuk deteksi cepat
positive_keywords = ['bagus', 'baik', 'senang', 'puas', 'terima kasih', 'recommended', 'mantap', 
                     'cepat', 'ramah', 'sukses', 'terbaik', 'sempurna', 'memuaskan']
negative_keywords = ['buruk', 'kecewa', 'lambat', 'lama', 'jelek', 'tidak', 'gagal', 'batal', 
                     'mahal', 'rugi', 'sial', 'parah', 'rusak', 'error']

def check_keyword_mismatch(text, sentiment):
    text_lower = text.lower()
    pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
    neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
    
    if sentiment == 'positive' and neg_count > pos_count and neg_count >= 2:
        return True, 'Labeled positive but has more negative keywords'
    elif sentiment == 'negative' and pos_count > neg_count and pos_count >= 2:
        return True, 'Labeled negative but has more positive keywords'
    elif sentiment == 'neutral' and (pos_count >= 3 or neg_count >= 3):
        return True, f'Labeled neutral but has strong sentiment keywords (pos:{pos_count}, neg:{neg_count})'
    
    return False, None

suspicious_cases = []
for idx, row in df.iterrows():
    is_suspicious, reason = check_keyword_mismatch(row['text'], row['sentiment'])
    if is_suspicious:
        suspicious_cases.append({
            'index': idx,
            'text': row['text'][:100] + '...' if len(row['text']) > 100 else row['text'],
            'label': row['sentiment'],
            'reason': reason
        })

print(f"\nDitemukan {len(suspicious_cases)} kasus yang mencurigakan dari keyword analysis")
print("(Ini hanya deteksi awal, perlu validasi manual)\n")

if len(suspicious_cases) > 0:
    print("Contoh 10 kasus pertama:")
    for i, case in enumerate(suspicious_cases[:10], 1):
        print(f"\n{i}. Label: {case['label']}")
        print(f"   Text: {case['text']}")
        print(f"   Reason: {case['reason']}")

# 4. Statistics per Sentiment
print("\n\n4. STATISTIK TEXT LENGTH PER SENTIMENT")
print("="*80)
df['text_length'] = df['text'].str.len()
for sentiment in ['negative', 'neutral', 'positive']:
    lengths = df[df['sentiment'] == sentiment]['text_length']
    print(f"\n{sentiment.capitalize()}:")
    print(f"  Mean length: {lengths.mean():.1f} chars")
    print(f"  Median length: {lengths.median():.1f} chars")
    print(f"  Min length: {lengths.min()} chars")
    print(f"  Max length: {lengths.max()} chars")

print("\n" + "="*80)
print("ANALISIS SELESAI")
print("="*80)
