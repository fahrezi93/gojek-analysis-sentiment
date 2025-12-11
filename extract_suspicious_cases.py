import pandas as pd
from datetime import datetime

# Load dataset
df = pd.read_csv('data/gojek_scraped_3class_RELABELED.csv')

print("="*80)
print("EKSTRAK KASUS MENCURIGAKAN - MANUAL REVIEW")
print("="*80)

# Keywords untuk deteksi
positive_keywords = ['bagus', 'baik', 'senang', 'puas', 'terima kasih', 'recommended', 'mantap', 
                     'cepat', 'ramah', 'sukses', 'terbaik', 'sempurna', 'memuaskan', 'suka',
                     'hebat', 'keren', 'oke', 'mantul', 'jos', 'top', 'makasih', 'thanks']
negative_keywords = ['buruk', 'kecewa', 'lambat', 'lama', 'jelek', 'tidak', 'gagal', 'batal', 
                     'mahal', 'rugi', 'sial', 'parah', 'rusak', 'error', 'susah', 'ribet',
                     'mengecewakan', 'payah', 'zonk', 'kesal', 'marah', 'benci']

def analyze_sentiment_keywords(text):
    """Analisis keyword dalam text"""
    text_lower = text.lower()
    pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
    neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
    return pos_count, neg_count

def check_mislabeling(text, sentiment):
    """Deteksi potensi mislabeling dengan reasoning"""
    pos_count, neg_count = analyze_sentiment_keywords(text)
    
    issues = []
    confidence = 'low'
    
    if sentiment == 'positive':
        if neg_count > pos_count and neg_count >= 2:
            confidence = 'high' if neg_count >= 3 else 'medium'
            issues.append(f'Labeled POSITIVE but has {neg_count} negative keywords vs {pos_count} positive')
    
    elif sentiment == 'negative':
        if pos_count > neg_count and pos_count >= 2:
            confidence = 'high' if pos_count >= 3 else 'medium'
            issues.append(f'Labeled NEGATIVE but has {pos_count} positive keywords vs {neg_count} negative')
    
    elif sentiment == 'neutral':
        if pos_count >= 3 or neg_count >= 3:
            confidence = 'medium'
            issues.append(f'Labeled NEUTRAL but has strong sentiment (pos:{pos_count}, neg:{neg_count})')
        elif pos_count >= 2 and neg_count >= 2:
            confidence = 'low'
            issues.append(f'Labeled NEUTRAL with mixed signals (pos:{pos_count}, neg:{neg_count}) - might be correct')
    
    if issues:
        return True, ' | '.join(issues), confidence, pos_count, neg_count
    return False, None, None, pos_count, neg_count

# Analisis semua data
print("\n🔍 Menganalisis dataset...")
suspicious_cases = []

for idx, row in df.iterrows():
    is_suspicious, reason, confidence, pos_kw, neg_kw = check_mislabeling(row['text'], row['sentiment'])
    if is_suspicious:
        suspicious_cases.append({
            'original_index': idx,
            'text': row['text'],
            'current_label': row['sentiment'],
            'reason': reason,
            'confidence': confidence,
            'pos_keywords': pos_kw,
            'neg_keywords': neg_kw,
            'suggested_label': '',  # Untuk diisi manual
            'notes': ''  # Untuk diisi manual
        })

print(f"✓ Ditemukan {len(suspicious_cases)} kasus mencurigakan")

# Sort berdasarkan confidence
suspicious_df = pd.DataFrame(suspicious_cases)
confidence_order = {'high': 0, 'medium': 1, 'low': 2}
suspicious_df['confidence_rank'] = suspicious_df['confidence'].map(confidence_order)
suspicious_df = suspicious_df.sort_values(['confidence_rank', 'current_label'])
suspicious_df = suspicious_df.drop('confidence_rank', axis=1)

# Simpan ke CSV
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'data/suspicious_cases_for_review_{timestamp}.csv'
suspicious_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✓ File tersimpan: {output_file}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nBy Confidence Level:")
for conf in ['high', 'medium', 'low']:
    count = len(suspicious_df[suspicious_df['confidence'] == conf])
    pct = count / len(suspicious_df) * 100 if len(suspicious_df) > 0 else 0
    print(f"  {conf.upper():8s}: {count:4d} cases ({pct:5.1f}%)")

print("\nBy Current Label:")
for label in ['negative', 'neutral', 'positive']:
    count = len(suspicious_df[suspicious_df['current_label'] == label])
    pct = count / len(suspicious_df) * 100 if len(suspicious_df) > 0 else 0
    print(f"  {label.capitalize():9s}: {count:4d} cases ({pct:5.1f}%)")

# Preview high confidence cases
print("\n" + "="*80)
print("PREVIEW: HIGH CONFIDENCE MISLABELING (10 kasus pertama)")
print("="*80)

high_conf = suspicious_df[suspicious_df['confidence'] == 'high'].head(10)
for i, (idx, row) in enumerate(high_conf.iterrows(), 1):
    print(f"\n{i}. [Index: {row['original_index']}]")
    print(f"   Current Label: {row['current_label'].upper()}")
    print(f"   Text: {row['text'][:150]}{'...' if len(row['text']) > 150 else ''}")
    print(f"   Reason: {row['reason']}")
    print(f"   Keywords: POS={row['pos_keywords']}, NEG={row['neg_keywords']}")

# Medium confidence cases
print("\n" + "="*80)
print("PREVIEW: MEDIUM CONFIDENCE (10 kasus pertama)")
print("="*80)

medium_conf = suspicious_df[suspicious_df['confidence'] == 'medium'].head(10)
for i, (idx, row) in enumerate(medium_conf.iterrows(), 1):
    print(f"\n{i}. [Index: {row['original_index']}]")
    print(f"   Current Label: {row['current_label'].upper()}")
    print(f"   Text: {row['text'][:150]}{'...' if len(row['text']) > 150 else ''}")
    print(f"   Reason: {row['reason']}")
    print(f"   Keywords: POS={row['pos_keywords']}, NEG={row['neg_keywords']}")

print("\n" + "="*80)
print("INSTRUKSI MANUAL REVIEW")
print("="*80)
print("""
1. Buka file: {output_file}
2. Review kolom 'text' dan 'current_label'
3. Isi kolom 'suggested_label' dengan label yang benar (negative/neutral/positive)
4. Isi kolom 'notes' jika ada catatan khusus
5. Prioritas review:
   - HIGH confidence: Kemungkinan besar salah label (review WAJIB)
   - MEDIUM confidence: Perlu validasi manual
   - LOW confidence: Optional, mungkin sudah benar

6. Setelah selesai review, simpan file dengan nama:
   suspicious_cases_REVIEWED_{timestamp}.csv
   
7. Jalankan script apply_manual_corrections.py untuk update dataset
""".format(output_file=output_file, timestamp=timestamp))

print("\n✓ Ekstraksi selesai!")
print("="*80)
