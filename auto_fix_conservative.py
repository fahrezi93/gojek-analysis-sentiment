import pandas as pd
import re
from datetime import datetime

print("="*80)
print("CONSERVATIVE AUTO-FIX - HIGH CONFIDENCE ONLY")
print("="*80)

# Load dataset
df = pd.read_csv('data/gojek_scraped_3class_RELABELED.csv')
print(f"\n✓ Dataset loaded: {len(df):,} rows")

# ========================================
# CONSERVATIVE SENTIMENT RULES
# ========================================

def is_clearly_positive(text):
    """Detect CLEARLY positive reviews with very high confidence"""
    text = str(text).lower()
    
    # Very strong positive patterns
    strong_pos_patterns = [
        r'sangat (bagus|baik|memuaskan|senang|puas|membantu)',
        r'(bagus|baik|memuaskan) (banget|sekali)',
        r'terima kasih (banyak|sekali|banget)',
        r'(sangat|amat) (senang|puas|memuaskan)',
        r'recommended (banget|sekali)',
        r'(sukses|mantap|hebat|sempurna|terbaik) (terus|selalu)',
        r'suka (banget|sekali)',
        r'top banget',
        r'mantap (jiwa|banget)',
        r'pelayanan (bagus|baik|memuaskan|ramah)',
        r'cepat dan (baik|bagus|ramah)',
        r'sangat membantu'
    ]
    
    # Must have strong positive pattern
    has_strong_pos = any(re.search(pattern, text) for pattern in strong_pos_patterns)
    
    # Should NOT have strong negative words
    strong_neg_words = ['buruk', 'kecewa', 'jelek', 'parah', 'gagal', 'rusak', 'zonk', 
                        'mengecewakan', 'payah', 'benci', 'menyesal', 'sial', 'error',
                        'tidak bisa', 'ga bisa', 'gak bisa']
    has_strong_neg = any(word in text for word in strong_neg_words)
    
    # Should NOT have adversatives (tapi, namun)
    adversatives = ['tapi', 'namun', 'tetapi', 'akan tetapi', 'cuma', 'cuman', 
                   'sayangnya', 'namun demikian']
    has_adversative = any(adv in text for adv in adversatives)
    
    # Should NOT have problem indicators
    problem_indicators = ['tolong', 'mohon', 'kenapa', 'masalah', 'susah']
    has_problem = any(indicator in text for indicator in problem_indicators)
    
    return has_strong_pos and not has_strong_neg and not has_adversative and not has_problem

def is_clearly_negative(text):
    """Detect CLEARLY negative reviews with very high confidence"""
    text = str(text).lower()
    
    # Very strong negative patterns
    strong_neg_patterns = [
        r'sangat (buruk|kecewa|mengecewakan|lambat|lama|susah)',
        r'(buruk|kecewa|jelek|parah|payah) (banget|sekali)',
        r'(sangat|amat) (kecewa|mengecewakan)',
        r'tidak (bagus|baik|memuaskan|puas|senang)',
        r'(aplikasi|driver|pengemudi|pelayanan) (buruk|jelek|parah)',
        r'kecewa (banget|sekali)',
        r'(lambat|lama) (banget|sekali)',
        r'gagal terus',
        r'error terus',
        r'tidak bisa (login|pesan|order)',
        r'batal terus'
    ]
    
    # Must have strong negative pattern
    has_strong_neg = any(re.search(pattern, text) for pattern in strong_neg_patterns)
    
    # Should NOT have strong positive words
    strong_pos_words = ['bagus', 'baik', 'senang', 'puas', 'sempurna', 'terbaik',
                        'memuaskan', 'hebat', 'mantap', 'recommended', 'sukses']
    has_strong_pos = any(word in text for word in strong_pos_words)
    
    # Should NOT have adversatives
    adversatives = ['tapi', 'namun', 'tetapi', 'akan tetapi', 'cuma', 'cuman']
    has_adversative = any(adv in text for adv in adversatives)
    
    return has_strong_neg and not has_strong_pos and not has_adversative

def is_thank_you_positive(text):
    """Detect thank you messages - usually positive"""
    text = str(text).lower()
    
    thank_patterns = [
        r'terima kasih',
        r'makasih',
        r'thanks',
        r'terimakasih'
    ]
    
    has_thanks = any(re.search(pattern, text) for pattern in thank_patterns)
    
    # Must be short and only thanks (no complaints)
    is_short = len(text) < 100
    
    complaint_words = ['tapi', 'namun', 'kecewa', 'buruk', 'jelek', 'parah', 
                      'lambat', 'lama', 'gagal', 'error', 'batal']
    has_complaint = any(word in text for word in complaint_words)
    
    return has_thanks and is_short and not has_complaint

def is_clearly_mixed_neutral(text):
    """Detect clearly mixed sentiment - should be neutral"""
    text = str(text).lower()
    
    # Has adversatives
    adversatives = ['tapi', 'namun', 'tetapi', 'cuma', 'cuman', 'sayangnya']
    has_adversative = any(adv in text for adv in adversatives)
    
    if not has_adversative:
        return False
    
    # Count positive and negative indicators
    pos_words = ['bagus', 'baik', 'senang', 'puas', 'suka', 'oke', 'mantap']
    neg_words = ['buruk', 'kecewa', 'jelek', 'lambat', 'lama', 'mahal', 'susah']
    
    pos_count = sum(1 for word in pos_words if word in text)
    neg_count = sum(1 for word in neg_words if word in text)
    
    # Must have both positive and negative
    return pos_count >= 1 and neg_count >= 1

# ========================================
# ANALYZE WITH CONSERVATIVE RULES
# ========================================

print("\n🔍 Analyzing with conservative rules...")
corrections = []

for idx, row in df.iterrows():
    text = row['text']
    current_label = row['sentiment']
    
    predicted_label = None
    reason = None
    
    # Check for clear patterns
    if current_label == 'negative':
        if is_clearly_positive(text):
            predicted_label = 'positive'
            reason = 'Clear positive pattern detected'
        elif is_thank_you_positive(text):
            predicted_label = 'positive'
            reason = 'Thank you message without complaint'
        elif is_clearly_mixed_neutral(text):
            predicted_label = 'neutral'
            reason = 'Mixed sentiment detected'
    
    elif current_label == 'positive':
        if is_clearly_negative(text):
            predicted_label = 'negative'
            reason = 'Clear negative pattern detected'
        elif is_clearly_mixed_neutral(text):
            predicted_label = 'neutral'
            reason = 'Mixed sentiment detected'
    
    elif current_label == 'neutral':
        if is_clearly_positive(text) and not is_clearly_mixed_neutral(text):
            predicted_label = 'positive'
            reason = 'Clear positive pattern, not mixed'
        elif is_clearly_negative(text) and not is_clearly_mixed_neutral(text):
            predicted_label = 'negative'
            reason = 'Clear negative pattern, not mixed'
    
    # Record correction
    if predicted_label and predicted_label != current_label:
        corrections.append({
            'index': idx,
            'text': text,
            'old_label': current_label,
            'new_label': predicted_label,
            'reason': reason
        })
    
    # Progress indicator
    if (idx + 1) % 5000 == 0:
        print(f"   Processed: {idx + 1:,} / {len(df):,} rows...")

print(f"\n✓ Analysis complete!")
print(f"   Conservative corrections found: {len(corrections):,}")

# ========================================
# SUMMARY
# ========================================

if len(corrections) == 0:
    print("\n✓ No high-confidence corrections needed!")
    print("   Dataset is already good quality.")
    exit(0)

print("\n" + "="*80)
print("SUMMARY OF CONSERVATIVE CORRECTIONS")
print("="*80)

corr_df = pd.DataFrame(corrections)

# By change type
print("\nBy Change Type:")
change_types = {}
for corr in corrections:
    change_type = f"{corr['old_label']} → {corr['new_label']}"
    change_types[change_type] = change_types.get(change_type, 0) + 1

for change_type, count in sorted(change_types.items(), key=lambda x: x[1], reverse=True):
    pct = count / len(corrections) * 100
    print(f"   {change_type:20s}: {count:4d} ({pct:5.1f}%)")

# ========================================
# PREVIEW SAMPLES
# ========================================

print("\n" + "="*80)
print("PREVIEW: SAMPLE CORRECTIONS (First 20)")
print("="*80)

for i, (idx, corr) in enumerate(corr_df.head(20).iterrows(), 1):
    text = corr['text']
    display_text = text[:150] + '...' if len(text) > 150 else text
    print(f"\n{i}. [Index: {corr['index']}]")
    print(f"   Text: {display_text}")
    print(f"   Change: {corr['old_label'].upper()} → {corr['new_label'].upper()}")
    print(f"   Reason: {corr['reason']}")

# More samples from different change types
print("\n" + "="*80)
print("MORE SAMPLES BY CHANGE TYPE")
print("="*80)

for change_type in sorted(change_types.keys()):
    samples = corr_df[
        (corr_df['old_label'] + ' → ' + corr_df['new_label']) == change_type
    ].head(3)
    
    if len(samples) > 0:
        print(f"\n{change_type}:")
        for idx, corr in samples.iterrows():
            text = corr['text'][:100] + '...' if len(corr['text']) > 100 else corr['text']
            print(f"   - {text}")

# ========================================
# APPLY CHANGES
# ========================================

print("\n" + "="*80)
print("⚠️  APPLY CONSERVATIVE CORRECTIONS?")
print("="*80)
print(f"\nTotal corrections: {len(corrections):,}")
print(f"Percentage of dataset: {len(corrections)/len(df)*100:.2f}%")
print("\nThese are HIGH CONFIDENCE corrections only.")

confirm = input("\nApply changes? (yes/no): ").strip().lower()

if confirm != 'yes':
    print("\n❌ Changes cancelled.")
    
    # Save preview
    preview_file = f'data/conservative_fix_preview_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    corr_df.to_csv(preview_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 Preview saved to: {preview_file}")
    exit(0)

# Apply corrections
print("\n🔄 Applying corrections...")
corrected_df = df.copy()

for corr in corrections:
    corrected_df.loc[corr['index'], 'sentiment'] = corr['new_label']

# Save
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'data/gojek_scraped_3class_CLEANED_{timestamp}.csv'
corrected_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✓ Cleaned dataset saved: {output_file}")

# Save log
log_file = f'data/conservative_fix_log_{timestamp}.csv'
corr_df.to_csv(log_file, index=False, encoding='utf-8-sig')
print(f"✓ Correction log saved: {log_file}")

# ========================================
# FINAL STATISTICS
# ========================================

print("\n" + "="*80)
print("BEFORE vs AFTER DISTRIBUTION")
print("="*80)

old_dist = df['sentiment'].value_counts()
new_dist = corrected_df['sentiment'].value_counts()

print(f"\n{'Label':<12} {'Before':<15} {'After':<15} {'Change':<12}")
print("-" * 60)
for label in ['negative', 'neutral', 'positive']:
    old_count = old_dist.get(label, 0)
    new_count = new_dist.get(label, 0)
    diff = new_count - old_count
    diff_str = f"+{diff}" if diff > 0 else str(diff)
    
    old_pct = old_count / len(df) * 100
    new_pct = new_count / len(corrected_df) * 100
    
    print(f"{label.capitalize():<12} {old_count:5,} ({old_pct:5.1f}%)   {new_count:5,} ({new_pct:5.1f}%)   {diff_str:<12}")

# Balance check
max_count = new_dist.max()
min_count = new_dist.min()
old_balance = old_dist.min() / old_dist.max()
new_balance = min_count / max_count

print(f"\nOld Balance Ratio: {old_balance:.3f}")
print(f"New Balance Ratio: {new_balance:.3f}")

if new_balance >= 0.8:
    status = "✓ SEIMBANG"
elif new_balance >= 0.5:
    status = "⚠ CUKUP SEIMBANG"
else:
    status = "✗ TIDAK SEIMBANG"
print(f"Status: {status}")

print("\n" + "="*80)
print("✅ CONSERVATIVE CLEANING COMPLETED!")
print("="*80)
print(f"\nCleaned dataset: {output_file}")
print("This dataset has been carefully cleaned with high-confidence corrections only.")
print("\n" + "="*80)
