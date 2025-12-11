import pandas as pd
import re
from datetime import datetime

print("="*80)
print("AUTO-FIX LABELS - ADVANCED SENTIMENT ANALYSIS")
print("="*80)

# Load dataset
df = pd.read_csv('data/gojek_scraped_3class_RELABELED.csv')
print(f"\n✓ Dataset loaded: {len(df):,} rows")

# ========================================
# SENTIMENT ANALYSIS ENGINE
# ========================================

# Expanded keywords with weights
POSITIVE_WORDS = {
    # Strong positive
    'bagus': 3, 'baik': 3, 'senang': 3, 'puas': 3, 'sempurna': 3, 'terbaik': 3,
    'memuaskan': 3, 'hebat': 3, 'mantap': 3, 'recommended': 3, 'sukses': 3,
    'excellent': 3, 'perfect': 3, 'amazing': 3, 'mantul': 3, 'jos': 3,
    
    # Moderate positive
    'terima kasih': 2, 'makasih': 2, 'thanks': 2, 'suka': 2, 'cepat': 2,
    'ramah': 2, 'oke': 2, 'ok': 2, 'lumayan': 2, 'lancar': 2, 'mudah': 2,
    'nyaman': 2, 'praktis': 2, 'membantu': 2, 'efisien': 2, 'top': 2,
    
    # Mild positive
    'keren': 1, 'asik': 1, 'seru': 1, 'gampang': 1, 'worth': 1, 'rekomendasi': 1
}

NEGATIVE_WORDS = {
    # Strong negative
    'buruk': 3, 'kecewa': 3, 'jelek': 3, 'parah': 3, 'gagal': 3, 'rusak': 3,
    'mengecewakan': 3, 'payah': 3, 'zonk': 3, 'benci': 3, 'menyesal': 3,
    'terrible': 3, 'awful': 3, 'kesal': 3, 'marah': 3, 'sial': 3,
    
    # Moderate negative
    'lambat': 2, 'lama': 2, 'susah': 2, 'ribet': 2, 'error': 2, 'mahal': 2,
    'rugi': 2, 'batal': 2, 'lemot': 2, 'ngelag': 2, 'lag': 2, 'macet': 2,
    
    # Mild negative  
    'kurang': 1, 'agak': 1
}

# Negation words that flip sentiment
NEGATIONS = ['tidak', 'bukan', 'belum', 'jangan', 'ga', 'gak', 'nggak', 'ngga', 'enggak']

# Intensifiers
INTENSIFIERS = ['sangat', 'amat', 'sekali', 'banget', 'bet', 'paling', 'super', 'very']

# Adversatives (but, however) - indicates mixed sentiment
ADVERSATIVES = ['tapi', 'namun', 'tetapi', 'akan tetapi', 'cuma', 'cuman', 'sayangnya', 
                'sayang', 'hanya', 'hanya saja', 'namun demikian']

def preprocess_text(text):
    """Normalize text"""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def has_negation_before(text, pos, window=3):
    """Check if there's a negation word before position"""
    words = text[:pos].split()
    recent_words = words[-window:] if len(words) >= window else words
    return any(neg in recent_words for neg in NEGATIONS)

def count_intensifiers(text):
    """Count intensifier words"""
    return sum(1 for word in INTENSIFIERS if word in text)

def has_adversative(text):
    """Check if text contains adversative conjunctions"""
    return any(adv in text for adv in ADVERSATIVES)

def analyze_sentiment_score(text):
    """
    Advanced sentiment analysis with scoring
    Returns: (score, pos_count, neg_count, is_mixed)
    """
    text = preprocess_text(text)
    words = text.split()
    
    pos_score = 0
    neg_score = 0
    pos_raw_count = 0
    neg_raw_count = 0
    
    # Check for adversatives (mixed sentiment indicator)
    has_mixed = has_adversative(text)
    
    # Intensity multiplier
    intensity_multiplier = 1 + (count_intensifiers(text) * 0.3)
    
    # Analyze each word
    for i, word in enumerate(words):
        # Check positive words
        if word in POSITIVE_WORDS:
            weight = POSITIVE_WORDS[word]
            # Check for negation
            context_start = max(0, i-3)
            context = ' '.join(words[context_start:i])
            
            if any(neg in context for neg in NEGATIONS):
                # Negated positive = negative
                neg_score += weight * intensity_multiplier
                neg_raw_count += 1
            else:
                pos_score += weight * intensity_multiplier
                pos_raw_count += 1
        
        # Check negative words
        elif word in NEGATIVE_WORDS:
            weight = NEGATIVE_WORDS[word]
            context_start = max(0, i-3)
            context = ' '.join(words[context_start:i])
            
            if any(neg in context for neg in NEGATIONS):
                # Negated negative = positive
                pos_score += weight * intensity_multiplier
                pos_raw_count += 1
            else:
                neg_score += weight * intensity_multiplier
                neg_raw_count += 1
    
    # Calculate final score
    final_score = pos_score - neg_score
    
    return final_score, pos_raw_count, neg_raw_count, has_mixed

def predict_sentiment(text, current_label):
    """
    Predict sentiment with confidence
    Returns: (predicted_label, confidence, reason)
    """
    score, pos_count, neg_count, has_mixed = analyze_sentiment_score(text)
    
    # Decision thresholds
    STRONG_THRESHOLD = 4
    MILD_THRESHOLD = 2
    
    # Mixed sentiment handling
    if has_mixed and pos_count >= 1 and neg_count >= 1:
        # Has both positive and negative with adversative
        if abs(score) < MILD_THRESHOLD:
            return 'neutral', 'high', f'Mixed sentiment (pos:{pos_count}, neg:{neg_count})'
    
    # Clear sentiment
    if score >= STRONG_THRESHOLD:
        return 'positive', 'high', f'Strong positive (score:{score:.1f})'
    elif score >= MILD_THRESHOLD:
        return 'positive', 'medium', f'Positive (score:{score:.1f})'
    elif score <= -STRONG_THRESHOLD:
        return 'negative', 'high', f'Strong negative (score:{score:.1f})'
    elif score <= -MILD_THRESHOLD:
        return 'negative', 'medium', f'Negative (score:{score:.1f})'
    else:
        # Near zero score
        if pos_count == 0 and neg_count == 0:
            return current_label, 'low', 'No clear sentiment keywords - keep current'
        return 'neutral', 'medium', f'Neutral (score:{score:.1f})'

# ========================================
# ANALYZE AND FIX
# ========================================

print("\n🔍 Analyzing all reviews...")
corrections = []
unchanged = 0

for idx, row in df.iterrows():
    text = row['text']
    current_label = row['sentiment']
    
    # Predict new label
    predicted_label, confidence, reason = predict_sentiment(text, current_label)
    
    # Only fix HIGH and MEDIUM confidence predictions that differ
    if predicted_label != current_label and confidence in ['high', 'medium']:
        corrections.append({
            'index': idx,
            'text': text[:120] + '...' if len(text) > 120 else text,
            'old_label': current_label,
            'new_label': predicted_label,
            'confidence': confidence,
            'reason': reason
        })
    else:
        unchanged += 1
    
    # Progress indicator
    if (idx + 1) % 1000 == 0:
        print(f"   Processed: {idx + 1:,} / {len(df):,} rows...")

print(f"\n✓ Analysis complete!")
print(f"   Changes to apply: {len(corrections):,}")
print(f"   Unchanged: {unchanged:,}")

# ========================================
# SUMMARY
# ========================================

print("\n" + "="*80)
print("SUMMARY OF CHANGES")
print("="*80)

if len(corrections) == 0:
    print("\n✓ No changes needed - dataset already optimal!")
    exit(0)

# By confidence
print("\nBy Confidence Level:")
corr_df = pd.DataFrame(corrections)
for conf in ['high', 'medium']:
    count = len(corr_df[corr_df['confidence'] == conf])
    pct = count / len(corrections) * 100 if len(corrections) > 0 else 0
    print(f"   {conf.upper():8s}: {count:4d} ({pct:5.1f}%)")

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
print("PREVIEW: HIGH CONFIDENCE CHANGES (First 15)")
print("="*80)

high_conf = [c for c in corrections if c['confidence'] == 'high'][:15]
for i, corr in enumerate(high_conf, 1):
    print(f"\n{i}. [Index: {corr['index']}]")
    print(f"   Text: {corr['text']}")
    print(f"   Change: {corr['old_label'].upper()} → {corr['new_label'].upper()}")
    print(f"   Reason: {corr['reason']}")

# ========================================
# APPLY CHANGES
# ========================================

print("\n" + "="*80)
print("⚠️  APPLY CHANGES?")
print("="*80)
print(f"\nTotal changes: {len(corrections):,}")
print(f"Percentage of dataset: {len(corrections)/len(df)*100:.2f}%")

confirm = input("\nApply these changes? (yes/no): ").strip().lower()

if confirm != 'yes':
    print("\n❌ Changes cancelled.")
    
    # Save preview for review
    preview_file = f'data/auto_fix_preview_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    pd.DataFrame(corrections).to_csv(preview_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 Preview saved to: {preview_file}")
    print("   Review and run again if needed.")
    exit(0)

# Apply corrections
print("\n🔄 Applying corrections...")
corrected_df = df.copy()

for corr in corrections:
    corrected_df.loc[corr['index'], 'sentiment'] = corr['new_label']

# Save corrected dataset
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'data/gojek_scraped_3class_AUTOFIXED_{timestamp}.csv'
corrected_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✓ Corrected dataset saved: {output_file}")

# Save correction log
log_file = f'data/auto_fix_log_{timestamp}.csv'
pd.DataFrame(corrections).to_csv(log_file, index=False, encoding='utf-8-sig')
print(f"✓ Correction log saved: {log_file}")

# ========================================
# FINAL STATISTICS
# ========================================

print("\n" + "="*80)
print("BEFORE vs AFTER DISTRIBUTION")
print("="*80)

old_dist = df['sentiment'].value_counts()
new_dist = corrected_df['sentiment'].value_counts()

print(f"\n{'Label':<12} {'Before':<12} {'After':<12} {'Change':<12}")
print("-" * 55)
for label in ['negative', 'neutral', 'positive']:
    old_count = old_dist.get(label, 0)
    new_count = new_dist.get(label, 0)
    diff = new_count - old_count
    diff_str = f"+{diff}" if diff > 0 else str(diff)
    
    old_pct = old_count / len(df) * 100
    new_pct = new_count / len(corrected_df) * 100
    
    print(f"{label.capitalize():<12} {old_count:5,} ({old_pct:5.1f}%)  {new_count:5,} ({new_pct:5.1f}%)  {diff_str:<12}")

print(f"\n{'Total':<12} {len(df):5,}         {len(corrected_df):5,}")

# Balance check
max_count = new_dist.max()
min_count = new_dist.min()
balance_ratio = min_count / max_count

print(f"\nOld Balance Ratio: {old_dist.min() / old_dist.max():.3f}")
print(f"New Balance Ratio: {balance_ratio:.3f}")

if balance_ratio >= 0.8:
    status = "✓ SEIMBANG"
elif balance_ratio >= 0.5:
    status = "⚠ CUKUP SEIMBANG"
else:
    status = "✗ TIDAK SEIMBANG"
print(f"Status: {status}")

print("\n" + "="*80)
print("✅ AUTO-FIX COMPLETED!")
print("="*80)
print(f"\nNew dataset: {output_file}")
print(f"Use this file for training your model.")
print("\n" + "="*80)
