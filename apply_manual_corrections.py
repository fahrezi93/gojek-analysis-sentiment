import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob

print("="*80)
print("APPLY MANUAL CORRECTIONS - UPDATE DATASET")
print("="*80)

# 1. Find reviewed file
print("\n🔍 Mencari file review...")
review_files = glob.glob('data/suspicious_cases_REVIEWED_*.csv')

if not review_files:
    print("\n❌ File review tidak ditemukan!")
    print("\nYang dicari: data/suspicious_cases_REVIEWED_*.csv")
    print("\nPastikan sudah:")
    print("1. Jalankan extract_suspicious_cases.py")
    print("2. Review file suspicious_cases_for_review_*.csv")
    print("3. Save dengan nama suspicious_cases_REVIEWED_*.csv")
    exit(1)

# Ambil file terbaru
review_file = max(review_files, key=os.path.getctime)
print(f"✓ File review ditemukan: {review_file}")

# 2. Load files
print("\n📂 Loading files...")
review_df = pd.read_csv(review_file)
original_df = pd.read_csv('data/gojek_scraped_3class_RELABELED.csv')

print(f"   Total review cases: {len(review_df)}")
print(f"   Original dataset: {len(original_df)} rows")

# 3. Validate review data
print("\n✓ Validasi data review...")
reviewed = review_df[review_df['suggested_label'].notna() & (review_df['suggested_label'] != '')]
print(f"   Cases yang sudah direview: {len(reviewed)}")
print(f"   Cases yang belum direview: {len(review_df) - len(reviewed)}")

if len(reviewed) == 0:
    print("\n⚠️  Belum ada case yang direview!")
    print("    Silakan isi kolom 'suggested_label' di file review.")
    exit(1)

# Validate label values
valid_labels = ['negative', 'neutral', 'positive']
invalid_labels = reviewed[~reviewed['suggested_label'].isin(valid_labels)]
if len(invalid_labels) > 0:
    print(f"\n⚠️  Ditemukan {len(invalid_labels)} label yang tidak valid:")
    for idx, row in invalid_labels.head(5).iterrows():
        print(f"      Index {row['original_index']}: '{row['suggested_label']}'")
    print(f"\n   Label harus salah satu dari: {valid_labels}")
    exit(1)

# 4. Prepare corrections
print("\n🔄 Mempersiapkan koreksi...")
corrections_made = []
unchanged_count = 0

for idx, row in reviewed.iterrows():
    original_index = int(row['original_index'])
    current_label = original_df.loc[original_index, 'sentiment']
    suggested_label = row['suggested_label'].strip().lower()
    
    if current_label != suggested_label:
        corrections_made.append({
            'index': original_index,
            'text': row['text'][:100] + '...' if len(str(row['text'])) > 100 else row['text'],
            'old_label': current_label,
            'new_label': suggested_label,
            'confidence': row['confidence'],
            'notes': row.get('notes', '')
        })
    else:
        unchanged_count += 1

print(f"   Perubahan yang akan diterapkan: {len(corrections_made)}")
print(f"   Label yang tetap sama: {unchanged_count}")

# 5. Show preview
if len(corrections_made) > 0:
    print("\n" + "="*80)
    print("PREVIEW PERUBAHAN (10 pertama)")
    print("="*80)
    
    for i, corr in enumerate(corrections_made[:10], 1):
        print(f"\n{i}. [Index: {corr['index']}] [{corr['confidence'].upper()}]")
        print(f"   Text: {corr['text']}")
        print(f"   {corr['old_label'].upper()} → {corr['new_label'].upper()}")
        if corr['notes']:
            print(f"   Notes: {corr['notes']}")

# 6. Confirm and apply
print("\n" + "="*80)
print("⚠️  KONFIRMASI PERUBAHAN")
print("="*80)
print(f"\nTotal perubahan: {len(corrections_made)}")

# Summary by change type
print("\nRingkasan perubahan:")
change_summary = {}
for corr in corrections_made:
    change_type = f"{corr['old_label']} → {corr['new_label']}"
    change_summary[change_type] = change_summary.get(change_type, 0) + 1

for change_type, count in sorted(change_summary.items()):
    print(f"   {change_type}: {count} cases")

# Apply changes
confirm = input("\nApply perubahan? (yes/no): ").strip().lower()

if confirm != 'yes':
    print("\n❌ Perubahan dibatalkan.")
    exit(0)

print("\n🔄 Applying corrections...")
corrected_df = original_df.copy()

for corr in corrections_made:
    corrected_df.loc[corr['index'], 'sentiment'] = corr['new_label']

# 7. Save corrected dataset
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'data/gojek_scraped_3class_CORRECTED_{timestamp}.csv'
corrected_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✓ Dataset terkoreksi tersimpan: {output_file}")

# 8. Save correction log
log_df = pd.DataFrame(corrections_made)
log_file = f'data/correction_log_{timestamp}.csv'
log_df.to_csv(log_file, index=False, encoding='utf-8-sig')
print(f"✓ Log koreksi tersimpan: {log_file}")

# 9. New distribution
print("\n" + "="*80)
print("DISTRIBUSI LABEL - SEBELUM vs SESUDAH")
print("="*80)

old_dist = original_df['sentiment'].value_counts()
new_dist = corrected_df['sentiment'].value_counts()

print(f"\n{'Label':<12} {'Sebelum':<10} {'Sesudah':<10} {'Perubahan':<10}")
print("-" * 50)
for label in ['negative', 'neutral', 'positive']:
    old_count = old_dist.get(label, 0)
    new_count = new_dist.get(label, 0)
    diff = new_count - old_count
    diff_str = f"+{diff}" if diff > 0 else str(diff)
    print(f"{label.capitalize():<12} {old_count:<10,} {new_count:<10,} {diff_str:<10}")

print(f"\n{'Total':<12} {len(original_df):<10,} {len(corrected_df):<10,}")

# Balance check
max_count = new_dist.max()
min_count = new_dist.min()
balance_ratio = min_count / max_count

print(f"\nBalance Ratio: {balance_ratio:.3f}")
if balance_ratio >= 0.8:
    print("✓ Dataset SEIMBANG")
elif balance_ratio >= 0.5:
    print("⚠ Dataset CUKUP SEIMBANG")
else:
    print("✗ Dataset TIDAK SEIMBANG")

print("\n" + "="*80)
print("✅ KOREKSI SELESAI!")
print("="*80)
print(f"\nFile baru: {output_file}")
print(f"Gunakan file ini untuk training model.")
