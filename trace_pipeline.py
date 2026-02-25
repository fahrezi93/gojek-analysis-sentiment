"""
Trace the complete data pipeline: where normalization happens
"""
import pandas as pd

print("=" * 70)
print("TRACE DATA PIPELINE")
print("=" * 70)

# Load all datasets
df3_raw = pd.read_csv('data/gojek_scraped_3class_all.csv')
df5_raw = pd.read_csv('data/gojek_scraped_5class_all.csv')
df3_bal = pd.read_csv('data/gojek_3class_BALANCED.csv')
df5_bal = pd.read_csv('data/gojek_5class_BALANCED_FIXED.csv')

slang_list = ['bgt', 'gk', 'gak', 'tp', 'yg', 'sdh', 'krn', 'dgn', 'utk', 'jg', 'pdhl', 'blm', 'lg']

def count_slang(df, col='text'):
    total = 0
    for s in slang_list:
        total += df[col].str.lower().str.contains(r'\b' + s + r'\b', regex=True, na=False).sum()
    return total

print("\n--- SLANG COUNT PER DATASET ---")
print(f"3-class RAW  (scraped_all):     {count_slang(df3_raw):>6} slang words")
print(f"3-class BALANCED:               {count_slang(df3_bal):>6} slang words")
print(f"5-class RAW  (scraped_all):     {count_slang(df5_raw):>6} slang words")
print(f"5-class BALANCED (FIXED):       {count_slang(df5_bal):>6} slang words")

print("\n--- COLUMN COMPARISON ---")
print(f"3-class RAW columns:  {df3_raw.columns.tolist()}")
print(f"5-class RAW columns:  {df5_raw.columns.tolist()}")
print(f"3-class BALANCED:     {df3_bal.columns.tolist()}")
print(f"5-class BALANCED:     {df5_bal.columns.tolist()}")

print("\n--- ROW COUNTS ---")
print(f"3-class RAW:      {len(df3_raw):>10,}")
print(f"5-class RAW:      {len(df5_raw):>10,}")
print(f"3-class BALANCED: {len(df3_bal):>10,}")
print(f"5-class BALANCED: {len(df5_bal):>10,}")

# Check if 3class raw and 5class raw share same texts
text3_set = set(df3_raw['text'].dropna().values)
text5_set = set(df5_raw['text'].dropna().values)
common = text3_set & text5_set
print(f"\n--- OVERLAP ---")
print(f"Common text between 3class raw and 5class raw: {len(common):,}")
print(f"Only in 3class raw: {len(text3_set - text5_set):,}")
print(f"Only in 5class raw: {len(text5_set - text3_set):,}")

# Check if 3class raw text looks already cleaned (lowercase, no emoji)
sample_upper_3 = df3_raw['text'].apply(lambda x: any(c.isupper() for c in str(x))).sum()
sample_upper_5 = df5_raw['text'].apply(lambda x: any(c.isupper() for c in str(x))).sum()
print(f"\n--- CASE CHECK ---")
print(f"3-class RAW texts with uppercase: {sample_upper_3} ({sample_upper_3/len(df3_raw)*100:.1f}%)")
print(f"5-class RAW texts with uppercase: {sample_upper_5} ({sample_upper_5/len(df5_raw)*100:.1f}%)")

# Show specific text samples side by side
# Find a text that exists in both (common)
if common:
    sample = list(common)[:3]
    print(f"\n--- COMMON TEXT SAMPLES ---")
    for t in sample:
        print(f"  TEXT: {str(t)[:100]}...")
        row3 = df3_raw[df3_raw['text'] == t].iloc[0]
        row5 = df5_raw[df5_raw['text'] == t].iloc[0]
        print(f"  3class sentiment: {row3.get('sentiment', '?')}, rating: {row3.get('rating', '?')}")
        print(f"  5class sentiment: {row5.get('sentiment', '?')}, rating: {row5.get('rating', '?')}")
        print()

# KEY CHECK: Does 3class raw data look like it was already cleaned by text_cleaner_indobert?
# text_cleaner_indobert does: lowercase, remove emoji, remove URL, normalize slang
# 3class raw: has slang (3024), no emoji, no URL -> PARTIALLY cleaned (emoji+URL removed, slang NOT removed)
# BUT balanced 3class: 0 slang -> normalization happened between raw and balanced

# So there must be an intermediate file: gojek_3class_PREPROCESSED.csv that had text_preprocessed column
print("\n--- PIPELINE RECONSTRUCTION ---")
print("STEP 1: scrape_gojek_advanced.py")
print("  -> clean_text_indobert() applied -> removed emoji, URL, lowercased")
print("  -> BUT slang normalization may have been PARTIAL or NOT in this function")
print(f"  -> 3class raw still has {count_slang(df3_raw)} slang words")
print()
print("STEP 2: A MISSING intermediate script (probably produced gojek_3class_PREPROCESSED.csv)")
print("  -> Applied slang normalization (full)")
print("  -> Created column 'text_preprocessed'")
print("  -> File now deleted")
print()
print("STEP 3: clean_and_balance_data.py")
print("  -> Read from PREPROCESSED.csv, took 'text_preprocessed' column")
print("  -> Applied spam removal, label correction, balancing")
print(f"  -> Result: 3class BALANCED has {count_slang(df3_bal)} slang words")
print()
print("FOR 5-CLASS:")
print("STEP 1: Same scraping (probably separate run)")
print(f"  -> 5class raw has {count_slang(df5_raw)} slang words") 
print("STEP 2: clean_and_balance_5class.py")
print("  -> Read from gojek_scraped_5class_RELABELED.csv (deleted)")
print("  -> NO slang normalization in this script")
print(f"  -> Result: 5class BALANCED has {count_slang(df5_bal)} slang words")
print()
print("=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("- 3-CLASS: Slang normalization WAS applied (via intermediate PREPROCESSED file)")
print("- 5-CLASS: Slang normalization was NOT applied")
print("  This explains why 5class balanced still has slang words!")
