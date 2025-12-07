"""
Extract dan simpan DATA KOTOR untuk analisis
"""

import pandas as pd
import re

# Simulated dirty data check (recreate logic dari scraper)
def is_dirty(text):
    """Check if text is dirty/low quality"""
    if not text or len(str(text).strip()) < 10:
        return True, "too_short"
    
    text_lower = str(text).lower()
    words = text_lower.split()
    
    if len(words) < 3:
        return True, "too_few_words"
    
    # Spam patterns
    spam_patterns = [
        r'(klik|visit|follow|subscribe).*link',
        r'(wa|whatsapp|hubungi|hub|call).*\d{4,}',
        r'(diskon|voucher|promo|gratis).*code',
        r'^(good|nice|ok|oke|mantap|bagus)$',  # Single word only
    ]
    
    for pattern in spam_patterns:
        if re.search(pattern, text_lower):
            return True, "spam_pattern"
    
    # Check repetition
    if len(set(words)) < len(words) * 0.3 and len(words) > 5:
        return True, "too_repetitive"
    
    # Check if mostly non-alpha
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars < len(text) * 0.3:
        return True, "not_enough_text"
    
    return False, "clean"

# Load data yang sudah ada
print("=" * 80)
print("🗑️ EXTRACT DATA KOTOR")
print("=" * 80)

# Coba dari data original
try:
    df_old = pd.read_csv('data/gojek_reviews_75k_FIXED_BALANCED.csv')
    print(f"\n📂 Checking: gojek_reviews_75k_FIXED_BALANCED.csv")
    print(f"   Rows: {len(df_old):,}")
    
    # Check for dirty
    if 'content_clean' in df_old.columns:
        check_col = 'content_clean'
    elif 'content' in df_old.columns:
        check_col = 'content'
    else:
        check_col = 'text'
    
    dirty_checks = df_old[check_col].apply(is_dirty)
    df_old['is_dirty'] = dirty_checks.apply(lambda x: x[0])
    df_old['dirty_reason'] = dirty_checks.apply(lambda x: x[1])
    
    df_dirty = df_old[df_old['is_dirty']].copy()
    df_clean = df_old[~df_old['is_dirty']].copy()
    
    print(f"\n📊 Results:")
    print(f"   ✅ Clean: {len(df_clean):,} ({len(df_clean)/len(df_old)*100:.1f}%)")
    print(f"   ❌ Dirty: {len(df_dirty):,} ({len(df_dirty)/len(df_old)*100:.1f}%)")
    
    if len(df_dirty) > 0:
        print(f"\n📋 Dirty reasons:")
        for reason, count in df_dirty['dirty_reason'].value_counts().items():
            print(f"      - {reason}: {count:,}")
        
        # Save dirty data
        dirty_file = 'data/processed/DATA_KOTOR.csv'
        df_dirty.to_csv(dirty_file, index=False, encoding='utf-8')
        print(f"\n💾 Saved dirty data: {dirty_file}")
        
        # Show samples
        print(f"\n📋 CONTOH DATA KOTOR:")
        for idx, row in df_dirty.head(10).iterrows():
            reason = row['dirty_reason']
            text = row[check_col][:80] if len(str(row[check_col])) > 80 else row[check_col]
            print(f"\n   [{reason}]")
            print(f"   \"{text}\"")
    else:
        print(f"\n✅ No dirty data found in this file!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n" + "=" * 80)
print("💡 INFO TENTANG DATA KOTOR:")
print("=" * 80)
print("""
Data kotor adalah data yang DI-REJECT karena:
1. Terlalu pendek (<10 karakter atau <3 kata)
2. Spam pattern (ada link, nomor WA, kode promo)
3. Terlalu repetitif (kata yang sama diulang terus)
4. Tidak cukup text (mostly angka/simbol)
5. Single word review ("bagus", "ok", "mantap" saja)

Data ini TIDAK COCOK untuk training karena:
- Tidak informatif
- Bisa bikin model overfitting
- Menurunkan akurasi

Data bersih yang sudah di-filter lebih bagus untuk training!
""")
print("=" * 80)
