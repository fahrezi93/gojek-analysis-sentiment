"""
Script untuk menggabungkan semua data dan augmentasi neutral
Target: 5,000 data per kelas dengan augmentation
"""

import pandas as pd
import numpy as np
import re
import os
import random

# ============================================
# CONFIGURATION
# ============================================
OUTPUT_DIR = 'data'
TARGET_PER_CLASS = 5000

# ============================================
# TEXT AUGMENTATION FUNCTIONS
# ============================================

# Sinonim Indonesia untuk augmentasi
INDONESIAN_SYNONYMS = {
    'bagus': ['baik', 'mantap', 'oke', 'lumayan'],
    'baik': ['bagus', 'mantap', 'oke'],
    'cepat': ['kilat', 'sigap', 'gesit'],
    'lambat': ['lama', 'pelan', 'lemot'],
    'mahal': ['kemahalan', 'tidak murah'],
    'murah': ['terjangkau', 'ekonomis'],
    'ramah': ['sopan', 'baik hati', 'friendly'],
    'jelek': ['buruk', 'tidak bagus', 'kurang'],
    'aplikasi': ['app', 'apk'],
    'driver': ['abang', 'kakak', 'pengemudi'],
    'makanan': ['makan', 'pesanan'],
    'enak': ['lezat', 'nikmat', 'sedap'],
    'tidak': ['gak', 'nggak', 'ga', 'tak'],
    'sangat': ['banget', 'sekali', 'amat'],
    'sudah': ['udah', 'dah'],
    'saya': ['aku', 'gue', 'gw'],
    'bisa': ['dapat', 'mampu'],
    'harga': ['tarif', 'biaya', 'ongkos'],
    'pelayanan': ['layanan', 'service'],
    'puas': ['senang', 'happy'],
    'kecewa': ['sedih', 'disappointed'],
}

def synonym_replacement(text, n=1):
    """Ganti n kata dengan sinonimnya"""
    words = text.split()
    new_words = words.copy()
    
    replaceable = [w for w in words if w.lower() in INDONESIAN_SYNONYMS]
    
    if not replaceable:
        return text
    
    n = min(n, len(replaceable))
    words_to_replace = random.sample(replaceable, n)
    
    for word in words_to_replace:
        synonyms = INDONESIAN_SYNONYMS.get(word.lower(), [])
        if synonyms:
            idx = words.index(word)
            new_words[idx] = random.choice(synonyms)
    
    return ' '.join(new_words)

def random_swap(text, n=1):
    """Swap n pasang kata random"""
    words = text.split()
    if len(words) < 2:
        return text
    
    new_words = words.copy()
    for _ in range(n):
        if len(new_words) >= 2:
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    
    return ' '.join(new_words)

def random_deletion(text, p=0.1):
    """Hapus kata dengan probabilitas p"""
    words = text.split()
    if len(words) <= 3:
        return text
    
    new_words = [w for w in words if random.random() > p]
    
    if len(new_words) == 0:
        return random.choice(words)
    
    return ' '.join(new_words)

def random_insertion(text, n=1):
    """Insert n kata random dari sinonim"""
    words = text.split()
    new_words = words.copy()
    
    all_synonyms = [s for syns in INDONESIAN_SYNONYMS.values() for s in syns]
    
    for _ in range(n):
        insert_word = random.choice(all_synonyms)
        insert_pos = random.randint(0, len(new_words))
        new_words.insert(insert_pos, insert_word)
    
    return ' '.join(new_words)

def augment_text(text, num_aug=4):
    """Generate multiple augmented versions of text"""
    augmented = []
    
    # Original
    augmented.append(text)
    
    # Synonym replacement
    for _ in range(num_aug):
        aug = synonym_replacement(text, n=random.randint(1, 2))
        if aug != text:
            augmented.append(aug)
    
    # Random swap
    for _ in range(num_aug // 2):
        aug = random_swap(text, n=1)
        if aug != text:
            augmented.append(aug)
    
    # Random deletion
    for _ in range(num_aug // 2):
        aug = random_deletion(text, p=0.1)
        if aug != text:
            augmented.append(aug)
    
    return list(set(augmented))  # Remove duplicates

# ============================================
# CLEANING FUNCTIONS
# ============================================
def clean_text(text):
    """Bersihkan text review"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    
    return text.strip()

def is_valid_review(text, min_words=3, max_words=500):
    """Cek apakah review valid"""
    if not text or not isinstance(text, str):
        return False
    
    words = text.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    
    if re.search(r'(.)\1{4,}', text):
        return False
    
    if text.replace(' ', '').isdigit():
        return False
    
    return True

# ============================================
# MAIN PROCESS
# ============================================
def main():
    print("=" * 60)
    print("🔧 DATA COMBINATION & AUGMENTATION")
    print("=" * 60)
    
    # Load all available data
    print("\n📥 Loading existing data...")
    
    all_data = []
    
    # 1. Data dari scraping baru
    if os.path.exists('data/gojek_reviews_scraped_all.csv'):
        df1 = pd.read_csv('data/gojek_reviews_scraped_all.csv')
        print(f"   ✓ gojek_reviews_scraped_all.csv: {len(df1)} rows")
        all_data.append(df1)
    
    # 2. Data 3class clean
    if os.path.exists('data/gojek_reviews_3class_clean.csv'):
        df2 = pd.read_csv('data/gojek_reviews_3class_clean.csv')
        print(f"   ✓ gojek_reviews_3class_clean.csv: {len(df2)} rows")
        all_data.append(df2)
    
    # 3. Data balanced 9997
    if os.path.exists('data/gojek_reviews_balanced_9997.csv'):
        df3 = pd.read_csv('data/gojek_reviews_balanced_9997.csv')
        # Clean this data
        if 'content_clean' not in df3.columns:
            df3['content_clean'] = df3['content'].apply(clean_text)
        print(f"   ✓ gojek_reviews_balanced_9997.csv: {len(df3)} rows")
        all_data.append(df3)
    
    if not all_data:
        print("❌ Tidak ada data yang tersedia!")
        return
    
    # Combine all data
    print("\n🔗 Combining data...")
    df = pd.concat(all_data, ignore_index=True)
    print(f"   Total combined: {len(df)}")
    
    # Ensure content_clean exists
    if 'content_clean' not in df.columns:
        df['content_clean'] = df['content'].apply(clean_text)
    else:
        # Fill missing content_clean
        mask = df['content_clean'].isna() | (df['content_clean'] == '')
        df.loc[mask, 'content_clean'] = df.loc[mask, 'content'].apply(clean_text)
    
    # Filter valid reviews
    df = df[df['content_clean'].apply(is_valid_review)]
    print(f"   After filtering: {len(df)}")
    
    # Remove duplicates based on content_clean
    df = df.drop_duplicates(subset=['content_clean'])
    print(f"   After dedup: {len(df)}")
    
    # Ensure sentiment column
    if 'sentiment' not in df.columns:
        def get_sentiment(score):
            if score <= 2:
                return 'negative'
            elif score >= 4:
                return 'positive'
            else:
                return 'neutral'
        df['sentiment'] = df['score'].apply(get_sentiment)
    
    # Filter only 3 classes
    df = df[df['sentiment'].isin(['negative', 'neutral', 'positive'])]
    
    print(f"\n📊 Distribution before augmentation:")
    print(df['sentiment'].value_counts())
    
    # ============================================
    # AUGMENTATION FOR MINORITY CLASS
    # ============================================
    print("\n🔄 Augmenting minority classes...")
    
    counts = df['sentiment'].value_counts().to_dict()
    max_count = max(counts.values())
    target = min(TARGET_PER_CLASS, max_count)
    
    augmented_dfs = []
    
    for sentiment in ['negative', 'neutral', 'positive']:
        df_class = df[df['sentiment'] == sentiment].copy()
        current_count = len(df_class)
        
        print(f"\n   {sentiment.upper()}: {current_count} samples")
        
        if current_count >= target:
            # Undersample
            df_sampled = df_class.sample(n=target, random_state=42)
            augmented_dfs.append(df_sampled)
            print(f"      → Undersampled to {target}")
        else:
            # Augment to reach target
            augmented_dfs.append(df_class)  # Keep original
            
            needed = target - current_count
            print(f"      → Need {needed} more samples via augmentation")
            
            # Augment
            augmented_texts = []
            original_rows = df_class.to_dict('records')
            
            while len(augmented_texts) < needed:
                row = random.choice(original_rows)
                text = row['content_clean']
                
                # Generate augmented versions
                aug_versions = augment_text(text, num_aug=4)
                
                for aug_text in aug_versions[1:]:  # Skip original
                    if len(augmented_texts) >= needed:
                        break
                    if is_valid_review(aug_text):
                        new_row = {
                            'userName': row.get('userName', 'augmented'),
                            'content': aug_text,
                            'content_clean': aug_text,
                            'score': row.get('score', 3),
                            'at': row.get('at', ''),
                            'sentiment': sentiment
                        }
                        augmented_texts.append(new_row)
            
            if augmented_texts:
                df_aug = pd.DataFrame(augmented_texts)
                augmented_dfs.append(df_aug)
                print(f"      → Added {len(augmented_texts)} augmented samples")
    
    # Combine all
    df_final = pd.concat(augmented_dfs, ignore_index=True)
    
    # Shuffle
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Keep needed columns
    columns_to_keep = ['userName', 'content', 'content_clean', 'score', 'at', 'sentiment']
    df_final = df_final[[c for c in columns_to_keep if c in df_final.columns]]
    
    print(f"\n📊 Final distribution:")
    print(df_final['sentiment'].value_counts())
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'gojek_reviews_final_augmented.csv')
    df_final.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("✅ COMPLETED!")
    print("=" * 60)
    print(f"""
📊 SUMMARY:
   • Total samples: {len(df_final)}
   • Negative: {df_final['sentiment'].value_counts().get('negative', 0)}
   • Neutral:  {df_final['sentiment'].value_counts().get('neutral', 0)}
   • Positive: {df_final['sentiment'].value_counts().get('positive', 0)}

💾 SAVED: {output_path}
""")

if __name__ == "__main__":
    main()
