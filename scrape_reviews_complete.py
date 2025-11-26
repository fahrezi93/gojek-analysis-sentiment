"""
Script untuk scraping review Gojek dari Google Play Store
Target: Minimal 5,000 data per kelas (negative, neutral, positive)

Jalankan: python scrape_reviews_complete.py
"""

import pandas as pd
import numpy as np
from google_play_scraper import reviews, Sort
from datetime import datetime
import time
import re
import os

# ============================================
# CONFIGURATION
# ============================================
APP_ID = 'com.gojek.app'  # Gojek app ID
TARGET_PER_CLASS = 5000   # Target per kelas
BATCH_SIZE = 200          # Reviews per batch
MAX_REVIEWS = 50000       # Maximum total reviews to fetch
OUTPUT_DIR = 'data'

# ============================================
# TEXT CLEANING FUNCTIONS
# ============================================
def clean_text(text):
    """Bersihkan text review"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep Indonesian chars
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def is_valid_review(text, min_words=3, max_words=500):
    """Cek apakah review valid"""
    if not text or not isinstance(text, str):
        return False
    
    words = text.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    
    # Terlalu banyak karakter berulang
    if re.search(r'(.)\1{4,}', text):
        return False
    
    # Hanya angka
    if text.replace(' ', '').isdigit():
        return False
    
    return True

def get_sentiment_from_score(score):
    """
    Convert score ke sentiment (menghindari data ambigu)
    - Score 1-2: negative (jelas negatif)
    - Score 3: SKIP (ambigu - tidak digunakan)
    - Score 4-5: positive (jelas positif)
    
    Untuk neutral, kita akan menggunakan analisis teks pada score 3
    yang memiliki kata-kata netral yang jelas
    """
    if score <= 2:
        return 'negative'
    elif score >= 4:
        return 'positive'
    else:
        return 'neutral_candidate'  # Score 3, perlu analisis lebih lanjut

def analyze_neutral_text(text):
    """
    Analisis teks untuk menentukan apakah benar-benar neutral
    Return True jika teks terlihat neutral
    """
    text_lower = text.lower()
    
    # Kata-kata positif yang kuat
    positive_words = [
        'bagus', 'mantap', 'keren', 'recommended', 'puas', 'senang', 
        'suka', 'cepat', 'ramah', 'nyaman', 'terbaik', 'top', 'josss',
        'mantul', 'oke banget', 'luar biasa', 'hebat', 'memuaskan'
    ]
    
    # Kata-kata negatif yang kuat
    negative_words = [
        'jelek', 'buruk', 'kecewa', 'lambat', 'parah', 'mengecewakan',
        'tidak recommended', 'mahal', 'lama', 'error', 'bug', 'masalah',
        'susah', 'ribet', 'payah', 'sampah', 'brengsek', 'bangsat',
        'uninstall', 'hapus', 'bintang 1', 'worst', 'terrible'
    ]
    
    # Hitung kata positif dan negatif
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    # Jika tidak ada kata positif/negatif yang kuat, kemungkinan neutral
    if pos_count == 0 and neg_count == 0:
        return True
    
    # Jika ada keduanya (mixed), bisa dianggap neutral
    if pos_count > 0 and neg_count > 0:
        return True
    
    return False

# ============================================
# SCRAPING FUNCTION
# ============================================
def scrape_reviews(app_id, count=1000, lang='id', country='id'):
    """Scrape reviews dari Google Play Store"""
    all_reviews = []
    continuation_token = None
    
    print(f"🔄 Scraping {count} reviews dari {app_id}...")
    
    fetched = 0
    while fetched < count:
        batch_count = min(BATCH_SIZE, count - fetched)
        
        try:
            result, continuation_token = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=batch_count,
                continuation_token=continuation_token
            )
            
            if not result:
                print("   Tidak ada review lagi")
                break
            
            all_reviews.extend(result)
            fetched += len(result)
            
            print(f"   Fetched: {fetched}/{count}", end='\r')
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\n   Error: {e}")
            time.sleep(2)
            continue
    
    print(f"\n✓ Total fetched: {len(all_reviews)} reviews")
    return all_reviews

# ============================================
# MAIN SCRAPING PROCESS
# ============================================
def main():
    print("=" * 60)
    print("🚀 GOJEK REVIEW SCRAPER - TARGET 5000/KELAS")
    print("=" * 60)
    print(f"App ID: {APP_ID}")
    print(f"Target per kelas: {TARGET_PER_CLASS}")
    print(f"Max reviews to fetch: {MAX_REVIEWS}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Scrape reviews
    print("\n📥 STEP 1: Scraping reviews...")
    raw_reviews = scrape_reviews(APP_ID, count=MAX_REVIEWS)
    
    if not raw_reviews:
        print("❌ Gagal mengambil reviews!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(raw_reviews)
    print(f"   Raw reviews: {len(df)}")
    
    # Keep only needed columns
    df = df[['userName', 'content', 'score', 'at']].copy()
    
    # Step 2: Clean data
    print("\n🧹 STEP 2: Cleaning data...")
    
    # Remove nulls
    df = df.dropna(subset=['content'])
    print(f"   After removing nulls: {len(df)}")
    
    # Clean text
    df['content_clean'] = df['content'].apply(clean_text)
    
    # Filter valid reviews
    df = df[df['content_clean'].apply(is_valid_review)]
    print(f"   After filtering invalid: {len(df)}")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['content_clean'])
    print(f"   After removing duplicates: {len(df)}")
    
    # Step 3: Assign sentiment
    print("\n🏷️ STEP 3: Assigning sentiment labels...")
    
    df['sentiment'] = df['score'].apply(get_sentiment_from_score)
    
    # For neutral candidates (score 3), analyze text
    neutral_mask = df['sentiment'] == 'neutral_candidate'
    df.loc[neutral_mask, 'is_truly_neutral'] = df.loc[neutral_mask, 'content_clean'].apply(analyze_neutral_text)
    
    # Keep only truly neutral reviews from score 3
    df.loc[neutral_mask & (df['is_truly_neutral'] == True), 'sentiment'] = 'neutral'
    df.loc[neutral_mask & (df['is_truly_neutral'] != True), 'sentiment'] = 'ambiguous'
    
    # Drop ambiguous
    df = df[df['sentiment'] != 'ambiguous']
    df = df[df['sentiment'] != 'neutral_candidate']
    
    # Drop helper column
    if 'is_truly_neutral' in df.columns:
        df = df.drop(columns=['is_truly_neutral'])
    
    print(f"\n📊 Distribution after cleaning:")
    print(df['sentiment'].value_counts())
    
    # Step 4: Balance data
    print("\n⚖️ STEP 4: Balancing data...")
    
    counts = df['sentiment'].value_counts()
    print(f"   Negative: {counts.get('negative', 0)}")
    print(f"   Neutral:  {counts.get('neutral', 0)}")
    print(f"   Positive: {counts.get('positive', 0)}")
    
    # Check if we have enough data
    min_count = counts.min()
    target = min(TARGET_PER_CLASS, min_count)
    
    print(f"\n   Target per class: {target}")
    
    # Balance by undersampling
    df_balanced = pd.DataFrame()
    for sentiment in ['negative', 'neutral', 'positive']:
        df_class = df[df['sentiment'] == sentiment]
        if len(df_class) >= target:
            df_sampled = df_class.sample(n=target, random_state=42)
        else:
            df_sampled = df_class  # Keep all if less than target
        df_balanced = pd.concat([df_balanced, df_sampled])
    
    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n📊 Final balanced distribution:")
    print(df_balanced['sentiment'].value_counts())
    
    # Step 5: Save data
    print("\n💾 STEP 5: Saving data...")
    
    # Save raw cleaned data (before balancing)
    raw_path = os.path.join(OUTPUT_DIR, 'gojek_reviews_scraped_all.csv')
    df.to_csv(raw_path, index=False)
    print(f"   ✓ All cleaned data: {raw_path} ({len(df)} rows)")
    
    # Save balanced data
    balanced_path = os.path.join(OUTPUT_DIR, 'gojek_reviews_3class_balanced.csv')
    df_balanced.to_csv(balanced_path, index=False)
    print(f"   ✓ Balanced data: {balanced_path} ({len(df_balanced)} rows)")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ SCRAPING COMPLETED!")
    print("=" * 60)
    print(f"""
📊 SUMMARY:
   • Total scraped: {len(raw_reviews)}
   • After cleaning: {len(df)}
   • Final balanced: {len(df_balanced)}
   
📁 FILES SAVED:
   • {raw_path}
   • {balanced_path}
   
🎯 PER CLASS:
   • Negative: {df_balanced['sentiment'].value_counts().get('negative', 0)}
   • Neutral:  {df_balanced['sentiment'].value_counts().get('neutral', 0)}
   • Positive: {df_balanced['sentiment'].value_counts().get('positive', 0)}
""")
    
    if target < TARGET_PER_CLASS:
        print(f"""
⚠️ WARNING: Target {TARGET_PER_CLASS} per kelas tidak tercapai!
   Hanya dapat {target} per kelas.
   
   Untuk mendapatkan lebih banyak data neutral (score 3):
   1. Scrape lebih banyak reviews (tingkatkan MAX_REVIEWS)
   2. Atau gunakan data augmentation
""")

if __name__ == "__main__":
    main()
