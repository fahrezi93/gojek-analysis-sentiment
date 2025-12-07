"""
Script untuk scraping review Gojek dengan RATING 3 SAJA (Neutral)
dari Google Play Store

Target: Menambah data neutral untuk balance dataset

Jalankan: python scrape_rating3_only.py

Dependencies: pip install google-play-scraper pandas
"""

import pandas as pd
import numpy as np
from google_play_scraper import reviews, Sort
from datetime import datetime
import time
import re
import os
import sys
from tqdm import tqdm

# ============================================
# CONFIGURATION
# ============================================
APP_ID = 'com.gojek.app'  # Gojek app ID
BATCH_SIZE = 200          # Reviews per batch
TARGET_RATING3 = 10000    # Target minimum rating 3 (neutral)
OUTPUT_DIR = 'data'
MAX_RETRIES = 5           # Maximum retries per batch
SLEEP_BETWEEN_BATCHES = 0.5  # Seconds between batches

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

# ============================================
# SCRAPING FUNCTION - IMPROVED
# ============================================
def scrape_reviews_rating3(app_id, target_rating3=10000, lang='id', country='id'):
    """
    Scrape reviews dari Google Play Store, fokus pada rating 3
    
    Args:
        app_id: Google Play app ID
        target_rating3: Target jumlah rating 3 yang diinginkan
        lang: Language
        country: Country
        
    Returns:
        DataFrame dengan reviews rating 3
    """
    all_reviews = []
    continuation_token = None
    
    print(f"🔄 Scraping reviews dari {app_id}")
    print(f"   Target rating 3: {target_rating3:,}")
    print(f"   Batch size: {BATCH_SIZE}")
    
    fetched_total = 0
    rating3_collected = 0
    retries = 0
    consecutive_empty = 0
    
    # Progress bar
    pbar = tqdm(total=target_rating3, desc="Rating 3 collected", unit="reviews")
    
    while rating3_collected < target_rating3:
        try:
            # Fetch batch
            result, continuation_token = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=BATCH_SIZE,
                continuation_token=continuation_token
            )
            
            if not result:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    print(f"\n   ⚠️ Tidak ada review baru setelah {consecutive_empty}x percobaan")
                    break
                time.sleep(2)
                continue
            
            # Filter rating 3 only
            rating3_batch = [r for r in result if r.get('score') == 3]
            all_reviews.extend(rating3_batch)
            
            fetched_total += len(result)
            rating3_collected = len(all_reviews)
            
            # Update progress bar
            pbar.update(len(rating3_batch))
            pbar.set_postfix({
                'total': fetched_total,
                'rating3': rating3_collected
            })
            
            # Reset counters
            retries = 0
            consecutive_empty = 0
            
            # Rate limiting
            time.sleep(SLEEP_BETWEEN_BATCHES)
            
            # Check if we reached target
            if rating3_collected >= target_rating3:
                print(f"\n   ✅ Target tercapai! {rating3_collected:,} rating 3")
                break
            
            # Check if no more data
            if continuation_token is None:
                print(f"\n   ⚠️ Tidak ada review lagi (total: {rating3_collected:,})")
                break
                
        except KeyboardInterrupt:
            print(f"\n\n⏹️ Scraping dihentikan oleh user")
            print(f"   Rating 3 collected: {rating3_collected:,}")
            break
            
        except Exception as e:
            retries += 1
            pbar.write(f"   ⚠️ Error (retry {retries}/{MAX_RETRIES}): {str(e)[:80]}")
            
            if retries >= MAX_RETRIES:
                print(f"\n   ❌ Max retries reached")
                break
            
            time.sleep(2 ** retries)  # Exponential backoff
            continue
    
    pbar.close()
    
    if not all_reviews:
        print("\n❌ Tidak ada rating 3 yang berhasil di-scrape!")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_reviews)
    
    print(f"\n✅ Scraping selesai:")
    print(f"   Total fetched: {fetched_total:,} reviews")
    print(f"   Rating 3: {len(df):,} reviews")
    
    return df

# ============================================
# MAIN FUNCTION - IMPROVED
# ============================================
def main():
    print("=" * 70)
    print("SCRAPING GOJEK REVIEWS - RATING 3 ONLY (NEUTRAL)")
    print("=" * 70)
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Scrape reviews
    print("📥 Fetching reviews from Google Play Store...\n")
    df_scraped = scrape_reviews_rating3(APP_ID, target_rating3=TARGET_RATING3)
    
    if df_scraped.empty:
        print("\n❌ Tidak ada review yang didapat!")
        return
    
    print(f"\n📊 Reviews scraped: {len(df_scraped):,}")
    
    # Clean and validate
    print("\n🧹 Cleaning dan validating reviews...")
    df_scraped['content_clean'] = df_scraped['content'].apply(clean_text)
    df_scraped['is_valid'] = df_scraped['content'].apply(is_valid_review)
    
    # Filter valid only
    df_valid = df_scraped[df_scraped['is_valid']].copy()
    print(f"   Valid reviews: {len(df_valid):,}")
    
    # Remove duplicates based on content
    print("\n🔍 Removing duplicates...")
    before = len(df_valid)
    df_valid = df_valid.drop_duplicates(subset=['content'], keep='first')
    print(f"   Removed {before - len(df_valid):,} duplicates")
    print(f"   Remaining: {len(df_valid):,}")
    
    if df_valid.empty:
        print("\n❌ Tidak ada data valid setelah cleaning!")
        return
    
    # Check existing data to avoid duplicates
    existing_file = os.path.join(OUTPUT_DIR, 'gojek_reviews_indobert_ready.csv')
    if os.path.exists(existing_file):
        print(f"\n📂 Checking existing data: {existing_file}")
        df_existing = pd.read_csv(existing_file)
        
        # Filter existing neutral only
        if 'sentiment_corrected' in df_existing.columns:
            df_existing_neutral = df_existing[df_existing['sentiment_corrected'] == 'neutral']
            print(f"   Existing neutral data: {len(df_existing_neutral):,}")
            
            # Remove duplicates with existing data
            existing_contents = set(df_existing['content_clean'].values)
            df_valid['is_duplicate_with_existing'] = df_valid['content_clean'].isin(existing_contents)
            
            duplicates_with_existing = df_valid['is_duplicate_with_existing'].sum()
            df_valid = df_valid[~df_valid['is_duplicate_with_existing']].copy()
            
            print(f"   Duplicates with existing: {duplicates_with_existing:,}")
            print(f"   New unique data: {len(df_valid):,}")
    
    if df_valid.empty:
        print("\n⚠️ Semua data sudah ada di dataset existing!")
        return
    
    # Prepare final dataframe
    import uuid
    df_final = pd.DataFrame({
        'review_id': [str(uuid.uuid4()) for _ in range(len(df_valid))],
        'content': df_valid['content'].values,
        'text': df_valid['content'].values,
        'content_clean': df_valid['content_clean'].values,
        'rating': 3,
        'sentiment': 'neutral',
        'sentiment_corrected': 'neutral'
    })
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(OUTPUT_DIR, f'gojek_rating3_scraped_{timestamp}.csv')
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n✅ HASIL SCRAPING RATING 3:")
    print(f"   Total data baru: {len(df_final):,}")
    print(f"   Saved to: {output_file}")
    
    # Also save as main file (append mode)
    main_output = os.path.join(OUTPUT_DIR, 'gojek_rating3_neutral_all.csv')
    
    # Append to existing if exists
    if os.path.exists(main_output):
        df_main = pd.read_csv(main_output)
        df_combined = pd.concat([df_main, df_final], ignore_index=True)
        # Remove duplicates
        df_combined = df_combined.drop_duplicates(subset=['content_clean'], keep='first')
        df_combined.to_csv(main_output, index=False, encoding='utf-8')
        print(f"   Appended to: {main_output}")
        print(f"   Total in file: {len(df_combined):,}")
    else:
        df_final.to_csv(main_output, index=False, encoding='utf-8')
        print(f"   Created: {main_output}")
    
    # Show statistics
    print("\n📊 STATISTICS:")
    print(f"   Average words: {df_final['content_clean'].str.split().str.len().mean():.1f}")
    print(f"   Min words: {df_final['content_clean'].str.split().str.len().min()}")
    print(f"   Max words: {df_final['content_clean'].str.split().str.len().max()}")
    
    # Show sample
    print("\n📋 Sample data (first 5):")
    print(df_final[['content', 'rating', 'sentiment']].head(5).to_string())
    
    # Instructions
    print("\n" + "=" * 70)
    print("✅ SCRAPING SELESAI!")
    print("=" * 70)
    print("\n📌 NEXT STEPS:")
    print(f"   1. Data baru tersimpan di: {output_file}")
    print(f"   2. Untuk merge dengan dataset utama, jalankan:")
    print(f"      python merge_rating3_data.py")
    print(f"   3. Atau manual merge ke gojek_reviews_indobert_ready.csv")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Script dihentikan oleh user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
