"""
Script Advanced untuk Scraping Gojek Reviews - Target 50K+ High Quality
- Fokus rating 3 (neutral) untuk balance dataset
- Filter spam, review pendek, dan low quality
- Multi-source scraping untuk maximize data

Jalankan: python scrape_gojek_50k_quality.py

Dependencies: pip install google-play-scraper pandas tqdm
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
import uuid

# ============================================
# CONFIGURATION
# ============================================
APP_ID = 'com.gojek.app'
TARGET_TOTAL = 50000        # Target total reviews
TARGET_RATING3_MIN = 15000  # Minimum rating 3 (neutral)

# Quality thresholds
MIN_WORDS = 5               # Minimum words (lebih strict)
MAX_WORDS = 300             # Maximum words
MIN_CHARS = 20              # Minimum characters
MAX_REPEAT_CHAR = 4         # Max karakter berulang (aaaa)

BATCH_SIZE = 200
MAX_RETRIES = 5
SLEEP_BETWEEN_BATCHES = 0.3
OUTPUT_DIR = 'data'

# ============================================
# ADVANCED TEXT QUALITY CHECKER
# ============================================
def calculate_text_quality_score(text):
    """
    Calculate quality score for text (0-100)
    Higher score = better quality
    """
    if not text or not isinstance(text, str):
        return 0
    
    score = 100.0
    text_lower = text.lower()
    words = text.split()
    
    # 1. Length check
    word_count = len(words)
    if word_count < MIN_WORDS:
        score -= 50
    elif word_count < 8:
        score -= 20
    elif word_count > MAX_WORDS:
        score -= 30
    
    # 2. Character repetition (aaaa, hahaha berlebihan)
    repetition_pattern = r'(.)\1{' + str(MAX_REPEAT_CHAR) + ',}'
    if re.search(repetition_pattern, text):
        score -= 30
    
    # 3. Excessive punctuation (!!!!, ????)
    if len(re.findall(r'[!?]{3,}', text)) > 0:
        score -= 15
    
    # 4. Too many emojis (lebih dari 5)
    emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]'
    emoji_count = len(re.findall(emoji_pattern, text))
    if emoji_count > 5:
        score -= 20
    
    # 5. Spam patterns
    spam_patterns = [
        r'\b(promo|diskon|voucher|gratis)\b.*\b(kode|code|kupon)\b',
        r'(wa|whatsapp|hub|hubungi|call).*\d{4,}',
        r'(cek|klik|visit|follow).*http',
        r'\b(good|nice|ok|oke|mantap)\b$',  # Single word only
    ]
    for pattern in spam_patterns:
        if re.search(pattern, text_lower):
            score -= 25
            break
    
    # 6. All caps (LIKE THIS)
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    if caps_ratio > 0.6:
        score -= 20
    
    # 7. Only numbers or symbols
    alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text) if len(text) > 0 else 0
    if alpha_ratio < 0.3:
        score -= 40
    
    # 8. Meaningful content check (has verbs/adjectives indicators)
    meaningful_words = [
        'bagus', 'jelek', 'baik', 'buruk', 'cepat', 'lambat',
        'ramah', 'kasar', 'puas', 'kecewa', 'senang', 'marah',
        'enak', 'lama', 'mudah', 'susah', 'murah', 'mahal',
        'bersih', 'kotor', 'nyaman', 'tidak', 'kurang', 'sangat',
        'banget', 'sekali', 'lumayan', 'biasa', 'oke', 'mantap',
        'recommended', 'rekomendasi', 'suka', 'benci', 'pelayanan',
        'driver', 'aplikasi', 'pesanan', 'sampai', 'cancel'
    ]
    has_meaningful = any(word in text_lower for word in meaningful_words)
    if has_meaningful:
        score += 10
    
    # 9. Sentence structure (ada spasi yang proper)
    if ' ' in text and len(words) > 3:
        score += 5
    
    return max(0, min(100, score))

def clean_text(text):
    """Clean text but preserve meaning"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = str(text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers (but keep review content)
    text = re.sub(r'\b\d{4,}\b', '', text)
    
    # Normalize repeated punctuation
    text = re.sub(r'([!?.]){3,}', r'\1\1', text)
    
    return text.strip()

def is_high_quality_review(text, min_score=50):
    """Check if review meets quality threshold"""
    if not text or len(text) < MIN_CHARS:
        return False, 0
    
    words = text.split()
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return False, 0
    
    quality_score = calculate_text_quality_score(text)
    return quality_score >= min_score, quality_score

# ============================================
# MULTI-STRATEGY SCRAPING
# ============================================
def scrape_with_strategy(app_id, strategy='newest', target=10000, focus_rating=None):
    """
    Scrape dengan berbagai strategi untuk maximize coverage
    
    Args:
        strategy: 'newest', 'most_relevant', 'rating'
        target: jumlah reviews yang diinginkan
        focus_rating: jika ada, prioritas rating tertentu (e.g., 3)
    """
    sort_map = {
        'newest': Sort.NEWEST,
        'most_relevant': Sort.MOST_RELEVANT,
        'rating': Sort.RATING
    }
    
    sort_type = sort_map.get(strategy, Sort.NEWEST)
    
    all_reviews = []
    continuation_token = None
    
    print(f"\n🔄 Strategy: {strategy.upper()}")
    if focus_rating:
        print(f"   Focus: Rating {focus_rating}")
    
    collected = 0
    fetched_total = 0
    retries = 0
    consecutive_empty = 0
    
    pbar = tqdm(total=target, desc=f"{strategy}", unit="reviews")
    
    while collected < target:
        try:
            result, continuation_token = reviews(
                app_id,
                lang='id',
                country='id',
                sort=sort_type,
                count=BATCH_SIZE,
                continuation_token=continuation_token
            )
            
            if not result:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    break
                time.sleep(2)
                continue
            
            # Filter by rating if specified
            if focus_rating:
                filtered = [r for r in result if r.get('score') == focus_rating]
            else:
                filtered = result
            
            all_reviews.extend(filtered)
            collected = len(all_reviews)
            fetched_total += len(result)
            
            pbar.update(len(filtered))
            pbar.set_postfix({'total': fetched_total, 'collected': collected})
            
            retries = 0
            consecutive_empty = 0
            
            time.sleep(SLEEP_BETWEEN_BATCHES)
            
            if collected >= target or continuation_token is None:
                break
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Strategy '{strategy}' interrupted by user")
            break
            
        except Exception as e:
            retries += 1
            if retries >= MAX_RETRIES:
                break
            time.sleep(2 ** retries)
            continue
    
    pbar.close()
    return all_reviews

def scrape_comprehensive(app_id, total_target=50000, rating3_target=15000):
    """
    Comprehensive scraping dengan multiple strategies
    """
    print("=" * 70)
    print("🚀 COMPREHENSIVE SCRAPING - GOJEK REVIEWS")
    print("=" * 70)
    print(f"   Total Target: {total_target:,} reviews")
    print(f"   Rating 3 Target: {rating3_target:,} minimum")
    print("=" * 70)
    
    all_reviews_dict = {}  # Use dict to auto-deduplicate by reviewId
    
    # Strategy 1: Focus on Rating 3 first (most important)
    print("\n📌 PHASE 1: Collecting Rating 3 (Neutral)")
    rating3_reviews = scrape_with_strategy(
        app_id, 
        strategy='newest',
        target=rating3_target,
        focus_rating=3
    )
    for review in rating3_reviews:
        review_id = review.get('reviewId', str(uuid.uuid4()))
        all_reviews_dict[review_id] = review
    
    print(f"   ✅ Phase 1: {len(rating3_reviews):,} rating 3 collected")
    
    # Strategy 2: Get more Rating 3 from different sort
    if len(rating3_reviews) < rating3_target:
        print("\n📌 PHASE 1B: More Rating 3 (Most Relevant)")
        more_rating3 = scrape_with_strategy(
            app_id,
            strategy='most_relevant',
            target=rating3_target - len(rating3_reviews),
            focus_rating=3
        )
        for review in more_rating3:
            review_id = review.get('reviewId', str(uuid.uuid4()))
            all_reviews_dict[review_id] = review
    
    current_total = len(all_reviews_dict)
    rating3_count = sum(1 for r in all_reviews_dict.values() if r.get('score') == 3)
    
    print(f"\n   Rating 3 collected so far: {rating3_count:,}")
    print(f"   Total unique reviews: {current_total:,}")
    
    # Strategy 3: Get all ratings (newest)
    if current_total < total_target:
        remaining = total_target - current_total
        print(f"\n📌 PHASE 2: Collecting All Ratings (Newest) - Target: {remaining:,}")
        all_ratings = scrape_with_strategy(
            app_id,
            strategy='newest',
            target=remaining
        )
        for review in all_ratings:
            review_id = review.get('reviewId', str(uuid.uuid4()))
            all_reviews_dict[review_id] = review
    
    # Strategy 4: Most relevant for quality
    current_total = len(all_reviews_dict)
    if current_total < total_target:
        remaining = total_target - current_total
        print(f"\n📌 PHASE 3: Quality Reviews (Most Relevant) - Target: {remaining:,}")
        quality_reviews = scrape_with_strategy(
            app_id,
            strategy='most_relevant',
            target=remaining
        )
        for review in quality_reviews:
            review_id = review.get('reviewId', str(uuid.uuid4()))
            all_reviews_dict[review_id] = review
    
    # Strategy 5: Rating sort for more rating 3
    current_total = len(all_reviews_dict)
    rating3_count = sum(1 for r in all_reviews_dict.values() if r.get('score') == 3)
    
    if rating3_count < rating3_target and current_total < total_target:
        remaining = min(total_target - current_total, rating3_target - rating3_count) * 3
        print(f"\n📌 PHASE 4: Rating Sort - Target: {remaining:,}")
        rating_sorted = scrape_with_strategy(
            app_id,
            strategy='rating',
            target=remaining
        )
        for review in rating_sorted:
            review_id = review.get('reviewId', str(uuid.uuid4()))
            all_reviews_dict[review_id] = review
    
    return list(all_reviews_dict.values())

# ============================================
# MAIN FUNCTION
# ============================================
def main():
    print("\n" + "=" * 70)
    print("GOJEK REVIEWS SCRAPING - 50K HIGH QUALITY")
    print("=" * 70)
    print()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Scrape comprehensive
    raw_reviews = scrape_comprehensive(
        APP_ID,
        total_target=TARGET_TOTAL,
        rating3_target=TARGET_RATING3_MIN
    )
    
    if not raw_reviews:
        print("\n❌ No reviews scraped!")
        return
    
    # Convert to DataFrame
    df_raw = pd.DataFrame(raw_reviews)
    
    print(f"\n" + "=" * 70)
    print("📊 SCRAPING COMPLETE - PROCESSING DATA")
    print("=" * 70)
    print(f"   Raw reviews: {len(df_raw):,}")
    
    # Show rating distribution
    print(f"\n📈 Rating Distribution (raw):")
    rating_dist = df_raw['score'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        percentage = count / len(df_raw) * 100
        print(f"   ⭐ {rating}: {count:,} ({percentage:.1f}%)")
    
    # Clean and validate
    print(f"\n🧹 Cleaning & Quality Check...")
    df_raw['content_clean'] = df_raw['content'].apply(clean_text)
    
    # Calculate quality scores
    print(f"   Calculating quality scores...")
    quality_results = df_raw['content'].apply(
        lambda x: is_high_quality_review(str(x), min_score=50)
    )
    df_raw['is_quality'] = quality_results.apply(lambda x: x[0])
    df_raw['quality_score'] = quality_results.apply(lambda x: x[1])
    
    # Filter quality reviews
    df_quality = df_raw[df_raw['is_quality']].copy()
    
    print(f"\n✅ Quality Filter Results:")
    print(f"   Raw: {len(df_raw):,}")
    print(f"   High Quality: {len(df_quality):,} ({len(df_quality)/len(df_raw)*100:.1f}%)")
    print(f"   Filtered out: {len(df_raw) - len(df_quality):,}")
    
    # Show quality rating distribution
    print(f"\n📈 Rating Distribution (quality filtered):")
    quality_rating_dist = df_quality['score'].value_counts().sort_index()
    for rating, count in quality_rating_dist.items():
        percentage = count / len(df_quality) * 100
        print(f"   ⭐ {rating}: {count:,} ({percentage:.1f}%)")
    
    # Remove duplicates
    print(f"\n🔍 Removing duplicates...")
    before_dedup = len(df_quality)
    df_quality = df_quality.drop_duplicates(subset=['content'], keep='first')
    duplicates = before_dedup - len(df_quality)
    print(f"   Removed: {duplicates:,} duplicates")
    print(f"   Final: {len(df_quality):,} unique reviews")
    
    # Check with existing data
    existing_file = os.path.join(OUTPUT_DIR, 'gojek_reviews_indobert_ready.csv')
    if os.path.exists(existing_file):
        print(f"\n📂 Checking with existing dataset...")
        df_existing = pd.read_csv(existing_file)
        existing_contents = set(df_existing['content_clean'].values)
        
        df_quality['is_new'] = ~df_quality['content_clean'].isin(existing_contents)
        new_count = df_quality['is_new'].sum()
        
        print(f"   Existing dataset: {len(df_existing):,}")
        print(f"   New unique data: {new_count:,}")
        print(f"   Duplicates with existing: {len(df_quality) - new_count:,}")
        
        df_quality = df_quality[df_quality['is_new']].copy()
    
    if df_quality.empty:
        print("\n⚠️ No new high-quality data after filtering!")
        return
    
    # Map sentiment
    sentiment_map = {1: 'negative', 2: 'negative', 3: 'neutral', 4: 'positive', 5: 'positive'}
    df_quality['sentiment'] = df_quality['score'].map(sentiment_map)
    df_quality['sentiment_corrected'] = df_quality['sentiment']
    
    # Prepare final dataframe
    df_final = pd.DataFrame({
        'review_id': [str(uuid.uuid4()) for _ in range(len(df_quality))],
        'content': df_quality['content'].values,
        'text': df_quality['content'].values,
        'content_clean': df_quality['content_clean'].values,
        'rating': df_quality['score'].values,
        'sentiment': df_quality['sentiment'].values,
        'sentiment_corrected': df_quality['sentiment_corrected'].values,
        'quality_score': df_quality['quality_score'].values
    })
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(OUTPUT_DIR, f'gojek_50k_quality_{timestamp}.csv')
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    
    # Statistics
    print(f"\n" + "=" * 70)
    print("✅ SCRAPING & PROCESSING COMPLETE")
    print("=" * 70)
    print(f"\n📊 FINAL STATISTICS:")
    print(f"   Total reviews: {len(df_final):,}")
    print(f"   Average quality score: {df_final['quality_score'].mean():.1f}/100")
    print(f"   Average words: {df_final['content_clean'].str.split().str.len().mean():.1f}")
    
    print(f"\n📈 Sentiment Distribution:")
    sent_dist = df_final['sentiment'].value_counts()
    for sent, count in sent_dist.items():
        percentage = count / len(df_final) * 100
        print(f"   {sent.upper():10s}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n💾 SAVED TO:")
    print(f"   {output_file}")
    
    # Show samples
    print(f"\n📋 Sample Reviews (High Quality):")
    for idx, row in df_final.head(3).iterrows():
        print(f"\n   [{row['sentiment'].upper()} | ⭐{row['rating']} | Q:{row['quality_score']:.0f}]")
        print(f"   \"{row['content'][:100]}...\"")
    
    print("\n" + "=" * 70)
    print("🎉 SUCCESS!")
    print("=" * 70)
    print(f"\n📌 NEXT STEPS:")
    print(f"   1. Review data quality: {output_file}")
    print(f"   2. Merge dengan dataset utama:")
    print(f"      python merge_rating3_data.py")
    print(f"   3. Re-train model dengan data yang lebih banyak & balance")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Scraping stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
