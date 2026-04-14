import pandas as pd
import numpy as np
from google_play_scraper import reviews, Sort
from datetime import datetime
import time
import os
import sys
from tqdm import tqdm
import uuid

# Import text cleaner
from text_cleaner_indobert import (
    clean_text_indobert,
    validate_text_quality,
    calculate_sentiment_score
)

# CONFIGURATION
APP_ID = 'com.gojek.app'
TARGET_TOTAL = 100000       
MIN_QUALITY_SCORE = 40

BATCH_SIZE = 200
MAX_RETRIES = 5
SLEEP_BETWEEN_BATCHES = 0.3
OUTPUT_DIR = 'data'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# QUALITY CHECKER
def calculate_review_quality(text: str, rating: int) -> float:
    """
    Calculate overall quality score for a review
    Returns 0-100
    """
    if not text or len(text) < 10:
        return 0
    
    score = 100.0
    words = text.split()
    
    # Length penalties
    if len(words) < 5:
        score -= 40
    elif len(words) > 200:
        score -= 20
    
    # Check text validity
    is_valid, reason = validate_text_quality(text)
    if not is_valid:
        if reason == "spam_pattern":
            score -= 60
        elif reason == "too_repetitive":
            score -= 40
        elif reason == "not_enough_text":
            score -= 50
        else:
            score -= 30
    
    # Calculate content quality
    sentiment_scores = calculate_sentiment_score(text)
    
    # Reward if sentiment aligns with rating
    if rating <= 2 and sentiment_scores['negative'] > 0.5:
        score += 10
    elif rating >= 4 and sentiment_scores['positive'] > 0.5:
        score += 10
    elif rating == 3 and sentiment_scores['neutral'] > 0.3:
        score += 10
    
    # Penalty for very short after cleaning
    if len(text) < 20:
        score -= 30
    
    return max(0, min(100, score))

# MULTI-STRATEGY SCRAPING
def scrape_with_strategy(app_id, strategy='newest', target=10000, focus_rating=None):
    """
    Scrape dengan berbagai strategi
    """
    sort_map = {
        'newest': Sort.NEWEST,
        'most_relevant': Sort.MOST_RELEVANT,
        'rating': Sort.RATING
    }
    
    sort_type = sort_map.get(strategy, Sort.NEWEST)
    
    all_reviews = []
    continuation_token = None
    
    strategy_display = strategy.upper()
    if focus_rating:
        strategy_display += f" (Rating {focus_rating})"
    
    collected = 0
    fetched_total = 0
    retries = 0
    consecutive_empty = 0
    
    pbar = tqdm(total=target, desc=strategy_display, unit="reviews")
    
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
            print(f"\n⏹️ Stopped by user")
            break
            
        except Exception as e:
            retries += 1
            if retries >= MAX_RETRIES:
                print(f"\n❌ Max retries reached: {e}")
                break
            time.sleep(2 ** retries)
            continue
    
    pbar.close()
    return all_reviews

def scrape_comprehensive_max(app_id, total_target=100000):
    """
    Scrape sebanyak-banyaknya dengan multiple strategies
    """
    print("=" * 80)
    print("🚀 COMPREHENSIVE SCRAPING - MAXIMIZE DATA")
    print("=" * 80)
    print(f"   Target: {total_target:,} reviews (or until exhausted)")
    print("=" * 80)
    
    all_reviews_dict = {}  # Deduplicate by reviewId
    
    strategies = [
        ('newest', None, 30000),
        ('most_relevant', None, 30000),
        ('rating', None, 20000),
        ('newest', 3, 10000),      # Extra focus on rating 3
        ('most_relevant', 3, 10000),
        ('newest', 1, 5000),
        ('newest', 5, 5000),
    ]
    
    for idx, (strategy, focus_rating, target) in enumerate(strategies, 1):
        if len(all_reviews_dict) >= total_target:
            print(f"\n✅ Target reached: {len(all_reviews_dict):,} reviews")
            break
        
        remaining = total_target - len(all_reviews_dict)
        actual_target = min(target, remaining)
        
        print(f"\n📌 PHASE {idx}/{len(strategies)}: {strategy.upper()}", end='')
        if focus_rating:
            print(f" (Rating {focus_rating})", end='')
        print(f" - Target: {actual_target:,}")
        
        batch_reviews = scrape_with_strategy(
            app_id,
            strategy=strategy,
            target=actual_target,
            focus_rating=focus_rating
        )
        
        for review in batch_reviews:
            review_id = review.get('reviewId', str(uuid.uuid4()))
            all_reviews_dict[review_id] = review
        
        print(f"   ✅ Total unique so far: {len(all_reviews_dict):,}")
        
        # Show rating distribution
        ratings = [r.get('score', 0) for r in all_reviews_dict.values()]
        rating_counts = pd.Series(ratings).value_counts().sort_index()
        print(f"   📊 Rating distribution:", end='')
        for rating, count in rating_counts.items():
            print(f" ⭐{rating}:{count}", end='')
        print()
    
    return list(all_reviews_dict.values())

# MAIN FUNCTION
def main():
    print("\n" + "=" * 80)
    print("🚀 GOJEK ADVANCED SCRAPER - INDOBERT READY")
    print("=" * 80)
    
    # Phase 1: Scrape
    print("\n" + "=" * 80)
    print("PHASE 1: SCRAPING")
    print("=" * 80)
    
    raw_reviews = scrape_comprehensive_max(APP_ID, TARGET_TOTAL)
    
    if not raw_reviews:
        print("\n❌ No reviews scraped!")
        return
    
    df_raw = pd.DataFrame(raw_reviews)
    
    print(f"\n" + "=" * 80)
    print("PHASE 2: CLEANING & PROCESSING")
    print("=" * 80)
    print(f"   Raw reviews scraped: {len(df_raw):,}")
    
    # Show raw rating distribution
    print(f"\n📈 Raw Rating Distribution:")
    for rating, count in df_raw['score'].value_counts().sort_index().items():
        percentage = count / len(df_raw) * 100
        print(f"   ⭐ {rating}: {count:,} ({percentage:.1f}%)")
    
    # Phase 2: Clean text
    print(f"\n🧹 Cleaning text for IndoBERT...")
    df_raw['content_original'] = df_raw['content']
    df_raw['content_clean'] = df_raw['content'].apply(clean_text_indobert)
    
    # Remove empty after cleaning
    before = len(df_raw)
    df_raw = df_raw[df_raw['content_clean'].str.len() > 0].copy()
    print(f"   Removed {before - len(df_raw):,} empty texts")
    print(f"   Remaining: {len(df_raw):,}")
    
    # Phase 3: Quality check
    print(f"\n✅ Quality validation...")
    validation_results = df_raw['content_clean'].apply(validate_text_quality)
    df_raw['is_valid'] = validation_results.apply(lambda x: x[0])
    df_raw['invalid_reason'] = validation_results.apply(lambda x: x[1])
    
    # Calculate quality score
    print(f"   Calculating quality scores...")
    df_raw['quality_score'] = df_raw.apply(
        lambda row: calculate_review_quality(row['content_clean'], row['score']),
        axis=1
    )
    
    # Filter by quality
    df_quality = df_raw[
        (df_raw['is_valid'] == True) & 
        (df_raw['quality_score'] >= MIN_QUALITY_SCORE)
    ].copy()
    
    print(f"\n   Quality Filter Results:")
    print(f"   ✅ High quality: {len(df_quality):,} ({len(df_quality)/len(df_raw)*100:.1f}%)")
    print(f"   ❌ Low quality:  {len(df_raw) - len(df_quality):,}")
    print(f"   📊 Avg quality score: {df_quality['quality_score'].mean():.1f}/100")
    
    # Remove duplicates
    print(f"\n🔍 Removing duplicates...")
    before = len(df_quality)
    df_quality = df_quality.drop_duplicates(subset=['content_clean'], keep='first')
    print(f"   Removed: {before - len(df_quality):,} duplicates")
    print(f"   Final unique: {len(df_quality):,}")
    
    if df_quality.empty:
        print("\n⚠️ No high-quality data after filtering!")
        return
    
    # Phase 4: Sentiment analysis & labeling
    print(f"\n" + "=" * 80)
    print("PHASE 3: SENTIMENT LABELING")
    print("=" * 80)
    
    print(f"🔍 Analyzing sentiment from text content...")
    sentiment_scores = df_quality['content_clean'].apply(calculate_sentiment_score)
    df_quality['pos_score'] = sentiment_scores.apply(lambda x: x['positive'])
    df_quality['neg_score'] = sentiment_scores.apply(lambda x: x['negative'])
    df_quality['neu_score'] = sentiment_scores.apply(lambda x: x['neutral'])
    
    # Create 3-class labels
    print(f"\n📊 Creating 3-CLASS labels...")
    
    def map_rating_to_sentiment_3class(rating, text=""):
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            scores = calculate_sentiment_score(text) if text else {}
            if scores.get('positive', 0) > 0.5 and scores.get('negative', 0) < 0.2:
                return 'positive'
            elif scores.get('negative', 0) > 0.5 and scores.get('positive', 0) < 0.2:
                return 'negative'
            return 'neutral'
        else:
            return 'positive'
    
    def map_rating_to_sentiment_5class(rating, text=""):
        if text:
            scores = calculate_sentiment_score(text)
            
            if rating == 1:
                if scores.get('negative', 0) > 0.7:
                    return 'very_negative'
                else:
                    return 'negative'
            
            if rating == 2:
                if scores.get('negative', 0) > 0.6:
                    return 'negative'
                elif scores.get('positive', 0) > 0.4:
                    return 'neutral'
                else:
                    return 'negative'
            
            if rating == 3:
                if scores.get('positive', 0) > 0.5:
                    return 'positive'
                elif scores.get('negative', 0) > 0.5:
                    return 'negative'
                else:
                    return 'neutral'
            
            if rating == 4:
                if scores.get('positive', 0) > 0.6:
                    return 'positive'
                elif scores.get('negative', 0) > 0.4:
                    return 'neutral'
                else:
                    return 'positive'
            
            if rating == 5:
                if scores.get('positive', 0) > 0.7:
                    return 'very_positive'
                else:
                    return 'positive'
        
        mapping = {1: 'very_negative', 2: 'negative', 3: 'neutral', 4: 'positive', 5: 'very_positive'}
        return mapping.get(rating, 'neutral')
    
    # Import function if not already
    try:
        from process_master_data import map_rating_to_sentiment_3class as m3, map_rating_to_sentiment_5class as m5
        map_rating_to_sentiment_3class = m3
        map_rating_to_sentiment_5class = m5
    except:
        pass  # Use local functions defined above
    
    df_quality['sentiment_3class'] = df_quality.apply(
        lambda row: map_rating_to_sentiment_3class(row['score'], row['content_clean']),
        axis=1
    )
    
    # Create 5-class labels
    print(f"📊 Creating 5-CLASS labels...")
    df_quality['sentiment_5class'] = df_quality.apply(
        lambda row: map_rating_to_sentiment_5class(row['score'], row['content_clean']),
        axis=1
    )
    
    # Phase 5: Save outputs
    print(f"\n" + "=" * 80)
    print("PHASE 4: SAVING RESULTS")
    print("=" * 80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Prepare datasets
    base_columns = ['content_clean', 'score', 'quality_score', 'pos_score', 'neg_score', 'neu_score']
    
    # 3-class dataset
    df_3class = df_quality[base_columns + ['sentiment_3class']].copy()
    df_3class.rename(columns={
        'content_clean': 'text',
        'score': 'rating',
        'sentiment_3class': 'sentiment'
    }, inplace=True)
    
    # 5-class dataset
    df_5class = df_quality[base_columns + ['sentiment_5class']].copy()
    df_5class.rename(columns={
        'content_clean': 'text',
        'score': 'rating',
        'sentiment_5class': 'sentiment'
    }, inplace=True)
    
    # Save both
    file_3class = os.path.join(OUTPUT_DIR, f'gojek_scraped_3class_{timestamp}.csv')
    file_5class = os.path.join(OUTPUT_DIR, f'gojek_scraped_5class_{timestamp}.csv')
    
    df_3class.to_csv(file_3class, index=False, encoding='utf-8')
    df_5class.to_csv(file_5class, index=False, encoding='utf-8')
    
    print(f"   ✅ 3-class dataset: {file_3class}")
    print(f"   ✅ 5-class dataset: {file_5class}")
    
    # Statistics
    print(f"\n" + "=" * 80)
    print("✅ SCRAPING COMPLETE - FINAL STATISTICS")
    print("=" * 80)
    
    print(f"\n📊 DATASET SIZES:")
    print(f"   Total scraped:  {len(df_raw):,}")
    print(f"   High quality:   {len(df_quality):,}")
    print(f"   Avg quality:    {df_quality['quality_score'].mean():.1f}/100")
    print(f"   Avg words:      {df_quality['content_clean'].str.split().str.len().mean():.1f}")
    
    print(f"\n📊 3-CLASS DISTRIBUTION:")
    for sent, count in df_3class['sentiment'].value_counts().items():
        percentage = count / len(df_3class) * 100
        print(f"   {sent:10s}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n📊 5-CLASS DISTRIBUTION:")
    for sent, count in df_5class['sentiment'].value_counts().items():
        percentage = count / len(df_5class) * 100
        print(f"   {sent:15s}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n📊 RATING DISTRIBUTION:")
    for rating, count in df_quality['score'].value_counts().sort_index().items():
        percentage = count / len(df_quality) * 100
        print(f"   ⭐ {rating}: {count:,} ({percentage:.1f}%)")
    
    # Show samples
    print(f"\n📋 SAMPLE HIGH-QUALITY REVIEWS:")
    for idx, row in df_3class.head(5).iterrows():
        print(f"\n   [{row['sentiment'].upper():8s} | ⭐{row['rating']} | Q:{row['quality_score']:.0f}]")
        print(f"   \"{row['text'][:100]}...\"")
    
    print("\n" + "=" * 80)
    print("🎉 SUCCESS - DATA READY FOR INDOBERT!")
    print("=" * 80)
    print(f"\n📌 NEXT STEPS:")
    print(f"   1. Process untuk balancing:")
    print(f"      python process_master_data.py")
    print(f"   2. Atau langsung train dengan file yang sudah ada")
    print(f"   3. Gunakan 3-class untuk general sentiment")
    print(f"   4. Gunakan 5-class untuk sentiment lebih detail")
    print("=" * 80)

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
