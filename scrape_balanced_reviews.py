"""
Script untuk Scraping Review Gojek dari Google Play Store
VERSI BALANCED - Scraping per rating untuk data seimbang

Target: 15.000 reviews SEIMBANG
- 3 kelas: masing-masing 5.000 (negative, neutral, positive)
- 5 kelas: masing-masing 3.000 per rating

Output:
- data/gojek_reviews_raw_balanced.csv
- data/gojek_reviews_3class_raw_balanced.csv  
- data/gojek_reviews_5class_raw_balanced.csv
"""

import pandas as pd
from google_play_scraper import reviews, Sort
from datetime import datetime
import time
import os

# =====================================================
# KONFIGURASI
# =====================================================
APP_ID = 'com.gojek.app'
BATCH_SIZE = 200
LANG = 'id'
COUNTRY = 'id'

# Target per rating untuk 5 kelas (total 15.000)
TARGET_PER_RATING = 3000  # 3.000 x 5 rating = 15.000

# Target per sentiment untuk 3 kelas (total 15.000)
TARGET_3CLASS = {
    'negative': 5000,   # rating 1-2
    'neutral': 5000,    # rating 3
    'positive': 5000    # rating 4-5
}

os.makedirs('data', exist_ok=True)

# =====================================================
# FUNGSI SCRAPING PER RATING
# =====================================================
def scrape_by_score(target_per_score=3000):
    """
    Scrape reviews dengan filter per score (1-5)
    Untuk mendapatkan data yang seimbang per rating
    """
    print('=' * 60)
    print('🔄 SCRAPING GOJEK REVIEWS - BALANCED MODE')
    print('=' * 60)
    print(f'App ID: {APP_ID}')
    print(f'Target per rating: {target_per_score:,} reviews')
    print(f'Total target: {target_per_score * 5:,} reviews')
    print('=' * 60)
    
    all_reviews = {1: [], 2: [], 3: [], 4: [], 5: []}
    
    start_time = time.time()
    
    for score in [1, 2, 3, 4, 5]:
        print(f'\n⭐ Scraping rating {score}...')
        continuation_token = None
        batch_num = 0
        consecutive_empty = 0
        
        while len(all_reviews[score]) < target_per_score:
            batch_num += 1
            
            try:
                result, continuation_token = reviews(
                    APP_ID,
                    lang=LANG,
                    country=COUNTRY,
                    sort=Sort.NEWEST,
                    count=BATCH_SIZE,
                    filter_score_with=score,  # Filter by specific score!
                    continuation_token=continuation_token
                )
                
                if not result:
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        print(f'\n   ⚠️ No more reviews for rating {score}')
                        print(f'   Got {len(all_reviews[score]):,} reviews (target: {target_per_score:,})')
                        break
                    time.sleep(1)
                    continue
                
                consecutive_empty = 0
                all_reviews[score].extend(result)
                
                # Progress
                current = len(all_reviews[score])
                pct = min(current / target_per_score * 100, 100)
                print(f'\r   📥 Rating {score}: {current:,}/{target_per_score:,} ({pct:.1f}%)', end='')
                
                # Rate limiting
                time.sleep(0.3)
                
                # Stop if we have enough
                if len(all_reviews[score]) >= target_per_score:
                    all_reviews[score] = all_reviews[score][:target_per_score]
                    break
                    
            except Exception as e:
                print(f'\n   ❌ Error: {e}')
                time.sleep(3)
                continue
        
        print(f'\n   ✅ Rating {score}: {len(all_reviews[score]):,} reviews collected')
    
    # Combine all reviews
    combined = []
    for score in [1, 2, 3, 4, 5]:
        combined.extend(all_reviews[score])
    
    elapsed = time.time() - start_time
    print(f'\n\n✅ Scraping completed in {elapsed/60:.1f} minutes')
    print(f'📊 Total reviews: {len(combined):,}')
    
    return combined, all_reviews

def process_to_dataframe(reviews_data):
    """Convert to DataFrame"""
    print('\n📝 Processing to DataFrame...')
    
    df = pd.DataFrame(reviews_data)
    
    # Select columns
    cols = ['reviewId', 'userName', 'content', 'score', 'thumbsUpCount', 
            'reviewCreatedVersion', 'at', 'replyContent', 'repliedAt']
    df = df[[c for c in cols if c in df.columns]]
    
    # Rename
    rename_map = {
        'reviewId': 'review_id',
        'userName': 'user_name', 
        'content': 'content',
        'score': 'rating',
        'thumbsUpCount': 'thumbs_up',
        'reviewCreatedVersion': 'app_version',
        'at': 'review_date',
        'replyContent': 'reply_content',
        'repliedAt': 'reply_date'
    }
    df = df.rename(columns=rename_map)
    
    df['scraped_at'] = datetime.now()
    
    print(f'✅ DataFrame: {len(df):,} rows')
    return df

def create_3class_balanced(df, target_per_class=5000):
    """
    Create balanced 3-class dataset
    """
    print('\n🏷️ Creating BALANCED 3-class labels...')
    
    df_copy = df.copy()
    
    # Map rating to sentiment
    def map_3class(rating):
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:
            return 'positive'
    
    df_copy['sentiment'] = df_copy['rating'].apply(map_3class)
    
    # Balance by undersampling
    balanced_dfs = []
    for sentiment in ['negative', 'neutral', 'positive']:
        df_class = df_copy[df_copy['sentiment'] == sentiment]
        
        if len(df_class) >= target_per_class:
            df_sampled = df_class.sample(n=target_per_class, random_state=42)
        else:
            # If not enough, take all
            df_sampled = df_class
            print(f'   ⚠️ {sentiment}: only {len(df_class)} available (target: {target_per_class})')
        
        balanced_dfs.append(df_sampled)
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print('\n📊 3-Class Balanced Distribution:')
    for sentiment, count in df_balanced['sentiment'].value_counts().items():
        print(f'   {sentiment}: {count:,}')
    
    return df_balanced

def create_5class_balanced(df, target_per_class=3000):
    """
    Create balanced 5-class dataset
    """
    print('\n🏷️ Creating BALANCED 5-class labels...')
    
    df_copy = df.copy()
    
    label_map = {
        1: 'sangat_negatif',
        2: 'negatif', 
        3: 'netral',
        4: 'positif',
        5: 'sangat_positif'
    }
    df_copy['sentiment'] = df_copy['rating'].map(label_map)
    
    # Balance by undersampling
    balanced_dfs = []
    for rating in [1, 2, 3, 4, 5]:
        sentiment = label_map[rating]
        df_class = df_copy[df_copy['rating'] == rating]
        
        if len(df_class) >= target_per_class:
            df_sampled = df_class.sample(n=target_per_class, random_state=42)
        else:
            df_sampled = df_class
            print(f'   ⚠️ {sentiment}: only {len(df_class)} available (target: {target_per_class})')
        
        balanced_dfs.append(df_sampled)
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print('\n📊 5-Class Balanced Distribution:')
    order = ['sangat_negatif', 'negatif', 'netral', 'positif', 'sangat_positif']
    for sentiment in order:
        if sentiment in df_balanced['sentiment'].values:
            count = len(df_balanced[df_balanced['sentiment'] == sentiment])
            print(f'   {sentiment}: {count:,}')
    
    return df_balanced

def save_data(df_raw, df_3class, df_5class):
    """Save all data"""
    print('\n💾 Saving data...')
    
    # Save raw
    df_raw.to_csv('data/gojek_reviews_raw_balanced.csv', index=False)
    print(f'   ✅ data/gojek_reviews_raw_balanced.csv ({len(df_raw):,} rows)')
    
    # Save 3-class
    df_3class.to_csv('data/gojek_reviews_3class_raw_balanced.csv', index=False)
    print(f'   ✅ data/gojek_reviews_3class_raw_balanced.csv ({len(df_3class):,} rows)')
    
    # Save 5-class
    df_5class.to_csv('data/gojek_reviews_5class_raw_balanced.csv', index=False)
    print(f'   ✅ data/gojek_reviews_5class_raw_balanced.csv ({len(df_5class):,} rows)')

def main():
    print('\n' + '=' * 60)
    print('📱 GOJEK REVIEW SCRAPER - BALANCED DATA')
    print('=' * 60)
    print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)
    
    # Scrape per rating
    reviews_data, reviews_by_score = scrape_by_score(target_per_score=TARGET_PER_RATING)
    
    if not reviews_data:
        print('❌ No reviews collected!')
        return
    
    # Show per-rating stats
    print('\n📊 Reviews per Rating:')
    for score in [1, 2, 3, 4, 5]:
        print(f'   Rating {score}: {len(reviews_by_score[score]):,}')
    
    # Process to DataFrame
    df_raw = process_to_dataframe(reviews_data)
    
    # Create balanced 3-class (5000 per class = 15000 total)
    df_3class = create_3class_balanced(df_raw, target_per_class=5000)
    
    # Create balanced 5-class (3000 per class = 15000 total)
    df_5class = create_5class_balanced(df_raw, target_per_class=3000)
    
    # Save
    save_data(df_raw, df_3class, df_5class)
    
    # Final summary
    print('\n' + '=' * 60)
    print('📊 FINAL SUMMARY')
    print('=' * 60)
    print(f'Raw data: {len(df_raw):,} reviews')
    print(f'3-Class balanced: {len(df_3class):,} reviews')
    print(f'5-Class balanced: {len(df_5class):,} reviews')
    
    print('\n' + '=' * 60)
    print('✅ BALANCED SCRAPING COMPLETED!')
    print('=' * 60)
    print('\n⚠️  Data ini BELUM di-cleaning!')
    print('   Tapi sudah SEIMBANG per kelas.')

if __name__ == '__main__':
    main()
