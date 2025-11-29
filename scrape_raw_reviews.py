"""
Script untuk Scraping Review Gojek dari Google Play Store
Menghasilkan data MENTAH (RAW) tanpa cleaning

Output:
- data/gojek_reviews_raw.csv (semua data mentah)
- data/gojek_reviews_3class_raw.csv (dengan label 3 kelas)
- data/gojek_reviews_5class_raw.csv (dengan label 5 kelas)
"""

import pandas as pd
from google_play_scraper import reviews, Sort
from datetime import datetime
import time
import os

# =====================================================
# KONFIGURASI
# =====================================================
APP_ID = 'com.gojek.app'  # Gojek app ID di Play Store
TOTAL_REVIEWS = 15000     # Target 15.000 reviews
BATCH_SIZE = 200          # Jumlah review per batch
LANG = 'id'               # Bahasa Indonesia
COUNTRY = 'id'            # Indonesia

# Buat folder data jika belum ada
os.makedirs('data', exist_ok=True)

# =====================================================
# FUNGSI SCRAPING
# =====================================================
def scrape_gojek_reviews(target_count=10000):
    """
    Scrape reviews dari Google Play Store
    """
    print('=' * 60)
    print('🔄 SCRAPING GOJEK REVIEWS FROM GOOGLE PLAY STORE')
    print('=' * 60)
    print(f'App ID: {APP_ID}')
    print(f'Target: {target_count} reviews')
    print(f'Language: {LANG}, Country: {COUNTRY}')
    print('-' * 60)
    
    all_reviews = []
    continuation_token = None
    batch_num = 0
    
    while len(all_reviews) < target_count:
        batch_num += 1
        try:
            # Scrape batch
            result, continuation_token = reviews(
                APP_ID,
                lang=LANG,
                country=COUNTRY,
                sort=Sort.NEWEST,  # Ambil yang terbaru
                count=BATCH_SIZE,
                continuation_token=continuation_token
            )
            
            if not result:
                print(f'⚠️ No more reviews available')
                break
            
            all_reviews.extend(result)
            print(f'📥 Batch {batch_num}: Got {len(result)} reviews | Total: {len(all_reviews)}')
            
            # Jika tidak ada token lanjutan, berarti sudah habis
            if continuation_token is None:
                print('✓ Reached end of reviews')
                break
            
            # Delay untuk menghindari rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f'❌ Error at batch {batch_num}: {e}')
            time.sleep(5)  # Wait longer on error
            continue
    
    print(f'\n✓ Total scraped: {len(all_reviews)} reviews')
    return all_reviews

def create_dataframe(reviews_data):
    """
    Convert reviews data ke DataFrame
    """
    data = []
    for review in reviews_data:
        data.append({
            'reviewId': review.get('reviewId', ''),
            'userName': review.get('userName', ''),
            'content': review.get('content', ''),  # Review text MENTAH
            'score': review.get('score', 0),       # Rating 1-5
            'thumbsUpCount': review.get('thumbsUpCount', 0),
            'reviewCreatedVersion': review.get('reviewCreatedVersion', ''),
            'at': review.get('at', ''),            # Tanggal review
            'replyContent': review.get('replyContent', ''),
            'repliedAt': review.get('repliedAt', '')
        })
    
    df = pd.DataFrame(data)
    return df

def add_3class_labels(df):
    """
    Tambahkan label 3 kelas berdasarkan rating:
    - negative: rating 1-2
    - neutral: rating 3
    - positive: rating 4-5
    """
    def get_sentiment_3class(score):
        if score <= 2:
            return 'negative'
        elif score == 3:
            return 'neutral'
        else:
            return 'positive'
    
    df['sentiment'] = df['score'].apply(get_sentiment_3class)
    return df

def add_5class_labels(df):
    """
    Tambahkan label 5 kelas berdasarkan rating:
    - sangat_negatif: rating 1
    - negatif: rating 2
    - netral: rating 3
    - positif: rating 4
    - sangat_positif: rating 5
    """
    label_map = {
        1: 'sangat_negatif',
        2: 'negatif',
        3: 'netral',
        4: 'positif',
        5: 'sangat_positif'
    }
    df['sentiment'] = df['score'].map(label_map)
    return df

# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == '__main__':
    start_time = datetime.now()
    print(f'🕐 Started at: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print()
    
    # 1. Scrape reviews
    reviews_data = scrape_gojek_reviews(TOTAL_REVIEWS)
    
    if not reviews_data:
        print('❌ No reviews scraped!')
        exit(1)
    
    # 2. Create base DataFrame
    df_raw = create_dataframe(reviews_data)
    
    print('\n' + '=' * 60)
    print('📊 DATA SUMMARY')
    print('=' * 60)
    print(f'Total reviews: {len(df_raw)}')
    print(f'\nRating distribution:')
    print(df_raw['score'].value_counts().sort_index())
    
    # 3. Save RAW data (tanpa label sentiment)
    raw_path = 'data/gojek_reviews_raw.csv'
    df_raw.to_csv(raw_path, index=False, encoding='utf-8')
    print(f'\n✓ Saved: {raw_path} ({len(df_raw)} rows)')
    
    # 4. Create 3-class version
    df_3class = df_raw.copy()
    df_3class = add_3class_labels(df_3class)
    
    path_3class = 'data/gojek_reviews_3class_raw.csv'
    df_3class.to_csv(path_3class, index=False, encoding='utf-8')
    print(f'✓ Saved: {path_3class} ({len(df_3class)} rows)')
    print(f'  Distribution:')
    print(df_3class['sentiment'].value_counts())
    
    # 5. Create 5-class version
    df_5class = df_raw.copy()
    df_5class = add_5class_labels(df_5class)
    
    path_5class = 'data/gojek_reviews_5class_raw.csv'
    df_5class.to_csv(path_5class, index=False, encoding='utf-8')
    print(f'\n✓ Saved: {path_5class} ({len(df_5class)} rows)')
    print(f'  Distribution:')
    print(df_5class['sentiment'].value_counts())
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print('\n' + '=' * 60)
    print('✅ SCRAPING COMPLETED!')
    print('=' * 60)
    print(f'Duration: {duration}')
    print(f'\n📁 Output files:')
    print(f'   1. {raw_path} - Data mentah tanpa label')
    print(f'   2. {path_3class} - Data dengan label 3 kelas')
    print(f'   3. {path_5class} - Data dengan label 5 kelas')
    print()
    print('⚠️  CATATAN: Data ini BELUM di-cleaning!')
    print('    Kolom "content" berisi teks review MENTAH dari user')
