"""
Scrape dan simpan RAW DATA (mentah/kotor) tanpa filtering
Plus juga save yang clean
"""

import pandas as pd
from google_play_scraper import reviews, Sort
from datetime import datetime
import time
import os
from tqdm import tqdm

APP_ID = 'com.gojek.app'
TARGET = 10000  # Target cepat untuk demo
BATCH_SIZE = 200

print("=" * 80)
print("🚀 SCRAPE RAW DATA (MENTAH/KOTOR)")
print("=" * 80)

all_reviews = []
continuation_token = None
collected = 0

print(f"\n📥 Scraping {TARGET:,} reviews...")
pbar = tqdm(total=TARGET, desc="Scraping", unit="reviews")

while collected < TARGET:
    try:
        result, continuation_token = reviews(
            APP_ID,
            lang='id',
            country='id',
            sort=Sort.NEWEST,
            count=BATCH_SIZE,
            continuation_token=continuation_token
        )
        
        if not result:
            break
        
        all_reviews.extend(result)
        collected = len(all_reviews)
        
        pbar.update(len(result))
        pbar.set_postfix({'total': collected})
        
        time.sleep(0.3)
        
        if collected >= TARGET or continuation_token is None:
            break
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Stopped by user")
        break
    except Exception as e:
        print(f"\n⚠️ Error: {e}")
        time.sleep(2)
        continue

pbar.close()

# Convert to DataFrame
df_raw = pd.DataFrame(all_reviews)

print(f"\n✅ Scraped: {len(df_raw):,} reviews")

# Save RAW DATA (MENTAH - tanpa cleaning)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
raw_file = f'data/RAW_DATA_KOTOR_{timestamp}.csv'

# Keep all columns
df_raw.to_csv(raw_file, index=False, encoding='utf-8')

print(f"\n💾 SAVED RAW DATA (MENTAH):")
print(f"   📄 {raw_file}")
print(f"   Rows: {len(df_raw):,}")

# Show what's inside
print(f"\n📊 Columns: {list(df_raw.columns)}")
print(f"\n📈 Rating distribution:")
print(df_raw['score'].value_counts().sort_index())

# Show raw samples
print(f"\n📋 SAMPLE RAW DATA (apa adanya):")
for idx, row in df_raw.head(10).iterrows():
    content = row['content'][:80] if len(str(row['content'])) > 80 else row['content']
    print(f"\n   [⭐{row['score']}] {row['userName']}")
    print(f"   \"{content}\"")

print(f"\n" + "=" * 80)
print("✅ RAW DATA TERSIMPAN!")
print("=" * 80)
print(f"""
File ini berisi:
- ✅ Semua data apa adanya (kotor + bersih)
- ✅ Dengan emoji, typo, URL, semua masih ada
- ✅ Belum di-clean sama sekali
- ✅ Rating 1-5 semua ada
- ✅ Username & timestamp masih ada

Gunakan untuk:
- Analisis data kotor
- Bandingkan sebelum & sesudah cleaning
- Research purposes
""")
print("=" * 80)
