import pandas as pd

# Check raw data
df_raw = pd.read_csv(r'd:\Skripsi\sentiment-analyst-ojol-review\data\gojek_scraped_3class_all.csv')
print(f"Raw 3class total: {len(df_raw)}")
print(f"Columns: {list(df_raw.columns)}")
if 'rating' in df_raw.columns:
    print(f"Ratings: {df_raw['rating'].value_counts().sort_index().to_dict()}")

print()

# Check balanced 3class
df_3b = pd.read_csv(r'd:\Skripsi\sentiment-analyst-ojol-review\data\gojek_3class_BALANCED.csv')
print(f"Balanced 3class total: {len(df_3b)}")
print(f"Columns: {list(df_3b.columns)}")
print(df_3b['sentiment'].value_counts())

print()

# Check balanced 5class
df_5b = pd.read_csv(r'd:\Skripsi\sentiment-analyst-ojol-review\data\gojek_5class_BALANCED_FIXED.csv')
print(f"Balanced 5class total: {len(df_5b)}")
print(f"Columns: {list(df_5b.columns)}")
print(df_5b['sentiment'].value_counts())

# Check raw 5class  
df_5raw = pd.read_csv(r'd:\Skripsi\sentiment-analyst-ojol-review\data\gojek_scraped_5class_all.csv')
print(f"\nRaw 5class total: {len(df_5raw)}")
