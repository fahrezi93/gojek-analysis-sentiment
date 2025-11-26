"""
Script untuk membersihkan dan menyeimbangkan data
gojek_reviews_relabelled_text_based.csv
"""
import pandas as pd
import numpy as np
from sklearn.utils import resample

# Load data
df = pd.read_csv('data/gojek_reviews_relabelled_text_based.csv')
print(f'Original data: {len(df):,}')
print(f'\nOriginal distribution:')
print(df['sentiment'].value_counts())

# 1. Remove score 3 yang labeled positive (ambiguous - 269 rows)
df_clean = df[~((df['score'] == 3) & (df['sentiment'] == 'positive'))].copy()
print(f'\nAfter removing ambiguous score 3 positive: {len(df_clean):,}')

# 2. Remove duplicates
df_clean = df_clean.drop_duplicates(subset=['content'])
print(f'After removing duplicates: {len(df_clean):,}')

# 3. Add word count and remove very short texts
df_clean['word_count'] = df_clean['content'].astype(str).apply(lambda x: len(x.split()))
df_clean = df_clean[df_clean['word_count'] >= 4]
print(f'After removing short texts: {len(df_clean):,}')

print(f'\nCleaned distribution:')
print(df_clean['sentiment'].value_counts())

# 4. Balance data dengan undersampling
sentiment_counts = df_clean['sentiment'].value_counts()
min_count = sentiment_counts.min()
print(f'\nMin class count: {min_count}')

df_balanced = pd.DataFrame()
for sentiment in df_clean['sentiment'].unique():
    df_class = df_clean[df_clean['sentiment'] == sentiment]
    df_sampled = resample(df_class, replace=False, n_samples=min_count, random_state=42)
    df_balanced = pd.concat([df_balanced, df_sampled])

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f'\nBalanced data: {len(df_balanced):,}')
print(f'\nBalanced distribution:')
print(df_balanced['sentiment'].value_counts())

# Save cleaned and balanced data
output_path = 'data/gojek_reviews_clean_balanced.csv'
df_balanced[['content', 'sentiment']].to_csv(output_path, index=False)
print(f'\n✅ Saved to: {output_path}')

# Also save the cleaned (but not balanced) version for reference
df_clean[['content', 'sentiment']].to_csv('data/gojek_reviews_cleaned.csv', index=False)
print(f'✅ Saved cleaned (unbalanced) to: data/gojek_reviews_cleaned.csv')
