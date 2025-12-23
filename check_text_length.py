import pandas as pd

df = pd.read_csv('data/gojek_5class_BALANCED_FIXED.csv')
df['text_len'] = df['text'].str.len()

print('=== DISTRIBUSI PANJANG TEKS PER KELAS ===\n')
for sentiment in ['very_positive', 'positive', 'neutral', 'negative', 'very_negative']:
    subset = df[df['sentiment'] == sentiment]['text_len']
    print(f'{sentiment.upper()}:')
    print(f'  Min: {subset.min()} | Max: {subset.max()} | Median: {subset.median():.0f}')
    print(f'  < 50 chars: {(subset < 50).sum()} | < 100 chars: {(subset < 100).sum()}')
    print()

print('=== CONTOH TEKS PENDEK (< 30 chars) PER KELAS ===\n')
for sentiment in ['very_positive', 'positive', 'neutral', 'negative', 'very_negative']:
    short = df[(df['sentiment'] == sentiment) & (df['text_len'] < 30)]['text'].head(5).tolist()
    print(f'{sentiment.upper()}:')
    for t in short:
        print(f'  - "{t}"')
    print()
