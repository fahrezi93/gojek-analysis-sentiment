import pandas as pd

# Cek file dataset
print("File di folder data:")
print("1. gojek_3class_BALANCED.csv")
df3 = pd.read_csv("data/gojek_3class_BALANCED.csv")
print(f"   - Total: {len(df3)} rows")
print(f"   - Columns: {list(df3.columns)}")
print(f"   - Labels: {df3['sentiment'].unique().tolist()}")

print()
print("2. gojek_5class_BALANCED_FIXED.csv")
df5 = pd.read_csv("data/gojek_5class_BALANCED_FIXED.csv")
print(f"   - Total: {len(df5)} rows")
print(f"   - Columns: {list(df5.columns)}")
print(f"   - Labels: {df5['sentiment'].unique().tolist()}")

print()
print("="*60)
print("INFO PENTING:")
print("="*60)
print("""
Berdasarkan analisis:
- Training 3-Kelas di Kaggle menggunakan: gojek_reviews_FINAL_POLISHED_READY.csv
- Training 5-Kelas di Kaggle menggunakan: gojek_5class_BALANCED_FIXED.csv

Yang kamu punya di lokal:
- gojek_3class_BALANCED.csv 
- gojek_5class_BALANCED_FIXED.csv

Pertanyaan: Apakah 'gojek_reviews_FINAL_POLISHED_READY.csv' sama dengan 'gojek_3class_BALANCED.csv'?
""")
