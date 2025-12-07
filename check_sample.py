import pandas as pd

def check_text_sample():
    df = pd.read_csv(r"d:\Skripsi\sentiment-analyst-ojol-review\data\gojek_scraped_3class_20251206_130028_cleaned.csv")
    print("Sample Text Content:")
    print(df['text'].head(10).tolist())

if __name__ == "__main__":
    check_text_sample()
