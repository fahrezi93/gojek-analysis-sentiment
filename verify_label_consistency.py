import pandas as pd
import os

def log_print(message, file):
    print(message)
    file.write(message + "\n")

def check_label_consistency(file_path, report_file):
    log_print(f"\n{'='*50}", report_file)
    log_print(f"Deep Analysis for: {os.path.basename(file_path)}", report_file)
    log_print(f"{'='*50}", report_file)
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        log_print(f"Error reading CSV: {e}", report_file)
        return

    # 1. Check direct correlation between Rating and Sentiment
    log_print("\n[1] Correlation Check (Rating vs Sentiment):", report_file)
    if 'rating' in df.columns and 'sentiment' in df.columns:
        crosstab = pd.crosstab(df['rating'], df['sentiment'])
        log_print(str(crosstab), report_file)
        
        # Determine if it's purely rating-based
        is_purely_rating_based = True
        # Logic: if for a given rating, there is more than one sentiment assigned, it's NOT purely rating based (mostly).
        # However, usually rating-based means: 1,2=neg, 3=neu, 4,5=pos.
        # Let's see if any rating has mixed sentiments.
        for rating in crosstab.index:
            sentiments_for_rating = crosstab.loc[rating]
            non_zero_sentiments = sentiments_for_rating[sentiments_for_rating > 0].count()
            if non_zero_sentiments > 1:
                is_purely_rating_based = False
                break
        
        if is_purely_rating_based:
            log_print("\nWARNING: Labels appear to be strictly derived from Rating!", report_file)
        else:
            log_print("\nNOTE: Labels vary within the same rating (Good sign, implies content analysis)", report_file)

    # 2. Keyword Analysis for Mismatches
    # Simple keywords list (Indonesian)
    pos_keywords = ['bagus', 'mantap', 'keren', 'suka', 'puas', 'terbaik', 'love', 'cepat', 'ramah']
    neg_keywords = ['jelek', 'kecewa', 'lambat', 'mahal', 'parah', 'gagal', 'rusak', 'susah', 'buruk', 'kapok']
    
    text_col = 'text' if 'text' in df.columns else 'content'
    
    log_print("\n[2] Content Mismatch Check (Keyword Heuristics):", report_file)
    
    # Check Neutral labeled rows
    if 'neutral' in df['sentiment'].unique():
        neu_df = df[df['sentiment'] == 'neutral']
        
        # Check for positive words in neutral
        pos_in_neu = neu_df[neu_df[text_col].str.contains('|'.join(pos_keywords), case=False, na=False)]
        log_print(f"\n  - 'Neutral' rows containing Strong Positive words: {len(pos_in_neu)} found", report_file)
        if len(pos_in_neu) > 0:
            log_print("    Examples:", report_file)
            log_print(str(pos_in_neu[[text_col, 'rating']].head(5)), report_file)
            
        # Check for negative words in neutral
        neg_in_neu = neu_df[neu_df[text_col].str.contains('|'.join(neg_keywords), case=False, na=False)]
        log_print(f"\n  - 'Neutral' rows containing Strong Negative words: {len(neg_in_neu)} found", report_file)
        if len(neg_in_neu) > 0:
            log_print("    Examples:", report_file)
            log_print(str(neg_in_neu[[text_col, 'rating']].head(5)), report_file)

    # Check Positive labeled rows for negative words (potential irony or mislabel)
    pos_labels = ['positive', 'very_positive']
    pos_df = df[df['sentiment'].isin(pos_labels)]
    if not pos_df.empty:
        neg_in_pos = pos_df[pos_df[text_col].str.contains('|'.join(neg_keywords), case=False, na=False)]
        log_print(f"\n  - 'Positive' rows containing Strong Negative words (Possible Mislabels): {len(neg_in_pos)} found", report_file)
        if len(neg_in_pos) > 0:
            log_print("    Examples:", report_file)
            log_print(str(neg_in_pos[[text_col, 'rating']].head(5)), report_file)

    # Check Negative labeled rows for positive words
    neg_labels = ['negative', 'very_negative']
    neg_df = df[df['sentiment'].isin(neg_labels)]
    if not neg_df.empty:
        pos_in_neg = neg_df[neg_df[text_col].str.contains('|'.join(pos_keywords), case=False, na=False)]
        log_print(f"\n  - 'Negative' rows containing Strong Positive words: {len(pos_in_neg)} found", report_file)
        if len(pos_in_neg) > 0:
            log_print("    Examples:", report_file)
            log_print(str(pos_in_neg[[text_col, 'rating']].head(5)), report_file)

def main():
    base_dir = r"d:\Skripsi\sentiment-analyst-ojol-review\data"
    files = [
        "gojek_scraped_3class_20251206_130028.csv",
        "gojek_scraped_5class_20251206_130028.csv"
    ]
    
    with open("label_consistency_report.txt", "w", encoding="utf-8") as f:
        for file in files:
            check_label_consistency(os.path.join(base_dir, file), f)

if __name__ == "__main__":
    main()
