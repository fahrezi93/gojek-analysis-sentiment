import pandas as pd
import os


def log_print(message, file):
    print(message)
    file.write(message + "\n")

def check_data_quality(file_path, report_file):
    log_print(f"\n{'='*50}", report_file)
    log_print(f"Checking file: {os.path.basename(file_path)}", report_file)
    log_print(f"{'='*50}", report_file)
    
    if not os.path.exists(file_path):
        log_print(f"Error: File not found at {file_path}", report_file)
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        log_print(f"Error reading CSV: {e}", report_file)
        return

    log_print(f"Total Rows: {len(df)}", report_file)
    log_print(f"Columns: {list(df.columns)}", report_file)
    
    # Check for missing values
    log_print("\n[-] Missing Values:", report_file)
    log_print(str(df.isnull().sum()), report_file)
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    log_print(f"\n[-] Duplicates: {duplicates}", report_file)
    
    # Check class distribution
    if 'sentiment' in df.columns:
        log_print("\n[-] Sentiment Distribution:", report_file)
        log_print(str(df['sentiment'].value_counts()), report_file)
    elif 'label' in df.columns:
        log_print("\n[-] Label Distribution:", report_file)
        log_print(str(df['label'].value_counts()), report_file)
        
    # Sample data
    log_print("\n[-] Sample Data (First 3 rows):", report_file)
    if 'sentiment' in df.columns:
        log_print(str(df[['text', 'sentiment']].head(3)), report_file)
    elif 'text' in df.columns:
        log_print(str(df[['text']].head(3)), report_file)
    else:
        log_print(str(df.head(3)), report_file)
    
    # text quality check
    log_print("\n[-] Text Quality Check (scan for potential cleaning needs):", report_file)
    text_col = 'text' if 'text' in df.columns else 'content'
    
    if text_col in df.columns:
        # Check for HTML tags (basic check)
        html_tags = df[text_col].str.contains(r'<[^>]+>', regex=True).sum()
        log_print(f"  - Rows with HTML tags: {html_tags}", report_file)
        
        # Check for uppercase characters (if not 0, might not be lowercased)
        has_uppercase = df[text_col].str.contains(r'[A-Z]', regex=True).sum()
        log_print(f"  - Rows with uppercase characters: {has_uppercase} (should be 0 if fully cleaned/lowercased)", report_file)

        
    log_print("\n", report_file)

def main():
    base_dir = r"d:\Skripsi\sentiment-analyst-ojol-review\data"
    files = [
        "gojek_scraped_3class_20251206_130028.csv",
        "gojek_scraped_5class_20251206_130028.csv"
    ]
    
    with open("data_quality_report.txt", "w", encoding="utf-8") as f:
        for file in files:
            check_data_quality(os.path.join(base_dir, file), f)


if __name__ == "__main__":
    main()
