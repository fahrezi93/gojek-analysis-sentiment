import pandas as pd
import os

def verify_cleaned():
    base_dir = r"d:\Skripsi\sentiment-analyst-ojol-review\data"
    files = [
        "gojek_scraped_3class_20251206_130028_cleaned.csv",
        "gojek_scraped_5class_20251206_130028_cleaned.csv"
    ]
    
    with open("cleaning_report_final.txt", "w") as report:
        report.write("="*50 + "\n")
        report.write("VERIFICATION OF CLEANED DATA\n")
        report.write("="*50 + "\n")
        
        for f in files:
            path = os.path.join(base_dir, f)
            if not os.path.exists(path):
                report.write(f"File not found: {f}\n")
                continue
                
            report.write(f"\nFile: {f}\n")
            df = pd.read_csv(path)
            report.write(f"Total Rows: {len(df)}\n")
            report.write("New Sentiment Distribution:\n")
            report.write(str(df['sentiment'].value_counts()) + "\n")
            
            # Check if 'sentiment_original' exists to see how many changed
            if 'sentiment_original' in df.columns:
                changes = (df['sentiment'] != df['sentiment_original']).sum()
                report.write(f"Total Changed from Original: {changes} rows ({changes/len(df)*100:.2f}%)\n")

if __name__ == "__main__":
    verify_cleaned()
