
import pandas as pd
import re

# File path
INPUT_FILE = r'd:\Skripsi\sentiment-analyst-ojol-review\data\gojek_5class_BALANCED_FIXED.csv'

# Keywords for checking contradictions
# Suspect: Label is positive/very_positive but contains negative words
neg_words = ['jelek', 'buruk', 'kecewa', 'lambat', 'susah', 'mahal', 'parah', 'gagal', 'tolak', 'hapus', 'uninstal']

# Suspect: Label is negative/very_negative but contains positive words
pos_words = ['bagus', 'baik', 'puas', 'cepat', 'lancar', 'mudah', 'suka', 'keren', 'mantap', 'terbaik']

def check_suspicious(row):
    text = str(row['text']).lower()
    label = row['sentiment']
    
    suspicious_reason = None
    
    # 1. Check Positive labels containing Negative words
    if label in ['positive', 'very_positive']:
        # Exclude negations (e.g., "tidak jelek") - simple check
        found_neg = [w for w in neg_words if w in text]
        if found_neg:
            # Check if it's negated "tidak jelek"
            real_issues = []
            for w in found_neg:
                if f"tidak {w}" not in text and f"gak {w}" not in text and f"bukan {w}" not in text:
                    real_issues.append(w)
            if real_issues:
                suspicious_reason = f"Label {label} but contains: {real_issues}"

    # 2. Check Negative labels containing Positive words
    elif label in ['negative', 'very_negative']:
        found_pos = [w for w in pos_words if w in text]
        if found_pos:
             # Check if it's negated "tidak bagus"
            real_issues = []
            for w in found_pos:
                if f"tidak {w}" not in text and f"gak {w}" not in text and f"bukan {w}" not in text:
                    real_issues.append(w)
            if real_issues:
                suspicious_reason = f"Label {label} but contains: {real_issues}"
                
    return suspicious_reason

def main():
    print(f"Auditing {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Apply check
    df['suspicious'] = df.apply(check_suspicious, axis=1)
    
    # Filter suspicious
    suspicious_df = df[df['suspicious'].notnull()]
    
    print(f"Total Rows: {len(df)}")
    print(f"Suspicious Rows Found: {len(suspicious_df)} ({len(suspicious_df)/len(df)*100:.2f}%)")
    
    if len(suspicious_df) > 0:
        print("\n=== SAMPLE CONTRADICTIONS (Top 20) ===")
        # Ambil sampel acak jika banyak, atau semua jika sedikit
        sample = suspicious_df.head(20)
        for idx, row in sample.iterrows():
            print(f"Text: {row['text']}")
            print(f"Current Label: {row['sentiment']}")
            print(f"Issue: {row['suspicious']}")
            print("-" * 50)
            
        print("\n\n=== SUMMARY OF ISSUES ===")
        print(suspicious_df['suspicious'].value_counts().head(10))
    else:
        print("Data looks very clean! No obvious keyword contradictions found.")

if __name__ == "__main__":
    main()
