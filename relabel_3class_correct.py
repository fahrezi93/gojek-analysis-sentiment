import pandas as pd
import re

class SentimentFixer:
    def __init__(self):
        # POSITIVE WORDS
        self.POS_WORDS = {
            'bagus', 'baik', 'oke', 'ok', 'mantap', 'mantab', 'keren', 'suka', 'puas', 'cinta', 
            'love', 'best', 'terbaik', 'juara', 'top', 'tahu', 'paham', 'mengerti',
            'cepat', 'lancar', 'wuzz', 'ngebut', 'sigap', 'gesit', 'sat set', 'tepat waktu', 'ontime',
            'ramah', 'sopan', 'senyum', 'baik hati', 'jujur', 'aman', 'nyaman', 'tenang', 'selamat',
            'murah', 'terjangkau', 'hemat', 'irit', 'promo', 'diskon', 'gratis', 'voucher', 'cashback',
            'mudah', 'gampang', 'praktis', 'simpel', 'simple', 'user friendly', 'jelas', 'lengkap',
            'membantu', 'solutif', 'berkah', 'bermanfaat', 'berguna', 'jos', 'gacor', 'semangat',
            'rapi', 'bersih', 'wangi', 'higienis', 'terpercaya', 'canggih', 'hebat', 'stabil', 'profesional',
            'recommended', 'recomended', 'rekomen', 'luar biasa', 'sempurna', 'good', 'nice'
        }

        # NEGATIVE WORDS
        self.NEG_WORDS = {
            'jelek', 'buruk', 'parah', 'payah', 'kacau', 'rusak', 'hancur', 'ancur', 'bubuk', 'sampah', 'ampas',
            'kecewa', 'kesal', 'kesel', 'marah', 'emosi', 'benci', 'muak', 'jijik', 'sebel', 'dongkol',
            'lambat', 'lama', 'lemot', 'lelet', 'ngaret', 'telat', 'macet', 'letoy', 'lola',
            'mahal', 'boros', 'mahal banget', 'naik harga', 'biaya mahal', 'tarif tinggi',
            'susah', 'sulit', 'ribet', 'bingung', 'berbelit', 'muter', 'rumit', 'kompleks',
            'error', 'eror', 'bug', 'glitch', 'crash', 'hang', 'blank', 'force close', 'keluar sendiri',
            'gagal', 'batal', 'cancel', 'tolak', 'ditolak', 'hilang', 'curang', 'tipu', 'penipu', 'bohong',
            'kasar', 'judes', 'galak', 'marah', 'bentak', 'bau', 'kotor', 'jorok', 'dekil',
            'uninstal', 'hapus', 'copot', 'buang', 'tinggalkan', 'pindah', 'kapok', 'nyesel', 'menyesal', 'jera',
            'gak guna', 'ga guna', 'percuma', 'sia sia', 'siasia', 'useless',
            'anjing', 'babi', 'bangsat', 'setan', 'taik', 'tai', 'monjet', 'goblok', 'tolol', 'bego', 'idiot',
            'menyusahkan', 'merugikan', 'mengecewakan', 'gaje', 'gak jelas', 'aneh', 'curiga', 'takut',
            'spam', 'iklan', 'mengganggu', 'berisik'
        }

        # NEGATION WORDS
        self.NEGATIONS = {
            'tidak', 'tak', 'gak', 'ga', 'nggak', 'bukan', 'jangan', 'kurang', 'belum', 'no', 'don\'t', 'dont', 'anti'
        }

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        return text.lower().strip()

    def get_lexicon_score(self, text):
        text = self.clean_text(text)
        words = re.findall(r'\w+', text)
        
        score = 0
        i = 0
        while i < len(words):
            word = words[i]
            is_negated = False
            
            if i > 0 and words[i-1] in self.NEGATIONS:
                is_negated = True
            
            points = 0
            if word in self.POS_WORDS:
                points = 1
            elif word in self.NEG_WORDS:
                points = -1
                
            if is_negated:
                points *= -1.5
                
            score += points
            i += 1
            
        return score

    def fix_label_3class(self, row):
        text = row['text'] if 'text' in row else row['clean_text']
        current_sentiment = row['sentiment']
        
        lex_score = self.get_lexicon_score(text)
        
        # Map sentiment to numeric for easier logic
        sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1}
        current_value = sentiment_map.get(current_sentiment, 0)
        
        # If no sentiment words detected, trust original label
        if lex_score == 0:
            return current_sentiment
        
        # Strong mismatch correction
        # If current is negative but text is strongly positive
        if current_value == -1 and lex_score >= 2:
            return 'positive'
        
        # If current is positive but text is strongly negative
        if current_value == 1 and lex_score <= -2:
            return 'negative'
        
        # Neutral corrections
        if current_value == 0:
            if lex_score >= 2:
                return 'positive'
            elif lex_score <= -2:
                return 'negative'
        
        # If mild mismatch, shift one level
        if current_value == -1 and lex_score > 0:
            return 'neutral'
        if current_value == 1 and lex_score < 0:
            return 'neutral'
            
        return current_sentiment

def main():
    input_file = 'data/gojek_scraped_3class_20251206_130028_FINAL_READY.csv'
    output_file = 'data/gojek_scraped_3class_RELABELED.csv'
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Total rows: {len(df)}")
    
    # Check columns
    print(f"Columns: {df.columns.tolist()}")
    
    fixer = SentimentFixer()
    
    print("Applying relabeling...")
    df['new_sentiment'] = df.apply(fixer.fix_label_3class, axis=1)
    
    # Count changes
    changes = (df['sentiment'] != df['new_sentiment']).sum()
    print(f"\nRows changed: {changes} ({changes/len(df)*100:.2f}%)")
    
    # Show distribution
    print("\n--- Original Distribution ---")
    print(df['sentiment'].value_counts())
    
    print("\n--- New Distribution ---")
    print(df['new_sentiment'].value_counts())
    
    # Show samples
    changed_rows = df[df['sentiment'] != df['new_sentiment']]
    if len(changed_rows) > 0:
        print("\n--- Sample Changes ---")
        text_col = 'text' if 'text' in df.columns else 'clean_text'
        print(changed_rows[[text_col, 'sentiment', 'new_sentiment']].head(10).to_string())
    
    # Save
    df_final = df.copy()
    df_final['sentiment'] = df_final['new_sentiment']
    df_final = df_final.drop(columns=['new_sentiment'])
    
    df_final.to_csv(output_file, index=False)
    print(f"\n✅ Saved relabeled data to {output_file}")
    print(f"Final row count: {len(df_final)}")

if __name__ == "__main__":
    main()
