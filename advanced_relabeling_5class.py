import pandas as pd
import re
import numpy as np

class SentimentFixer5Class:
    def __init__(self):
        # 1. POSITIVE WORDS
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

        # 2. NEGATIVE WORDS
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
            'gak guna', 'ga guna', 'percuma', 'sia sia', 'siasia', 'useless', 'sampah',
            'anjing', 'babi', 'bangsat', 'setan', 'taik', 'tai', 'monjet', 'goblok', 'tolol', 'bego', 'idiot',
            'menyusahkan', 'merugikan', 'mengecewakan', 'gaje', 'gak jelas', 'aneh', 'curiga', 'takut',
            'spam', 'iklan', 'mengganggu', 'berisik'
        }

        # 3. NEGATION WORDS
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

    def fix_label_5class(self, row):
        text = row['text'] if 'text' in row else row['clean_text']
        current_label = row['sentiment']
        
        # Map label to score for comparison
        label_map = {
            'very_negative': 1,
            'negative': 2,
            'neutral': 3,
            'positive': 4,
            'very_positive': 5
        }
        reverse_map = {v: k for k, v in label_map.items()}
        
        current_score = label_map.get(current_label, 3)
        lex_score = self.get_lexicon_score(text)
        
        new_score = current_score

        # LOGIC: Only fix GROSS errors
        # Thresholds: Strong Positive > 1.5, Strong Negative < -1.5
        
        if current_score <= 2: # Currently Negative
            if lex_score >= 2: # Text is Strongly Positive
                new_score = 5 # Flip to Very Positive
            elif lex_score >= 1:
                new_score = 4 # Flip to Positive
                
        elif current_score >= 4: # Currently Positive
            if lex_score <= -2: # Text is Strongly Negative
                new_score = 1 # Flip to Very Negative
            elif lex_score <= -1:
                new_score = 2 # Flip to Negative
                
        elif current_score == 3: # Currently Neutral
            if lex_score >= 2:
                new_score = 5
            elif lex_score >= 1:
                new_score = 4
            elif lex_score <= -2:
                new_score = 1
            elif lex_score <= -1:
                new_score = 2

        return reverse_map[new_score]

def main():
    input_file = 'data/gojek_scraped_5class_20251206_130028_FINAL_READY.csv'
    output_file = 'data/gojek_scraped_5class_RELABELED.csv'
    
    print(f"Reading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("File not found!")
        return

    fixer = SentimentFixer5Class()
    
    print("Relabeling...")
    df['new_sentiment'] = df.apply(fixer.fix_label_5class, axis=1)
    
    # Check changes
    changes = df[df['sentiment'] != df['new_sentiment']]
    print(f"Total rows: {len(df)}")
    print(f"Rows changed: {len(changes)} ({len(changes)/len(df)*100:.2f}%)")
    
    print("\n--- Sample Changes ---")
    print(changes[['text', 'sentiment', 'new_sentiment']].head(10).to_string())
    
    # Save
    # We replace the old sentiment with the new one for the final file
    df_final = df.copy()
    df_final['sentiment'] = df_final['new_sentiment']
    df_final = df_final.drop(columns=['new_sentiment'])
    
    df_final.to_csv(output_file, index=False)
    print(f"\nSaved relabeled data to {output_file}")

if __name__ == "__main__":
    main()
