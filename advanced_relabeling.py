
import pandas as pd
import os
import re

class SentimentFixer:
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

        # 3. NEGATION WORDS (Flip polarity)
        self.NEGATIONS = {
            'tidak', 'tak', 'gak', 'ga', 'nggak', 'bukan', 'jangan', 'kurang', 'belum', 'no', 'don\'t', 'dont', 'anti'
        }
        
        # 4. BOOSTER WORDS (Intensifiers - optional, for simple scoring we treated roughly)
        self.BOOSTERS = {
            'sangat', 'banget', 'sekali', 'super', 'terlalu', 'paling', 'benar', 'bener', 'parah'
        }

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        return text.lower().strip()

    def get_lexicon_score(self, text):
        text = self.clean_text(text)
        words = re.findall(r'\w+', text)
        
        score = 0
        
        # Iterate words with window to check negation
        i = 0
        while i < len(words):
            word = words[i]
            is_negated = False
            
            # Check negation in previous word
            if i > 0 and words[i-1] in self.NEGATIONS:
                is_negated = True
            
            # Additional check: "kurang ajar" is negative, but "kurang" is negation. 
            # Simple heuristic: if word found in pos/neg dict, apply score.
            
            points = 0
            if word in self.POS_WORDS:
                points = 1
            elif word in self.NEG_WORDS:
                points = -1
                
            if is_negated:
                points *= -1.5 # Flip and slightly boost impact of negated sentiment
                
            score += points
            i += 1
            
        return score

    def fix_label_3class(self, row):
        text = row['text'] if 'text' in row else row['content']
        rating = row['rating']
        
        # Calculate content-based score
        lex_score = self.get_lexicon_score(text)
        
        # Decision Logic:
        # 1. Pure Rating Backup: If text has NO sentiment words, trust rating.
        if lex_score == 0:
            if rating >= 4: return 'positive'
            if rating <= 2: return 'negative'
            return 'neutral'
            
        # 2. Strong Text Signal Overrides Rating
        # e.g. User gives 1 star but says "Aplikasi sangat bagus banget" (Score > 2) -> Positive
        # e.g. User gives 5 star but says "Jelek, mahal, lemot" (Score < -2) -> Negative
        
        if lex_score >= 2:
            return 'positive'
        elif lex_score <= -2:
            return 'negative'
            
        # 3. Weak Text Signal vs Rating
        # If score is mild (1 or -1)
        if lex_score > 0:
            if rating <= 2: return 'positive' # "Lumayan" (score 1) with 1 star -> Likely Positive text, rating mismatch
            return 'positive'
        
        if lex_score < 0:
            if rating >= 4: return 'negative' # "Agak lambat" (score -1) with 5 star -> Negative text detail
            return 'negative'
            
        return 'neutral' # Should be covered by lex_score==0 logic, but fallback.

    def fix_label_5class(self, row):
        # For 5 class, we map scores to Very Pos/Pos/Neu/Neg/Very Neg
        text = row['text'] if 'text' in row else row['content']
        rating = row['rating']
        
        lex_score = self.get_lexicon_score(text)
        
        if lex_score == 0:
            if rating == 5: return 'very_positive'
            if rating == 4: return 'positive'
            if rating == 3: return 'neutral'
            if rating == 2: return 'negative'
            return 'very_negative' # rating 1
            
        # Score Mapping
        if lex_score >= 3: return 'very_positive'
        if 1 <= lex_score < 3: return 'positive'
        if lex_score == 0: return 'neutral' # Unlikely to reach here
        if -2 <= lex_score < 0: return 'negative'
        if lex_score < -2: return 'very_negative'
        
        return 'neutral' # Fallback

def process_files():
    base_dir = r"d:\Skripsi\sentiment-analyst-ojol-review\data"
    files = {
        '3class': "gojek_scraped_3class_20251206_130028.csv",
        '5class': "gojek_scraped_5class_20251206_130028.csv"
    }
    
    fixer = SentimentFixer()
    
    for key, filename in files.items():
        file_path = os.path.join(base_dir, filename)
        if not os.path.exists(file_path):
            print(f"Skipping {filename} (Not found)")
            continue
            
        print(f"\nProcessing {filename}...")
        df = pd.read_csv(file_path)
        
        original_dist = df['sentiment'].value_counts()
        print("Original Distribution:")
        print(original_dist)
        
        # Apply Fix
        if key == '3class':
            df['sentiment_fixed'] = df.apply(fixer.fix_label_3class, axis=1)
        else:
            df['sentiment_fixed'] = df.apply(fixer.fix_label_5class, axis=1)
            
        # Check changes
        changes = (df['sentiment'] != df['sentiment_fixed']).sum()
        print(f"Total Rows re-labeled: {changes} ({changes/len(df)*100:.2f}%)")
        
        new_dist = df['sentiment_fixed'].value_counts()
        print("New Distribution:")
        print(new_dist)
        
        # Save Fixed File
        # We replace the 'sentiment' column with the fixed one, but keep 'sentiment_original' for reference? 
        # User wants it fixed. Let's overwrite 'sentiment'.
        df['sentiment_original'] = df['sentiment']
        df['sentiment'] = df['sentiment_fixed']
        df.drop(columns=['sentiment_fixed'], inplace=True)
        
        new_filename = filename.replace('.csv', '_cleaned.csv')
        output_path = os.path.join(base_dir, new_filename)
        df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to: {new_filename}")

if __name__ == "__main__":
    process_files()
