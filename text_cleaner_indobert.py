"""
Text Cleaner & Normalizer untuk IndoBERT
Membersihkan dan normalize text dengan sangat ketat untuk akurasi tinggi
"""

import re
import pandas as pd
from typing import Tuple, Dict

# ============================================
# INDONESIAN SLANG NORMALIZATION
# ============================================
SLANG_DICT = {
    # Common variations
    'gak': 'tidak', 'ga': 'tidak', 'g': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak',
    'gk': 'tidak', 'tdk': 'tidak', 'gx': 'tidak', 'ndak': 'tidak',
    
    # Positive words
    'mantap': 'bagus', 'mantul': 'bagus', 'mantab': 'bagus', 'keren': 'bagus',
    'kereen': 'bagus', 'okee': 'oke', 'oks': 'oke', 'okeh': 'oke',
    'top': 'bagus', 'jos': 'bagus', 'josss': 'bagus', 'juara': 'bagus',
    'recommended': 'direkomendasikan', 'rekomended': 'direkomendasikan',
    'reccomend': 'direkomendasikan', 'recomend': 'direkomendasikan',
    'bangett': 'banget', 'bgt': 'banget', 'bngtt': 'banget', 'bgtt': 'banget',
    'bngst': 'banget', 'puas': 'puas', 'puass': 'puas', 'puasss': 'puas',
    'seneng': 'senang', 'seneng banget': 'sangat senang', 'suka': 'suka',
    'sukaa': 'suka', 'nyaman': 'nyaman', 'enak': 'enak', 'best': 'terbaik',
    'the best': 'terbaik', 'terbest': 'terbaik',
    
    # Negative words  
    'jelek': 'buruk', 'jelek banget': 'sangat buruk', 'ancur': 'buruk',
    'parah': 'buruk', 'payah': 'buruk', 'zonk': 'buruk', 'ngaco': 'buruk',
    'kecewa': 'kecewa', 'kecewa banget': 'sangat kecewa', 'kesel': 'kesal',
    'marah': 'marah', 'bete': 'kesal', 'nyebelin': 'menjengkelkan',
    'nyebalin': 'menjengkelkan', 'nyesel': 'menyesal', 'kapok': 'jera',
    'lama': 'lambat', 'lelet': 'lambat', 'lambat': 'lambat',
    'gaje': 'buruk', 'anjing': 'buruk', 'anjir': 'buruk', 'sial': 'buruk',
    
    # Neutral words
    'biasa': 'biasa', 'aja': 'saja', 'aj': 'saja', 'doang': 'saja',
    'kok': 'kenapa', 'knp': 'kenapa', 'knapa': 'kenapa', 'gmn': 'bagaimana',
    'gimana': 'bagaimana', 'gmana': 'bagaimana', 'bgmn': 'bagaimana',
    
    # Service related
    'driver': 'pengemudi', 'cancel': 'batal', 'order': 'pesan',
    'orderan': 'pesanan', 'promo': 'promosi', 'voucher': 'voucher',
    'gopay': 'gopay', 'gofood': 'gofood', 'goride': 'goride',
    'gosend': 'gosend', 'gocar': 'gocar', 'gojek': 'gojek',
    
    # Common typos
    'tapi': 'tapi', 'tp': 'tapi', 'dgn': 'dengan', 'dg': 'dengan',
    'sm': 'sama', 'sma': 'sama', 'dh': 'sudah', 'udh': 'sudah',
    'udah': 'sudah', 'sdh': 'sudah', 'blm': 'belum', 'blom': 'belum',
    'lg': 'lagi', 'lgi': 'lagi', 'lgu': 'lagi', 'jd': 'jadi',
    'jdi': 'jadi', 'krn': 'karena', 'krna': 'karena', 'karna': 'karena',
    'mgkn': 'mungkin', 'mungkn': 'mungkin', 'hrs': 'harus',
    'pdhl': 'padahal', 'pdahal': 'padahal', 'yg': 'yang', 'sy': 'saya',
    'sya': 'saya', 'aq': 'saya', 'ak': 'saya', 'gw': 'saya',
    'gue': 'saya', 'w': 'saya', 'ane': 'saya',
    
    # Time related
    'skrg': 'sekarang', 'skrng': 'sekarang', 'skg': 'sekarang',
    'kmrn': 'kemarin', 'kmren': 'kemarin', 'bsk': 'besok',
    'besok': 'besok', 'td': 'tadi', 'tdi': 'tadi',
    
    # Numbers
    'x': 'kali', 'brp': 'berapa', 'brapa': 'berapa',
    
    # Confirmation
    'ga': 'tidak', 'gak': 'tidak', 'engga': 'tidak', 'enggak': 'tidak',
    'nope': 'tidak', 'ga': 'tidak', 'yup': 'ya', 'yoi': 'ya',
    'iya': 'ya', 'iye': 'ya', 'yah': 'ya',
}

# ============================================
# SENTIMENT KEYWORDS (untuk koreksi label)
# ============================================
POSITIVE_KEYWORDS = {
    'bagus', 'baik', 'memuaskan', 'puas', 'senang', 'suka', 'cepat', 'ramah',
    'nyaman', 'enak', 'murah', 'oke', 'mantap', 'recommended', 'terbaik',
    'sempurna', 'excellent', 'hebat', 'luar biasa', 'profesional', 'sopan',
    'tepat waktu', 'lancar', 'mudah', 'membantu', 'praktis', 'efisien',
    'terima kasih', 'thanks', 'makasih', 'sukses', 'keren', 'top', 'jos',
}

NEGATIVE_KEYWORDS = {
    'buruk', 'jelek', 'kecewa', 'kesal', 'marah', 'lambat', 'lama', 'cancel',
    'tidak profesional', 'kasar', 'tidak ramah', 'error', 'bermasalah', 'rusak',
    'parah', 'payah', 'zonk', 'mengecewakan', 'menyebalkan', 'ngaco', 'gaje',
    'tidak puas', 'kapok', 'jera', 'menyesal', 'susah', 'ribet', 'bingung',
    'tidak jelas', 'tidak bisa', 'tidak mau', 'ditolak', 'dibatalkan', 'sial',
    'anjing', 'tai', 'sampah', 'bodoh', 'gila', 'tolol', 'brengsek',
}

NEUTRAL_KEYWORDS = {
    'biasa', 'lumayan', 'cukup', 'standar', 'kadang', 'terkadang', 'mungkin',
    'sepertinya', 'seharusnya', 'harusnya', 'kenapa', 'bagaimana', 'apa',
    'tolong', 'mohon', 'bisa', 'tidak bisa', 'gimana', 'cara',
}

# ============================================
# TEXT CLEANING FUNCTIONS
# ============================================
def normalize_slang(text: str) -> str:
    """Normalize Indonesian slang to standard words"""
    words = text.split()
    normalized = []
    
    for word in words:
        word_lower = word.lower()
        # Check if word is in slang dict
        if word_lower in SLANG_DICT:
            normalized.append(SLANG_DICT[word_lower])
        else:
            normalized.append(word)
    
    return ' '.join(normalized)

def clean_text_indobert(text: str) -> str:
    """
    Clean text specifically for IndoBERT
    - Remove noise but keep meaningful content
    - Normalize slang
    - Fix spacing
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = str(text).strip()
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # 3. Remove email
    text = re.sub(r'\S+@\S+', '', text)
    
    # 4. Remove phone numbers (preserve context)
    text = re.sub(r'\b\d{10,}\b', '', text)
    
    # 5. Remove mentions/hashtags (but keep text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    
    # 6. Normalize repeated characters (looove -> love, hahahaha -> haha)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # 7. Remove excessive punctuation but keep one
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    
    # 8. Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # 9. Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # 10. Remove special characters (keep alphanumeric, spaces, basic punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
    
    # 11. Normalize slang
    text = normalize_slang(text)
    
    # 12. Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # 13. Strip
    text = text.strip()
    
    # 14. Remove if too short after cleaning
    if len(text) < 10:
        return ""
    
    return text

def calculate_sentiment_score(text: str) -> Dict[str, float]:
    """
    Calculate sentiment score based on keywords
    Returns: {'positive': score, 'negative': score, 'neutral': score}
    """
    text_lower = text.lower()
    words = set(text_lower.split())
    
    pos_count = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in text_lower)
    neg_count = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in text_lower)
    neu_count = sum(1 for keyword in NEUTRAL_KEYWORDS if keyword in text_lower)
    
    # Check for negation
    negation_words = ['tidak', 'bukan', 'jangan', 'belum', 'tanpa']
    has_negation = any(neg in words for neg in negation_words)
    
    # If negation exists, flip positive/negative
    if has_negation:
        # "tidak bagus" -> negative
        if pos_count > 0:
            neg_count += pos_count
            pos_count = 0
    
    total = pos_count + neg_count + neu_count
    if total == 0:
        return {'positive': 0, 'negative': 0, 'neutral': 1}
    
    return {
        'positive': pos_count / total,
        'negative': neg_count / total,
        'neutral': neu_count / total
    }

def correct_sentiment_label(text: str, rating: int, current_sentiment: str) -> str:
    """
    Correct sentiment label based on text content and rating
    Uses both keyword analysis and rating
    """
    scores = calculate_sentiment_score(text)
    
    # Strong indicators
    if scores['positive'] > 0.6 and rating >= 4:
        return 'positive'
    if scores['negative'] > 0.6 and rating <= 2:
        return 'negative'
    
    # Rating 3 is tricky - analyze content carefully
    if rating == 3:
        # If clearly positive content
        if scores['positive'] > 0.5 and scores['negative'] < 0.2:
            return 'positive'
        # If clearly negative content
        elif scores['negative'] > 0.5 and scores['positive'] < 0.2:
            return 'negative'
        # Otherwise neutral
        else:
            return 'neutral'
    
    # For rating 1-2, should be negative
    if rating <= 2:
        # Unless has very positive words (sarcasm or mistake)
        if scores['positive'] > 0.7:
            return 'neutral'  # Mixed signal
        return 'negative'
    
    # For rating 4-5, should be positive
    if rating >= 4:
        # Unless has very negative words (constructive criticism)
        if scores['negative'] > 0.7:
            return 'neutral'  # Mixed signal
        return 'positive'
    
    # Default to current
    return current_sentiment

def validate_text_quality(text: str) -> Tuple[bool, str]:
    """
    Validate if text is good quality for training
    Returns: (is_valid, reason)
    """
    if not text or len(text.strip()) < 10:
        return False, "too_short"
    
    words = text.split()
    if len(words) < 3:
        return False, "too_few_words"
    
    if len(words) > 300:
        return False, "too_long"
    
    # Check if text is meaningful (has some alphabetic characters)
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars < len(text) * 0.5:
        return False, "not_enough_text"
    
    # Check for spam patterns
    spam_patterns = [
        r'(klik|visit|follow|subscribe).*link',
        r'(wa|whatsapp|hubungi).*\d{4,}',
        r'(diskon|voucher|promo).*code',
    ]
    
    for pattern in spam_patterns:
        if re.search(pattern, text.lower()):
            return False, "spam_pattern"
    
    # Check for repeated single words
    if len(set(words)) < len(words) * 0.3 and len(words) > 5:
        return False, "too_repetitive"
    
    return True, "valid"

# ============================================
# BATCH PROCESSING
# ============================================
def process_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process entire dataframe
    Returns: (clean_df, dirty_df)
    """
    print("🧹 Processing data...")
    
    # Clean text
    df['content_clean'] = df['content'].apply(clean_text_indobert)
    
    # Validate quality
    validation_results = df['content_clean'].apply(validate_text_quality)
    df['is_valid'] = validation_results.apply(lambda x: x[0])
    df['invalid_reason'] = validation_results.apply(lambda x: x[1])
    
    # Separate clean and dirty
    df_clean = df[df['is_valid']].copy()
    df_dirty = df[~df['is_valid']].copy()
    
    # For clean data, correct sentiment labels
    if not df_clean.empty:
        print("🔧 Correcting sentiment labels...")
        df_clean['sentiment_original'] = df_clean['sentiment']
        df_clean['sentiment_corrected'] = df_clean.apply(
            lambda row: correct_sentiment_label(
                row['content_clean'], 
                row['rating'], 
                row['sentiment']
            ), 
            axis=1
        )
        
        # Calculate sentiment scores for reference
        sentiment_scores = df_clean['content_clean'].apply(calculate_sentiment_score)
        df_clean['pos_score'] = sentiment_scores.apply(lambda x: x['positive'])
        df_clean['neg_score'] = sentiment_scores.apply(lambda x: x['negative'])
        df_clean['neu_score'] = sentiment_scores.apply(lambda x: x['neutral'])
    
    print(f"✅ Clean: {len(df_clean)}, ❌ Dirty: {len(df_dirty)}")
    
    return df_clean, df_dirty

if __name__ == "__main__":
    # Test
    test_texts = [
        "Aplikasi bagus banget! Driver ramah dan cepat sampai.",
        "Gak bisa dipake, eror terus anjing parah bgt!!",
        "Biasa aja sih, kadang bagus kadang lama",
        "mantap bro, recommended deh pokoknya TOP",
        "sy kecewa bgt sm pelayanannya, cancel trs",
    ]
    
    print("Testing text cleaner...\n")
    for text in test_texts:
        cleaned = clean_text_indobert(text)
        scores = calculate_sentiment_score(cleaned)
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}")
        print(f"Scores:   {scores}")
        print()
