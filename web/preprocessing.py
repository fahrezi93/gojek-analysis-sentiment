"""
TextNormalizer - Modul untuk membersihkan dan normalisasi teks ulasan Gojek
"""

import re
import string
from typing import List, Union


class TextNormalizer:
    """
    Kelas untuk menangani pembersihan dan normalisasi teks ulasan.
    Termasuk penghapusan noise, normalisasi slang, dan persiapan tokenisasi.
    """
    
    def __init__(self):
        """Inisialisasi TextNormalizer dengan dictionary slang dan pattern cleaning"""
        self.slang_dict = {
            # Slang positif
            'mantap': 'mantap',
            'mantul': 'mantap',
            'mantab': 'mantap',
            'okee': 'oke',
            'okeee': 'oke',
            'okeh': 'oke',
            'rekomend': 'rekomendasi',
            'rekomen': 'rekomendasi',
            'makasi': 'terima kasih',
            'makasih': 'terima kasih',
            'thx': 'terima kasih',
            'thanks': 'terima kasih',
            'tengkyu': 'terima kasih',
            'top': 'bagus',
            'jos': 'bagus',
            'oke': 'baik',
            
            # Slang negatif
            'gak': 'tidak',
            'ga': 'tidak',
            'ngga': 'tidak',
            'nggak': 'tidak',
            'gk': 'tidak',
            'tdk': 'tidak',
            'kecewa': 'kecewa',
            'jelek': 'buruk',
            'parah': 'buruk',
            'mengecewakan': 'kecewa',
            'zonk': 'buruk',
            'payah': 'buruk',
            'lelet': 'lambat',
            'lemot': 'lambat',
            
            # Slang umum
            'yg': 'yang',
            'dgn': 'dengan',
            'dg': 'dengan',
            'utk': 'untuk',
            'tdk': 'tidak',
            'tq': 'terima kasih',
            'sm': 'sama',
            'dr': 'dari',
            'ke': 'ke',
            'gt': 'gitu',
            'gitu': 'begitu',
            'bgt': 'banget',
            'bener': 'benar',
            'bner': 'benar',
            'aja': 'saja',
            'aj': 'saja',
            'udh': 'sudah',
            'udah': 'sudah',
            'blm': 'belum',
            'blom': 'belum',
            'telat': 'terlambat',
            'nyampe': 'sampai',
            'nyampai': 'sampai',
            'pesen': 'pesan',
            'nih': 'ini',
            'pdhal': 'padahal',
            'lgi': 'lagi',
            'males': 'malas',
            'sdh': 'sudah',
            'krn': 'karena',
            'tp': 'tapi',
            'apps': 'aplikasi',
            'emg': 'memang',
            'emang': 'memang',
            'bkn': 'bukan',
            'lg': 'lagi',
            'klo': 'kalau',
            'kalo': 'kalau',
            'hrs': 'harus',
            'bnr': 'benar',
            'skrg': 'sekarang',
            'skrng': 'sekarang',
            'org': 'orang',
            'jgn': 'jangan',
            'jng': 'jangan',
            'msh': 'masih',
            'masi': 'masih',
            'knp': 'kenapa',
            'knapa': 'kenapa',
            'gmn': 'gimana',
            'gimana': 'bagaimana',
            'biar': 'agar',
            'bs': 'bisa',
            'dpt': 'dapat',
            
            # Kata-kata terkait Gojek
            'gojek': 'gojek',
            'gocar': 'gocar',
            'goride': 'goride',
            'gofood': 'gofood',
            'gosend': 'gosend',
            'driver': 'pengemudi',
            'drivernya': 'pengemudinya',
            'aplikasi': 'aplikasi',
            'app': 'aplikasi',
        }
        
        # Compile regex patterns untuk efisiensi
        self.url_pattern = re.compile(r'http\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        self.number_pattern = re.compile(r'\d+')
        self.punctuation_pattern = re.compile(f'[{re.escape(string.punctuation)}]')
        self.whitespace_pattern = re.compile(r'\s+')
    
    def remove_noise(self, text: str) -> str:
        """
        Menghapus noise dari teks (URL, mention, hashtag, emoji, dll)
        
        Args:
            text: Teks input yang akan dibersihkan
            
        Returns:
            Teks yang sudah dibersihkan dari noise
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove mentions
        text = self.mention_pattern.sub('', text)
        
        # Remove hashtags
        text = self.hashtag_pattern.sub('', text)
        
        # Remove emojis
        text = self.emoji_pattern.sub('', text)
        
        # Remove numbers (optional, tergantung kebutuhan)
        # text = self.number_pattern.sub('', text)
        
        # Remove punctuation
        text = self.punctuation_pattern.sub(' ', text)
        
        # Remove extra whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        return text.strip()
    
    def normalize_slang(self, text: str) -> str:
        """
        Normalisasi kata-kata slang ke bentuk baku
        
        Args:
            text: Teks yang akan dinormalisasi
            
        Returns:
            Teks dengan kata slang yang sudah dinormalisasi
        """
        if not isinstance(text, str):
            return ""
        
        words = text.split()
        normalized_words = [self.slang_dict.get(word, word) for word in words]
        return ' '.join(normalized_words)
    
    def clean_text(self, text: str) -> str:
        """
        Pipeline lengkap untuk membersihkan teks
        
        Args:
            text: Teks input yang akan dibersihkan
            
        Returns:
            Teks yang sudah bersih dan siap untuk tokenisasi
        """
        if not isinstance(text, str):
            return ""
        
        # Step 1: Remove noise
        text = self.remove_noise(text)
        
        # Step 2: Normalize slang
        text = self.normalize_slang(text)
        
        # Step 3: Final cleanup
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Membersihkan batch teks sekaligus
        
        Args:
            texts: List teks yang akan dibersihkan
            
        Returns:
            List teks yang sudah dibersihkan
        """
        return [self.clean_text(text) for text in texts]
    
    def get_slang_dict(self) -> dict:
        """
        Mendapatkan dictionary slang yang digunakan
        
        Returns:
            Dictionary mapping slang ke kata baku
        """
        return self.slang_dict.copy()
    
    def add_slang_words(self, slang_dict: dict):
        """
        Menambahkan kata slang baru ke dictionary
        
        Args:
            slang_dict: Dictionary dengan format {slang: kata_baku}
        """
        self.slang_dict.update(slang_dict)


if __name__ == "__main__":
    # Test TextNormalizer
    normalizer = TextNormalizer()
    
    test_texts = [
        "Mantul bgt drivernya ramah 😊👍 #gojek",
        "gak rekomen deh, lelet bgt aplikasinya 😡",
        "Makasih ya driver gocar nya baik bgt, top markotop!",
        "Kecewa bgt sm pelayanannya, gk profesional",
        "@gojekIndonesia tolong perbaiki app nya yg sering error"
    ]
    
    print("=" * 60)
    print("Testing TextNormalizer")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        cleaned = normalizer.clean_text(text)
        print(f"\n{i}. Original: {text}")
        print(f"   Cleaned:  {cleaned}")
