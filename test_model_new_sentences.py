"""
Script untuk menguji model IndoBERT dengan kalimat-kalimat baru (di luar dataset)
"""

import sys
import os

# Add web directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web'))

from predictor import ModelHandler
from preprocessing import TextCleaner

# Test cases: (kalimat, expected_sentiment)
TEST_CASES = [
    # POSITIF (Expected: Positif)
    ("Drivernya sopan banget, sampai tujuan dengan selamat dan tepat waktu", "Positif"),
    ("Gojek emang paling top deh, promo nya banyak dan murah meriah", "Positif"),
    ("Saya sangat terbantu dengan layanan goride, driver ramah dan motor bersih", "Positif"),
    ("Mantul sih ini aplikasi, gercep banget pesanan langsung diantar", "Positif"),
    ("Terima kasih gojek sudah menemani perjalanan saya setiap hari, sangat membantu", "Positif"),
    
    # NEGATIF (Expected: Negatif)
    ("Aplikasi nya error mulu, mau pesen ojek susah banget", "Negatif"),
    ("Driver nya kasar dan ugal-ugalan di jalan, kapok pake gojek", "Negatif"),
    ("Sudah transfer gopay tapi saldo tidak masuk, pelayanan customer service mengecewakan", "Negatif"),
    ("Makin hari makin mahal, promo udah jarang, mending naik angkot", "Negatif"),
    ("Pesanan gofood saya dibatalin sama driver tanpa alasan, rugi waktu dan tenaga", "Negatif"),
    
    # NETRAL (Expected: Netral)
    ("Yah lumayan lah buat transportasi sehari-hari", "Netral"),
    ("Mungkin bisa ditambahkan fitur chat dengan driver sebelum order", "Netral"),
    ("Saya biasanya pakai gojek kalau mau ke kantor pagi hari", "Netral"),
    ("Kenapa ya promo gopay food tidak bisa dipakai di resto tertentu", "Netral"),
    ("Ada plus minusnya sih, kadang dapat driver bagus kadang enggak", "Netral"),
    
    # TRICKY CASES
    ("Wah bagus banget nih aplikasi, cuma butuh 1 jam buat dapat driver", "Negatif"),  # Sarkasme
    ("Aplikasinya udah bagus, cuma kalau bisa tambahin fitur lacak driver lebih akurat lagi", "Positif"),
    ("Hmm sebenarnya bisa lebih baik lagi sih pelayanannya", "Negatif"),
    ("Gojek tidak pernah mengecewakan saya, selalu on time", "Positif"),  # Double negation
    ("Dulu sih enak, sekarang makin banyak driver yang cancel orderan", "Negatif"),
    
    # TAMBAHAN - Kasus kontras
    ("awalnya emang bagus, tapi makin kesini makin lemot aja", "Negatif"),
]

def main():
    print("=" * 80)
    print("🧪 TEST MODEL INDOBERT 3-KELAS DENGAN DATA BARU")
    print("=" * 80)
    
    # Initialize
    print("\n⏳ Loading model...")
    handler = ModelHandler(model_type="3class")
    cleaner = TextCleaner()
    
    if not handler.load_model():
        print("❌ Gagal load model!")
        return
    
    print("✅ Model loaded successfully!\n")
    
    # Run tests
    results = []
    correct = 0
    total = len(TEST_CASES)
    
    print("=" * 80)
    print("📊 HASIL TESTING")
    print("=" * 80)
    
    for i, (text, expected) in enumerate(TEST_CASES, 1):
        # Preprocess
        cleaned = cleaner.clean_text(text)
        
        # Predict
        result = handler.predict_sentiment(cleaned)
        predicted = result['simplified_label']
        confidence = result['confidence_percentage']
        
        # Check if correct
        is_correct = predicted.lower() == expected.lower()
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        results.append({
            'text': text,
            'expected': expected,
            'predicted': predicted,
            'confidence': confidence,
            'correct': is_correct
        })
        
        # Print result
        print(f"\n{status} Test {i}/{total}")
        print(f"   Teks: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
        print(f"   Expected: {expected}")
        print(f"   Predicted: {predicted} ({confidence:.1f}%)")
    
    # Summary
    accuracy = correct / total * 100
    
    print("\n" + "=" * 80)
    print("📈 RINGKASAN HASIL")
    print("=" * 80)
    print(f"\n   Total test cases: {total}")
    print(f"   Benar: {correct}")
    print(f"   Salah: {total - correct}")
    print(f"   Akurasi: {accuracy:.1f}%")
    
    # Show wrong predictions
    wrong = [r for r in results if not r['correct']]
    if wrong:
        print(f"\n❌ PREDIKSI YANG SALAH ({len(wrong)}):")
        for r in wrong:
            print(f"\n   Teks: \"{r['text'][:70]}...\"")
            print(f"   Expected: {r['expected']} | Predicted: {r['predicted']} ({r['confidence']:.1f}%)")
    
    print("\n" + "=" * 80)
    print("🎯 ANALISIS:")
    print("=" * 80)
    
    # Categorize results
    pos_correct = sum(1 for r in results if r['expected'] == 'Positif' and r['correct'])
    pos_total = sum(1 for r in results if r['expected'] == 'Positif')
    
    neg_correct = sum(1 for r in results if r['expected'] == 'Negatif' and r['correct'])
    neg_total = sum(1 for r in results if r['expected'] == 'Negatif')
    
    neu_correct = sum(1 for r in results if r['expected'] == 'Netral' and r['correct'])
    neu_total = sum(1 for r in results if r['expected'] == 'Netral')
    
    print(f"\n   Positif: {pos_correct}/{pos_total} benar ({pos_correct/pos_total*100:.0f}%)")
    print(f"   Negatif: {neg_correct}/{neg_total} benar ({neg_correct/neg_total*100:.0f}%)")
    print(f"   Netral:  {neu_correct}/{neu_total} benar ({neu_correct/neu_total*100:.0f}%)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
