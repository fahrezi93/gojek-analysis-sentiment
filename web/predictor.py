"""
ModelPredictor - Modul untuk melakukan inferensi dengan model IndoBERT
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple, Union
import os


class ModelPredictor:
    """
    Kelas untuk melakukan prediksi sentimen menggunakan model IndoBERT.
    Mendukung skema 3-kelas (Positif, Netral, Negatif) dan 5-kelas (Rating 1-5).
    """
    
    def __init__(self, model_path: str = None, model_type: str = "3class"):
        """
        Inisialisasi ModelPredictor
        
        Args:
            model_path: Path ke direktori model yang sudah dilatih
            model_type: Tipe model - "3class" atau "5class"
        """
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Label mapping
        self.label_mapping_3class = {
            0: "Negatif",
            1: "Netral",
            2: "Positif"
        }
        
        self.label_mapping_5class = {
            0: "Rating 1 (Sangat Negatif)",
            1: "Rating 2 (Negatif)",
            2: "Rating 3 (Netral)",
            3: "Rating 4 (Positif)",
            4: "Rating 5 (Sangat Positif)"
        }
        
        # Simplified label for display
        self.simplified_labels_5class = {
            0: "Rating 1",
            1: "Rating 2",
            2: "Rating 3",
            3: "Rating 4",
            4: "Rating 5"
        }
        
        # Set default model path jika tidak diberikan
        if model_path is None:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if model_type == "3class":
                # Menggunakan model Skenario 2 (8 epoch, LR 2e-5) - Akurasi 97.62%
                model_path = os.path.join(base_path, "saved_model_indobert_3class_scenario2")
            else:
                # Menggunakan model Skenario 5 (8 epoch, LR 2e-5) - Akurasi 98.69%
                model_path = os.path.join(base_path, "saved_model_indobert_5class_scenario5")
        
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """
        Memuat model dan tokenizer dari disk
        """
        try:
            print(f"Loading model from: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "indobenchmark/indobert-base-p1"
            )
            
            # Load model
            num_labels = 3 if self.model_type == "3class" else 5
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=num_labels
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Model loaded successfully on {self.device}")
            print(f"✓ Model type: {self.model_type}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            return False
    
    def predict_single(self, text: str) -> Dict:
        """
        Melakukan prediksi untuk satu teks
        
        Args:
            text: Teks ulasan yang akan diprediksi
            
        Returns:
            Dictionary berisi label, confidence score, dan probabilitas semua kelas
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model belum dimuat! Panggil load_model() terlebih dahulu.")
        
        # Tokenisasi
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Prediksi
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
        # Get predicted class and confidence
        confidence, predicted_class = torch.max(probabilities, dim=-1)
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        
        # Get all probabilities
        all_probs = probabilities[0].cpu().numpy()
        
        # Get label mapping
        label_map = (self.label_mapping_3class if self.model_type == "3class" 
                     else self.label_mapping_5class)
        simplified_map = (self.label_mapping_3class if self.model_type == "3class"
                         else self.simplified_labels_5class)
        
        return {
            "text": text,
            "predicted_class": predicted_class,
            "predicted_label": label_map[predicted_class],
            "simplified_label": simplified_map[predicted_class],
            "confidence": confidence,
            "confidence_percentage": confidence * 100,
            "all_probabilities": {
                simplified_map[i]: float(prob) * 100 
                for i, prob in enumerate(all_probs)
            }
        }
    
    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        """
        Melakukan prediksi untuk batch teks
        
        Args:
            texts: List teks yang akan diprediksi
            batch_size: Ukuran batch untuk processing
            
        Returns:
            List dictionary hasil prediksi
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model belum dimuat! Panggil load_model() terlebih dahulu.")
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenisasi batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Prediksi
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Process results
            confidences, predicted_classes = torch.max(probabilities, dim=-1)
            
            # Get label mapping
            label_map = (self.label_mapping_3class if self.model_type == "3class" 
                        else self.label_mapping_5class)
            simplified_map = (self.label_mapping_3class if self.model_type == "3class"
                             else self.simplified_labels_5class)
            
            # Create results for each item in batch
            for j, text in enumerate(batch_texts):
                predicted_class = predicted_classes[j].item()
                confidence = confidences[j].item()
                all_probs = probabilities[j].cpu().numpy()
                
                results.append({
                    "text": text,
                    "predicted_class": predicted_class,
                    "predicted_label": label_map[predicted_class],
                    "simplified_label": simplified_map[predicted_class],
                    "confidence": confidence,
                    "confidence_percentage": confidence * 100,
                    "all_probabilities": {
                        simplified_map[i]: float(prob) * 100 
                        for i, prob in enumerate(all_probs)
                    }
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Mendapatkan informasi tentang model
        
        Returns:
            Dictionary berisi informasi model
        """
        return {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "device": str(self.device),
            "num_labels": 3 if self.model_type == "3class" else 5,
            "labels": (list(self.label_mapping_3class.values()) 
                      if self.model_type == "3class" 
                      else list(self.label_mapping_5class.values())),
            "is_loaded": self.model is not None
        }
    
    def get_sentiment_emoji(self, label: str) -> str:
        """
        Mendapatkan emoji yang sesuai dengan label sentimen
        
        Args:
            label: Label sentimen
            
        Returns:
            Emoji string
        """
        emoji_map = {
            "Negatif": "😞",
            "Netral": "😐",
            "Positif": "😊",
            "Rating 1": "😡",
            "Rating 2": "😞",
            "Rating 3": "😐",
            "Rating 4": "😊",
            "Rating 5": "😍"
        }
        
        return emoji_map.get(label, "")


if __name__ == "__main__":
    # Test ModelPredictor
    print("=" * 60)
    print("Testing ModelPredictor")
    print("=" * 60)
    
    # Test 3-class model
    print("\n[3-Class Model]")
    predictor_3class = ModelPredictor(model_type="3class")
    
    if predictor_3class.load_model():
        test_texts = [
            "Pelayanan sangat memuaskan, driver ramah dan cepat!",
            "Biasa saja, tidak ada yang spesial",
            "Kecewa banget, driver tidak sopan dan aplikasi error terus"
        ]
        
        for text in test_texts:
            result = predictor_3class.predict_single(text)
            print(f"\nTeks: {text}")
            print(f"Prediksi: {result['simplified_label']} ({result['confidence_percentage']:.1f}%)")
            print(f"Probabilitas semua kelas: {result['all_probabilities']}")
