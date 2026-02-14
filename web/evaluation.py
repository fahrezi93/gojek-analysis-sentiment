"""
PerformanceEvaluator - Modul untuk evaluasi performa model
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import List, Dict, Tuple
import plotly.graph_objects as go
import plotly.express as px


class PerformanceEvaluator:
    """
    Kelas untuk mengevaluasi performa model dengan menghitung metrik
    (Akurasi, Presisi, Recall, F1-Score) dan membuat visualisasi.
    """
    
    def __init__(self, model_type: str = "3class"):
        """
        Inisialisasi PerformanceEvaluator
        
        Args:
            model_type: Tipe model - "3class" atau "5class"
        """
        self.model_type = model_type
        
        # Label names
        self.label_names_3class = ["Negatif", "Netral", "Positif"]
        self.label_names_5class = ["Rating 1", "Rating 2", "Rating 3", "Rating 4", "Rating 5"]
        
        self.label_names = (self.label_names_3class if model_type == "3class" 
                           else self.label_names_5class)
        
        # Results storage
        self.y_true = None
        self.y_pred = None
        self.metrics = {}
    
    def calculate_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict:
        """
        Menghitung semua metrik evaluasi
        
        Args:
            y_true: Label ground truth
            y_pred: Label hasil prediksi
            
        Returns:
            Dictionary berisi semua metrik
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_true, self.y_pred)
        
        # For multi-class, use macro average (konsisten dengan training notebook)
        precision = precision_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(self.y_true, self.y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(self.y_true, self.y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(self.y_true, self.y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        self.metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
            "f1_per_class": f1_per_class,
            "confusion_matrix": cm,
            "num_samples": len(y_true)
        }
        
        return self.metrics
    
    def get_metrics_summary(self) -> Dict:
        """
        Mendapatkan ringkasan metrik dalam format yang mudah dibaca
        
        Returns:
            Dictionary berisi ringkasan metrik
        """
        if not self.metrics:
            return {}
        
        summary = {
            "Akurasi": f"{self.metrics['accuracy'] * 100:.2f}%",
            "Presisi": f"{self.metrics['precision'] * 100:.2f}%",
            "Recall": f"{self.metrics['recall'] * 100:.2f}%",
            "F1-Score": f"{self.metrics['f1_score'] * 100:.2f}%",
            "Jumlah Sampel": self.metrics['num_samples']
        }
        
        return summary
    
    def get_per_class_metrics(self) -> pd.DataFrame:
        """
        Mendapatkan metrik per kelas dalam format DataFrame
        
        Returns:
            DataFrame berisi metrik per kelas
        """
        if not self.metrics:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            "Kelas": self.label_names,
            "Presisi": [f"{p*100:.2f}%" for p in self.metrics['precision_per_class']],
            "Recall": [f"{r*100:.2f}%" for r in self.metrics['recall_per_class']],
            "F1-Score": [f"{f*100:.2f}%" for f in self.metrics['f1_per_class']]
        })
        
        return df
    
    def create_confusion_matrix_plot(self, title: str = "Confusion Matrix") -> go.Figure:
        """
        Membuat visualisasi confusion matrix menggunakan Plotly
        
        Args:
            title: Judul plot
            
        Returns:
            Plotly Figure object
        """
        if not self.metrics:
            return None
        
        cm = self.metrics['confusion_matrix']
        
        # Create annotations for heatmap
        annotations = []
        for i in range(len(self.label_names)):
            for j in range(len(self.label_names)):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=str(cm[i, j]),
                        showarrow=False,
                        font=dict(color="white" if cm[i, j] > cm.max()/2 else "black", size=14)
                    )
                )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=self.label_names,
            y=self.label_names,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Count")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Prediksi",
            yaxis_title="Ground Truth",
            annotations=annotations,
            width=600,
            height=600,
            font=dict(size=12)
        )
        
        return fig
    
    def create_metrics_bar_chart(self) -> go.Figure:
        """
        Membuat bar chart untuk metrik utama
        
        Returns:
            Plotly Figure object
        """
        if not self.metrics:
            return None
        
        metrics_names = ["Akurasi", "Presisi", "Recall", "F1-Score"]
        metrics_values = [
            self.metrics['accuracy'] * 100,
            self.metrics['precision'] * 100,
            self.metrics['recall'] * 100,
            self.metrics['f1_score'] * 100
        ]
        
        # Color mapping
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        fig = go.Figure(data=[
            go.Bar(
                x=metrics_names,
                y=metrics_values,
                text=[f"{v:.2f}%" for v in metrics_values],
                textposition='auto',
                marker_color=colors
            )
        ])
        
        fig.update_layout(
            title="Metrik Evaluasi Model",
            xaxis_title="Metrik",
            yaxis_title="Nilai (%)",
            yaxis_range=[0, 100],
            height=400,
            font=dict(size=12)
        )
        
        return fig
    
    def create_per_class_metrics_chart(self) -> go.Figure:
        """
        Membuat grouped bar chart untuk metrik per kelas
        
        Returns:
            Plotly Figure object
        """
        if not self.metrics:
            return None
        
        fig = go.Figure()
        
        # Add bars for each metric
        fig.add_trace(go.Bar(
            name='Presisi',
            x=self.label_names,
            y=self.metrics['precision_per_class'] * 100,
            text=[f"{v*100:.1f}%" for v in self.metrics['precision_per_class']],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Recall',
            x=self.label_names,
            y=self.metrics['recall_per_class'] * 100,
            text=[f"{v*100:.1f}%" for v in self.metrics['recall_per_class']],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='F1-Score',
            x=self.label_names,
            y=self.metrics['f1_per_class'] * 100,
            text=[f"{v*100:.1f}%" for v in self.metrics['f1_per_class']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Metrik Per Kelas",
            xaxis_title="Kelas",
            yaxis_title="Nilai (%)",
            barmode='group',
            yaxis_range=[0, 100],
            height=450,
            font=dict(size=12)
        )
        
        return fig
    
    def create_prediction_distribution(self) -> go.Figure:
        """
        Membuat pie chart untuk distribusi prediksi
        
        Returns:
            Plotly Figure object
        """
        if self.y_pred is None:
            return None
        
        # Count predictions per class
        unique, counts = np.unique(self.y_pred, return_counts=True)
        
        # Create labels with counts
        labels = [f"{self.label_names[i]}<br>({counts[list(unique).index(i)] if i in unique else 0} sampel)" 
                 for i in range(len(self.label_names))]
        
        values = [counts[list(unique).index(i)] if i in unique else 0 
                 for i in range(len(self.label_names))]
        
        fig = go.Figure(data=[go.Pie(
            labels=[self.label_names[i] for i in range(len(self.label_names))],
            values=values,
            hole=.3,
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Distribusi Prediksi Model",
            height=450,
            font=dict(size=12)
        )
        
        return fig
    
    def generate_classification_report(self) -> str:
        """
        Generate classification report dalam format text
        
        Returns:
            String berisi classification report
        """
        if self.y_true is None or self.y_pred is None:
            return ""
        
        report = classification_report(
            self.y_true, 
            self.y_pred, 
            target_names=self.label_names,
            digits=4
        )
        
        return report
    
    def export_results_to_csv(self, output_path: str):
        """
        Export hasil evaluasi ke CSV
        
        Args:
            output_path: Path file output
        """
        if not self.metrics:
            return
        
        # Create summary DataFrame
        summary_data = {
            "Metrik": ["Akurasi", "Presisi", "Recall", "F1-Score"],
            "Nilai": [
                f"{self.metrics['accuracy']*100:.2f}%",
                f"{self.metrics['precision']*100:.2f}%",
                f"{self.metrics['recall']*100:.2f}%",
                f"{self.metrics['f1_score']*100:.2f}%"
            ]
        }
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)
        
        print(f"✓ Hasil evaluasi disimpan ke: {output_path}")


if __name__ == "__main__":
    # Test PerformanceEvaluator
    print("=" * 60)
    print("Testing PerformanceEvaluator")
    print("=" * 60)
    
    # Simulate some predictions for 3-class
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 100)
    y_pred = y_true.copy()
    # Add some noise
    noise_indices = np.random.choice(100, 20, replace=False)
    y_pred[noise_indices] = np.random.randint(0, 3, 20)
    
    evaluator = PerformanceEvaluator(model_type="3class")
    metrics = evaluator.calculate_metrics(y_true.tolist(), y_pred.tolist())
    
    print("\nMetrik Summary:")
    summary = evaluator.get_metrics_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\nMetrik Per Kelas:")
    print(evaluator.get_per_class_metrics())
