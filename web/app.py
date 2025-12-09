"""
Sentiment Analysis Web App - Analisis Sentimen Ulasan Gojek dengan IndoBERT
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import TextNormalizer
from predictor import ModelPredictor
from evaluation import PerformanceEvaluator


class SentimentUI:
    """
    Kelas utama untuk mengatur antarmuka Streamlit dan logika aplikasi
    """
    
    def __init__(self):
        """Inisialisasi SentimentUI"""
        self.setup_page_config()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Konfigurasi halaman Streamlit"""
        st.set_page_config(
            page_title="Analisis Sentimen Gojek - IndoBERT",
            page_icon="🚗",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS untuk styling
        st.markdown("""
            <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #00AA13;
                text-align: center;
                padding: 1rem 0;
                margin-bottom: 2rem;
            }
            .sub-header {
                font-size: 1.5rem;
                font-weight: bold;
                color: #333333 !important;
                margin-top: 2rem;
                margin-bottom: 1rem;
            }
            .metric-card {
                background-color: #f0f2f6;
                color: #333333 !important;
                padding: 1rem;
                border-radius: 0.5rem;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .stButton>button {
                width: 100%;
                background-color: #00AA13;
                color: white !important;
                font-weight: bold;
                border-radius: 0.5rem;
                padding: 0.5rem 1rem;
                border: none;
            }
            .stButton>button:hover {
                background-color: #008f0f;
                color: white !important;
            }
            .sentiment-positive {
                color: #00AA13 !important;
                font-weight: bold;
                font-size: 1.2rem;
            }
            .sentiment-negative {
                color: #DC3545 !important;
                font-weight: bold;
                font-size: 1.2rem;
            }
            .sentiment-neutral {
                color: #FFC107 !important;
                font-weight: bold;
                font-size: 1.2rem;
            }
            .info-box {
                background-color: #E3F2FD;
                color: #000000 !important;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #2196F3;
                margin: 1rem 0;
            }
            /* Fix for dark mode text issues */
            .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, li, span {
                color: inherit; 
            }
            /* Ensure text inside info-box is always black */
            .info-box p, .info-box li, .info-box span, .info-box h1, .info-box h2, .info-box h3 {
                color: #000000 !important;
            }
            /* Ensure text inside metric-card is always dark */
            .metric-card p, .metric-card h1, .metric-card h2, .metric-card h3, .metric-card span {
                color: #333333 !important;
            }
            </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Inisialisasi session state"""
        if 'predictor' not in st.session_state:
            st.session_state.predictor = None
        if 'normalizer' not in st.session_state:
            st.session_state.normalizer = TextNormalizer()
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'model_type' not in st.session_state:
            st.session_state.model_type = "3class"
        if 'prediction_results' not in st.session_state:
            st.session_state.prediction_results = None
        if 'evaluation_results' not in st.session_state:
            st.session_state.evaluation_results = None
    
    def render_sidebar(self):
        """Render sidebar dengan logo dan navigasi"""
        with st.sidebar:
            # Logo and Title
            st.markdown("""
                <div style='text-align: center; padding: 1rem 0;'>
                    <h1 style='color: #00AA13; margin-bottom: 0;'>🚗</h1>
                    <h2 style='color: #00AA13; margin-top: 0;'>Gojek</h2>
                    <h3 style='color: #666;'>Sentiment Analysis</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Navigation
            page = st.radio(
                "📍 Navigasi",
                ["🏠 Beranda", "🔮 Analisis Sentimen", "📊 Evaluasi Model", "ℹ️ Tentang"],
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            # Model Selection
            st.markdown("### ⚙️ Pengaturan Model")
            model_type = st.selectbox(
                "Pilih Skema Model",
                ["Skema 3-Kelas (Positif, Netral, Negatif)", 
                 "Skema 5-Kelas (Rating 1-5)"],
                key="model_selector"
            )
            
            # Convert selection to model type
            new_model_type = "3class" if "3-Kelas" in model_type else "5class"
            
            # Load model button
            if st.button("🔄 Load Model", key="load_model_btn"):
                self.load_model(new_model_type)
            
            # Model status
            if st.session_state.model_loaded:
                st.success(f"✅ Model {st.session_state.model_type} siap")
            else:
                st.warning("⚠️ Model belum dimuat")
            
            st.markdown("---")
            
            # Information
            st.markdown("### 📖 Informasi")
            st.markdown("""
                **Model:** IndoBERT Base  
                **Dataset:** Ulasan Gojek  
                **Framework:** PyTorch + Transformers
            """)
            
            return page
    
    def load_model(self, model_type: str):
        """Load model berdasarkan tipe yang dipilih"""
        with st.spinner(f"⏳ Memuat model {model_type}..."):
            try:
                # Initialize predictor
                predictor = ModelPredictor(model_type=model_type)
                
                # Load model
                success = predictor.load_model()
                
                if success:
                    st.session_state.predictor = predictor
                    st.session_state.model_type = model_type
                    st.session_state.model_loaded = True
                    st.success(f"✅ Model {model_type} berhasil dimuat!")
                else:
                    st.error("❌ Gagal memuat model. Periksa path model.")
                    st.session_state.model_loaded = False
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.model_loaded = False
    
    def render_home_page(self):
        """Render halaman beranda"""
        st.markdown('<p class="main-header">🚗 Analisis Sentimen Ulasan Gojek</p>', 
                   unsafe_allow_html=True)
        
        # Hero Section
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #00AA13 0%, #008f0f 100%); 
                     border-radius: 1rem; color: white; margin: 2rem 0;'>
                    <h2>Powered by IndoBERT</h2>
                    <p style='font-size: 1.1rem; margin-top: 1rem;'>
                        Sistem analisis sentimen berbasis deep learning untuk memahami 
                        feedback pelanggan Gojek secara otomatis dan akurat
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Features
        st.markdown('<p class="sub-header">✨ Fitur Utama</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class='metric-card'>
                    <h3>🔮 Analisis Real-time</h3>
                    <p>Prediksi sentimen instan untuk teks manual atau file CSV</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='metric-card'>
                    <h3>📊 Evaluasi Lengkap</h3>
                    <p>Metrik komprehensif dengan confusion matrix dan visualisasi</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class='metric-card'>
                    <h3>🎯 Dual Schema</h3>
                    <p>Pilih antara 3-kelas (Positif/Netral/Negatif) atau 5-kelas (Rating 1-5)</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown('<p class="sub-header">📈 Performa Model</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Akurasi", "92%", "↑ 2%")
        with col2:
            st.metric("Presisi", "91%", "↑ 1%")
        with col3:
            st.metric("Recall", "90%", "→")
        with col4:
            st.metric("F1-Score", "91%", "↑ 1%")
        
        st.markdown("---")
        
        # Getting Started
        st.markdown('<p class="sub-header">🚀 Cara Memulai</p>', unsafe_allow_html=True)
        
        st.markdown("""
            <div class='info-box'>
                <h4>Langkah-langkah:</h4>
                <ol>
                    <li>Pilih <b>Skema Model</b> di sidebar (3-Kelas atau 5-Kelas)</li>
                    <li>Klik tombol <b>🔄 Load Model</b> untuk memuat model</li>
                    <li>Navigasi ke halaman <b>🔮 Analisis Sentimen</b> untuk prediksi</li>
                    <li>Atau ke halaman <b>📊 Evaluasi Model</b> untuk testing</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
    
    def render_analysis_page(self):
        """Render halaman analisis sentimen"""
        st.markdown('<p class="main-header">🔮 Analisis Sentimen Ulasan Gojek</p>', 
                   unsafe_allow_html=True)
        
        # Check if model is loaded
        if not st.session_state.model_loaded:
            st.warning("⚠️ Silakan load model terlebih dahulu dari sidebar!")
            return
        
        # Input method selection
        st.markdown("### 📝 Pilih Metode Input")
        
        input_method = st.radio(
            "Metode Input:",
            ["💬 Teks Manual", "📄 Upload File CSV"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        if input_method == "💬 Teks Manual":
            self.render_manual_input()
        else:
            self.render_csv_input()
    
    def render_manual_input(self):
        """Render input manual untuk analisis"""
        st.markdown("### 💬 Masukkan Teks Ulasan")
        
        # Text input
        user_text = st.text_area(
            "Ketik ulasan di sini...",
            height=150,
            placeholder="Contoh: Pelayanan sangat memuaskan, driver ramah dan cepat!",
            key="manual_text_input"
        )
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_btn = st.button("🔍 ANALISIS SENTIMEN", key="analyze_manual_btn", use_container_width=True)
        
        if analyze_btn and user_text.strip():
            self.analyze_single_text(user_text)
        elif analyze_btn:
            st.warning("⚠️ Mohon masukkan teks terlebih dahulu!")
    
    def analyze_single_text(self, text: str):
        """Analisis sentimen untuk single text"""
        with st.spinner("🔄 Menganalisis sentimen..."):
            try:
                # Preprocess text
                cleaned_text = st.session_state.normalizer.clean_text(text)
                
                # Predict
                result = st.session_state.predictor.predict_single(cleaned_text)
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Hasil Prediksi")
                
                # Main result card
                sentiment = result['simplified_label']
                confidence = result['confidence_percentage']
                emoji = st.session_state.predictor.get_sentiment_emoji(sentiment)
                
                # Determine sentiment class for styling
                if "Positif" in sentiment or "Rating 5" in sentiment or "Rating 4" in sentiment:
                    sentiment_class = "sentiment-positive"
                elif "Negatif" in sentiment or "Rating 1" in sentiment or "Rating 2" in sentiment:
                    sentiment_class = "sentiment-negative"
                else:
                    sentiment_class = "sentiment-neutral"
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                             padding: 2rem; border-radius: 1rem; color: white; text-align: center;'>
                            <h2>{emoji}</h2>
                            <h3>Sentimen: {sentiment}</h3>
                            <h4>Confidence: {confidence:.1f}%</h4>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### 📈 Distribusi Probabilitas")
                    prob_df = pd.DataFrame([
                        {"Kelas": k, "Probabilitas (%)": v} 
                        for k, v in result['all_probabilities'].items()
                    ])
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
                # Show cleaned text
                with st.expander("🔍 Lihat Detail Preprocessing"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Teks Original:**")
                        st.info(text)
                    with col2:
                        st.markdown("**Teks Setelah Preprocessing:**")
                        st.success(cleaned_text)
                
            except Exception as e:
                st.error(f"❌ Error saat analisis: {str(e)}")
    
    def render_csv_input(self):
        """Render CSV upload untuk analisis batch"""
        st.markdown("### 📄 Upload File CSV")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Pilih file CSV yang berisi kolom 'text' atau 'review'",
            type=['csv'],
            key="csv_uploader"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                # Show preview
                st.markdown("#### 📋 Preview Data")
                st.dataframe(df.head(10), use_container_width=True)
                
                st.info(f"📊 Total data: {len(df)} baris")
                
                # Identify text column
                text_col = None
                for col in ['text', 'review', 'ulasan', 'komentar', 'comment']:
                    if col in df.columns:
                        text_col = col
                        break
                
                if text_col is None:
                    st.error("❌ Tidak ditemukan kolom 'text' atau 'review' dalam file CSV!")
                    return
                
                st.success(f"✅ Kolom teks ditemukan: `{text_col}`")
                
                # Analyze button
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    analyze_btn = st.button("🔍 ANALISIS SEMUA DATA", key="analyze_csv_btn", use_container_width=True)
                
                if analyze_btn:
                    self.analyze_batch_data(df, text_col)
                    
            except Exception as e:
                st.error(f"❌ Error membaca file: {str(e)}")
    
    def analyze_batch_data(self, df: pd.DataFrame, text_col: str):
        """Analisis sentimen untuk batch data"""
        with st.spinner("🔄 Menganalisis data... Mohon tunggu..."):
            try:
                # Preprocess texts
                texts = df[text_col].fillna("").astype(str).tolist()
                cleaned_texts = st.session_state.normalizer.preprocess_batch(texts)
                
                # Predict
                results = st.session_state.predictor.predict_batch(cleaned_texts)
                
                # Create results DataFrame
                results_df = pd.DataFrame([
                    {
                        "Text Original": texts[i],
                        "Text Cleaned": cleaned_texts[i],
                        "Prediksi": results[i]['simplified_label'],
                        "Confidence (%)": f"{results[i]['confidence_percentage']:.2f}"
                    }
                    for i in range(len(results))
                ])
                
                # Store in session state
                st.session_state.prediction_results = results_df
                
                # Display results
                st.markdown("---")
                st.markdown("### ✅ Hasil Analisis")
                
                st.success(f"✨ Berhasil menganalisis {len(results_df)} data!")
                
                # Summary statistics
                st.markdown("#### 📊 Ringkasan Hasil")
                
                sentiment_counts = results_df['Prediksi'].value_counts()
                
                cols = st.columns(len(sentiment_counts))
                for idx, (sentiment, count) in enumerate(sentiment_counts.items()):
                    with cols[idx]:
                        percentage = (count / len(results_df)) * 100
                        emoji = st.session_state.predictor.get_sentiment_emoji(sentiment)
                        st.metric(
                            f"{emoji} {sentiment}",
                            f"{count} data",
                            f"{percentage:.1f}%"
                        )
                
                # Show detailed results
                st.markdown("#### 📋 Detail Hasil Prediksi")
                st.dataframe(results_df, use_container_width=True, height=400)
                
                # Download button
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Hasil (CSV)",
                    data=csv,
                    file_name="hasil_analisis_sentimen.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error saat analisis: {str(e)}")
    
    def render_evaluation_page(self):
        """Render halaman evaluasi model"""
        st.markdown('<p class="main-header">📊 Evaluasi Kinerja Model IndoBERT</p>', 
                   unsafe_allow_html=True)
        
        # Check if model is loaded
        if not st.session_state.model_loaded:
            st.warning("⚠️ Silakan load model terlebih dahulu dari sidebar!")
            return
        
        st.markdown("### 📤 Unggah Data Uji (Test Set)")
        
        st.markdown("""
            <div class='info-box'>
                <b>Format file CSV:</b><br>
                • Harus memiliki kolom <code>text</code> atau <code>review</code> (teks ulasan)<br>
                • Harus memiliki kolom <code>label</code> (ground truth label dalam bentuk angka)<br>
                • Label: 0=Negatif, 1=Netral, 2=Positif (untuk 3-kelas)<br>
                • Label: 0=Rating 1, 1=Rating 2, ..., 4=Rating 5 (untuk 5-kelas)
            </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Pilih file CSV data uji",
            type=['csv'],
            key="eval_csv_uploader"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                # Show preview
                st.markdown("#### 📋 Preview Data")
                st.dataframe(df.head(10), use_container_width=True)
                
                st.info(f"📊 Total data: {len(df)} baris")
                
                # Identify columns
                text_col = None
                for col in ['text', 'review', 'ulasan', 'komentar']:
                    if col in df.columns:
                        text_col = col
                        break
                
                if text_col is None:
                    st.error("❌ Tidak ditemukan kolom teks!")
                    return
                
                if 'label' not in df.columns:
                    st.error("❌ Tidak ditemukan kolom 'label'!")
                    return
                
                st.success(f"✅ Kolom teks: `{text_col}`, Kolom label: `label`")
                
                # Evaluate button
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    eval_btn = st.button("🎯 MULAI EVALUASI", key="eval_btn", use_container_width=True)
                
                if eval_btn:
                    self.evaluate_model(df, text_col)
                    
            except Exception as e:
                st.error(f"❌ Error membaca file: {str(e)}")
    
    def evaluate_model(self, df: pd.DataFrame, text_col: str):
        """Evaluasi model dengan data test"""
        with st.spinner("🔄 Mengevaluasi model... Mohon tunggu..."):
            try:
                # Preprocess texts
                texts = df[text_col].fillna("").astype(str).tolist()
                cleaned_texts = st.session_state.normalizer.preprocess_batch(texts)
                
                # Get ground truth labels
                y_true = df['label'].tolist()
                
                # Predict
                results = st.session_state.predictor.predict_batch(cleaned_texts)
                y_pred = [r['predicted_class'] for r in results]
                
                # Calculate metrics
                evaluator = PerformanceEvaluator(model_type=st.session_state.model_type)
                metrics = evaluator.calculate_metrics(y_true, y_pred)
                
                # Store in session state
                st.session_state.evaluation_results = {
                    'evaluator': evaluator,
                    'metrics': metrics,
                    'y_true': y_true,
                    'y_pred': y_pred
                }
                
                # Display results
                st.markdown("---")
                st.markdown("### 📈 Laporan Kinerja")
                
                st.success("✨ Evaluasi selesai!")
                
                # Main metrics
                st.markdown("#### 🎯 Metrik Utama")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Akurasi", f"{metrics['accuracy']*100:.0f}%")
                with col2:
                    st.metric("Presisi", f"{metrics['precision']*100:.0f}%")
                with col3:
                    st.metric("Recall", f"{metrics['recall']*100:.0f}%")
                with col4:
                    st.metric("F1-Score", f"{metrics['f1_score']*100:.0f}%")
                
                st.markdown("---")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Metrik Per Kelas")
                    fig_bar = evaluator.create_per_class_metrics_chart()
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    st.markdown("#### 🥧 Distribusi Prediksi")
                    fig_pie = evaluator.create_prediction_distribution()
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Confusion Matrix
                st.markdown("#### 🔥 Confusion Matrix")
                fig_cm = evaluator.create_confusion_matrix_plot()
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Per-class metrics table
                st.markdown("#### 📋 Detail Metrik Per Kelas")
                per_class_df = evaluator.get_per_class_metrics()
                st.dataframe(per_class_df, use_container_width=True, hide_index=True)
                
                # Classification report
                with st.expander("📄 Lihat Classification Report Lengkap"):
                    report = evaluator.generate_classification_report()
                    st.text(report)
                
            except Exception as e:
                st.error(f"❌ Error saat evaluasi: {str(e)}")
                st.exception(e)
    
    def render_about_page(self):
        """Render halaman about"""
        st.markdown('<p class="main-header">ℹ️ Tentang Aplikasi</p>', 
                   unsafe_allow_html=True)
        
        # About Project
        st.markdown("### 📱 Tentang Proyek")
        st.markdown("""
            <div class='info-box'>
                Aplikasi ini merupakan sistem analisis sentimen berbasis <b>Deep Learning</b> 
                yang dirancang khusus untuk menganalisis ulasan pelanggan <b>Gojek</b>. 
                Sistem ini menggunakan model <b>IndoBERT</b> yang telah di-fine-tune 
                dengan dataset ulasan Gojek dari Google Play Store.
            </div>
        """, unsafe_allow_html=True)
        
        # Technology Stack
        st.markdown("### 🛠️ Teknologi yang Digunakan")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                **Model & Framework:**
                - IndoBERT Base
                - PyTorch
                - Transformers (HuggingFace)
            """)
        
        with col2:
            st.markdown("""
                **Web Framework:**
                - Streamlit
                - Plotly
                - Pandas
            """)
        
        with col3:
            st.markdown("""
                **Preprocessing:**
                - NLTK
                - Regex
                - Custom Normalizer
            """)
        
        st.markdown("---")
        
        # Model Architecture
        st.markdown("### 🏗️ Arsitektur Model")
        
        st.markdown("""
            <div class='metric-card'>
                <h4>IndoBERT (Indonesian BERT)</h4>
                <p>Model transformer pre-trained pada korpus bahasa Indonesia</p>
                <ul style='text-align: left; display: inline-block;'>
                    <li>12 layer transformer</li>
                    <li>768 hidden dimensions</li>
                    <li>12 attention heads</li>
                    <li>Fine-tuned untuk sentiment analysis</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Features
        st.markdown("### ✨ Fitur Aplikasi")
        
        features = [
            ("🔮", "Analisis Sentimen Real-time", "Prediksi sentimen instant untuk teks manual atau batch"),
            ("📊", "Evaluasi Model Komprehensif", "Metrik lengkap: Accuracy, Precision, Recall, F1-Score"),
            ("📈", "Visualisasi Interaktif", "Confusion matrix dan chart distribusi sentimen"),
            ("💾", "Export Hasil", "Download hasil analisis dalam format CSV"),
            ("🎯", "Dual Schema", "Pilihan klasifikasi 3-kelas atau 5-kelas"),
            ("🧹", "Text Preprocessing", "Pembersihan teks otomatis dengan normalisasi slang")
        ]
        
        for i in range(0, len(features), 2):
            col1, col2 = st.columns(2)
            
            with col1:
                emoji, title, desc = features[i]
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>{emoji} {title}</h3>
                        <p>{desc}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            if i + 1 < len(features):
                with col2:
                    emoji, title, desc = features[i+1]
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h3>{emoji} {title}</h3>
                            <p>{desc}</p>
                        </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Dataset Info
        st.markdown("### 📚 Dataset")
        st.markdown("""
            <div class='info-box'>
                <b>Sumber Data:</b> Google Play Store - Ulasan Aplikasi Gojek<br>
                <b>Jumlah Data:</b> ~50,000 ulasan<br>
                <b>Label:</b> 3-Kelas (Positif/Netral/Negatif) & 5-Kelas (Rating 1-5)<br>
                <b>Preprocessing:</b> Cleaning, normalisasi slang, tokenisasi
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Contact/Credit
        st.markdown("### 👨‍💻 Developer")
        st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <h4>Skripsi - Analisis Sentimen Ulasan Gojek</h4>
                <p>Menggunakan Model IndoBERT untuk Klasifikasi Sentimen</p>
                <p style='color: #666; margin-top: 1rem;'>
                    © 2024 | Built with ❤️ using Streamlit
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Menjalankan aplikasi"""
        # Render sidebar dan get selected page
        page = self.render_sidebar()
        
        # Render page based on selection
        if page == "🏠 Beranda":
            self.render_home_page()
        elif page == "🔮 Analisis Sentimen":
            self.render_analysis_page()
        elif page == "📊 Evaluasi Model":
            self.render_evaluation_page()
        elif page == "ℹ️ Tentang":
            self.render_about_page()


def main():
    """Main function"""
    app = SentimentUI()
    app.run()


if __name__ == "__main__":
    main()
