"""
Streamlit App - Analisis Sentimen Review Ojek Online
Menggunakan model IndoBERT untuk menganalisis sentimen dari teks review.
"""

import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os
import sys

# Add parent directory to path for model access
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Analisis Sentimen Ojol",
    page_icon="🛵",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .header-title {
        text-align: center;
        color: #1E3A8A;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        text-align: center;
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .result-positive {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        border: 2px solid #10B981;
    }
    
    .result-negative {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border: 2px solid #EF4444;
    }
    
    .result-neutral {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border: 2px solid #F59E0B;
    }
    
    .sentiment-emoji {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .sentiment-label {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .confidence-text {
        font-size: 1.1rem;
        color: #475569;
    }
    
    /* Probability bars */
    .prob-container {
        background: #F1F5F9;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    
    .prob-bar {
        height: 30px;
        border-radius: 15px;
        margin: 8px 0;
        transition: width 0.5s ease;
    }
    
    /* Input area */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #E2E8F0;
        font-size: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Info box */
    .info-box {
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Example chips */
    .example-chip {
        display: inline-block;
        background: #F1F5F9;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .example-chip:hover {
        background: #E2E8F0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #94A3B8;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# MODEL DEFINITION
# ============================================
class IndoBERTSentimentClassifier(nn.Module):
    """IndoBERT-based sentiment classifier for Indonesian text."""
    
    def __init__(self, model_name, num_classes, dropout_rate=0.4, freeze_layers=4):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Freeze embedding & bottom layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(freeze_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        # Classifier head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        x = self.dropout(pooled)
        return self.fc(x)


# ============================================
# MODEL LOADING WITH CACHING
# ============================================
@st.cache_resource
def load_model():
    """Load the trained sentiment analysis model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Path to model
    model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(model_dir, 'models', 'indobert_sentiment_3class.pt')
    tokenizer_path = os.path.join(model_dir, 'models', 'tokenizer')
    
    # Check if model exists
    if not os.path.exists(model_path):
        return None, None, None, f"Model tidak ditemukan di: {model_path}"
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get('config', {})
        
        # Load tokenizer
        if os.path.exists(tokenizer_path):
            tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        else:
            tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
        
        # Initialize and load model
        model = IndoBERTSentimentClassifier(
            model_name='indobenchmark/indobert-base-p1',
            num_classes=config.get('num_classes', 3),
            dropout_rate=config.get('dropout_rate', 0.4),
            freeze_layers=config.get('freeze_layers', 4)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        label_names = checkpoint.get('label_names', ['negative', 'neutral', 'positive'])
        
        return model, tokenizer, device, None
        
    except Exception as e:
        return None, None, None, f"Error loading model: {str(e)}"


def predict_sentiment(text, model, tokenizer, device):
    """Predict sentiment for given text."""
    label_names = ['negative', 'neutral', 'positive']
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    
    return {
        'sentiment': label_names[predicted_class],
        'confidence': probabilities[predicted_class].item(),
        'probabilities': {
            label: prob.item() 
            for label, prob in zip(label_names, probabilities)
        }
    }


# ============================================
# MAIN APP
# ============================================
def main():
    # Header
    st.markdown('<h1 class="header-title">🛵 Analisis Sentimen Review Ojol</h1>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">Analisis sentimen review aplikasi ojek online menggunakan IndoBERT</p>', unsafe_allow_html=True)
    
    # Load model
    model, tokenizer, device, error = load_model()
    
    if error:
        st.error(f"""
        ⚠️ **Model belum tersedia!**
        
        {error}
        
        Pastikan Anda sudah melatih model dengan menjalankan notebook `sentiment_training_3class_final.ipynb` terlebih dahulu.
        
        Model akan disimpan di folder `models/`.
        """)
        
        st.info("""
        💡 **Langkah-langkah:**
        1. Buka file `sentiment_training_3class_final.ipynb`
        2. Jalankan semua cell untuk melatih model
        3. Model akan disimpan otomatis di `models/indobert_sentiment_3class.pt`
        4. Refresh halaman ini setelah model selesai dilatih
        """)
        return
    
    # Device info
    device_name = "GPU 🎮" if torch.cuda.is_available() else "CPU 💻"
    st.caption(f"🖥️ Running on: {device_name}")
    
    # Divider
    st.markdown("---")
    
    # Input section
    st.subheader("📝 Masukkan Teks Review")
    
    # Example reviews
    st.markdown("**Contoh review yang bisa dicoba:**")
    
    examples = [
        "Aplikasi sangat bagus, driver ramah dan tepat waktu",
        "Pelayanan buruk, driver lama dan tidak sopan",
        "Biasa saja, standar seperti aplikasi lainnya",
        "Suka banget sama promo-promonya, murah meriah!",
        "Pesanan sering dibatalkan driver, mengecewakan",
    ]
    
    # Create columns for example buttons
    cols = st.columns(2)
    selected_example = None
    
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"💬 {example[:35]}...", key=f"example_{i}"):
                selected_example = example
    
    # Text input
    if selected_example:
        text_input = st.text_area(
            "Ketik atau paste review di sini:",
            value=selected_example,
            height=120,
            placeholder="Contoh: Aplikasi sangat membantu, driver ramah dan cepat sampai..."
        )
    else:
        text_input = st.text_area(
            "Ketik atau paste review di sini:",
            height=120,
            placeholder="Contoh: Aplikasi sangat membantu, driver ramah dan cepat sampai..."
        )
    
    # Analyze button
    analyze_clicked = st.button("🔍 Analisis Sentimen", type="primary")
    
    # Process analysis
    if analyze_clicked:
        if not text_input.strip():
            st.warning("⚠️ Mohon masukkan teks review terlebih dahulu!")
        else:
            with st.spinner("🔄 Menganalisis sentimen..."):
                result = predict_sentiment(text_input, model, tokenizer, device)
            
            st.markdown("---")
            st.subheader("📊 Hasil Analisis")
            
            # Sentiment emoji and colors
            sentiment_config = {
                'positive': {'emoji': '😊', 'color': '#10B981', 'bg': 'result-positive', 'label': 'POSITIF'},
                'negative': {'emoji': '😠', 'color': '#EF4444', 'bg': 'result-negative', 'label': 'NEGATIF'},
                'neutral': {'emoji': '😐', 'color': '#F59E0B', 'bg': 'result-neutral', 'label': 'NETRAL'}
            }
            
            config = sentiment_config[result['sentiment']]
            
            # Result card
            st.markdown(f"""
            <div class="result-card {config['bg']}">
                <div class="sentiment-emoji">{config['emoji']}</div>
                <div class="sentiment-label" style="color: {config['color']}">
                    {config['label']}
                </div>
                <div class="confidence-text">
                    Tingkat keyakinan: <strong>{result['confidence']*100:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability breakdown
            st.markdown("### 📈 Probabilitas per Kelas")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                prob_neg = result['probabilities']['negative'] * 100
                st.metric(
                    label="😠 Negatif",
                    value=f"{prob_neg:.1f}%"
                )
                st.progress(result['probabilities']['negative'])
            
            with col2:
                prob_neu = result['probabilities']['neutral'] * 100
                st.metric(
                    label="😐 Netral",
                    value=f"{prob_neu:.1f}%"
                )
                st.progress(result['probabilities']['neutral'])
            
            with col3:
                prob_pos = result['probabilities']['positive'] * 100
                st.metric(
                    label="😊 Positif",
                    value=f"{prob_pos:.1f}%"
                )
                st.progress(result['probabilities']['positive'])
            
            # Analysis details
            with st.expander("🔎 Detail Analisis"):
                st.write("**Teks yang dianalisis:**")
                st.info(text_input)
                
                st.write("**Hasil prediksi:**")
                st.json({
                    'sentiment': result['sentiment'],
                    'confidence': f"{result['confidence']*100:.2f}%",
                    'probabilities': {
                        k: f"{v*100:.2f}%" for k, v in result['probabilities'].items()
                    }
                })
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ Tentang Aplikasi")
        st.markdown("""
        Aplikasi ini menggunakan model **IndoBERT** yang telah dilatih untuk menganalisis sentimen review aplikasi ojek online dalam bahasa Indonesia.
        
        **Kelas Sentimen:**
        - 😊 **Positif**: Review yang menunjukkan kepuasan
        - 😐 **Netral**: Review yang biasa saja atau ambigu
        - 😠 **Negatif**: Review yang menunjukkan ketidakpuasan
        
        **Teknologi:**
        - 🤖 IndoBERT (Indonesian BERT)
        - 🔥 PyTorch
        - 🌐 Streamlit
        
        **Tips:**
        - Gunakan bahasa Indonesia untuk hasil terbaik
        - Review yang lebih panjang dan detail akan memberikan hasil yang lebih akurat
        """)
        
        st.markdown("---")
        st.markdown("**📊 Statistik Model:**")
        if model is not None:
            st.success("✅ Model berhasil dimuat")
            st.caption(f"Device: {device}")
        else:
            st.error("❌ Model belum dimuat")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>🛵 Sentiment Analysis for Ojek Online Reviews</p>
        <p>Built with ❤️ using Streamlit & IndoBERT</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
