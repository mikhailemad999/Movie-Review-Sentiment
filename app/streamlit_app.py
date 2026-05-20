"""
streamlit_app.py — Production Streamlit dashboard.
"""
import os
import requests
import streamlit as st
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title="Movie Sentiment Analyser",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.environ.get("SENTIMENT_API_URL", "http://localhost:8000")

# --- Custom CSS for Premium Design ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B, #C0392B, #8E44AD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        padding-top: 1rem;
    }

    .sub-header {
        text-align: center;
        color: #B2BABB;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 3rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.45);
    }

    .positive {
        border-left: 6px solid #2ECC71;
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.1), rgba(0, 0, 0, 0));
    }

    .negative {
        border-left: 6px solid #E74C3C;
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.1), rgba(0, 0, 0, 0));
    }

    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
        border: none;
        background: linear-gradient(135deg, #6C3483, #8E44AD);
        color: white;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #8E44AD, #9B59B6);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(142, 68, 173, 0.4);
    }
    
    .stTextArea>div>div>textarea {
        background-color: rgba(0, 0, 0, 0.2);
        color: white;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    .stTextArea>div>div>textarea:focus {
        border-color: #8E44AD;
        box-shadow: 0 0 0 1px #8E44AD;
    }
</style>
""", unsafe_allow_html=True)

# --- Gauge Chart Component ---
def make_gauge(probability: float) -> go.Figure:
    color = "#2ECC71" if probability >= 0.5 else "#E74C3C"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 42, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
            "bar": {"color": color, "thickness": 0.4},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        height=300,
        margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "family": "Inter"}
    )
    return fig

def main():
    st.markdown('<h1 class="main-header">🎬 Movie Sentiment Analyser</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Enterprise Review Classification</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown("---")
        
        # Check backend health
        backend_status = "🔴 Offline"
        try:
            res = requests.get(f"{API_URL}/health", timeout=2)
            if res.status_code == 200:
                backend_status = "🟢 Online"
        except Exception:
            pass
            
        st.markdown(f"**Backend API:** {backend_status}")
        st.markdown("---")
        st.markdown("## 📖 About")
        st.markdown("""
        Powered by a **Bidirectional GRU** with self-attention.
        
        Deployed with FastAPI, Docker, and MLflow for true production readiness.
        """)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### 📝 Enter a Review")
        example = "This film was an absolute masterpiece! The acting was flawless and the cinematography was breathtaking."
        review = st.text_area("Your review:", value=example, height=200, label_visibility="collapsed")
        
        c1, c2 = st.columns([3, 1])
        analyze = c1.button("🔮 Analyze Sentiment", type="primary")
        clear = c2.button("🗑 Clear")
        
        if clear:
            st.rerun()

    with col2:
        if analyze and review.strip():
            with st.spinner("Analyzing..."):
                t0 = time.time()
                try:
                    response = requests.post(f"{API_URL}/predict", json={"text": review}, timeout=5)
                    response.raise_for_status()
                    data = response.json()
                    
                    label = data["label"].capitalize()
                    prob = data["probability"]
                    conf = data["confidence"]
                    
                    is_pos = (label == "Positive")
                    cls = "positive" if is_pos else "negative"
                    emoji = "😊" if is_pos else "😞"
                    
                    st.markdown(f"""
                    <div class="glass-card {cls}">
                        <h2 style="margin:0; font-size: 2rem;">{label} {emoji}</h2>
                        <p style="color: #B2BABB; margin-top: 10px;">
                            Confidence: <strong style="color:white;">{conf*100:.1f}%</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.plotly_chart(make_gauge(prob), use_container_width=True)
                
                except requests.exceptions.RequestException as e:
                    st.error(f"⚠️ Failed to connect to API: {e}")
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 4rem 2rem;">
                <h3 style="color: #7F8C8D; margin:0;">Waiting for review...</h3>
                <p style="color: #5D6D7E; margin-top: 10px;">Click analyze to see the AI verdict.</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
