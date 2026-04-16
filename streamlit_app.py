import streamlit as st
from pipeline import NewsPipeline
import pandas as pd
import plotly.express as px
from collections import Counter
import re

# Page configuration for a professional wide look
st.set_page_config(
    page_title="News Intelligence Dashboard",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- PREMIUM CSS STYLING ---
st.markdown("""
<style>
    /* Main Background & Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=Inter:wght@300;400;700&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(10, 15, 30) 0%, rgb(1, 5, 10) 90.2%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }

    /* Glassmorphic Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        border: 1px solid rgba(255, 255, 255, 0.2);
        background: rgba(255, 255, 255, 0.05);
    }

    /* Header Styling */
    .main-title {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        color: #888;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }

    /* Category Glow Effects */
    .glow-Politics-World { border-left: 5px solid #FFD700; box-shadow: -10px 0 20px -10px rgba(255, 215, 0, 0.3); }
    .glow-Sports { border-left: 5px solid #39FF14; box-shadow: -10px 0 20px -10px rgba(57, 255, 20, 0.3); }
    .glow-Business { border-left: 5px solid #BF00FF; box-shadow: -10px 0 20px -10px rgba(191, 0, 255, 0.3); }
    .glow-Tech { border-left: 5px solid #00FFFF; box-shadow: -10px 0 20px -10px rgba(0, 255, 255, 0.3); }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.4);
    }

    /* Sidebar Tweaks */
    .css-1d391kg { background-color: rgba(10, 15, 30, 0.95); }
</style>
""", unsafe_allow_html=True)

# --- DATA & PIPELINE INITIALIZATION ---
@st.cache_resource
def get_pipeline():
    return NewsPipeline()

pipeline = get_pipeline()

SAMPLE_ARTICLES = {
    "Politics": "World leaders gathered at the United Nations today to discuss the escalating climate crisis. The summit aimed to finalize a global agreement on carbon emission reductions, with several major economies pledging significant investments in renewable energy infrastructure. Tensions remained high, however, as developing nations called for more financial support to manage the transition and adapt to the impacts of a warming planet.",
    "Sports": "In a thrilling championship final, the underdog team secured a historic victory in the final minutes of extra time. The star striker scored a magnificent goal from outside the box, sending the spectators into a frenzy. This win marks the club's first major trophy in over three decades, and celebrations are expected to last throughout the night across the city.",
    "Tech": "A groundbreaking new AI model has been unveiled, demonstrating near-human performance on complex logical reasoning tasks. Unlike previous generations, this architecture focuses on causal inference and structured knowledge representation, significantly reducing the tendency to 'hallucinate' incorrect facts. Experts believe this represents a major leap toward Artificial General Intelligence (AGI).",
    "Business": "The global tech giants reported record-breaking quarterly earnings yesterday, defying analyst expectations despite economic headwinds. The surge was driven by massive growth in cloud computing services and a renewed interest in advertising spend. Stock prices for the major players jumped by nearly 8% in pre-market trading, leading a broader rally in the global indices."
}

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.markdown("<h2 style='font-family: Outfit;'>⚙️ Control Center</h2>", unsafe_allow_html=True)
    st.info("System uses TF-IDF + Logistic Regression")
    
    st.markdown("### 📊 Metrics")
    st.metric("Model Fidelity", "90.66%")
    st.metric("Total Vocabulary", "5,000 Tokens")
    
    st.markdown("### 📝 Summary Settings")
    summary_len = st.slider("Target Sentences", 1, 5, 2)
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Test Bench")
    st.caption("Load a sample article to see the system in action:")
    
    selected_sample = st.selectbox("Select a Topic", list(SAMPLE_ARTICLES.keys()), key="sample_topic_select")
    
    if st.button("Load Selected Sample"):
        st.session_state.main_input_area = SAMPLE_ARTICLES[selected_sample]
        st.rerun()

# --- MAIN INTERFACE ---
st.markdown("<h1 class='main-title'>News Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>State-of-the-art News Classification & Summarization Engine</p>", unsafe_allow_html=True)

# Input Layout
col_main, col_stats = st.columns([1.8, 1.2], gap="large")

with col_main:
    st.markdown("### 📰 Input Feed")
    article_input = st.text_area(
        "Enter news content or title + description:",
        height=320,
        placeholder="Paste your news article here...",
        key="main_input_area"
    )
    
    if st.button("🚀 Analyze Now"):
        if article_input.strip():
            with st.spinner("Decoding article features..."):
                # Processing
                result = pipeline.process_article(article_input, summary_sentences=summary_len)
                probs = pipeline.get_probabilities(article_input)
                
                # Store results in session
                st.session_state.result = result
                st.session_state.probs = probs
                st.session_state.analyzed = True
        else:
            st.error("Please provide some text to analyze.")

# Results Display
if st.session_state.get('analyzed'):
    res = st.session_state.result
    probs = st.session_state.probs
    cat = res['category'].replace("/", "-") # For CSS class mapping
    
    with col_main:
        st.markdown(f"""
        <div class="glass-card glow-{cat}">
            <p style="color: #aaa; font-size: 0.9rem; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 2px;">Predicted Category</p>
            <h2 style="margin: 0; color: #fff; font-family: Outfit;">{res['category']}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ✍️ Intelligence Summary")
        st.markdown(f"""
        <div class="glass-card">
            <p style="font-size: 1.1rem; line-height: 1.6; color: #eee;">{res['summary']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_stats:
        st.markdown("### 📈 Confidence Analysis")
        # Confidence Chart
        prob_df = pd.DataFrame(list(probs.items()), columns=['Category', 'Probability'])
        fig = px.bar(prob_df, x='Probability', y='Category', orientation='h', 
                     color='Category', color_discrete_sequence=px.colors.qualitative.Antique)
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#aaa",
            xaxis_title="Confidence level",
            yaxis_title=None,
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🏷️ Keyword Distribution")
        # Keyword extraction (simple)
        words = re.findall(r'\w+', article_input.lower())
        stop_words = set(['the', 'and', 'a', 'to', 'of', 'in', 'i', 'is', 'it', 'for', 'on', 'with', 'as', 'at', 'by', 'today', 'news'])
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
        word_counts = Counter(filtered_words).most_common(8)
        
        if word_counts:
            kw_df = pd.DataFrame(word_counts, columns=['Keyword', 'Count'])
            fig_kw = px.pie(kw_df, values='Count', names='Keyword', hole=.4,
                           color_discrete_sequence=px.colors.sequential.RdBu)
            fig_kw.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#aaa",
                showlegend=False,
                height=300,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_kw, use_container_width=True)

else:
    with col_stats:
        st.markdown("""
        <div class="glass-card" style="text-align: center; color: #888;">
            <p>Ready for input...</p>
            <p style="font-size: 0.8rem;">Select a sample or paste custom text to generate charts and insights.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><hr><p style='text-align: center; color: #555;'>NLP News Intelligence System v1.0 | Project Submission</p>", unsafe_allow_html=True)
