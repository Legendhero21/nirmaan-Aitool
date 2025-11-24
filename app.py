"""
app.py
IntroScore AI - Communication Assessment Tool
Clean, functional UI for case study demonstration
"""

import streamlit as st
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

# Import custom modules
from scoring_logic import ScoringEngine
from ai_models import get_sentiment_score, check_semantic_similarity, analyze_tone

# Page configuration
st.set_page_config(
    page_title="IntroScore AI - Nirmaan Education",
    page_icon="🎯",
    layout="wide"
)

# Dark theme with proper contrast
st.markdown("""
<style>
    /* Dark background */
    .main {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e293b 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e293b 100%);
    }
    
    /* Main content area */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: rgba(30, 41, 59, 0.7);
        border-radius: 10px;
        margin-top: 1rem;
    }
    
    /* Headers */
    h1 { color: #f8fafc !important; font-weight: 700; }
    h2 { color: #e2e8f0 !important; font-weight: 600; margin-top: 2rem; }
    h3 { color: #cbd5e1 !important; font-weight: 500; }
    p, label { color: #cbd5e1 !important; }
    
    /* Score card */
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }
    .score-number { font-size: 4rem; font-weight: 800; color: #ffffff; margin: 0; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3); }
    .score-label { font-size: 1.2rem; color: #f8fafc; margin-top: 0.5rem; font-weight: 600; }
    
    /* Criterion boxes */
    .criterion-box {
        background: rgba(51, 65, 85, 0.8);
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 0.75rem;
    }
    .criterion-title { color: #f8fafc; font-weight: 600; font-size: 1rem; margin-bottom: 0.5rem; }
    .criterion-feedback { color: #cbd5e1; font-size: 0.95rem; line-height: 1.6; }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white; border: none; border-radius: 8px; padding: 0.75rem 1.5rem; font-weight: 600;
    }
    
    /* Inputs */
    .stTextArea textarea, .stNumberInput input {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 2px solid #475569 !important;
        border-radius: 8px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1e3a8a 0%, #1e293b 100%); }
    [data-testid="stSidebar"] * { color: #f8fafc !important; }
    
    /* Metrics */
    [data-testid="stMetricValue"] { color: #f8fafc !important; }
    [data-testid="stMetricLabel"] { color: #cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)

# Load sample
@st.cache_data
def load_sample_transcript():
    try:
        with open("Sample-text-for-case-study.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except:
        return """Hello everyone, myself Muskan, studying in class 8th B section from Christ Public School. 
I am 13 years old. I live with my family. There are 3 people in my family, me, my mother and my father.
One special thing about my family is that they are very kind hearted to everyone and soft spoken. 
One thing I really enjoy is play, playing cricket and taking wickets.
A fun fact about me is that I see in mirror and talk by myself. 
One thing people don't know about me is that I once stole a toy from one of my cousin.
My favorite subject is science because it is very interesting. 
Through science I can explore the whole world and make the discoveries and improve the lives of others. 
Thank you for listening."""

def create_radar_chart(results):
    categories = []
    scores = []
    max_scores = []
    
    for crit, data in results['breakdown'].items():
        short_name = crit.split('(')[0].strip()
        categories.append(short_name)
        scores.append(data['score'])
        max_scores.append(data['max'])
    
    percentages = [(s/m)*100 for s, m in zip(scores, max_scores)]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=percentages,
        theta=categories,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line=dict(color='#3b82f6', width=3)
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor='#1e293b',
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=11, color='#f8fafc'), gridcolor='#475569'),
            angularaxis=dict(tickfont=dict(size=12, color='#f8fafc'), gridcolor='#475569', linecolor='#64748b')
        ),
        showlegend=False,
        margin=dict(t=20, b=20, l=40, r=40),
        height=350,
        paper_bgcolor='#1e293b',
        plot_bgcolor='#1e293b',
        font=dict(color='#f8fafc')
    )
    return fig

def display_tone_bars(tones: Dict[str, float]):
    """Create horizontal bar chart for tones"""
    if not tones:
        return None
    
    # Prepare data
    df = pd.DataFrame({
        'Tone': list(tones.keys())[:5], # Top 5
        'Score': list(tones.values())[:5]
    })
    
    fig = px.bar(
        df, x='Score', y='Tone', orientation='h',
        color='Score', color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis=dict(showgrid=False, gridcolor='#475569', tickfont=dict(color='#f8fafc')),
        yaxis=dict(showgrid=False, tickfont=dict(color='#f8fafc')),
        margin=dict(t=0, b=0, l=0, r=0),
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        coloraxis_showscale=False
    )
    return fig

def export_json(results, transcript, duration):
    return json.dumps({
        "timestamp": datetime.now().isoformat(),
        "transcript": transcript,
        "duration_seconds": duration,
        "results": results
    }, indent=2)

# Main app
def main():
    # Top header with logo
    header_col1, header_col2 = st.columns([1, 6])
    with header_col1:
        try:
            st.image("nirmaan.png", width=100)
        except:
            st.markdown("### 🎯")
    
    with header_col2:
        st.markdown("""
        <div style="margin-top: 1rem;">
            <h1 style="margin: 0; color: #f8fafc;">IntroScore AI</h1>
            <p style="margin: 0.3rem 0 0 0; color: #cbd5e1; font-size: 1.1rem;">AI-Powered Communication Assessment Tool</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 Instructions")
        st.markdown("1. Enter transcript text\n2. Set speech duration\n3. Click Evaluate\n4. Review scores")
        st.markdown("---")
        st.markdown("### 🎯 Scoring Criteria")
        st.markdown("- **Content**: 40 pts\n- **Speech Rate**: 10 pts\n- **Grammar**: 10 pts\n- **Vocab**: 10 pts\n- **Clarity**: 15 pts\n- **Engagement**: 15 pts")
        st.markdown("---")
        
        sample_text = load_sample_transcript()
        if st.button("📥 Load Sample Transcript"):
            st.session_state.transcript = sample_text
            st.session_state.duration = 52
            st.success("✅ Sample loaded")
            st.rerun()
        
        st.markdown("---")
        st.caption("Developed for Nirmaan Education")
    
    # Main input
    col1, col2 = st.columns([3, 1])
    with col1:
        transcript = st.text_area("Transcript Text", value=st.session_state.get('transcript', ''), height=250)
    with col2:
        duration = st.number_input("Duration (seconds)", min_value=10, value=st.session_state.get('duration', 60), step=5)
        word_count = len(transcript.split()) if transcript else 0
        st.metric("📝 Words", word_count)
        if duration > 0: st.metric("⚡ WPM", f"{int((word_count/duration)*60)}")
    
    # Evaluate button
    if st.button("🚀 Evaluate Introduction", type="primary"):
        if not transcript or len(transcript.strip()) < 20:
            st.error("⚠️ Please enter at least 20 characters")
        else:
            with st.spinner("🤖 Analyzing transcript (Tone, Sentiment, Rules)..."):
                try:
                    # 1. Run AI Analysis
                    sentiment = get_sentiment_score(transcript)
                    semantics = check_semantic_similarity(transcript, ["Science", "Ambition", "Family", "Hobbies"])
                    tones = analyze_tone(transcript)
                    
                    # 2. Calculate Scores
                    engine = ScoringEngine(transcript, duration)
                    results = engine.calculate_all(sentiment)
                    
                    # 3. Save to Session
                    st.session_state.results = results
                    st.session_state.semantics = semantics
                    st.session_state.tones = tones
                    st.session_state.transcript = transcript
                    st.session_state.duration = duration
                    
                    st.success("✅ Analysis complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Display results
    if 'results' in st.session_state:
        st.markdown("---")
        results = st.session_state.results
        score = results['total_score']
        
        st.header("📊 Results")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-number">{score}</div>
                <div class="score-label">Overall Score / 100</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🎭 Tone Analysis")
            if 'tones' in st.session_state and st.session_state.tones:
                tone_fig = display_tone_bars(st.session_state.tones)
                st.plotly_chart(tone_fig, use_container_width=True)
                top_tone = list(st.session_state.tones.keys())[0]
                st.info(f"Dominant Tone: **{top_tone}**")
            else:
                st.caption("Tone analysis unavailable")

        with col2:
            st.markdown("**Performance Breakdown**")
            radar = create_radar_chart(results)
            st.plotly_chart(radar, use_container_width=True)
        
        # Detailed breakdown
        st.markdown("---")
        st.subheader("📝 Detailed Feedback")
        for criterion, data in results['breakdown'].items():
            progress = data['score'] / data['max']
            st.markdown(f"""
            <div class="criterion-box">
                <div class="criterion-title">{criterion}: {data['score']}/{data['max']} points</div>
                <div class="criterion-feedback">{data['feedback']}</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(progress)
        
        # Semantic analysis
        if 'semantics' in st.session_state and st.session_state.semantics:
            st.markdown("---")
            st.subheader("🔍 Semantic Theme Detection")
            sem_cols = st.columns(len(st.session_state.semantics))
            for col, (theme, score) in zip(sem_cols, st.session_state.semantics.items()):
                with col: st.metric(theme, f"{score*100:.0f}%")
        
        # Export
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            json_data = export_json(results, st.session_state.transcript, st.session_state.duration)
            st.download_button("📥 Download Results (JSON)", data=json_data, file_name="results.json", mime="application/json")
        with col2:
            if st.button("🔄 New Assessment"):
                for key in ['results', 'semantics', 'tones', 'transcript', 'duration']:
                    if key in st.session_state: del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main()