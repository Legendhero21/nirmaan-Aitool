"""
app.py
IntroScore AI - Communication Assessment Tool
Clean, functional UI for case study demonstration
"""

import streamlit as st
import json
from datetime import datetime
import plotly.graph_objects as go
from typing import Dict, Any

# Import custom modules
from scoring_logic import ScoringEngine
from ai_models import get_sentiment_score, check_semantic_similarity

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
    
    /* Main content area - dark with light text */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: rgba(30, 41, 59, 0.7);
        border-radius: 10px;
        margin-top: 1rem;
    }
    
    /* Headers - light text on dark bg */
    h1 {
        color: #f8fafc !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #e2e8f0 !important;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #cbd5e1 !important;
        font-weight: 500;
    }
    
    /* Paragraph text */
    p, label {
        color: #cbd5e1 !important;
    }
    
    /* Score card */
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }
    
    .score-number {
        font-size: 4rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .score-label {
        font-size: 1.2rem;
        color: #f8fafc;
        margin-top: 0.5rem;
        font-weight: 600;
    }
    
    /* Criterion boxes - dark with light text */
    .criterion-box {
        background: rgba(51, 65, 85, 0.8);
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .criterion-title {
        color: #f8fafc;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .criterion-feedback {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
    }
    
    /* Text areas - dark bg with light text */
    .stTextArea textarea {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 2px solid #475569 !important;
        border-radius: 8px;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3) !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #64748b !important;
    }
    
    /* Number input - dark theme */
    .stNumberInput input {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 2px solid #475569 !important;
        border-radius: 8px;
    }
    
    .stNumberInput input:focus {
        border-color: #3b82f6 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #1e293b 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }
    
    /* Metrics - light text */
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
    }
    
    /* Divider */
    hr {
        border-color: #475569 !important;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
    }
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
    """Create performance radar chart"""
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
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=11, color='#f8fafc'),
                gridcolor='#475569',
                tickcolor='#f8fafc'
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='#f8fafc'),
                gridcolor='#475569',
                linecolor='#64748b'
            )
        ),
        showlegend=False,
        margin=dict(t=20, b=20, l=40, r=40),
        height=350,
        paper_bgcolor='#1e293b',
        plot_bgcolor='#1e293b',
        font=dict(color='#f8fafc')
    )
    
    return fig

def export_json(results, transcript, duration):
    """Export results as JSON"""
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
        st.markdown("""
        1. Enter transcript text
        2. Set speech duration
        3. Click Evaluate
        4. Review detailed scores
        """)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Scoring Criteria")
        st.markdown("""
        - **Content & Structure**: 40 pts
        - **Speech Rate**: 10 pts
        - **Grammar**: 10 pts
        - **Vocabulary**: 10 pts
        - **Clarity (Fillers)**: 15 pts
        - **Engagement**: 15 pts
        """)
        
        st.markdown("---")
        
        sample_text = load_sample_transcript()
        if st.button("📥 Load Sample Transcript"):
            st.session_state.transcript = sample_text
            st.session_state.duration = 52
            st.success("✅ Sample loaded")
            st.rerun()
        
        st.markdown("---")
        st.caption("Developed for Nirmaan Education")
    
    # Main input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        transcript = st.text_area(
            "Transcript Text",
            value=st.session_state.get('transcript', ''),
            height=250,
            placeholder="Enter or paste your self-introduction transcript here..."
        )
    
    with col2:
        duration = st.number_input(
            "Duration (seconds)",
            min_value=10,
            max_value=300,
            value=st.session_state.get('duration', 60),
            step=5
        )
        
        word_count = len(transcript.split()) if transcript else 0
        st.metric("📝 Words", word_count)
        
        if duration > 0 and word_count > 0:
            wpm = (word_count / duration) * 60
            st.metric("⚡ WPM", f"{wpm:.0f}")
    
    # Evaluate button
    if st.button("🚀 Evaluate Introduction", type="primary"):
        if not transcript or len(transcript.strip()) < 20:
            st.error("⚠️ Please enter at least 20 characters")
        else:
            with st.spinner("🤖 Analyzing transcript..."):
                try:
                    # Run AI analysis
                    sentiment = get_sentiment_score(transcript)
                    semantics = check_semantic_similarity(
                        transcript, 
                        ["Science", "Ambition", "Family", "Hobbies"]
                    )
                    
                    # Calculate scores
                    engine = ScoringEngine(transcript, duration)
                    results = engine.calculate_all(sentiment)
                    
                    # Save to session
                    st.session_state.results = results
                    st.session_state.semantics = semantics
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
        
        # Score and chart
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-number">{score}</div>
                <div class="score-label">Overall Score / 100</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
            st.metric("📄 Word Count", results.get('word_count', 'N/A'))
            st.metric("⏱️ Duration", f"{results.get('duration_seconds', 'N/A')}s")
        
        with col2:
            st.markdown("**Performance Breakdown**")
            radar = create_radar_chart(results)
            st.plotly_chart(radar, width='stretch')
        
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
            st.caption("AI-detected themes using sentence embeddings")
            
            sem_cols = st.columns(len(st.session_state.semantics))
            for col, (theme, score) in zip(sem_cols, st.session_state.semantics.items()):
                with col:
                    st.metric(theme, f"{score*100:.0f}%")
        
        # Export
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            json_data = export_json(results, st.session_state.transcript, st.session_state.duration)
            st.download_button(
                label="📥 Download Results (JSON)",
                data=json_data,
                file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            if st.button("🔄 New Assessment"):
                for key in ['results', 'semantics', 'transcript', 'duration']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main()
