# AI Communication Scoring Tool

## Project Overview
This tool was developed as a submission for the **Nirmaan Education AI Internship Case Study**. It automates the evaluation of student self-introduction transcripts based on a multi-criteria rubric.

The solution utilizes a hybrid approach:
1.  **Rule-Based Logic:** For strict adherence to rubric constraints (Speech Rate, Grammar, Keyword Presence).
2.  **NLP & Transformer Models:** For semantic analysis, tone detection, and sentiment scoring.

## Deployed Link
https://nirmaan-aitool.streamlit.app/

## Repository Structure
```text
├── app.py                 # Main Streamlit application (Frontend & State Management)
├── scoring_logic.py       # Rubric calculation engine (Rule-based logic)
├── ai_models.py           # AI Inference engine (HuggingFace Transformers)
├── requirements.txt       # Python dependencies
├── packages.txt           # System-level dependencies (Java for LanguageTool)
└── README.md              # Project documentation
```
## Key Features & Implementation
Hybrid Scoring Engine: Combines deterministic rules with probabilistic AI models to generate a final score (0-100).

Speech Rate Analysis: Calculates Words Per Minute (WPM) based on transcript length and user-provided duration.

Semantic Search: Uses sentence-transformers (MiniLM) to identify conceptual matches for topics like "Family" or "Ambition" even when exact keywords are missing.

Tone Analysis: Implements Zero-Shot Classification (DeBERTa) to detect speaker confidence and enthusiasm.

Resiliency: Includes error handling decorators to prevent application crashes if AI models fail to load or inputs are malformed.

## Tech Stack
Frontend: Streamlit

Language Processing: Python 3.9+, Spacy, NLTK

AI/ML Models:

Sentiment: distilbert-base-uncased-finetuned-sst-2-english

Tone: MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33

Semantic: sentence-transformers/all-MiniLM-L6-v2

Grammar: LanguageTool (via language-tool-python)

Visualization: Plotly (Radar Charts, Bar Charts)#