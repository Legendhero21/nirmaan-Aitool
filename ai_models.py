"""
ai_models.py
Handles deep learning models for Sentiment, Tone, and Semantic Analysis.
Uses HuggingFace Transformers and Sentence-Transformers with robust error handling.
"""

# Fix OpenMP library conflict in Anaconda environments
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
from typing import Dict, List, Optional, Tuple
import torch
import logging
from functools import wraps
import time

# Lazy imports with fallback handling
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.error("transformers not installed. Install with: pip install transformers")

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.error("sentence-transformers not installed")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- FEATURE FLAGS ---
ENABLE_TONE_ANALYSIS = False  # Requires PyTorch 2.6+, disabled due to security restrictions
ENABLE_SEMANTIC_ANALYSIS = True  # Optional but working

# --- MODEL CONFIGURATION ---
# Distilled models for faster CPU inference in deployment
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
TONE_MODEL = "valhalla/distilbart-mnli-12-3"  # Zero-shot classifier (disabled)
SEMANTIC_MODEL = "all-MiniLM-L6-v2"  # 80MB, fast inference

# Token limits for each model (prevents memory issues)
SENTIMENT_MAX_TOKENS = 512
TONE_MAX_TOKENS = 1024
SEMANTIC_MAX_TOKENS = 256

# Device detection (automatically use GPU if available)
DEVICE = 0 if torch.cuda.is_available() else -1
DEVICE_NAME = "GPU" if DEVICE == 0 else "CPU"

logger.info(f"🖥️ Running models on: {DEVICE_NAME}")


# --- UTILITY DECORATORS ---
def handle_model_errors(default_return):
    """Decorator to handle model inference errors gracefully"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                return default_return
        return wrapper
    return decorator


def validate_text_input(min_length: int = 1, max_length: int = 10000):
    """Decorator to validate text input before processing"""
    def decorator(func):
        @wraps(func)
        def wrapper(text: str, *args, **kwargs):
            # Handle None/empty
            if not text or not isinstance(text, str):
                logger.warning(f"{func.__name__} received invalid input: {type(text)}")
                return func("", *args, **kwargs)  # Pass empty string to function
            
            # Strip whitespace
            text = text.strip()
            
            # Check length constraints
            if len(text) < min_length:
                logger.warning(f"{func.__name__}: Text too short ({len(text)} chars)")
                return func("", *args, **kwargs)
            
            if len(text) > max_length:
                logger.warning(f"{func.__name__}: Text truncated from {len(text)} to {max_length} chars")
                text = text[:max_length]
            
            return func(text, *args, **kwargs)
        return wrapper
    return decorator


# --- MODEL LOADING FUNCTIONS ---
@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    """
    Loads the sentiment analysis pipeline with error handling.
    
    Returns:
        Sentiment pipeline or None if loading fails
        
    Raises:
        RuntimeError: If transformers library is unavailable
    """
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("Transformers library not available. Cannot load sentiment model.")
    
    try:
        with st.spinner("🔄 Loading Sentiment Model (DistilBERT)..."):
            logger.info("Loading Sentiment Model...")
            start_time = time.time()
            
            model = pipeline(
                "text-classification",
                model=SENTIMENT_MODEL,
                return_all_scores=True,
                device=DEVICE
            )
            
            load_time = time.time() - start_time
            logger.info(f"✅ Sentiment Model loaded in {load_time:.2f}s")
            return model
            
    except Exception as e:
        logger.error(f"❌ Failed to load sentiment model: {e}")
        st.error("⚠️ Sentiment analysis unavailable. Using fallback scoring.")
        return None


@st.cache_resource(show_spinner=False)
def load_tone_pipeline():
    """
    Loads the Zero-Shot Classification pipeline for complex tone detection.
    **CURRENTLY DISABLED** due to PyTorch version requirements.
    
    Returns:
        Tone classification pipeline or None if loading fails
    """
    if not ENABLE_TONE_ANALYSIS:
        logger.info("⏭️ Tone analysis disabled (requires PyTorch 2.6+)")
        return None
    
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers library unavailable for tone analysis")
        return None
    
    try:
        with st.spinner("🔄 Loading Tone Analysis Model..."):
            logger.info("Loading Zero-Shot Tone Model...")
            start_time = time.time()
            
            model = pipeline(
                "zero-shot-classification",
                model=TONE_MODEL,
                device=DEVICE
            )
            
            load_time = time.time() - start_time
            logger.info(f"✅ Tone Model loaded in {load_time:.2f}s")
            return model
            
    except Exception as e:
        logger.error(f"❌ Failed to load tone model: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_semantic_model():
    """
    Loads Sentence-Transformer for semantic similarity checks.
    
    Returns:
        SentenceTransformer model or None if loading fails
    """
    if not ENABLE_SEMANTIC_ANALYSIS:
        logger.info("⏭️ Semantic analysis disabled")
        return None
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.warning("Sentence-transformers library unavailable")
        return None
    
    try:
        with st.spinner("🔄 Loading Semantic Model (MiniLM)..."):
            logger.info("Loading Semantic Model...")
            start_time = time.time()
            
            # Device selection for sentence-transformers
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = SentenceTransformer(SEMANTIC_MODEL, device=device)
            
            load_time = time.time() - start_time
            logger.info(f"✅ Semantic Model loaded in {load_time:.2f}s on {device.upper()}")
            return model
            
    except Exception as e:
        logger.error(f"❌ Failed to load semantic model: {e}")
        return None


# --- INFERENCE FUNCTIONS ---
@validate_text_input(min_length=5)
@handle_model_errors(default_return=0.5)  # Neutral score on error
def get_sentiment_score(text: str) -> float:
    """
    Analyzes text to find the probability of 'POSITIVE' sentiment.
    Used for the 'Engagement' rubric score.
    
    Args:
        text: Input transcript text (automatically validated and truncated)
    
    Returns:
        float: Probability between 0.0 and 1.0 (0.5 = neutral fallback)
        
    Edge Cases Handled:
        - Empty/None input → returns 0.5
        - Model loading failure → returns 0.5
        - Text > 512 tokens → automatically truncated
        - Special characters → handled by tokenizer
    """
    if not text:
        logger.debug("Empty text provided to sentiment analysis")
        return 0.5  # Neutral score for empty input
    
    classifier = load_sentiment_pipeline()
    
    if classifier is None:
        logger.warning("Sentiment model unavailable, returning neutral score")
        return 0.5
    
    try:
        # Truncate to model's token limit (conservative character estimate)
        truncated_text = text[:SENTIMENT_MAX_TOKENS * 4]
        
        # Run inference
        results = classifier(truncated_text)
        
        # Extract score for 'POSITIVE' label
        # Format: [[{'label': 'NEGATIVE', 'score': 0.1}, {'label': 'POSITIVE', 'score': 0.9}]]
        if not results or not results[0]:
            return 0.5
        
        for res in results[0]:
            if res.get('label') == 'POSITIVE':
                score = res.get('score', 0.5)
                logger.debug(f"Sentiment score: {score:.3f}")
                return float(score)
        
        # If POSITIVE label not found (shouldn't happen)
        logger.warning("POSITIVE label not found in sentiment results")
        return 0.5
        
    except Exception as e:
        logger.error(f"Sentiment inference failed: {e}")
        return 0.5


@validate_text_input(min_length=10)
@handle_model_errors(default_return={})
def analyze_tone(text: str, candidate_labels: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Classifies text into custom tone labels using Zero-Shot Learning.
    This is the 'Wow Factor' visual feature.
    **CURRENTLY DISABLED** due to PyTorch version requirements.
    
    Args:
        text: Input transcript text
        candidate_labels: Custom tone labels to detect (default: predefined set)
    
    Returns:
        dict: Tone labels mapped to confidence scores, sorted by score descending
        Empty dict if analysis fails or disabled
        
    Example:
        >>> analyze_tone("I'm so excited!")
        {'Enthusiastic': 0.92, 'Confident': 0.45, 'Grateful': 0.23, ...}
    """
    if not ENABLE_TONE_ANALYSIS:
        logger.debug("Tone analysis is disabled")
        return {}
    
    if not text:
        logger.debug("Empty text provided to tone analysis")
        return {}
    
    classifier = load_tone_pipeline()
    
    if classifier is None:
        logger.warning("Tone model unavailable")
        return {}
    
    # Default tone labels (can be customized)
    if candidate_labels is None:
        candidate_labels = [
            "Confident",
            "Enthusiastic", 
            "Grateful",
            "Nervous",
            "Monotone",
            "Professional"
        ]
    
    try:
        # Truncate to model's token limit
        truncated_text = text[:TONE_MAX_TOKENS * 4]
        
        # Run zero-shot classification
        result = classifier(truncated_text, candidate_labels, multi_label=False)
        
        if not result or 'labels' not in result or 'scores' not in result:
            logger.warning("Invalid tone analysis result format")
            return {}
        
        # Create sorted dictionary by score (highest first)
        tone_dict = dict(zip(result['labels'], result['scores']))
        sorted_tones = dict(sorted(tone_dict.items(), key=lambda x: x[1], reverse=True))
        
        logger.debug(f"Top tone: {result['labels'][0]} ({result['scores'][0]:.3f})")
        return sorted_tones
        
    except Exception as e:
        logger.error(f"Tone analysis failed: {e}")
        return {}


@validate_text_input(min_length=5)
@handle_model_errors(default_return={})
def check_semantic_similarity(
    transcript: str, 
    reference_concepts: List[str],
    threshold: float = 0.3
) -> Dict[str, float]:
    """
    Checks if the transcript semantically matches a list of concepts.
    Useful for validating 'Content' if exact keywords are missing.
    
    Args:
        transcript: Student's introduction text
        reference_concepts: List of concepts to check (e.g., ["family", "hobbies"])
        threshold: Minimum similarity score to consider a match (0.0 to 1.0)
    
    Returns:
        dict: Concept → similarity score mapping (only scores above threshold)
        
    Example:
        >>> check_semantic_similarity(
        ...     "I live with my parents and sister",
        ...     ["family members", "household", "pets"]
        ... )
        {'family members': 0.78, 'household': 0.65}
    """
    # FIXED: Changed 'text' to 'transcript' to match parameter name
    if not transcript or not reference_concepts:
        logger.debug("Empty input to semantic similarity check")
        return {}
    
    model = load_semantic_model()
    
    if model is None:
        logger.warning("Semantic model unavailable")
        return {}
    
    try:
        # Truncate transcript for efficiency
        truncated_text = transcript[:SEMANTIC_MAX_TOKENS * 4]
        
        # Encode transcript once (optimization)
        transcript_emb = model.encode(
            truncated_text,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        results = {}
        
        for concept in reference_concepts:
            if not concept or not isinstance(concept, str):
                continue
                
            # Encode concept
            concept_emb = model.encode(
                concept,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            # Calculate cosine similarity
            similarity = util.cos_sim(transcript_emb, concept_emb)
            score = float(similarity[0][0])
            
            # Only include if above threshold
            if score >= threshold:
                results[concept] = round(score, 3)
        
        logger.debug(f"Found {len(results)} semantic matches above {threshold}")
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
    except Exception as e:
        logger.error(f"Semantic similarity check failed: {e}")
        return {}


# --- BATCH PROCESSING (for future scalability) ---
def batch_sentiment_analysis(texts: List[str]) -> List[float]:
    """
    Analyze sentiment for multiple texts efficiently.
    
    Args:
        texts: List of transcript texts
        
    Returns:
        List of positivity scores (same order as input)
    """
    if not texts:
        return []
    
    classifier = load_sentiment_pipeline()
    if classifier is None:
        return [0.5] * len(texts)
    
    try:
        # Process in batch (much faster than individual calls)
        results = classifier([t[:SENTIMENT_MAX_TOKENS * 4] for t in texts])
        
        scores = []
        for result in results:
            pos_score = next((r['score'] for r in result if r['label'] == 'POSITIVE'), 0.5)
            scores.append(pos_score)
        
        return scores
        
    except Exception as e:
        logger.error(f"Batch sentiment analysis failed: {e}")
        return [0.5] * len(texts)


# --- HEALTH CHECK ---
def check_models_health() -> Dict[str, bool]:
    """
    Verify all models can be loaded successfully.
    Useful for debugging deployment issues.
    
    Returns:
        dict: Model name → availability status
    """
    health = {
        "transformers_installed": TRANSFORMERS_AVAILABLE,
        "sentence_transformers_installed": SENTENCE_TRANSFORMERS_AVAILABLE,
        "cuda_available": torch.cuda.is_available(),
        "sentiment_model": False,
        "tone_model": False,
        "semantic_model": False
    }
    
    try:
        model = load_sentiment_pipeline()
        health["sentiment_model"] = model is not None
    except Exception:
        pass
    
    try:
        model = load_tone_pipeline()
        health["tone_model"] = model is not None
    except Exception:
        pass
    
    try:
        model = load_semantic_model()
        health["semantic_model"] = model is not None
    except Exception:
        pass
    
    return health


# --- TEST FUNCTION ---
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Running AI Model Tests")
    print("=" * 60)
    
    # Test text
    sample_text = "I am really excited to be here! I love solving problems and helping others."
    
    print(f"\n📄 Test Text: '{sample_text}'\n")
    
    # 1. Test Sentiment (REQUIRED)
    print("1️⃣ Testing Sentiment Analysis...")
    positivity = get_sentiment_score(sample_text)
    print(f"   ✅ Positivity Score: {positivity:.4f}")
    
    # 2. Test Tone (OPTIONAL - skip if disabled)
    if ENABLE_TONE_ANALYSIS:
        print("\n2️⃣ Testing Tone Analysis...")
        tones = analyze_tone(sample_text)
        if tones:
            top_tone = max(tones, key=tones.get)
            print(f"   ✅ Top Tone: {top_tone} ({tones[top_tone]:.3f})")
            print(f"   📊 All Tones: {tones}")
        else:
            print("   ⚠️ Tone analysis unavailable")
    else:
        print("\n2️⃣ Tone Analysis: ⏭️ Skipped (requires PyTorch 2.6+)")
    
    # 3. Test Semantic Similarity (OPTIONAL)
    if ENABLE_SEMANTIC_ANALYSIS:
        print("\n3️⃣ Testing Semantic Similarity...")
        concepts = ["enthusiasm", "problem solving", "collaboration", "family"]
        similarities = check_semantic_similarity(sample_text, concepts)
        if similarities:
            print(f"   ✅ Matches Found: {similarities}")
        else:
            print("   ⚠️ No semantic matches above threshold")
    else:
        print("\n3️⃣ Semantic Similarity: ⏭️ Skipped (disabled)")
    
    # 4. Edge Case Tests
    print("\n4️⃣ Testing Edge Cases...")
    edge_cases = [
        ("", "Empty string"),
        (None, "None input"),
        ("Hi", "Very short text"),
        ("A" * 10000, "Very long text")
    ]
    
    for text, description in edge_cases:
        try:
            score = get_sentiment_score(text)
            print(f"   ✅ {description}: {score:.2f}")
        except Exception as e:
            print(f"   ❌ {description}: {e}")
    
    # 5. Model Health Check
    print("\n5️⃣ Model Health Check...")
    health = check_models_health()
    for component, status in health.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {component}: {status}")
    
    print("\n" + "=" * 60)
    print("🎉 Test Suite Complete!")
    print("=" * 60)
