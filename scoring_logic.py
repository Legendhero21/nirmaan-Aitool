

import re
import logging
from typing import Dict, Tuple, Any, Optional
from functools import lru_cache

# Lazy imports for optional dependencies
try:
    import language_tool_python
    LANGUAGE_TOOL_AVAILABLE = True
except ImportError:
    LANGUAGE_TOOL_AVAILABLE = False
    logging.warning("language_tool_python not available. Using fallback mode.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Grammar checking limited.")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScoringEngine:
    """
    Analyzes and scores student self-introduction transcripts based on a multi-criteria rubric.
    
    Criteria (100 points total):
    - Content & Structure (40): Salutation, keywords, flow
    - Speech Rate (10): Words per minute analysis
    - Grammar (10): Error detection
    - Vocabulary (10): Type-Token Ratio
    - Clarity (15): Filler word frequency
    - Engagement (15): Sentiment positivity (passed externally)
    """
    
    # Constants - Define once for reusability and easy updates
    OPTIMAL_WPM_MIN = 111
    OPTIMAL_WPM_MAX = 140
    
    # Salutation keywords organized by score tier
    SALUTATIONS = {
        5: ["excited to introduce", "feeling great", "delighted", "thrilled"],
        4: ["good morning", "good afternoon", "good evening", "hello everyone", "greetings"],
        2: ["hi everyone", "hello", "hey there", "hi all"],
    }
    
    # Content keywords (Mandatory = 4pts each, Optional = 2pts each)
    MANDATORY_KEYWORDS = {
        "Name": ["myself", "name is", "i am", "i'm", "my name"],
        "Age": ["years old", "age is", "aged", "year old"],
        "School": ["school", "class", "study", "studying", "student", "college", "university", "grade"],
        "Family": ["family", "mother", "father", "parents", "siblings", "brother", "sister", "mom", "dad"],
        "Hobbies": ["hobby", "hobbies", "playing", "enjoy", "like to", "love to", "interest", "passionate"],
    }
    
    OPTIONAL_KEYWORDS = {
        "Origin": ["from", "live in", "born in", "native", "hometown", "city"],
        "Goal": ["goal", "dream", "become", "ambition", "aspire", "aim", "want to be", "future"],
        "Unique": ["fact", "secret", "unique", "special", "interesting thing", "quirk"],
        "Strength": ["strength", "achievement", "won", "good at", "skilled", "proud", "accomplished"],
    }
    
    # Filler words for clarity analysis
    FILLER_WORDS = [
        "um", "uh", "like", "you know", "so", "actually", "basically", 
        "right", "hmm", "ah", "er", "well", "kind of", "sort of", "i mean"
    ]
    
    # Compiled regex patterns for performance
    _PUNCTUATION_PATTERN = re.compile(r'[^\w\s]')
    _WORD_BOUNDARY_PATTERN = re.compile(r'\b(?:' + '|'.join(FILLER_WORDS) + r')\b')
    
    def __init__(self, transcript: str, duration_seconds: float):
        """
        Initialize scoring engine with transcript and audio duration.
        
        Args:
            transcript: Raw text transcript of the speech
            duration_seconds: Duration of the audio in seconds
            
        Raises:
            ValueError: If transcript is empty or duration is invalid
        """
        # Input validation
        if not transcript or not transcript.strip():
            raise ValueError("Transcript cannot be empty")
        if duration_seconds <= 0:
            raise ValueError("Duration must be positive")
            
        self.raw_text = transcript.strip()
        
        # Clean text for word counting (remove punctuation, preserve spaces)
        self.cleaned_text = self._PUNCTUATION_PATTERN.sub('', self.raw_text)
        self.words = self.cleaned_text.split()
        self.word_count = len(self.words)
        
        # Ensure minimum duration to prevent division errors
        self.duration = max(duration_seconds, 1)
        
        # Lazy initialization of grammar checker
        self._grammar_tool: Optional[language_tool_python.LanguageTool] = None
        self._use_language_tool = LANGUAGE_TOOL_AVAILABLE
        
        logger.info(f"Initialized ScoringEngine: {self.word_count} words, {self.duration}s duration")

    @property
    def grammar_tool(self):
        """Lazy load LanguageTool only when needed (saves ~2-3 seconds on init)"""
        if self._grammar_tool is None and self._use_language_tool:
            try:
                self._grammar_tool = language_tool_python.LanguageTool('en-US')
                logger.info("LanguageTool initialized successfully")
            except Exception as e:
                logger.error(f"LanguageTool initialization failed: {e}")
                self._use_language_tool = False
        return self._grammar_tool

    def __del__(self):
        """Cleanup: Close LanguageTool connection to free resources"""
        if self._grammar_tool is not None:
            try:
                self._grammar_tool.close()
            except Exception:
                pass

    def _get_wpm_score(self) -> Tuple[int, str]:
        """
        Calculate Speech Rate and map to rubric points.
        
        Rubric ranges (WPM -> Points):
        - 111-140: 10 (optimal)
        - 141-160 or 81-110: 6 (acceptable)
        - >160 or <80: 2 (poor)
        
        Returns:
            Tuple of (score, feedback_message)
        """
        if self.word_count == 0:
            return 0, "⚠️ No speech detected."
        
        wpm = (self.word_count / self.duration) * 60
        wpm = round(wpm, 1)
        
        # Apply rubric scoring logic
        if self.OPTIMAL_WPM_MIN <= wpm <= self.OPTIMAL_WPM_MAX:
            return 10, f"✅ Perfect pace ({wpm} WPM)."
        elif 141 <= wpm <= 160:
            return 6, f"⚡ Slightly fast ({wpm} WPM). Aim for {self.OPTIMAL_WPM_MIN}-{self.OPTIMAL_WPM_MAX}."
        elif 81 <= wpm <= 110:
            return 6, f"🐌 Slightly slow ({wpm} WPM). Aim for {self.OPTIMAL_WPM_MIN}-{self.OPTIMAL_WPM_MAX}."
        elif wpm > 160:
            return 2, f"🚨 Too fast ({wpm} WPM). Slow down significantly."
        else:
            return 2, f"🚨 Too slow ({wpm} WPM). Speak more fluently."

    def _get_content_score(self) -> Tuple[int, str]:
        """
        Analyze Salutation, Keywords, and Flow based on rubric.
        
        Components:
        - Salutation (max 5): Opening quality
        - Keywords (max 30): Mandatory (4pts each) + Optional (2pts each)
        - Flow (max 5): Structural coherence
        
        Returns:
            Tuple of (score, feedback_message)
        """
        text_lower = self.raw_text.lower()
        score = 0
        feedback = []
        
        # === 1. Salutation Analysis (Max 5 points) ===
        salutation_pts = 0
        for points, keywords in sorted(self.SALUTATIONS.items(), reverse=True):
            if any(phrase in text_lower for phrase in keywords):
                salutation_pts = points
                feedback.append(
                    "🎯 Excellent opening energy." if points == 5 else
                    "👍 Professional salutation used." if points == 4 else
                    "💬 Casual salutation. Try 'Hello everyone'."
                )
                break
        
        if salutation_pts == 0:
            feedback.append("❌ No clear salutation found.")
        
        score += salutation_pts
        
        # === 2. Keyword Presence (Max 30 points) ===
        # Mandatory keywords (4 points each, max 20)
        found_mandatory = []
        for category, keywords in self.MANDATORY_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                score += 4
                found_mandatory.append(category)
        
        # Optional keywords (2 points each, max 10)
        found_optional = []
        for category, keywords in self.OPTIONAL_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                score += 2
                found_optional.append(category)
        
        feedback.append(
            f"📋 Covered {len(found_mandatory)}/5 mandatory topics" + 
            (f" + {len(found_optional)} optional" if found_optional else "")
        )
        
        # === 3. Flow Check (Max 5 points) ===
        flow_score = 0
        if len(self.words) > 5:
            # Check structural coherence: Start -> End
            start_segment = " ".join(self.words[:10]).lower()
            end_segment = " ".join(self.words[-15:]).lower()
            
            has_opening = any(s in start_segment for s in ["hi", "hello", "good", "myself", "greetings"])
            has_closing = any(e in end_segment for e in ["thank", "listening", "that's all", "conclud"])
            
            if has_opening and has_closing:
                flow_score = 5
                feedback.append("🔄 Good structural flow (Opening → Body → Closing).")
            elif has_opening:
                flow_score = 3
                feedback.append("⚠️ Good start, but abrupt ending.")
            else:
                flow_score = 1
                feedback.append("📉 Structure needs improvement.")
        
        score += flow_score
        
        return min(score, 40), " | ".join(feedback)  # Cap at max 40

    def _get_grammar_score(self) -> Tuple[int, str]:
        """
        Count grammar errors using LanguageTool (or TextBlob fallback).
        
        Rubric formula: 1 - min(errors_per_100_words / 10, 1)
        Mapped to: 10/8/6/4/2 points
        
        Returns:
            Tuple of (score, feedback_message)
        """
        if self.word_count == 0:
            return 0, "⚠️ No text to analyze."
        
        error_count = 0
        
        # Try LanguageTool first (more accurate)
        if self._use_language_tool and self.grammar_tool:
            try:
                matches = self.grammar_tool.check(self.raw_text)
                error_count = len(matches)
            except Exception as e:
                logger.warning(f"LanguageTool check failed: {e}")
                self._use_language_tool = False
        
        # Fallback to TextBlob if LanguageTool unavailable
        if not self._use_language_tool and TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(self.raw_text)
                corrected = blob.correct()
                # Estimate errors by comparing original vs corrected
                error_count = sum(
                    1 for orig, corr in zip(self.raw_text.split(), str(corrected).split()) 
                    if orig.lower() != corr.lower()
                )
            except Exception as e:
                logger.warning(f"TextBlob fallback failed: {e}")
                error_count = 0
        
        # Calculate normalized metric
        errors_per_100 = (error_count / self.word_count) * 100
        raw_metric = 1 - min(errors_per_100 / 10, 1)
        
        # Map to rubric scores
        if raw_metric >= 0.9:
            return 10, f"✅ Excellent grammar ({error_count} error{'s' if error_count != 1 else ''})."
        elif raw_metric >= 0.7:
            return 8, f"👍 Good grammar ({error_count} error{'s' if error_count != 1 else ''})."
        elif raw_metric >= 0.5:
            return 6, f"⚠️ Fair grammar. {error_count} error{'s' if error_count != 1 else ''} found."
        elif raw_metric >= 0.3:
            return 4, f"📉 Needs work. {error_count} error{'s' if error_count != 1 else ''} found."
        else:
            return 2, f"🚨 Poor grammar. {error_count} error{'s' if error_count != 1 else ''} detected."

    def _get_vocabulary_score(self) -> Tuple[int, str]:
        """
        Calculate Type-Token Ratio (TTR) for vocabulary richness.
        
        TTR = unique_words / total_words
        
        Rubric ranges:
        - ≥0.7: 10 (excellent for speech)
        - 0.5-0.69: 8 (good)
        - 0.3-0.49: 6 (fair)
        - <0.3: 4 (repetitive)
        
        Returns:
            Tuple of (score, feedback_message)
        """
        if self.word_count == 0:
            return 0, "⚠️ No text to analyze."
        
        unique_words = len(set(word.lower() for word in self.words))
        ttr = unique_words / self.word_count
        
        # Note: High TTR (>0.7) is actually excellent for speech transcripts
        if ttr >= 0.7:
            return 10, f"✅ Excellent vocabulary (TTR: {ttr:.2f})."
        elif ttr >= 0.5:
            return 8, f"👍 Good vocabulary variety (TTR: {ttr:.2f})."
        elif ttr >= 0.3:
            return 6, f"⚠️ Moderate variety. Some repetitive usage (TTR: {ttr:.2f})."
        else:
            return 4, f"📉 Very repetitive language (TTR: {ttr:.2f})."

    def _get_clarity_score(self) -> Tuple[int, str]:
        """
        Calculate Filler Word Rate as percentage of total words.
        
        Rubric ranges (filler %):
        - 0-3%: 15 (excellent)
        - 4-6%: 12 (good)
        - 7-9%: 9 (fair)
        - 10-12%: 6 (poor)
        - >12%: 3 (very poor)
        
        Returns:
            Tuple of (score, feedback_message)
        """
        if self.word_count == 0:
            return 0, "⚠️ No text to analyze."
        
        text_lower = self.raw_text.lower()
        
        # Use pre-compiled regex for performance
        filler_matches = self._WORD_BOUNDARY_PATTERN.findall(text_lower)
        filler_count = len(filler_matches)
        
        rate = (filler_count / self.word_count) * 100
        
        # Map to rubric scores
        if rate <= 3:
            return 15, f"✅ Clear speech ({rate:.1f}% fillers, {filler_count} total)."
        elif rate <= 6:
            return 12, f"👍 Mostly clear ({rate:.1f}% fillers, {filler_count} total)."
        elif rate <= 9:
            return 9, f"⚠️ Noticeable fillers ({rate:.1f}%, {filler_count} instances)."
        elif rate <= 12:
            return 6, f"📉 Distracting fillers ({rate:.1f}%, {filler_count} instances)."
        else:
            return 3, f"🚨 Too many fillers ({rate:.1f}%, {filler_count} instances)."

    def calculate_all(self, sentiment_positive_prob: float) -> Dict[str, Any]:
        """
        Aggregate all criterion scores into final rubric result.
        
        Args:
            sentiment_positive_prob: Positivity score from AI model (0.0 to 1.0)
        
        Returns:
            Dictionary containing total_score and detailed breakdown by criterion
            
        Raises:
            ValueError: If sentiment probability is out of valid range
        """
        # Validate sentiment input
        if not 0.0 <= sentiment_positive_prob <= 1.0:
            raise ValueError(f"Sentiment probability must be 0-1, got {sentiment_positive_prob}")
        
        # Calculate individual criterion scores
        wpm_score, wpm_feed = self._get_wpm_score()
        content_score, content_feed = self._get_content_score()
        grammar_score, grammar_feed = self._get_grammar_score()
        vocab_score, vocab_feed = self._get_vocabulary_score()
        clarity_score, clarity_feed = self._get_clarity_score()
        
        # === Engagement Score (from AI model) ===
        # Rubric: ≥0.9(15), 0.7-0.89(12), 0.5-0.69(9), 0.3-0.49(6), <0.3(3)
        if sentiment_positive_prob >= 0.9:
            engage_score = 15
        elif sentiment_positive_prob >= 0.7:
            engage_score = 12
        elif sentiment_positive_prob >= 0.5:
            engage_score = 9
        elif sentiment_positive_prob >= 0.3:
            engage_score = 6
        else:
            engage_score = 3
        
        engage_feed = f"Positivity Score: {sentiment_positive_prob:.2f}"
        
        # === Calculate Total (Max 100) ===
        # Content(40) + Speech Rate(10) + Grammar(10) + Vocab(10) + Clarity(15) + Engagement(15)
        total_score = sum([
            min(content_score, 40),  # Safety cap
            wpm_score,
            grammar_score,
            vocab_score,
            clarity_score,
            engage_score
        ])
        
        return {
            "total_score": min(round(total_score), 100),  # Final safety cap
            "word_count": self.word_count,
            "duration_seconds": self.duration,
            "breakdown": {
                "Content & Structure": {
                    "score": min(content_score, 40),
                    "max": 40,
                    "feedback": content_feed
                },
                "Speech Rate": {
                    "score": wpm_score,
                    "max": 10,
                    "feedback": wpm_feed
                },
                "Grammar": {
                    "score": grammar_score,
                    "max": 10,
                    "feedback": grammar_feed
                },
                "Vocabulary": {
                    "score": vocab_score,
                    "max": 10,
                    "feedback": vocab_feed
                },
                "Clarity (Fillers)": {
                    "score": clarity_score,
                    "max": 15,
                    "feedback": clarity_feed
                },
                "Engagement": {
                    "score": engage_score,
                    "max": 15,
                    "feedback": engage_feed
                }
            }
        }


# === Utility function for standalone testing ===
def test_scoring_engine():
    """Quick test with sample transcript"""
    sample = """Hello everyone, myself Muskan, studying in class 8th B section from Christ Public School. 
    I am 13 years old. I live with my family. There are 3 people in my family, me, my mother and my father.
    One special thing about my family is that they are very kind hearted to everyone and soft spoken. 
    One thing I really enjoy is play, playing cricket and taking wickets.
    A fun fact about me is that I see in mirror and talk by myself. 
    One thing people don't know about me is that I once stole a toy from one of my cousin.
    My favorite subject is science because it is very interesting. 
    Through science I can explore the whole world and make the discoveries and improve the lives of others. 
    Thank you for listening."""
    
    engine = ScoringEngine(sample, duration_seconds=60)
    results = engine.calculate_all(sentiment_positive_prob=0.65)
    
    print(f"\n{'='*50}")
    print(f"Total Score: {results['total_score']}/100")
    print(f"{'='*50}")
    for criterion, data in results['breakdown'].items():
        print(f"{criterion}: {data['score']}/{data['max']}")
        print(f"  └─ {data['feedback']}\n")


if __name__ == "__main__":
    test_scoring_engine()