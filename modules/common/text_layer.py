"""Text Common Layer — Unified text representation from all input modalities.

Merges inputs from Sign Language Recognition (glosses), Speech-to-Text,
and direct Text input into a common text representation that can be
processed by the LLM for grammar correction and sentence formation.
"""


class TextCommonLayer:
    """Unifies all input modalities into text for LLM processing.
    
    This is the central convergence point in the architecture —
    all inputs (sign, speech, text) pass through here before
    being sent to the LLM for processing.
    """

    def __init__(self):
        self.word_buffer = []       # Accumulated words from sign detection
        self.pending_source = None  # Source of pending input

    def add_sign_gloss(self, gloss):
        """Add a recognized sign language gloss to the buffer.
        
        Args:
            gloss: The recognized ASL gloss (word).
        """
        self.word_buffer.append(gloss.lower())
        self.pending_source = 'sign'

    def get_sign_buffer(self):
        """Get the current sign word buffer contents."""
        return list(self.word_buffer)

    def flush_sign_buffer(self):
        """Flush and return the sign word buffer.
        
        Returns:
            tuple: (text, source) — the joined words and 'sign'.
        """
        if not self.word_buffer:
            return None, None
        text = ' '.join(self.word_buffer)
        self.word_buffer.clear()
        return text, 'sign'

    def process_speech_input(self, text):
        """Process speech-to-text input.
        
        Speech input is already natural language, so it just
        passes through with minimal processing.
        
        Args:
            text: Recognized speech text.
            
        Returns:
            tuple: (text, source) — cleaned text and 'speech'.
        """
        cleaned = text.strip()
        if not cleaned:
            return None, None
        return cleaned, 'speech'

    def process_text_input(self, text):
        """Process direct text (keyboard) input.
        
        Args:
            text: Typed text from the user.
            
        Returns:
            tuple: (text, source) — cleaned text and 'text'.
        """
        cleaned = text.strip()
        if not cleaned:
            return None, None
        return cleaned, 'text'

    def clear(self):
        """Clear the word buffer."""
        self.word_buffer.clear()
        self.pending_source = None
