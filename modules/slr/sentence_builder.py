import time

class SentenceBuilder:
    """Collects individual predicted signs and groups them into sentences.
    
    If the user pauses signing for a specific duration (timeout),
    the collected words are flushed as a complete sentence ready
    for LLM translation.
    """
    def __init__(self, pause_threshold_sec=3.0):
        self.pause_threshold = pause_threshold_sec
        self.words = []
        self.last_update_time = time.time()

    def add_word(self, word):
        """Add a word to the current sentence sequence."""
        # Avoid immediate duplicates
        if not self.words or self.words[-1] != word:
            self.words.append(word)
        self.last_update_time = time.time()

    def check_flush(self):
        """Check if the user has paused long enough to flush the sentence.
        
        Returns:
            list: The sequence of words if flush is triggered, else None.
        """
        if not self.words:
            return None

        time_since_last = time.time() - self.last_update_time
        if time_since_last > self.pause_threshold:
            sentence = list(self.words)
            self.words = []
            return sentence
            
        return None

    def force_flush(self):
        """Force flush the sentence immediately."""
        if not self.words:
            return None
        sentence = list(self.words)
        self.words = []
        return sentence
