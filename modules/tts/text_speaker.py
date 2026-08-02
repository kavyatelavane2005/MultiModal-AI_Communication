"""Text-to-Speech Module using pyttsx3.

Converts text to spoken audio through the system speakers.
Works completely offline using the system's built-in TTS engine.
"""
import threading
import queue

import pyttsx3


class TextSpeaker:
    """Converts text to speech using the system TTS engine."""

    def __init__(self, rate=170, volume=0.9):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

        voices = self.engine.getProperty('voices')
        self._voices = voices

        if voices:
            self.engine.setProperty('voice', voices[0].id)
            print(f"TTS initialized with voice: {voices[0].name}")

        self._speech_queue = queue.Queue()
        self._worker_thread = None
        self._running = False

    def speak(self, text, block=True):
        """Convert text to speech.

        Args:
            text: The text string to speak.
            block: If True, waits until speech finishes.
        """
        if not text or not text.strip():
            return

        if block:
            self._speak_sync(text)
        else:
            self._speech_queue.put(text)
            self._ensure_worker_running()

    def _speak_sync(self, text):
        """Speak text synchronously (blocks until done)."""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except RuntimeError:
            self.engine = pyttsx3.init()
            self.engine.say(text)
            self.engine.runAndWait()

    def _ensure_worker_running(self):
        """Start the background speech worker if not running."""
        if not self._running:
            self._running = True
            self._worker_thread = threading.Thread(
                target=self._speech_worker, daemon=True
            )
            self._worker_thread.start()

    def _speech_worker(self):
        """Background thread that processes the speech queue."""
        engine = pyttsx3.init()
        engine.setProperty('rate', self.engine.getProperty('rate'))
        engine.setProperty('volume', self.engine.getProperty('volume'))

        while self._running:
            try:
                text = self._speech_queue.get(timeout=1.0)
                engine.say(text)
                engine.runAndWait()
                self._speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS error: {e}")

    def set_rate(self, rate):
        """Set speech rate in words per minute."""
        self.engine.setProperty('rate', rate)

    def set_volume(self, volume):
        """Set volume level (0.0 to 1.0)."""
        self.engine.setProperty('volume', max(0.0, min(1.0, volume)))

    def list_voices(self):
        """Print all available TTS voices."""
        print("Available voices:")
        for i, voice in enumerate(self._voices):
            print(f"  [{i}] {voice.name} (ID: {voice.id})")

    def stop(self):
        """Stop any ongoing speech."""
        try:
            self.engine.stop()
        except Exception:
            pass

    def close(self):
        """Clean up resources."""
        self._running = False
        self.stop()


if __name__ == '__main__':
    print("=" * 50)
    print("  Text-to-Speech Test")
    print("=" * 50)

    tts = TextSpeaker()
    tts.list_voices()

    phrases = [
        "Hello! The sign language recognition system is working.",
        "I detected the sign for BOOK with 97 percent confidence.",
        "This is a multi-modal communication system.",
    ]

    for i, phrase in enumerate(phrases):
        print(f"  [{i+1}] Speaking: {phrase}")
        tts.speak(phrase)
        print(f"       Done.")

    tts.close()
    print("Done.")
