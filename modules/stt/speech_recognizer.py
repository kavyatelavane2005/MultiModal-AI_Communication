"""Speech-to-Text Module using Google Speech Recognition API.

Captures audio from the microphone and converts it to text
using Google's free Speech Recognition API.

Usage:
    from modules.stt import SpeechRecognizer

    stt = SpeechRecognizer()
    text = stt.listen()          # Blocks until speech is detected
    print(f"You said: {text}")

    # Or non-blocking with callback:
    stt.listen_in_background(callback=lambda text: print(text))
"""
import threading
import queue

import speech_recognition as sr


class SpeechRecognizer:
    """Converts spoken words to text using Google Speech API.
    
    Uses the SpeechRecognition library with Google's free API.
    Requires an internet connection.
    """

    def __init__(self, energy_threshold=300, pause_threshold=0.8,
                 language="en-US"):
        """
        Args:
            energy_threshold: Minimum audio energy to consider as speech.
                              Lower = more sensitive. Default 300.
            pause_threshold: Seconds of silence before considering speech
                             complete. Default 0.8.
            language: Language code for recognition. Default "en-US".
        """
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.dynamic_energy_threshold = True
        self.language = language

        # For background listening
        self._stop_listening = None
        self._text_queue = queue.Queue()
        self._callback = None

        # Test microphone availability
        try:
            self.microphone = sr.Microphone()
            print("Microphone initialized successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Could not access microphone: {e}\n"
                f"Make sure a microphone is connected and PyAudio is installed."
            )

    def calibrate(self, duration=2):
        """Calibrate the recognizer for ambient noise.
        
        Call this once before listening to adjust for background noise.
        
        Args:
            duration: Seconds to listen for ambient noise. Default 2.
        """
        print(f"Calibrating for ambient noise ({duration}s)...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=duration)
        print(f"Energy threshold set to: {self.recognizer.energy_threshold:.0f}")

    def listen(self, timeout=None, phrase_time_limit=10):
        """Listen for speech and return the recognized text.
        
        This is a BLOCKING call — it waits until speech is detected.
        
        Args:
            timeout: Max seconds to wait for speech to start. None = wait forever.
            phrase_time_limit: Max seconds for a single phrase. Default 10.
        
        Returns:
            str: The recognized text, or None if recognition failed.
        """
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            
            print("Recognizing...")
            text = self.recognizer.recognize_google(audio, language=self.language)
            return text

        except sr.WaitTimeoutError:
            print("Listening timed out — no speech detected.")
            return None
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition API error: {e}")
            print("Make sure you have an internet connection.")
            return None

    def listen_in_background(self, callback=None, phrase_time_limit=10):
        """Start listening for speech in a background thread.
        
        Args:
            callback: Function to call with recognized text.
                      Signature: callback(text: str)
            phrase_time_limit: Max seconds for a single phrase. Default 10.
        """
        self._callback = callback

        def _bg_callback(recognizer, audio):
            try:
                text = recognizer.recognize_google(audio, language=self.language)
                self._text_queue.put(text)
                if self._callback:
                    self._callback(text)
            except sr.UnknownValueError:
                pass  # Couldn't understand — skip
            except sr.RequestError as e:
                print(f"Speech API error: {e}")

        self._stop_listening = self.recognizer.listen_in_background(
            self.microphone,
            _bg_callback,
            phrase_time_limit=phrase_time_limit
        )
        print("Background listening started.")

    def stop_background_listening(self, wait=True):
        """Stop the background listening thread."""
        if self._stop_listening:
            self._stop_listening(wait_for_stop=wait)
            self._stop_listening = None
            print("Background listening stopped.")

    def get_queued_text(self):
        """Get any text recognized in the background (non-blocking).
        
        Returns:
            str or None: The next recognized text, or None if queue is empty.
        """
        try:
            return self._text_queue.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        """Clean up resources."""
        self.stop_background_listening(wait=False)


# ---- Standalone test ----
if __name__ == '__main__':
    print("=" * 50)
    print("  Speech-to-Text Test")
    print("=" * 50)
    print()

    stt = SpeechRecognizer()
    stt.calibrate(duration=2)

    print()
    print("Speak something (3 attempts):")
    print()

    for i in range(3):
        text = stt.listen(timeout=5, phrase_time_limit=10)
        if text:
            print(f"  [{i+1}] You said: \"{text}\"")
        else:
            print(f"  [{i+1}] (nothing detected)")
        print()

    stt.close()
    print("Done.")
