"""Test Script: Text-to-Speech and Speech-to-Text modules."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_tts():
    """Test Text-to-Speech."""
    print("=" * 50)
    print("  Test: Text-to-Speech")
    print("=" * 50)

    from modules.tts.text_speaker import TextSpeaker

    tts = TextSpeaker()
    tts.list_voices()
    print()

    phrases = [
        "Hello! I am the multi-modal communication system.",
        "I can convert text to speech for blind users.",
        "The sign language recognition detected the word BOOK.",
    ]

    for i, phrase in enumerate(phrases):
        print(f"  [{i + 1}] Speaking: {phrase}")
        tts.speak(phrase)
        print(f"       Done.")

    tts.close()
    print("\n  TTS test complete!\n")


def test_stt():
    """Test Speech-to-Text."""
    print("=" * 50)
    print("  Test: Speech-to-Text")
    print("=" * 50)

    from modules.stt.speech_recognizer import SpeechRecognizer

    stt = SpeechRecognizer()
    stt.calibrate(duration=2)

    print("\n  Speak something (3 attempts, 5s timeout each):\n")

    for i in range(3):
        text = stt.listen(timeout=5, phrase_time_limit=10)
        if text:
            print(f"  [{i + 1}] You said: \"{text}\"\n")
        else:
            print(f"  [{i + 1}] (nothing detected)\n")

    stt.close()
    print("  STT test complete!\n")


def test_roundtrip():
    """Speak into mic -> text -> speak back."""
    print("=" * 50)
    print("  Test: Round-Trip (Speak -> Text -> Speak Back)")
    print("=" * 50)

    from modules.stt.speech_recognizer import SpeechRecognizer
    from modules.tts.text_speaker import TextSpeaker

    stt = SpeechRecognizer()
    tts = TextSpeaker()
    stt.calibrate(duration=2)

    print("\n  Speak something and I will repeat it back (3 attempts):\n")

    for i in range(3):
        text = stt.listen(timeout=5, phrase_time_limit=10)
        if text:
            print(f"  [{i + 1}] You said: \"{text}\"")
            response = f"You said: {text}"
            print(f"       Speaking back: \"{response}\"")
            tts.speak(response)
            print(f"       Done.\n")
        else:
            print(f"  [{i + 1}] (nothing detected)\n")

    stt.close()
    tts.close()
    print("  Round-trip test complete!\n")


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "tts"

    if mode == "tts":
        test_tts()
    elif mode == "stt":
        test_stt()
    elif mode == "roundtrip":
        test_roundtrip()
    elif mode == "all":
        test_tts()
        test_stt()
        test_roundtrip()
    else:
        print(f"Usage: python test_speech.py [tts|stt|roundtrip|all]")
        print(f"  Default: tts")
        test_tts()

    print("All tests done!")
