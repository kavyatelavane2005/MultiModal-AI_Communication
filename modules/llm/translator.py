import os
import base64
import json
import urllib.request
import urllib.error

class ASLTranslator:
    """Uses Groq API to translate a sequence of ASL glosses
    into a coherent, grammatically correct English sentence.
    Also supports vision-based refinement for the Ensemble mode.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.groq_api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        self.text_model = "qwen/qwen3.6-27b"
        self.vision_model = "qwen/qwen-vl-plus:free"

        self.text_system_prompt = (
            "You are an expert American Sign Language (ASL) translator. "
            "I will provide a sequence of predicted ASL glosses (words) from a visual AI model. "
            "The visual model has an accuracy of ~60%, so some words might be slightly misclassified "
            "(e.g., 'look' instead of 'book', 'fine' instead of 'five'). "
            "Your job is to fix any obvious visual misclassifications based on context, "
            "apply proper English grammar (ASL grammar is different, e.g. Time-Topic-Comment), "
            "and output ONLY the final conversational English sentence. "
            "Do not include any explanations or quotes, just the translated text."
        )

        self.general_system_prompt = (
            "You are a communication assistant. The user is sending a message "
            "in a multi-modal communication system for people with disabilities. "
            "Clean up any grammar issues, fix obvious speech recognition errors, "
            "and output ONLY the corrected, natural English sentence. "
            "If the input is already correct, return it as-is. "
            "Do not add explanations, just the text."
        )

        self.vision_system_prompt = (
            "You are an expert American Sign Language (ASL) recognition system. "
            "You will receive an image of a person performing an ASL sign, along with "
            "the top predictions from a skeleton-based AI model. "
            "Your job is to look at the hand shape, palm orientation, finger positions, "
            "and body posture in the image, and determine which of the given predictions "
            "is the most likely correct ASL sign. "
            "Respond with ONLY a JSON object in this exact format: "
            "{\"sign\": \"<word>\", \"confidence\": <0-100>, \"reason\": \"<brief reason>\"} "
            "Do not include markdown formatting or code blocks."
        )

    def setup(self, api_key):
        """Initialize the Groq client with the given API key."""
        self.api_key = api_key

    def _make_request(self, payload):
        """Helper to make REST API call to Groq."""
        if not self.api_key:
            raise ValueError("Groq API key not configured.")

        req = urllib.request.Request(
            self.groq_api_url,
            data=json.dumps(payload).encode('utf-8'),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['choices'][0]['message']['content']
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            raise Exception(f"HTTP {e.code}: {error_body}")
        except Exception as e:
            raise Exception(str(e))

    def translate(self, words_list):
        """Translate a list of ASL glosses into a sentence.
        
        Args:
            words_list (list): List of strings (glosses).
            
        Returns:
            str: Translated English sentence.
        """
        if not self.api_key:
            return "Error: Groq API key not configured."
            
        if not words_list:
            return ""

        input_text = " ".join(words_list)
        
        payload = {
            "model": self.text_model,
            "messages": [
                {"role": "system", "content": self.text_system_prompt},
                {"role": "user", "content": f"Glosses: [{input_text}]"}
            ],
            "temperature": 0.3
        }

        try:
            response_text = self._make_request(payload)
            return response_text.strip()
        except Exception as e:
            return f"Translation error: {str(e)}"

    def translate_general(self, text):
        """Process general text (speech/keyboard) through LLM for cleanup.
        
        Args:
            text (str): Input text from speech or keyboard.
            
        Returns:
            str: Cleaned/corrected English sentence.
        """
        if not self.api_key:
            return text  # Pass through if no API key
            
        if not text or not text.strip():
            return ""

        payload = {
            "model": self.text_model,
            "messages": [
                {"role": "system", "content": self.general_system_prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.2
        }

        try:
            response_text = self._make_request(payload)
            return response_text.strip()
        except Exception as e:
            return text  # Return original on error

    def refine_with_image(self, image_b64, top5_predictions):
        """Use Groq Vision to refine skeleton model predictions.
        
        Args:
            image_b64 (str): Base64-encoded JPEG image of the signer.
            top5_predictions (list): List of (gloss, confidence%) tuples.
            
        Returns:
            dict: {'sign': str, 'confidence': int, 'reason': str}
                  or {'error': str} on failure.
        """
        if not self.api_key:
            return {"error": "Groq API key not configured."}

        try:
            top5_str = "\n".join([f"  {i+1}. {g} ({c}%)" for i, (g, c) in enumerate(top5_predictions)])
            prompt = (
                f"Here is an image of a person performing an ASL sign.\n\n"
                f"The skeleton-based AI model predicted these top 5 possibilities:\n"
                f"{top5_str}\n\n"
                f"Based on the hand shape, finger positions, and palm orientation visible "
                f"in the image, which prediction is most likely correct? "
                f"You may also suggest a sign NOT in the list if you are very confident."
            )

            payload = {
                "model": self.vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.vision_system_prompt + "\n\n" + prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]
                    }
                ],
                "temperature": 0.2
            }

            response_text = self._make_request(payload)
            response_text = response_text.strip()
            
            # Clean markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            
            result = json.loads(response_text)
            return result

        except json.JSONDecodeError:
            return {"sign": response_text[:30], "confidence": 50, "reason": "Parsed from text"}
        except Exception as e:
            return {"error": str(e)}

