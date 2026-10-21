"""Gemini-backed sentence formation for the sign communication pipeline."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from dotenv import load_dotenv

load_dotenv()


class ASLTranslator:
    """Translate recognized ASL glosses and clean up typed or spoken text."""

    def __init__(self, api_key=None):
        self.api_key = (
            api_key if api_key is not None
            else os.getenv("GEMINI_API_KEY")
        )
        self.last_error = None
        self.last_error_code = None
        self.last_error_reason = None
        self.api_base = "https://generativelanguage.googleapis.com/v1beta"
        self.text_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.vision_model = os.getenv("GEMINI_VISION_MODEL", self.text_model)

        self.text_system_prompt = (
            "Turn an ordered sequence of predicted American Sign Language (ASL) glosses "
            "into one concise, natural English sentence. The glosses are uncertain labels "
            "from a visual model, not a complete English sentence. Preserve their supported "
            "meaning; reorder or add only small grammatical words needed for readable English. "
            "Do not invent details or replace a gloss with a different sign based only on a guess. "
            "Return only the sentence, with no explanation or quotation marks."
        )

        self.general_system_prompt = (
            "You are a communication assistant. Clean up grammar and obvious speech recognition "
            "errors in the user's message. Return only the corrected, natural English sentence. "
            "If the input is already correct, return it as-is. Do not add explanations."
        )

        self.vision_system_prompt = (
            "You are helping review a visual sign recognition result. Compare the visible hand "
            "shape, palm orientation, finger positions, and body posture with the provided model "
            "predictions. Choose a listed prediction unless another sign is clearly more likely. "
            'Return only JSON in this format: {"sign":"word","confidence":0,"reason":"brief reason"}.'
        )

    def _record_error(self, error):
        """Store only the HTTP status and safe provider error category."""
        if isinstance(error, urllib.error.HTTPError):
            self.last_error_code = error.code
            self.last_error_reason = getattr(error, "gemini_error_reason", None)
            label = f"HTTP {error.code}"
        elif isinstance(error, TimeoutError) or (
            isinstance(error, urllib.error.URLError)
            and isinstance(error.reason, TimeoutError)
        ):
            self.last_error_code = "timeout"
            label = "request timed out"
        elif isinstance(error, urllib.error.URLError):
            self.last_error_code = "network"
            label = "network error"
        else:
            self.last_error_code = "request"
            label = "request error"

        self.last_error = label
        print(f"[LLM Translator] Gemini {label}")

    def _make_request(self, payload, model=None):
        """Call Gemini generateContent and return concatenated text output."""
        if not self.api_key:
            raise ValueError("Gemini API key is not configured.")

        model_name = urllib.parse.quote(model or self.text_model, safe="-_.")
        endpoint = f"{self.api_base}/models/{model_name}:generateContent"
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))

            text_parts = []
            for candidate in result.get("candidates", []):
                content = candidate.get("content", {})
                for part in content.get("parts", []):
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)

            if not text_parts:
                raise ValueError("Gemini response contained no text output.")
            return "".join(text_parts)
        except urllib.error.HTTPError as error:
            # Parse only the provider's stable status label; never log or retain
            # the full response body, which may contain request details.
            try:
                body = error.read(65536)
                details = json.loads(body.decode("utf-8")).get("error", {})
                if isinstance(details, dict):
                    error.gemini_error_reason = details.get("status") or details.get("reason")
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, OSError):
                pass
            error.close()
            raise

    @staticmethod
    def _payload(system_prompt, user_text, max_output_tokens=128):
        return {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": max_output_tokens,
            },
        }

    def translate(self, words_list):
        """Turn an ordered list of recognized sign glosses into a sentence."""
        if not words_list:
            return ""

        self.last_error = None
        self.last_error_code = None
        self.last_error_reason = None
        fallback_text = " ".join(words_list).title()
        if not self.api_key:
            return fallback_text

        payload = self._payload(
            self.text_system_prompt,
            f"ASL glosses: [{ ' '.join(words_list) }]",
            max_output_tokens=128,
        )
        try:
            return self._make_request(payload).strip()
        except Exception as error:
            self._record_error(error)
            return fallback_text

    def translate_general(self, text):
        """Clean up a message entered by keyboard or speech recognition."""
        if not text or not text.strip():
            return ""

        self.last_error = None
        self.last_error_code = None
        self.last_error_reason = None
        if not self.api_key:
            return text

        payload = self._payload(
            self.general_system_prompt,
            text,
            max_output_tokens=256,
        )
        try:
            return self._make_request(payload).strip()
        except Exception as error:
            self._record_error(error)
            return text

    def refine_with_image(self, image_b64, top5_predictions):
        """Use Gemini vision to review the SPOTER model's top sign predictions."""
        if not self.api_key:
            return {"error": "Gemini API key is not configured."}

        top5_text = "\n".join(
            f"{index + 1}. {gloss} ({confidence}%)"
            for index, (gloss, confidence) in enumerate(top5_predictions)
        )
        prompt = (
            "Review this image of a person signing. The skeleton model's top predictions are:\n"
            f"{top5_text}\n\n"
            "Based on the visible hand shape, finger positions, palm orientation, and body posture, "
            "select the most likely prediction. Suggest an unlisted sign only if clearly supported."
        )
        payload = {
            "systemInstruction": {"parts": [{"text": self.vision_system_prompt}]},
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/jpeg", "data": image_b64}},
                ],
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 256,
                "responseMimeType": "application/json",
            },
        }

        try:
            response_text = self._make_request(payload, model=self.vision_model).strip()
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {
                "sign": response_text[:30],
                "confidence": 50,
                "reason": "Parsed from text",
            }
        except Exception as error:
            self._record_error(error)
            return {"error": self.last_error or "Gemini request failed."}
