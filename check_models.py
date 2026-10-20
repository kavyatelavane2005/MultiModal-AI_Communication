import google.generativeai as genai
import os

genai.configure(api_key=os.environ.get('GEMINI_API_KEY', 'fake_key'))

try:
    models = genai.list_models()
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error: {e}")
