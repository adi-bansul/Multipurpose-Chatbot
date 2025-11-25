import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# This loop will print every available model that supports the 'generateContent' method
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)