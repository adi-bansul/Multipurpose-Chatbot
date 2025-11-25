import openai
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv

# --- SETUP ---
# Load environment variables from a .env file
load_dotenv()

# Set the API key safely
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
openai.api_key = api_key

# --- CORE FUNCTIONS ---

def get_text_response(prompt):
    """Gets a text response from the GPT model."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

def transcribe_audio(audio_file_path):
    """Transcribes an audio file using Whisper."""
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcription.text
    except FileNotFoundError:
        return f"Error: The file '{audio_file_path}' was not found."
    except Exception as e:
        return f"An error occurred during transcription: {e}"

def generate_image(prompt):
    """Generates an image using DALL-E 3."""
    try:
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        return response.data[0].url
    except Exception as e:
        return f"An error occurred during image generation: {e}"

def extract_text_from_pdf(pdf_file_path):
    """Extracts text from a PDF file."""
    try:
        doc = fitz.open(pdf_file_path)
        text = "".join(page.get_text() for page in doc)
        return text
    except FileNotFoundError:
        return f"Error: The file '{pdf_file_path}' was not found."
    except Exception as e:
        return f"Error reading PDF: {e}"

# --- EXAMPLE USAGE ---
# This block will only run when you execute `python chatbot.py` directly
if __name__ == "__main__":
    print("--- 1. Testing Text-to-Text ---")
    user_question = "What is the capital of France?"
    answer = get_text_response(user_question)
    print(f"User: {user_question}")
    print(f"Bot: {answer}\n")

    print("--- 2. Testing Audio-to-Text ---")
    transcribed_text = transcribe_audio("my_audio.mp3")
    print(f"Transcribed Text: {transcribed_text}\n")

    print("--- 3. Testing Image Generation ---")
    image_prompt = "A cute, fluffy cat programming on a laptop, digital art style"
    generated_image_link = generate_image(image_prompt)
    print(f"Image prompt: {image_prompt}")
    print(f"Generated Image URL: {generated_image_link}\n")

    print("--- 4. Testing PDF-to-Text and Summarization ---")
    pdf_text = extract_text_from_pdf("my_document.pdf")
    if "Error" not in pdf_text:
        summary_prompt = f"Please summarize the following document in three key points:\n\n{pdf_text[:4000]}"
        summary = get_text_response(summary_prompt)
        print(f"PDF Summary:\n{summary}")
    else:
        print(pdf_text)