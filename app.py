import gradio as gr
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
import google.generativeai as genai
from huggingface_hub import InferenceClient
import uuid
from PIL import Image
import io
import re
import pandas as pd
import matplotlib.pyplot as plt
import camelot

# --- New Imports for RAG & Data Viz ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- SAFE SETUP ---
load_dotenv()

# Configure Google AI API
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("Google AI API key not found.")
genai.configure(api_key=google_api_key)

# Configure Hugging Face API
hf_token = os.getenv("HUGGING_FACE_TOKEN")
if not hf_token:
    print("Warning: Hugging Face token not found. Audio and Image functions will not work.")
    hf_client = None
else:
    hf_client = InferenceClient(token=hf_token)


# --- CORE FUNCTIONS ---

def get_gemini_response_stream(prompt, history):
    """Gets a streaming text response from the Gemini model with conversation history."""
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        chat = model.start_chat(history=history)
        response = chat.send_message(prompt, stream=True)
        
        full_response = ""
        for chunk in response:
            if hasattr(chunk, 'text'):
                full_response += chunk.text
                yield full_response
    except Exception as e:
        yield f"An error occurred with Gemini: {e}"

def transcribe_audio_hf(audio_file_path):
    """Transcribes audio using a Hugging Face Whisper model."""
    if not hf_client: return "Hugging Face token is missing."
    if not audio_file_path: return "Please upload an audio file first."
    try:
        # --- The Fix ---
        # Using a very small, fast, and stable model
        response = hf_client.automatic_speech_recognition(
            audio=audio_file_path, 
            model="openai/whisper-tiny.en"
        )
        return response['text']
    except Exception as e:
        return f"An error occurred during transcription: {e}"

def generate_image_hf(prompt):
    """Generates an image using a Hugging Face Stable Diffusion model."""
    if not hf_client: return "Hugging Face token is missing."
    if not prompt: return None
    try:
        image_response = hf_client.text_to_image(prompt=prompt, model="stabilityai/stable-diffusion-xl-base-1.0")
        filename = f"generated_image_{uuid.uuid4().hex}.png"
        image_response.save(filename)
        return filename
    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        return None

# Global variables to hold the processed PDF data
vector_store_retriever = None
extracted_tables = []

def process_pdf_for_rag_and_data(pdf_file_obj):
    """Processes the uploaded PDF for both RAG and data extraction."""
    global vector_store_retriever, extracted_tables
    if not pdf_file_obj:
        return "Please upload a PDF first.", gr.update(interactive=False)
    
    try:
        text = "".join(page.get_text() for page in fitz.open(pdf_file_obj.name))
        if text.strip():
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = Chroma.from_texts(chunks, embedding_model)
            vector_store_retriever = vector_store.as_retriever()
        
        tables = camelot.read_pdf(pdf_file_obj.name, pages='all', flavor='lattice')
        extracted_tables = [table.df for table in tables]
        
        status_message = f"PDF processed. Found {len(extracted_tables)} tables. You can now ask questions."
        return status_message, gr.update(interactive=True)
    
    except Exception as e:
        return f"Failed to process PDF: {e}", gr.update(interactive=False)

def get_rag_or_viz_response(history):
    """Determines whether to give a text answer (RAG) or a visualization."""
    global vector_store_retriever, extracted_tables
    question = history[-1][0]

    if not vector_store_retriever and not extracted_tables:
        history[-1][1] = "Please upload and process a PDF file first."
        return history

    viz_keywords = ['chart', 'plot', 'graph', 'visualize', 'bar', 'pie', 'line']
    if any(keyword in question.lower() for keyword in viz_keywords) and extracted_tables:
        try:
            table_summaries = "\n".join([f"Table {i+1}:\n{df.head().to_string()}" for i, df in enumerate(extracted_tables)])
            
            prompt = f"""
            You are a data analyst. Based on the following user question and the provided table data, write Python code using Matplotlib to generate a chart that answers the question.
            The data for the tables is provided as pandas DataFrames in a list called `dfs`. You can access the first table with `dfs[0]`, the second with `dfs[1]`, etc.
            
            User Question: "{question}"
            
            Available Table Summaries:
            {table_summaries}
            
            Rules:
            - ONLY write Python code. Do not include any explanation or markdown formatting.
            - Assume the data is in a list of DataFrames called `dfs`.
            - Save the plot to a file named 'plot.png'.
            - Use `plt.tight_layout()` before saving.
            - Example of good code:
              import pandas as pd
              import matplotlib.pyplot as plt
              plt.figure(figsize=(10, 6))
              dfs[0]['Sales'].plot(kind='bar')
              plt.title('Sales Figures')
              plt.ylabel('Amount')
              plt.xticks(rotation=45)
              plt.tight_layout()
              plt.savefig('plot.png')
            """
            
            model = genai.GenerativeModel('gemini-2.5-pro')
            response = model.generate_content(prompt)
            python_code = response.text.strip().replace('`python', '').replace('`', '')
            
            exec_globals = {'dfs': extracted_tables, 'pd': pd, 'plt': plt}
            exec(python_code, exec_globals)
            
            history[-1][1] = (f"Generated plot for: '{question}'", 'plot.png')

        except Exception as e:
            history[-1][1] = f"Sorry, I failed to generate the visualization. Error: {e}"
        
        return history

    else:
        try:
            relevant_docs = vector_store_retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            prompt = f"Based ONLY on the following context, answer the question. If the answer is not in the context, say so.\n\nContext:\n{context}\n\nQuestion: {question}"

            model = genai.GenerativeModel('gemini-2.5-pro')
            response = model.generate_content(prompt)
            history[-1][1] = response.text
        except Exception as e:
            history[-1][1] = f"An error occurred: {e}"
            
        return history

def transcribe_and_respond(audio_file_path):
    """Transcribes audio and then gets a response from Gemini."""
    if not audio_file_path:
        return "Please record or upload audio first.", "No input provided."

    transcribed_text = transcribe_audio_hf(audio_file_path)
    if "error" in str(transcribed_text).lower():
        return transcribed_text, "Could not get an answer due to transcription error."
    
    model = genai.GenerativeModel('gemini-2.5-pro')
    response = model.generate_content(transcribed_text)
    
    return transcribed_text, response.text

# --- GRADIO USER INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Multipurpose GenAI Chatbot 🚀")
    gr.Markdown("Powered by Google Gemini and Hugging Face.")

    with gr.Tab("📝 Text Chatbot (with Memory)"):
        gr.Markdown("This chatbot remembers the conversation. Try asking follow-up questions!")
        chatbot_ui = gr.Chatbot(label="Conversation", height=500)
        chat_history_state = gr.State(value=[])
        chatbot_input = gr.Textbox(label="Your Question")
        
        def stream_response(prompt, history):
            history.append({'role': 'user', 'parts': [prompt]})
            display_history = [[msg['parts'][0], ""] if msg['role'] == 'user' else [None, msg['parts'][0]] for msg in history]
            
            response_generator = get_gemini_response_stream(prompt, history)
            bot_response = ""
            for partial_response in response_generator:
                bot_response = partial_response
                display_history[-1][1] = bot_response
                yield display_history, ""
            history.append({'role': 'model', 'parts': [bot_response]})

        chatbot_input.submit(fn=stream_response, inputs=[chatbot_input, chat_history_state], outputs=[chatbot_ui, chatbot_input])

    with gr.Tab("🎤 Audio Q&A"):
        gr.Markdown("Record or upload audio to get both the transcription and an AI-powered answer.")
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Your Spoken Question or Command")
        
        with gr.Row():
            transcription_output = gr.Textbox(label="Transcription Result", lines=5)
            answer_output = gr.Textbox(label="AI Answer", lines=5)
            
        gr.Button("Submit Audio", variant="primary").click(
            fn=transcribe_and_respond,
            inputs=audio_input,
            outputs=[transcription_output, answer_output]
        )

    with gr.Tab("🎨 Image Generation"):
        img_input = gr.Textbox(label="Describe the image you want to create", lines=3)
        img_output = gr.Image(label="Generated Image (Stable Diffusion)")
        gr.Button("Generate", variant="primary").click(fn=generate_image_hf, inputs=img_input, outputs=img_output)
    
    with gr.Tab("📄 PDF Q&A and Visualization"):
        gr.Markdown("Upload a PDF to ask text questions or request charts from its tables.")
        
        with gr.Row():
            pdf_input = gr.File(label="Upload your PDF")
            process_status = gr.Textbox(label="Status", interactive=False)
        
        process_btn = gr.Button("Process PDF", variant="secondary")

        pdf_chatbot_ui = gr.Chatbot(label="Ask Questions about the PDF", height=400, render_markdown=True)
        pdf_question_input = gr.Textbox(label="Your Question", interactive=False)
        
        process_btn.click(
            fn=process_pdf_for_rag_and_data,
            inputs=pdf_input,
            outputs=[process_status, pdf_question_input]
        )
        
        def add_text_to_pdf_history(text, history):
            return history + [[text, None]]

        pdf_question_input.submit(
            fn=add_text_to_pdf_history,
            inputs=[pdf_question_input, pdf_chatbot_ui],
            outputs=pdf_chatbot_ui,
        ).then(
            fn=get_rag_or_viz_response,
            inputs=pdf_chatbot_ui,
            outputs=pdf_chatbot_ui
        ).then(
            lambda: "",
            None,
            pdf_question_input
        )

demo.launch()