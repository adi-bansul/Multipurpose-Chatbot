Here is a **clean, professional, and complete README.md** for your project based on the functionality in your uploaded files (`app.py`, `chatbot.py`, `check_models.py`).
You can directly copy-paste this into **README.md**.

---

# 🚀 Multipurpose GenAI Chatbot

A powerful **multi-modal** chatbot application built using **Gradio**, **Google Gemini**, **Hugging Face models**, and **OpenAI APIs**, supporting:

* 📝 Text Chat with memory
* 🎤 Audio Q&A (Speech → Text → AI Response)
* 🎨 Image Generation (Stable Diffusion)
* 📄 PDF Question Answering (RAG) + Visualizations
* 🔎 Model Checker (for Google Gemini)

---

## 📌 Features

### **1. Text Chatbot with Persistent Conversation Memory**

* Powered by **Gemini 2.5 Pro**
* Streams responses in real time
* Maintains context using conversation history

### **2. Audio Q&A**

* Upload or record audio
* Transcribes audio using Hugging Face **Whisper**
* Gemini answers the transcribed question

### **3. Image Generation**

* Uses **Stable Diffusion XL** (`stabilityai/stable-diffusion-xl-base-1.0`)
* Generates high-quality images from text prompts

### **4. PDF Q&A with RAG + Data Visualization**

* Extracts text & tables automatically
* Builds a Chroma vector store for contextual RAG
* Automatically generates **Matplotlib visualizations** based on PDF tables
* Supports charts like bar, line, pie, etc.

### **5. Model Checker**

List all available Google Gemini models that support `generateContent`.

---

# 🛠️ Tech Stack

### **Backend / Core**

* Python
* Gradio

### **AI / ML**

* Google Gemini API
* Hugging Face Whisper (Audio)
* Stable Diffusion XL (Image generation)
* LangChain + ChromaDB (RAG)
* Camelot (PDF table extraction)
* PyMuPDF (PDF text extraction)
* Matplotlib / Pandas (Data visualization)

---

# 📂 Project Structure

```
📦 GenAI-Multipurpose-Chatbot
├── app.py               # Main multipurpose Gradio app
├── chatbot.py           # Test script for OpenAI models
├── check_models.py      # Check available Gemini models
├── requirements.txt     # Required dependencies
├── README.md            # Project documentation
└── .env                 # API keys (not included in repo)
```

---

# 🔧 Setup & Installation

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/genai-multipurpose-chatbot.git
cd genai-multipurpose-chatbot
```

### **2. Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### **3. Install Dependencies**

Create a `requirements.txt` (example below):

```
gradio
fitz
pymupdf
python-dotenv
google-generativeai
huggingface_hub
Pillow
pandas
matplotlib
camelot-py
langchain
chromadb
sentence-transformers
```

Then install:

```bash
pip install -r requirements.txt
```

### **4. Add API Keys**

Create a **.env** file in the root directory:

```
GOOGLE_API_KEY=your_google_key
HUGGING_FACE_TOKEN=your_hf_token
OPENAI_API_KEY=your_openai_key
```

---

# ▶️ Running the App

### **Start the Gradio UI**

```bash
python app.py
```

The application will launch at:

```
http://127.0.0.1:7860
```

---

# 🧪 Additional Scripts

### **Run OpenAI test functions**

```bash
python chatbot.py
```

### **List available Gemini models**

```bash
python check_models.py
```

---

# 📘 Usage Instructions

### **Text Chatbot**

* Enter a question
* The AI responds with memory-aware answers

### **Audio Q&A**

* Upload or record audio
* The app transcribes speech using Whisper
* Gemini responds to the transcription

### **Image Generation**

* Enter a prompt
* View Stable Diffusion XL generated image

### **PDF Q&A + Visualization**

1. Upload a PDF
2. Click **Process PDF**
3. Ask questions like:

   * “Summarize section 2”
   * “Create a bar chart of sales from the table”
4. Visualizations are saved as `plot.png`

---

# 📄 License

This project is released under the **MIT License**.
