
# 📚 ResearchMate AI

ResearchMate AI is a Generative AI-powered research paper assistant. Users can upload a research paper in PDF format and ask questions about its content.

The application uses Google Gemini to generate answers based on relevant passages retrieved from the uploaded paper. It also provides PDF page references to help users verify the information.

## Features

- Upload and read research paper PDFs
- Ask questions in simple English
- Get AI-generated answers using Google Gemini
- Retrieve relevant passages using semantic search
- View source page references
- Simple web interface built with Gradio

## Tech Stack

- Python
- Generative AI and Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Google Gemini API
- Sentence Transformers
- NumPy
- PyMuPDF
- Gradio

## Run Locally

1. Clone this repository.
2. Install the required packages:

   `pip install -r requirements.txt`

3. Set the `GEMINI_API_KEY` environment variable with your own Gemini API key.
4. Start the application:

   `python app.py`

## Developer

**Harshita Mishra**

- [LinkedIn](https://www.linkedin.com/in/harshita-m-910149265/)
- [GitHub](https://github.com/Harshita0412)

## Note

A valid Gemini API key is required. API usage limits apply. The application is a prototype and may not answer every question correctly, so verify important information against the original research paper.
