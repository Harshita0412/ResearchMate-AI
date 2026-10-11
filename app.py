
import os
import gradio as gr
import pymupdf
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai

# ------------------------------
# CONFIGURATION
# ------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Add it in your hosting platform's Secrets."
    )

client = genai.Client(api_key=GEMINI_API_KEY)

print("Loading ResearchMate AI search model...")
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
print("ResearchMate AI is ready!")

# ------------------------------
# PDF PROCESSING
# ------------------------------

def process_pdf(pdf_file):
    if pdf_file is None:
        return None, "Please upload a PDF first."

    try:
        pages = []

        with pymupdf.open(pdf_file) as doc:
            for number, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append({"page": number, "text": text})

        if not pages:
            return None, "No readable text found. Try a text-based PDF."

        chunks = []
        for page in pages:
            text = page["text"]
            for start in range(0, len(text), 850):
                piece = text[start:start + 1000]
                if piece.strip():
                    chunks.append({
                        "page": page["page"],
                        "text": piece
                    })
                if start + 1000 >= len(text):
                    break

        embeddings = embedding_model.encode(
            [item["text"] for item in chunks],
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return {
            "chunks": chunks,
            "embeddings": embeddings
        }, f"✅ PDF ready! {len(pages)} pages and {len(chunks)} passages."

    except Exception as e:
        return None, f"PDF error: {type(e).__name__}: {e}"

# ------------------------------
# AI QUESTION ANSWERING (RAG)
# ------------------------------

def answer_question(question, paper_data, history):
    history = history or []

    if paper_data is None:
        return "Please upload and process a PDF first.", history

    if not question or not question.strip():
        return "Please enter a question.", history

    try:
        chunks = paper_data["chunks"]

        query_vector = embedding_model.encode(
            [question.strip()],
            normalize_embeddings=True
        )[0]

        scores = paper_data["embeddings"] @ query_vector
        best_indices = np.argsort(scores)[::-1][:5]

        context = "\n\n".join(
            f"PDF page {chunks[i]['page']}:\n{chunks[i]['text']}"
            for i in best_indices
        )

        prompt = f"""
You are ResearchMate AI, a research-paper assistant.

Answer using only the supplied paper passages.
Use clear, simple English.
Cite the relevant PDF page numbers.
Never invent facts.
If the passages do not answer the question, say so.

PAPER PASSAGES:
{context}

QUESTION:
{question.strip()}
"""

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        answer = (
            f"{response.text}\n\n"
            f"**Retrieved source pages:** "
            f"{sorted({chunks[i]['page'] for i in best_indices})}"
        )

        updated_history = history + [
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": answer}
        ]

        return answer, updated_history

    except Exception as e:
        return f"Answer error: {type(e).__name__}: {e}", history

# ------------------------------
# WEBSITE DESIGN
# ------------------------------

custom_css = """
.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
    font-family: Inter, sans-serif !important;
}
#hero {
    text-align: center;
    padding: 28px 10px 20px;
}
#hero h1 {
    font-size: 38px;
    font-weight: 800;
    letter-spacing: -1px;
}
#hero p {
    font-size: 16px;
    opacity: 0.75;
}
#paper-panel, #chat-panel {
    border: 1px solid #e6e2f5 !important;
    border-radius: 18px !important;
    padding: 20px !important;
}
#process-btn, #ask-btn {
    background: #7051f5 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}
#footer {
    text-align: center;
    padding: 24px 10px;
    margin-top: 24px;
    border-top: 1px solid #e6e2f5;
}
#footer a {
    color: #7051f5;
    text-decoration: none;
    font-weight: 600;
    margin: 0 10px;
}
"""

with gr.Blocks(title="ResearchMate AI | Research Assistant") as demo:

    gr.HTML("""
    <div id="hero">
        <h1>📚 ResearchMate AI</h1>
        <p>Your intelligent companion for understanding research papers.</p>
        <p>Upload a paper. Explore the evidence. Ask better questions.</p>
    </div>
    """)

    paper_state = gr.State(value=None)
    chat_history = gr.State(value=[])

    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            with gr.Group(elem_id="paper-panel"):
                gr.Markdown("## 📄 Your research paper")
                gr.Markdown("Upload a PDF to get started.")

                pdf_upload = gr.File(
                    label="Upload research paper",
                    file_types=[".pdf"],
                    type="filepath"
                )

                process_button = gr.Button(
                    "✨ Process my paper",
                    elem_id="process-btn"
                )

                upload_status = gr.Markdown("Waiting for your PDF...")

                gr.Markdown("---")
                gr.Markdown("### 💡 Suggested questions")

                suggestions = [
                    "Summarize this paper",
                    "Explain the methodology",
                    "What are the key findings?",
                    "What are the limitations?"
                ]

                suggestion_buttons = [
                    gr.Button(s, size="sm") for s in suggestions
                ]

        with gr.Column(scale=2, min_width=320):
            with gr.Group(elem_id="chat-panel"):
                gr.Markdown("## 💬 Ask ResearchMate")

                question_box = gr.Textbox(
                    label="Your question",
                    placeholder="What exactly does this paper say?",
                    lines=3
                )

                ask_button = gr.Button(
                    "Ask ResearchMate →",
                    elem_id="ask-btn"
                )

                answer_box = gr.Markdown(
                    "Upload and process a paper to begin."
                )

                history_display = gr.Chatbot(height=280)

    gr.HTML("""
    <div id="footer">
        <p>Built with ❤️ by <strong>Harshita Mishra</strong></p>
        <p>
            <a href="https://www.linkedin.com/in/harshita-m-910149265/"
               target="_blank" rel="noopener noreferrer">LinkedIn ↗</a>
            ·
            <a href="https://github.com/Harshita0412"
               target="_blank" rel="noopener noreferrer">GitHub ↗</a>
        </p>
        <p>Research smarter. Discover more.</p>
    </div>
    """)

    process_button.click(
        fn=process_pdf,
        inputs=[pdf_upload],
        outputs=[paper_state, upload_status]
    )

    ask_button.click(
        fn=answer_question,
        inputs=[question_box, paper_state, chat_history],
        outputs=[answer_box, chat_history]
    ).then(
        fn=lambda h: h,
        inputs=[chat_history],
        outputs=[history_display]
    )

    question_box.submit(
        fn=answer_question,
        inputs=[question_box, paper_state, chat_history],
        outputs=[answer_box, chat_history]
    ).then(
        fn=lambda h: h,
        inputs=[chat_history],
        outputs=[history_display]
    )

    for button, suggestion in zip(suggestion_buttons, suggestions):
        button.click(
            fn=lambda s=suggestion: s,
            inputs=[],
            outputs=[question_box]
        )

demo.launch(
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="slate"),
    css=custom_css
)
