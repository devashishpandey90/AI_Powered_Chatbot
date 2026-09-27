# 🎬 AI-Powered YouTube Video Chatbot — RAG

An AI-powered **YouTube Video Chatbot** built with **Streamlit, LangChain, Ollama, FAISS, and YouTube Transcript API**.

The application allows users to paste a YouTube video URL, automatically extract the video ID, fetch the video's transcript, detect whether the transcript is **Hindi or English**, build a vector-based RAG pipeline, and ask questions about the video through an interactive chatbot.

The chatbot answers questions **only from the retrieved transcript context**, helping reduce irrelevant or hallucinated responses.

---

<img width="1919" height="983" alt="image" src="https://github.com/user-attachments/assets/78d4961f-9995-48a3-968f-f80964c62e25" />


## 🚀 Features

* 🔗 **YouTube URL Support**

  * Paste a YouTube URL instead of manually entering the video ID.
  * Supports:

    * `youtube.com/watch`
    * `youtu.be`
    * `youtube.com/embed`
    * `youtube.com/shorts`
    * `youtube.com/live`

* 🆔 **Automatic Video ID Extraction**

  * Automatically extracts the 11-character YouTube video ID from the provided URL.

* 📝 **Automatic Transcript Retrieval**

  * Fetches the available YouTube transcript using `youtube-transcript-api`.

* 🌐 **Hindi / English Language Detection**

  * Detects the transcript language using:

    * YouTube caption metadata
    * Devanagari script detection
    * `langdetect` fallback

* 🧠 **Retrieval-Augmented Generation (RAG)**

  * Transcript is divided into smaller chunks.
  * Chunks are converted into embeddings.
  * Embeddings are stored in a FAISS vector database.
  * Relevant chunks are retrieved for every user question.

* 🤖 **Local LLM**

  * Uses **Mistral 7B** through Ollama for answer generation.

* 🔍 **Semantic Search**

  * Uses `nomic-embed-text` to generate embeddings.
  * Retrieves the most relevant transcript chunks using FAISS.

* 💬 **Interactive Chat Interface**

  * Streamlit chat interface.
  * Maintains conversation history during the session.

* 🎨 **Premium Streamlit UI**

  * Pastel lavender, blue, pink, and white theme.
  * Responsive layout.
  * Video preview.
  * Language and chunk-count indicators.
  * Processing status indicators.

* 🔒 **Context-Grounded Answers**

  * The LLM is instructed to answer only from the retrieved transcript context.
  * If the required information is unavailable, it responds that it does not know based on the provided transcript.

---

# 🏗️ Architecture

```text
                 ┌──────────────────────┐
                 │   YouTube Video URL  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Extract Video ID     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Fetch Video Metadata │
                 │  YouTube oEmbed      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Fetch Transcript     │
                 │ YouTube Transcript   │
                 │ API                  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Language Detection   │
                 │ Hindi / English      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Text Chunking        │
                 │ Recursive Splitter   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Embeddings           │
                 │ nomic-embed-text     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ FAISS Vector Store   │
                 └──────────┬───────────┘
                            │
                       User Question
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Similarity Search    │
                 │ Top K = 4            │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Transcript Context   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Mistral 7B           │
                 │ Ollama               │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Hindi / English      │
                 │ Answer               │
                 └──────────────────────┘
```

---

# 🧠 How RAG Works in This Project

The application follows a standard Retrieval-Augmented Generation pipeline.

### 1. Load Transcript

The YouTube transcript is fetched using:

```python
YouTubeTranscriptApi()
```

The transcript snippets are combined into one text document.

---

### 2. Split Transcript into Chunks

The transcript is divided into smaller pieces using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

This prevents the entire transcript from being passed to the LLM at once.

---

### 3. Generate Embeddings

Each chunk is converted into a numerical vector using:

```text
nomic-embed-text
```

through Ollama.

Embeddings allow the application to compare the semantic meaning of the user's question with transcript chunks.

---

### 4. Store Embeddings in FAISS

The embeddings are stored in a FAISS vector store:

```python
FAISS.from_documents(chunks, embeddings)
```

FAISS enables fast similarity-based retrieval.

---

### 5. Retrieve Relevant Context

When the user asks a question:

```python
retrieved_docs = retriever.invoke(question)
```

The application retrieves the **top 4 relevant transcript chunks**.

---

### 6. Generate Answer

The retrieved chunks are inserted into the prompt and sent to Mistral 7B.

The prompt instructs the model to:

* Use only transcript information.
* Avoid outside knowledge.
* Answer in the detected transcript language.
* Say that it doesn't know when the answer is not available.

---

# 🛠️ Tech Stack

| Technology             | Purpose                             |
| ---------------------- | ----------------------------------- |
| Python                 | Core programming language           |
| Streamlit              | Web UI                              |
| LangChain              | RAG pipeline orchestration          |
| Ollama                 | Local LLM and embedding runtime     |
| Mistral 7B             | LLM                                 |
| nomic-embed-text       | Text embeddings                     |
| FAISS                  | Vector database / similarity search |
| YouTube Transcript API | Transcript extraction               |
| LangDetect             | Language detection fallback         |
| Requests               | YouTube metadata retrieval          |

---

# 📁 Project Structure

```text
youtube-video-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
└── .gitignore
```

> Rename `app.py` according to the actual filename of your Streamlit application.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/youtube-video-chatbot.git
```

```bash
cd youtube-video-chatbot
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

---

# 📦 Install Dependencies

Create a `requirements.txt` file:

```txt
streamlit
requests
youtube-transcript-api
langchain
langchain-core
langchain-community
langchain-ollama
langchain-text-splitters
faiss-cpu
langdetect
```

Install them:

```bash
pip install -r requirements.txt
```

---

# 🦙 Install Ollama

This project uses Ollama to run the LLM and embedding model locally.

Install Ollama from:

```text
https://ollama.com
```

After installation, verify:

```bash
ollama --version
```

---

# 🤖 Download Required Models

Pull the Mistral model:

```bash
ollama pull mistral:7b
```

Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

Verify installed models:

```bash
ollama list
```

You should see something similar to:

```text
mistral:7b
nomic-embed-text
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

Then:

1. Paste a YouTube URL.
2. Click **Process Video**.
3. Wait for the transcript and vector store to be created.
4. Ask questions about the video.
5. Receive answers based on the video transcript.

---

# 💬 Example

### Input

```text
https://www.youtube.com/watch?v=VIDEO_ID
```

The application automatically:

```text
YouTube URL
      ↓
Video ID Extraction
      ↓
Transcript Extraction
      ↓
Language Detection
      ↓
Text Chunking
      ↓
Embeddings
      ↓
FAISS
      ↓
Question
      ↓
Relevant Chunks
      ↓
Mistral 7B
      ↓
Answer
```

---

# 🌐 Language Support

The application currently supports automatic **Hindi and English** response generation.

For example:

### Hindi Transcript

```text
User:
इस वीडियो में RAG क्या है?
```

The chatbot responds in Hindi.

### English Transcript

```text
User:
What is RAG according to this video?
```

The chatbot responds in English.

The response language is controlled through the prompt:

```python
Respond in {language}.
```

---

# 🔐 Context-Grounded Prompt

The chatbot uses a language-aware prompt:

```text
You are an AI assistant that answers questions about a YouTube video,
using its transcript as the only source of truth.

Answer ONLY using the information available in the transcript context.

Do NOT use outside knowledge.

If the answer is not present in the context,
say you don't know based on the provided transcript.
```

This makes the application a **grounded RAG chatbot** rather than a general-purpose chatbot.

---

# 📊 RAG Configuration

Current chunking configuration:

```python
chunk_size = 1000
chunk_overlap = 200
```

Retriever configuration:

```python
search_type = "similarity"
k = 4
```

Models:

```text
LLM:
mistral:7b

Embedding:
nomic-embed-text
```

---

# ✨ Key Functions

### `extract_video_id()`

Extracts the YouTube video ID from different YouTube URL formats.

```python
extract_video_id(url)
```

---

### `get_video_info()`

Retrieves video metadata such as:

* Video title
* Channel/author
* Thumbnail

using YouTube oEmbed.

---

### `fetch_transcript()`

Fetches the transcript and selects an available English/Hindi caption track when possible.

---

### `detect_language()`

Determines whether the transcript should be treated as Hindi or English.

---

### `build_retriever()`

Creates the complete retrieval pipeline:

```text
Transcript
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS
    ↓
Retriever
```

---

### `process_video()`

Coordinates the complete video-processing pipeline:

```text
URL
 ↓
Video ID
 ↓
Metadata
 ↓
Transcript
 ↓
Language
 ↓
Embeddings
 ↓
FAISS
 ↓
Ready for Chat
```

---

# ⚡ Performance Considerations

Because the application uses local Ollama models, processing time depends on:

* CPU/GPU performance
* Available RAM
* Transcript length
* Number of transcript chunks
* Mistral model size

The first request can take longer because the local model may need to be loaded into memory.

---

# 🚧 Limitations

* The chatbot requires an available YouTube transcript.
* Videos with disabled captions cannot be processed.
* Videos without usable transcripts cannot be processed.
* The current application processes one active video at a time.
* FAISS is created in memory for the current Streamlit session.
* Ollama must be running locally.
* Mistral 7B requires sufficient system resources.

---

# 🔮 Future Improvements

Possible future enhancements include:

* 🎙️ Speech-to-text for videos without transcripts
* 📄 PDF/document upload support
* 💾 Persistent FAISS vector database
* 👥 Multi-video knowledge base
* 🧠 Conversation-aware RAG
* ⏱️ Timestamp-based answers
* 🔎 Clickable transcript references
* 📌 Source citation for every answer
* ⚡ Streaming LLM responses
* ☁️ Cloud deployment
* 🌍 More language support
* 🎧 Audio/video file upload
* 📊 Video summarization
* 📝 Automatic notes and key-point extraction
* 🤖 AI-generated quizzes from the video

---

# 🔒 Security

Do not commit sensitive credentials or configuration files.

Add the following to `.gitignore`:

```gitignore
.venv/
__pycache__/
.env
*.pyc
.streamlit/secrets.toml
```

---

# 📸 Application Flow

```text
        🎬 AI Video Chatbot
                 │
                 ▼
       Paste YouTube URL
                 │
                 ▼
        🚀 Process Video
                 │
        ┌────────┴────────┐
        ▼                 ▼
   Video Metadata      Transcript
                           │
                           ▼
                   Language Detection
                           │
                           ▼
                     RAG Pipeline
                           │
                           ▼
                     FAISS Search
                           │
                           ▼
                      Mistral 7B
                           │
                           ▼
                    💬 AI Response
```

---

# 👨‍💻 Author

**Devashish Pandey**

AI/ML Developer | Python | Generative AI | RAG | LLM | LangChain | FastAPI | SQL | RPA Automation

---

# ⭐ Project Highlights

This project demonstrates practical implementation of:

* Python
* Generative AI
* Large Language Models
* Retrieval-Augmented Generation
* LangChain
* Vector Embeddings
* FAISS
* Ollama
* Mistral 7B
* Semantic Search
* Prompt Engineering
* YouTube Transcript Processing
* Hindi/English Language Detection
* Streamlit
* AI Chatbot Development

---

## 📌 One-Line Project Description

> **An AI-powered YouTube Video Chatbot that uses Retrieval-Augmented Generation (RAG), FAISS, Ollama, and Mistral 7B to answer questions from YouTube transcripts in Hindi or English.**
