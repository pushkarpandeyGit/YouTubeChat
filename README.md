# YouTube Video Q&A Chatbot

This project is a Python-based chatbot that can answer questions about YouTube videos using their transcripts. It combines **LangChain**, **HuggingFace models**, and **FAISS** to provide context-aware answers. The chatbot strictly uses the video transcript to generate responses and does not rely on external knowledge.

---

## Features

- Fetch transcripts from YouTube videos.
- Split transcripts into smaller, manageable chunks for processing.
- Convert chunks into embeddings using HuggingFace sentence-transformers.
- Store embeddings in FAISS for fast semantic search.
- Retrieve relevant chunks based on user questions.
- Generate context-aware answers using a HuggingFace LLM.
- Works with videos that have transcripts or captions available.

---

## How It Works

1. **Fetch Transcript**  
   The chatbot uses `YouTubeTranscriptApi` to fetch the transcript of a video. If a transcript is not available, it notifies the user.

2. **Split Transcript**  
   Large transcripts are split into smaller chunks using `RecursiveCharacterTextSplitter` to make them easier to process and embed.

3. **Create Embeddings and Vector Store**  
   Each chunk is converted into embeddings using HuggingFace sentence transformers. The embeddings are stored in **FAISS**, allowing efficient similarity-based retrieval.

4. **Retrieve Relevant Chunks**  
   When a user asks a question, the system searches the vector store for the most relevant transcript chunks.

5. **Generate Answers Using LLM**  
   The retrieved chunks are passed to a HuggingFace LLM via LangChain. A structured prompt ensures the LLM answers **only using the video context**.  

6. **Return Answer**  
   The LLM generates an answer, or if the information is not in the transcript, it returns: `"Not mentioned in the video."`

---

