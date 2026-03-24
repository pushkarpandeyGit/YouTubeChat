# YouTube Video Q&A Chatbot 🤖

[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A Python chatbot that answers questions about YouTube videos using **video transcripts**, **LangChain**, and **HuggingFace models**.  
It only answers based on the video content and does not use external knowledge.

---

## Features

- Fetches transcripts from YouTube videos.
- Splits transcripts into chunks for efficient processing.
- Converts chunks into embeddings using HuggingFace sentence transformers.
- Stores embeddings in FAISS for fast semantic search.
- Answers questions strictly using the video transcript.
- Uses a HuggingFace LLM (e.g., Mistral-7B-Instruct) to generate answers.

---

## Requirements

- Python 3.9+
- Install dependencies:

```bash
pip install langchain_huggingface youtube-transcript-api langchain_core langchain_community faiss-cpu sentence-transformers
