import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# ------------------------------
# CONFIG
# ------------------------------
hf_token = "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"   # 🔑 replace this

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=700,
    do_sample=False,
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="YouTube Chatbot", page_icon="🎥")
st.title("🎥 YouTube Video Chatbot")

video_id = st.text_input("Enter YouTube Video ID:")
question = st.text_input("Ask a question about the video:")

# ------------------------------
# MAIN LOGIC
# ------------------------------
if st.button("Get Answer"):

    if not video_id or not question:
        st.warning("Please enter both Video ID and Question")
    else:
        try:
            with st.spinner("Fetching transcript..."):

                ytt_api = YouTubeTranscriptApi()
                transcript_list = ytt_api.fetch(video_id)
                transcript = " ".join(chunk.text for chunk in transcript_list)

            # Split text
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.create_documents([transcript])

            # Embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Vector store
            vector_store = FAISS.from_documents(chunks, embeddings)

            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

            # Prompt
            prompt = PromptTemplate(
                template="""
You are a YouTube video assistant.

Answer ONLY using the provided context.
Do NOT use any outside knowledge.

If the answer is not in the context, say:
"Not mentioned in the video."

Context:
{context}

Question: {question}

Answer:
""",
                input_variables=["context", "question"]
            )

            with st.spinner("Thinking..."):

                retrieved_docs = retriever.invoke(question)
                context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

                final_prompt = prompt.invoke({
                    "context": context_text,
                    "question": question
                })

                answer = model.invoke(final_prompt)

            st.success("Answer:")
            st.write(answer.content)

        except TranscriptsDisabled:
            st.error("❌ No captions available for this video.")

        except NoTranscriptFound:
            st.error("❌ No transcript found.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")