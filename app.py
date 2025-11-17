import streamlit as st
import tempfile, os
from openai import AzureOpenAI
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------------------------
# Streamlit UI Setup
# -------------------------
st.set_page_config(page_title="Policy Q&A Agent", page_icon="📘")
st.title("📘 Policy Q&A Agent")

# -------------------------
# Config / Keys (Use env variable ideally)
# -------------------------
api_key = "your key here"   
endpoint = "https://kraft017.openai.azure.com/"
api_version = "2024-12-01-preview"

chat_model = "gpt-4o-mini"
embed_model = "text-embedding-3-large"

# -------------------------
# File Upload UI
# -------------------------
uploaded_files = st.file_uploader(
    "Upload policy documents",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# -------------------------
# Initialize client
# -------------------------
client = None
if api_key:
    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version
    )

# -------------------------
# Safe initialization
# -------------------------
docs = []
all_chunks = []
chunk_store = []

# -------------------------
# Document ingestion
# -------------------------
if client and uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Extract text
        if uploaded_file.type == "application/pdf":
            reader = PdfReader(tmp_path)
            text = "".join([page.extract_text() or "" for page in reader.pages])
        else:
            with open(tmp_path, "r", encoding="utf-8") as f:
                text = f.read()

        os.remove(tmp_path)
        docs.append(text)

# -------------------------
# Chunking function
# -------------------------
def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Chunk all docs
if docs:
    for d in docs:
        all_chunks.extend(chunk_text(d))

# -------------------------
# Embeddings
# -------------------------
if all_chunks:
    emb_response = client.embeddings.create(
        model=embed_model,
        input=all_chunks
    )
    embeddings = [e.embedding for e in emb_response.data]
    chunk_store = list(zip(all_chunks, embeddings))
else:
    st.info("📄 Upload documents to start asking questions.")
    st.stop()

# -------------------------
# Chat history
# -------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous chat
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# Chat input
# -------------------------
user_query = st.chat_input("Ask a policy question...")
if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Embed query
    q_embed = client.embeddings.create(
        model=embed_model,
        input=[user_query]
    ).data[0].embedding

    # Retrieve top chunks
    sims = cosine_similarity([q_embed], [c[1] for c in chunk_store])[0]
    top_idx = np.argsort(sims)[-3:][::-1]
    context = "\n\n".join([chunk_store[i][0] for i in top_idx])

    # Chat completion
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer using ONLY the policy context provided."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
        ]
    )

    answer = completion.choices[0].message.content

    # Store + Display
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
