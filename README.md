✨ Policy Q&A Agent – Intelligent Policy Document Assistant

The Policy Q&A Agent is an AI-powered assistant built using Python and a modern RAG (Retrieval-Augmented Generation) approach. It allows users to upload policy documents (PDFs) and interact with them through a natural conversational interface. Designed for HR teams, employees, and compliance departments, the agent enables quick answers to policy-related questions without manually searching through lengthy documents.

🔍 Key Features
Document Upload (PDF Support): Drag-and-drop policy documents via the Streamlit UI.
Automatic Text Extraction: Efficient PDF parsing and chunking for accurate retrieval.
Vector Embeddings + Semantic Search: Uses embeddings to find the most relevant policy sections.
Agentic RAG Pipeline: Combines retrieval and reasoning for precise and context-aware answers.
Chat Interface: Clean Streamlit-based chat UI for easy interaction.
Modular Python Code: Clear file structure for embedding, retrieval, and LLM calls.
Extendable: Add more documents, switch models, or improve retrieval logic with ease.

🛠️ Tech Stack

Python 3.x
Streamlit (UI & chat interface)
OpenAI / Azure OpenAI API
PyPDF2 (PDF extraction)
Vector Store (FAISS / custom embeddings)
Scikit-learn for cosine similarity

🚀 Use Cases

HR policy Q&A (leave policy, travel policy, code of conduct, etc.)
Compliance and regulatory document assistant
SOP/Guideline Q&A for internal teams
Quick knowledge lookup across large PDF documents

📦 How It Works

Upload a policy document
The agent processes, extracts, and stores embeddings
Ask any question via chat
The agent retrieves relevant policy chunks
LLM generates a precise and well-referenced answer

#Steps

Step 1 - Create a virtual environment 
py -m venv PolicyAgent

Step 2 - Activate virtual environment 
PolicyAgent\scripts\activate

Step 3 - Install necessary libraries 
pip install --upgrade pip
pip install openai faiss-cpu streamlit PyPDF2
pip install pydantic_core
pip install cosine_similarity
pip install scikit-learn

Step 4 = create app.py or map this folder

Step 5 - Let's run app
streamlit run app.py


