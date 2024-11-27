import streamlit as st
from manager import VectorDBManager
from PyPDF2 import PdfReader

# Qdrant configuration
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
QDRANT_URL = st.secrets["QDRANT_URL"]

# Initialize the manager
manager = VectorDBManager(QDRANT_URL, QDRANT_API_KEY, index_dir="persistent_indexdir")

# Collection configuration
collection_name = "example_collection3"

# Streamlit app
st.title("Document Search Application")

# Part 1: Upload Document
st.header("Upload Document")
uploaded_file = st.file_uploader("Upload a PDF file containing structured information", type=["pdf"])
chunk_size = st.number_input("Enter chunk size (tokens):", min_value=10, value=50)
page_size = st.number_input("Enter page size (words):", min_value=50, value=200)

if uploaded_file:
    # Read the PDF and extract text
    pdf_reader = PdfReader(uploaded_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

    st.write("PDF uploaded successfully.")
    
    # Clear the database embeddings
    manager.clear_collection(collection_name)

    # Upload the document to the collection
    program_data = {
        "id": 1,
        "description": "Uploaded document",
        "objective": "Searchable structured document",
        "scope": "Uploaded document scope",
        "name": "Uploaded Document",
        "amiko_id": 1,
        "duration": 0,
        "requires_rag": True,
        "steps": {
            "step_1": {
                "step_id": "001",
                "tag": "document_content",
                "name": "Uploaded Document Content",
                "content": pdf_text,
                "description": "Content from the uploaded PDF document."
            }
        }
    }
    manager.create_collection(collection_name, vector_size=1536)
    manager.add_program(collection_name, program_id="1", program_data=program_data, chunk_size=chunk_size, page_size=page_size)
    st.success("Document uploaded and processed successfully.")

# Part 2: Search Query
st.header("Search Query")
search_query = st.text_input("Enter your search query:")
threshold = st.slider("Set score threshold:", 0.0, 1.0, 0.7)

if search_query:
    results = manager.hierarchical_search(
        query=search_query,
        collection_name=collection_name,
        program_id="1",
        limit=3
    )

    if not results:
        st.write("No search results found.")
    else:
        st.write("Search Results:")
        for result in results:
            if result["score"] >= threshold:
                st.write(f"**Score:** {result['score']:.2f}")
                st.write(f"**Content:** {result['content']}")
                st.write(f"**Context:** {result['context']}")
                st.write("---")
