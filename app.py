import streamlit as st
import requests
import os
from utils.pdf_utils import extract_text_from_pdf

API_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="PDF RAG Bot", layout="wide")
st.title("📚 Advanced PDF RAG Engine")

# Ensure a temporary directory exists for UI uploads
TEMP_DIR = "./temp_ui_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize Session States
if "pdf_context" not in st.session_state:
    st.session_state.pdf_context = None
if "current_file_path" not in st.session_state:
    st.session_state.current_file_path = None
if "direct_chat_history" not in st.session_state:
    st.session_state.direct_chat_history = []
if "global_chat_history" not in st.session_state:
    st.session_state.global_chat_history = []

# Create Tabs for different interactive modes
tab_global, tab_upload = st.tabs(["🌐 Search & Chat (Global PDF Store)", "📄 Upload & Analyze New PDF"])

# ==========================================
# TAB 1: GLOBAL PDF STORE CHAT
# ==========================================
with tab_global:
    st.header("Chat with your existing PDF Knowledge Base")
    st.markdown("Ask questions, and the AI will search across all embedded PDFs to find the answer.")
    
    # Display Global Chat History
    for chat in st.session_state.global_chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])
            if "metrics" in chat:
                st.caption(f"Cost: ${chat['metrics']['execution_cost_usd']} | Tokens: {chat['metrics']['total_tokens']}")

    # Global Chat Input
    if global_prompt := st.chat_input("E.g., 'What PDFs mention machine learning?' or 'Summarize our financial reports'"):
        st.session_state.global_chat_history.append({"role": "user", "content": global_prompt})
        with st.chat_message("user"):
            st.write(global_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching the global database..."):
                payload = {"query": global_prompt, "limit": 4}
                res = requests.post(f"{API_URL}/chat", json=payload)
                
                if res.status_code == 200:
                    data = res.json()
                    st.write(data["answer"])
                    st.caption(f"Cost: ${data['metrics']['execution_cost_usd']} | Tokens: {data['metrics']['total_tokens']}")
                    st.session_state.global_chat_history.append({
                        "role": "assistant", 
                        "content": data["answer"],
                        "metrics": data["metrics"]
                    })
                else:
                    st.error("Failed to get response from backend.")

# ==========================================
# TAB 2: UPLOAD & ANALYZE SINGLE PDF
# ==========================================
with tab_upload:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Upload Document")
        uploaded_file = st.file_uploader("Upload a new PDF for isolation analysis", type=["pdf"])
        
        if uploaded_file is not None and st.session_state.current_file_path != os.path.join(TEMP_DIR, uploaded_file.name):
            temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Extracting text..."):
                st.session_state.pdf_context = extract_text_from_pdf(temp_path)
                st.session_state.current_file_path = temp_path
                st.session_state.direct_chat_history = [] 
            st.success("PDF loaded into memory!")

        st.divider()
        
        # New Feature: Find Similar PDFs
        st.subheader("2. Cross-Reference")
        if st.session_state.pdf_context:
            if st.button("🔍 Find Similar PDFs in Database"):
                with st.spinner("Analyzing semantic similarity..."):
                    # Use the first 1500 characters of the PDF as the similarity query to avoid blowing up the payload
                    preview_text = st.session_state.pdf_context[:1500] 
                    res = requests.post(f"{API_URL}/similar_sources", json={"query_text": preview_text, "limit": 3})
                    
                    if res.status_code == 200:
                        similar_files = res.json().get("similar_pdfs", [])
                        if similar_files:
                            st.write("**Related files already in database:**")
                            for sf in similar_files:
                                st.markdown(f"- `{sf}`")
                        else:
                            st.info("No similar documents found in the current store.")
                    else:
                        st.error("Error finding similar documents.")

        st.divider()
        
        # Existing Feature: Ingest to Database
        st.subheader("3. Persistent Storage")
        if st.session_state.current_file_path:
            if st.button("💾 Add PDF to Global Dataset", type="primary"):
                with st.spinner("Uploading and embedding..."):
                    with open(st.session_state.current_file_path, "rb") as f:
                        files = {"file": (os.path.basename(st.session_state.current_file_path), f, "application/pdf")}
                        response = requests.post(f"{API_URL}/ingest_single", files=files)
                    
                    if response.status_code == 200:
                        st.success("Successfully added and embedded into the dataset!")
                        os.remove(st.session_state.current_file_path)
                        st.session_state.pdf_context = None
                        st.session_state.current_file_path = None
                    else:
                        st.error(f"Error: {response.text}")

    with col2:
        st.subheader("Direct Document Chat")
        if st.session_state.pdf_context:
            st.info(f"Currently chatting with: **{os.path.basename(st.session_state.current_file_path)}**")
            
            # Display Direct Chat History
            for chat in st.session_state.direct_chat_history:
                with st.chat_message(chat["role"]):
                    st.write(chat["content"])
                    if "metrics" in chat:
                        st.caption(f"Cost: ${chat['metrics']['execution_cost_usd']} | Tokens: {chat['metrics']['total_tokens']}")

            # Direct Chat Input
            if direct_prompt := st.chat_input("Ask a question specific to this uploaded PDF..."):
                st.session_state.direct_chat_history.append({"role": "user", "content": direct_prompt})
                with st.chat_message("user"):
                    st.write(direct_prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing document..."):
                        payload = {
                            "query": direct_prompt,
                            "context": st.session_state.pdf_context
                        }
                        res = requests.post(f"{API_URL}/chat_direct", json=payload)
                        
                        if res.status_code == 200:
                            data = res.json()
                            st.write(data["answer"])
                            st.caption(f"Cost: ${data['metrics']['execution_cost_usd']} | Tokens: {data['metrics']['total_tokens']}")
                            st.session_state.direct_chat_history.append({
                                "role": "assistant", 
                                "content": data["answer"],
                                "metrics": data["metrics"]
                            })
                        else:
                            st.error("Failed to get response from backend.")
        else:
            st.markdown("👈 **Upload a document on the left to start a direct session.**")