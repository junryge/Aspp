import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
MODEL_PATH = "Qwen3-4B-Q8_0.gguf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "./chroma_db_data"

st.set_page_config(page_title="Local NotebookLM", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .stSidebar { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    .stChatMessage { background-color: #ffffff; border-radius: 10px; padding: 10px; margin-bottom: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    h1 { color: #202124; font-family: 'Google Sans', sans-serif; }
</style>
""", unsafe_allow_html=True)

# --- Functions ---

@st.cache_resource
def get_llm():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None
   
    try:
        llm = LlamaCpp(
            model_path=MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=4096,
            temperature=0.1,
            max_tokens=2048,
            verbose=False,
            f16_kv=True,
        )
        return llm
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

@st.cache_resource
def get_embedding_model():
    local_path = "./local_embedding_model"
    if os.path.exists(local_path):
        return HuggingFaceEmbeddings(model_name=local_path)
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def process_uploaded_files(uploaded_files):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
   
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
       
        try:
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_file_path)
            elif uploaded_file.name.endswith(".txt") or uploaded_file.name.endswith(".md") or uploaded_file.name.endswith(".py"):
                loader = TextLoader(tmp_file_path, encoding='utf-8')
            else:
                loader = UnstructuredFileLoader(tmp_file_path)
               
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            os.remove(tmp_file_path)
           
    return text_splitter.split_documents(documents)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Main Interface ---

def main():
    st.title("📚 Local NotebookLM")
   
    if "messages" not in st.session_state:
        st.session_state.messages = []
       
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
   
    # Sidebar
    with st.sidebar:
        st.header("Sources")
        uploaded_files = st.file_uploader("Upload documents (PDF, TXT, Code)", accept_multiple_files=True)
       
        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                chunks = process_uploaded_files(uploaded_files)
                if chunks:
                    embeddings = get_embedding_model()
                    vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings
                    )
                    st.session_state.vectorstore = vectorstore
                    st.success(f"Processed {len(chunks)} chunks!")

    # Chat Area
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something about your documents..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
       
        if st.session_state.vectorstore:
            with st.spinner("Thinking..."):
                try:
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                    context_docs = retriever.invoke(prompt)
                    context_text = format_docs(context_docs)
                   
                    llm = get_llm()
                    if llm:
                        template = """<|im_start|>system
You are a helpful AI assistant. Answer the user's question strictly based on the provided context. If the answer is not in the context, say so.
Context:
{context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
                        prompt_template = PromptTemplate.from_template(template)
                        formatted_prompt = prompt_template.format(context=context_text, question=prompt)
                       
                        answer = llm.invoke(formatted_prompt)
                       
                        st.chat_message("assistant").markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                         st.error("LLM not loaded.")
                except Exception as e:
                    st.error(f"Error during generation: {e}")
        else:
            st.warning("Please upload and process documents first to start chatting!")

if __name__ == "__main__":
    main()