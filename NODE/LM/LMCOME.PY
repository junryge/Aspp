import os
os.environ["HF_HUB_OFFLINE"] = "1"

import streamlit as st
import tempfile
import requests
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_PATH = "Qwen3-4B-Q8_0.gguf"
VECTOR_STORE_PATH = "./chroma_db_data"

# LLM 설정
LLM_MODE = "api"  # "local" 또는 "api"
API_URL = "http://dev.assistant.llm.skhynix.com/v1/chat/completions"
API_MODEL = "Qwen3-Coder-30B-A3B-Instruct"
API_TOKEN = None

st.set_page_config(page_title="Local NotebookLM", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .stSidebar { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
</style>
""", unsafe_allow_html=True)

# ========================================
# API 토큰 로드
# ========================================
def load_api_token():
    global API_TOKEN
    token_path = "token.txt"
    
    if os.path.exists(token_path):
        try:
            with open(token_path, "r") as f:
                API_TOKEN = f.read().strip()
            logger.info("✅ API 토큰 로드 완료")
            return True
        except Exception as e:
            logger.error(f"❌ API 토큰 로드 실패: {e}")
            return False
    else:
        logger.warning(f"⚠️ 토큰 파일 없음: {token_path}")
        return False

# ========================================
# API LLM 호출
# ========================================
def call_api_llm(prompt, context):
    global API_TOKEN
    
    if not API_TOKEN:
        if not load_api_token():
            return "API 토큰 없음. token.txt 파일 확인해."
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    
    payload = {
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": f"You are a helpful AI assistant. Answer based on the context.\n\nContext:\n{context}"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 2048
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"API 호출 실패: {e}")
        return f"API 오류: {e}"

# ========================================
# 로컬 LLM
# ========================================
@st.cache_resource
def get_local_llm():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=4096,
        temperature=0.1,
        max_tokens=2048,
        verbose=False,
        f16_kv=True,
    )

@st.cache_resource
def get_embedding_model():
    local_path = "./local_embedding_model"
    if not os.path.exists(local_path):
        st.error("로컬 임베딩 모델 없음: ./local_embedding_model")
        return None
    return HuggingFaceEmbeddings(model_name=local_path)

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
            elif uploaded_file.name.endswith((".txt", ".md", ".py")):
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

def main():
    st.title("📚 Local NotebookLM")
    
    # 시작할 때 토큰 로드
    if LLM_MODE == "api" and not API_TOKEN:
        load_api_token()
   
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
   
    with st.sidebar:
        st.header("Sources")
        uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)
        
        st.header("LLM 설정")
        st.info(f"모드: **{LLM_MODE.upper()}**")
        if LLM_MODE == "api":
            st.caption(f"API: {API_MODEL}")
            if API_TOKEN:
                st.success("✅ 토큰 로드됨")
            else:
                st.error("❌ 토큰 없음")
       
        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing..."):
                chunks = process_uploaded_files(uploaded_files)
                if chunks:
                    embeddings = get_embedding_model()
                    if embeddings:
                        st.session_state.vectorstore = Chroma.from_documents(
                            documents=chunks, embedding=embeddings
                        )
                        st.success(f"Processed {len(chunks)} chunks!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
       
        if st.session_state.vectorstore:
            with st.spinner("Thinking..."):
                try:
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                    context_docs = retriever.invoke(prompt)
                    context_text = format_docs(context_docs)
                    
                    if LLM_MODE == "api":
                        # API 방식
                        answer = call_api_llm(prompt, context_text)
                    else:
                        # 로컬 GGUF 방식
                        llm = get_local_llm()
                        if llm:
                            template = """<|im_start|>system
You are a helpful AI assistant. Answer based on the context.
Context:
{context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
                            formatted_prompt = template.format(context=context_text, question=prompt)
                            answer = llm.invoke(formatted_prompt)
                        else:
                            answer = "로컬 LLM 로드 실패"
                   
                    st.chat_message("assistant").markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("문서 먼저 업로드해!")

if __name__ == "__main__":
    main()