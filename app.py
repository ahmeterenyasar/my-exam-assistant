import json
from pathlib import Path

import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COURSES_FILE = CHROMA_DIR / "courses.json"

# Sayfa ayarları
st.set_page_config(page_title="Yerel RAG Sınav Asistanı", page_icon="📚")
st.title("📚 Yerel Sınav Asistanı (Llama 3)")

if not CHROMA_DIR.exists() or not COURSES_FILE.exists():
    st.error("Vektor veritabani bulunamadi. Once ingest.py calistirmalisin.")
    st.code("python ingest.py --reset")
    st.stop()


@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        collection_name="ders_notlari",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


@st.cache_data
def load_courses() -> list[str]:
    payload = json.loads(COURSES_FILE.read_text(encoding="utf-8"))
    return payload.get("courses", [])


def build_rag_chain(vectorstore: Chroma, selected_course: str):
    search_kwargs = {"k": 4}
    if selected_course != "Tum dersler":
        search_kwargs["filter"] = {"course": selected_course}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    llm = Ollama(model="llama3")

    prompt = ChatPromptTemplate.from_template("""
    Sen bir sınav hazırlık asistanısın. Aşağıdaki bağlamı kullanarak soruyu cevapla.
    Eğer cevap bu notlarda yoksa, uydurma, "Notlarda bu bilgi yok" de.
    
    Bağlam: {context}
    
    Soru: {input}
    Cevap:
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


courses = load_courses()
if not courses:
    st.warning("Ders bulunamadi. PDF'leri data/<ders_adi>/ altina koyup ingest.py calistir.")
    st.stop()

vectorstore = load_vectorstore()

selected_course = st.sidebar.selectbox(
    "Ders Sec",
    ["Tum dersler"] + courses,
)

rag_chain = build_rag_chain(vectorstore, selected_course)

# --- 2. CHAT ARAYÜZÜ ---
# Sohbet geçmişini tutmak için session_state kullanıyoruz
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_course" not in st.session_state:
    st.session_state.last_course = selected_course

if st.session_state.last_course != selected_course:
    st.session_state.messages = []
    st.session_state.last_course = selected_course

# Geçmiş mesajları ekranda göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan yeni soru al
if prompt := st.chat_input("Ders notlarıyla ilgili ne öğrenmek istersin?"):
    # Kullanıcının sorusunu ekrana bas
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Modeli çalıştırıp cevabı üret
    with st.chat_message("assistant"):
        with st.spinner("Notlar taranıyor ve cevap üretiliyor..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response.get("answer", "Cevap olusturulamadi.")
            st.markdown(answer)

            with st.expander("Kullanılan Kaynaklar (Referanslar)"):
                for doc in response.get("context", []):
                    source = doc.metadata.get("source", "Bilinmeyen kaynak")
                    page = doc.metadata.get("page", "?")
                    snippet = doc.page_content[:200].replace("\n", " ")
                    st.write(f"📄 {source} | sayfa: {page}")
                    st.caption(snippet + "...")

    # Asistanın cevabını geçmişe kaydet
    st.session_state.messages.append({"role": "assistant", "content": answer})
