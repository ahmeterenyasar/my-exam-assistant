import json
from pathlib import Path

import streamlit as st
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COURSES_FILE = CHROMA_DIR / "courses.json"

# Page configuration
st.set_page_config(page_title="Local RAG Exam Assistant", page_icon="📚")
st.title("📚 Local Exam Assistant (Llama 3)")

if not CHROMA_DIR.exists() or not COURSES_FILE.exists():
    st.error("Vector database not found. Please run ingest.py first.")
    st.code("python ingest.py --reset")
    st.stop()


@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        collection_name="course_notes",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


@st.cache_data
def load_courses() -> list[str]:
    payload = json.loads(COURSES_FILE.read_text(encoding="utf-8"))
    return payload.get("courses", [])


def build_rag_chain(vectorstore: Chroma, selected_course: str):
    search_kwargs = {"k": 4}
    if selected_course != "All courses":
        search_kwargs["filter"] = {"course": selected_course}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    llm = Ollama(model="llama3")

    prompt = ChatPromptTemplate.from_template("""
    You are an exam preparation assistant. Answer the question using the context below.
    If the answer is not in the notes, do not make it up and say "This information is not in the notes".
    
    Context: {context}
    
    Question: {input}
    Answer:
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    return rag_chain


courses = load_courses()
if not courses:
    st.warning("No courses found. Add PDF files to data/<course_name>/ and run ingest.py.")
    st.stop()

vectorstore = load_vectorstore()

selected_course = st.sidebar.selectbox(
    "Select Course",
    ["All courses"] + courses,
)

rag_chain = build_rag_chain(vectorstore, selected_course)

# --- 2. CHAT INTERFACE ---
# Using session_state to keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_course" not in st.session_state:
    st.session_state.last_course = selected_course

if st.session_state.last_course != selected_course:
    st.session_state.messages = []
    st.session_state.last_course = selected_course

# Display previous messages on screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get new question from user
if prompt := st.chat_input("What would you like to learn about the course notes?"):
    # Display user's question on screen
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run model and generate answer
    with st.chat_message("assistant"):
        with st.spinner("Scanning notes and generating answer..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response.get("answer", "Answer could not be generated.")
            st.markdown(answer)

            with st.expander("Sources Used (References)"):
                for doc in response.get("context", []):
                    source = doc.metadata.get("file_name", "Unknown source")
                    page_num = doc.metadata.get("page")
                    display_page = page_num + 1 if isinstance(page_num, int) else "?"
                    snippet = doc.page_content[:200].replace("\n", " ")
                    st.write(f"📄 {source} | page: {display_page}")
                    st.caption(snippet + "...")

    # Save assistant's answer to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
