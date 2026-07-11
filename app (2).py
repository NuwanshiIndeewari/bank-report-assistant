# ---- Fix for ChromaDB's sqlite requirement on Streamlit Cloud ----
# (Streamlit Cloud ships an older sqlite; this swaps in a newer one.)
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import glob
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Groq API key comes from Streamlit "Secrets" (not written in the code)
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Map file prefixes to nice bank names
BANK_NAMES = {
    "commercial": "Commercial Bank",
    "sampath": "Sampath Bank",
    "hnb": "HNB",
    "boc": "Bank of Ceylon",
    "ndb": "NDB",
    "dfcc": "DFCC Bank",
    "seylan": "Seylan Bank",
    "ntb": "Nations Trust Bank",
}

# Build the vector database once and cache it (won't re-embed on every click)
@st.cache_resource(show_spinner="Loading and embedding reports (first run only)...")
def build_vectorstore():
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    all_chunks = []
    for path in glob.glob("reports/*.pdf"):
        fname = os.path.basename(path).replace(".pdf", "")
        try:
            prefix, year = fname.rsplit("_", 1)   # e.g. "commercial_2023"
        except ValueError:
            continue
        company = BANK_NAMES.get(prefix.lower(), prefix)
        chunks = splitter.split_documents(PyPDFLoader(path).load())
        for c in chunks:
            c.metadata["company"] = company
            c.metadata["year"] = year
        all_chunks.extend(chunks)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = Chroma.from_documents(all_chunks, embeddings)
    companies = sorted({c.metadata["company"] for c in all_chunks})
    years = sorted({c.metadata["year"] for c in all_chunks})
    return vs, companies, years

@st.cache_resource
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are a financial analyst assistant for {company}'s {year} annual report.
Answer the question using ONLY the context below.
If the answer is not in the context, reply: "I could not find that in {company}'s {year} report."
Be concise and quote figures exactly as they appear.

Context:
{context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def ask_bank(vectorstore, llm, company, year, question):
    retriever = vectorstore.as_retriever(search_kwargs={
        "k": 8,
        "filter": {"$and": [{"company": {"$eq": company}}, {"year": {"$eq": year}}]},
    })
    docs = retriever.invoke(question)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "company": company, "year": year,
        "question": question, "context": format_docs(docs),
    })
    pages = sorted({d.metadata.get("page", "?") for d in docs})
    return answer, pages

# ------------------- UI -------------------
st.set_page_config(page_title="SL Banks Annual Report Assistant", page_icon="🏦")
st.title("🏦 Sri Lankan Banks — Annual Report Assistant")
st.write("Select a **bank** and a **year**, type a question, and get an answer from that annual report.")

vectorstore, company_list, year_list = build_vectorstore()
llm = get_llm()

if not company_list:
    st.warning("No reports found. Add PDFs to a `reports/` folder named like `commercial_2023.pdf`.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    company = st.selectbox("Bank", company_list)
with col2:
    year = st.selectbox("Year", year_list, index=len(year_list) - 1)

question = st.text_input("Your question", placeholder="e.g. What was the net profit? What were total assets?")

if st.button("Ask", type="primary"):
    if not question.strip():
        st.info("Please type a question first.")
    else:
        with st.spinner("Thinking..."):
            answer, pages = ask_bank(vectorstore, llm, company, year, question)
        st.markdown(f"### {company} — {year}")
        st.write(answer)
        st.caption(f"Source pages: {pages}")
