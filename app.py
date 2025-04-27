import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Initialize the LLM and embeddings (using FastEmbed instead of sentence-transformers)
llm = ChatOllama(model="mistral")
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Streamlit UI setup
st.set_page_config(page_title="Insurance Policy Chatbot", page_icon=":shield:")
st.title("AI-Powered Insurance Policy Assistant")
st.write("Ask me anything about health, life, auto, or home insurance policies!")

# Sidebar for PDF upload
with st.sidebar:
    st.header("Configuration")
    uploaded_files = st.file_uploader(
        "Upload insurance policy PDFs", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                # Save uploaded files temporarily and load them
                docs = []
                temp_dir = tempfile.mkdtemp()
                for file in uploaded_files:
                    temp_filepath = os.path.join(temp_dir, file.name)
                    with open(temp_filepath, "wb") as f:
                        f.write(file.getvalue())
                    loader = PyPDFLoader(temp_filepath)
                    docs.extend(loader.load())
                
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(docs)
                
                # Create vector store
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
                vectorstore.persist()
                st.success("Documents processed and knowledge base created!")
        else:
            st.warning("Please upload at least one PDF file.")

# Initialize or load vector store
if os.path.exists("./chroma_db"):
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
else:
    retriever = None

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know about our insurance policies?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if not retriever:
            st.warning("Please upload and process insurance policy documents first.")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "I need insurance policy documents to answer questions. Please upload them in the sidebar."
            })
        else:
            with st.spinner("Researching your question..."):
                # RAG setup
                template = """You are a helpful insurance policy assistant. 
                Answer the question based only on the following context, which is insurance policy documentation:
                {context}
                
                Question: {question}
                
                If you don't know the answer, say you don't know. 
                Be precise and professional in your responses."""
                prompt_template = ChatPromptTemplate.from_template(template)
                
                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )
                
                response = chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })