from dotenv import load_dotenv
import streamlit as st
import pickle
import dill
import time
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()
st.title("News Research Tool")
st.sidebar.title("New Article URLs")

llm = OpenAI(temperature=0.9, max_tokens=500)

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Text Splitter Working ....")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n",".",","],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)

    # Create Embeddings
    embeddings = OpenAIEmbeddings()
    main_placeholder.text("Embedder Working ....")
    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    ''''# Save the FAISS index to a pickle file
    main_placeholder.text("Pickle Dump working ....")
    with open(file_path,"wb") as f:
        dill.dump(vectorstore_openai, f)'''

    print(vectorstore_openai)

    query = main_placeholder.text_input("Question: ")
    if query:
        print("Inside query if condition")
        chain = RetrievalQAWithSourcesChain(llm=llm, retriever=vectorstore_openai.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        # {"answer":"", "sources"=[]}
        print(result)
        st.header("Answer")
        st.subheader(result['answer'])

        # Display sources, if available
        sources = result.get("sources")
        if sources:
            st.subheader("Sources")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)
