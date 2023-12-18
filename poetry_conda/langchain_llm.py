import pickle

from keys import open_api_key
import os
from langchain.llms.openai import OpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
import langchain

os.environ['OPENAI_API_KEY'] = open_api_key

# Initialize LLM with required params
llm = OpenAI(temperature=0.9, max_tokens=500)

list_of_urls = [
    "https://www.moneycontrol.com/news/business/s-finishes-flat-on-day-11912101.html",
    "https://www.moneycontrol.com/news/business/tata-motors-reports-highest-ever-monthly-retail-sales-in-november-11891501.html"
]

loaders = UnstructuredURLLoader(urls=list_of_urls)
data = loaders.load()
print(len(data), type(data[0]))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

# As the data is of document type, we can directly use split_documents over split_text in order to get the chunks.
docs = text_splitter.split_documents(data)
print(len(docs))

embeddings = OpenAIEmbeddings()
vector_index_openai = FAISS.from_documents(docs, embeddings)

'''file_path = "vector_index.pkl"
with open(file_path, "wb") as f:
    pickle.dump(vector_index_openai, f)

if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorIndex = pickle.load(f)'''

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_index_openai.as_retriever())
print(chain)

#query = "Who is Feb Bank of New York President ?"
#query = "Who is John Williams ?"
query = "What is dow jones level ?"
langchain.debug = True
chain({"question": query}, return_only_outputs=True)