import os 
import streamlit as st
import pickle 
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader 
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

urls = []
file_path = "FAISS_OpenAI.pkl"
load_dotenv ()


st.title("Equity News Research Tool 📰")

st.sidebar.title("News Articles URLs ")

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)


url_clicked_button = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
if url_clicked_button:
   #Loading Data
   loader = UnstructuredURLLoader(urls=urls)
   main_placeholder.text("Loading the data 🧑‍💻🧑‍💻")
   data = loader.load()

   main_placeholder.text("Splitting the documents 🧑‍💻🧑‍💻")
   text_splitter = RecursiveCharacterTextSplitter(
       separators=['\n\n','\n','.',','],
       chunk_size = 1000
   )

   docs = text_splitter.split_documents(data)


   embeddings = OpenAIEmbeddings()
   openai_vector = FAISS.from_documents(docs, embeddings)
   main_placeholder.text("Fetching the news masala 🧑‍💻🧑‍💻")
   
   with open(file_path,"wb") as f:
      pickle.dump(openai_vector,f)
llm = OpenAI(temperature=0.8,max_tokens=500)      
query = main_placeholder.text_input("Question:")

if query:
   if os.path.exists(file_path):
      with open(file_path,"rb") as f:
         vectorstore = pickle.load(f)
         chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorstore.as_retriever())
         result = chain({"question":query},return_only_outputs=True)
         st.header("Answer")
         st.write(result["answer"])

         sources = result.get("sources","")
         if sources:
            st.subheader("Sources")
            sources_list = sources.split("\n")
            for source in sources_list:
               st.write(source)




    



