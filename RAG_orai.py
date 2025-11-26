import os.path

import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from gitdb.fun import chunk_size
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain

load_dotenv()
def main():
    st.header("Hello")
    pdf = st.file_uploader("Upload your pdf here!", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text+=page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}"):
            VectorStore = FAISS.load_local(f"{store_name}",OpenAIEmbeddings(),allow_dangerous_deserialization=True)
            st.write("Embeddings loaded from the disk.")
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            VectorStore.save_local(f"{store_name}")
            st.write("Embeddings Computation Completed.")

        query = st.text_input("Ask question about your pdf!")

        if query:
            docs = VectorStore.similarity_search(query=query, k=5)

            llm = OpenAI()
            chain = load_qa_chain(llm = llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question = query)
            st.write(response)

if __name__ == '__main__':
    main()