from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback

# Load environment variables
load_dotenv()

from langchain.globals import set_debug
set_debug(True)

from langchain.globals import set_verbose

set_verbose(True)

def process_text(text):
    # Split the text into chunks using langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    # embeddings = OpenAIEmbeddings()
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="",
        openai_api_version="2024-04-01-preview",
    )
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase


def main():
    st.title("Chat with your PDF 💬")

    pdf = st.file_uploader('Upload your PDF Document', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # Text variable will store the pdf text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Create the knowledge base object
        knowledgeBase = process_text(text)

        query = st.text_input('Ask a question to the PDF')
        cancel_button = st.button('Cancel')

        if cancel_button:
            st.stop()

        if query:
            docs = knowledgeBase.similarity_search(query)
            # llm = OpenAI()
            # llm = AzureOpenAI(
            #     azure_endpoint="https://stingers-openai.openai.azure.com",
            #     deployment_name="financial-instruments",
            #     model_name="gpt-35-turbo",
            #     temperature=0,
            #     stop=["\nObservation"]
            # )
            llm = AzureChatOpenAI(
                azure_endpoint="",
                openai_api_type="azure",
                openai_api_key="",
                deployment_name="",
                model_name="gpt-35-turbo",
                temperature=0.3,
                api_version="2024-04-01-preview"
            )
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)

            st.write(response)


if __name__ == "__main__":
    main()
