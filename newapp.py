
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
data_path = "data/"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_text_from_pdf(pdf_path):
    loader = DirectoryLoader(data_path, glob ="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()
    return documents

def get_text_chunk(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024,chunk_overlap = 128)
    return text_splitter.split_documents(documents)

def get_vector_store(text_chunks):
    embedding_model = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_db = FAISS.from_documents(text_chunks,embedding_model)
    vector_db.save_local("faiss_index")
    return vector_db
def conversational_chains():
    prompt_template = """
    # Character
    You serve as an intelligent assistant who possesses many skills and knowledge areas.
    You have proven your competence by passing the Turing test. You are known for your friendly and approachable manner.
    You have the following skills:


    # Skills:
    ## Skill 1: Effectively use your context to answer queries.
    - Consider all information available from context and effectively combine them in order to answer user's query.
    Your answer to Human should be focused on the following context:

    ## Skill 2: Answering questions
    - You can answer questions based on the context provided. You can also ask questions to clarify the context.
    
    ## Skill 3: Combine or Ignore all document knowledge that are not relevant to the context.
    Output any relevant information that might be relevant to the context using the following document knowledge:
    
    #Context:
    CONTEXT: {context} \n
    QUESTION: {question} \n
    ANSWER: 
    """
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=key, temperature = 0.8 )
    prompt = PromptTemplate(template = prompt_template, 
                            input_variables = ['context', 'question']
                            )
    chains = load_qa_chain(llm = llm, chain_type = 'stuff', prompt = prompt )
    return chains
def user_input(user_question):
    embedding_model = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    db = FAISS.load_local('faiss_index', embedding_model,allow_dangerous_deserialization = True)
    docs = db.similarity_search(user_question)
    chain = conversational_chains()

    response = chain(
        {  
            "input_documents":docs, "question": user_question
        },
        return_only_outputs = True 
    )
    print(response)
    st.write("Reply: ",response['output_text'])
def main():
    st.set_page_config("Chat with me")
    st.header("CHAT WITH GEMINI PRO")

    user_question = st.text_input("Ask a question")

    if user_question: 
        user_input(user_question)
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDFs files here:")
        if st.button("Submit and Process:"):
            with st.spinner("Processing...."):
                documents =  get_text_from_pdf(pdf_docs) 
                text_chunks = get_text_chunk(documents)
                get_vector_store(text_chunks)
                st.success("Done")
if __name__ == "__main__":
    main()
