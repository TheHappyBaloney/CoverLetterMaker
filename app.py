# Package import

import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import io
import shutil

load_dotenv()

# API Configuration

genai.configure(api_key = os.environ['GOOGLE_API_KEY'])

# Function to extract text from PDF

def get_pdf_text(pdf_bytes):
    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to split text into chunks

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to get vector store

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    vector_store.save_local("faiss_index")

# Function for the Prompt Template

def get_qa_chain(company_name, role_name, job_desc):
    prompt_template = f"""
    Context: \n {{context}}?\n
    Description: \n {{Description}}?\n
    Based on the resume PDF file provided and the context of the {job_desc} for the role of {role_name}, generate a cover letter for the user. 
    Address the letter to "Dear Team {company_name}". 
    Start with the sentence: "Based on the job posting for the role of {role_name}, I would like to express my candidacy, as I believe my current skill set and future career goals align well with the position offered." 
    Highlight the user's work experience mentioned in the resume pdf(if any), projects(if any), volunteer work, extracurricular activities, and soft skills. 
    Focus particularly on the projects, discuss the technologies used in these projects. Conclude by expressing enthusiasm for the opportunity and looking forward to discussing the application further. 
    Try to add good keywords, numbers and phrases to make the cover letter more appealing.
    Highlight the skills and projects more that are relevant to the job description but do not add anything if not already in pdf.
    End the letter with the user's name.
    If the user has no work experience, mention that they are a fresh graduate and are eager to learn and contribute to the company. 
    If the user has no projects, mention that they are eager to work on projects and learn new technologies at the company.
    If the details about user are not available in the provided context, just say "Resume data insufficient. Please modify resume."
    Do not provide wrong information in the cover letter.
    Do not fabricate any information.\n\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature=0.5)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["Context", "Description"])
    
    qa_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return qa_chain

# Function to get user input

def user_input(job_desc, company_name, role_name):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(job_desc)

    chain = get_qa_chain(company_name, role_name, job_desc)
    
    response = chain(
        {
            "input_documents": docs,
            "Description": job_desc,
            "Company_Name": company_name,
            "Role_Name": role_name
        },
        return_only_outputs = True)
    
    # formatted_response = response.replace("[Company_Name]", company_name).replace("[role]", role_name) 
    print(response)
    st.write("Cover Letter: ", response["output_text"])


# Streamlit App Configuration

def main():
    st.title("Jobless Developer's Aid 👩🏾‍💻")
    st.header("Upload resume PDF file, enter the job description and generate a cover letter.")
    
    # Delete faiss_index if it exists
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
    
    # File upload
    pdf_docs = st.file_uploader("Upload Resume PDF and Click on Submit & Process", type="pdf")
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            if pdf_docs is not None:
                pdf_bytes = pdf_docs.read()
                text = get_pdf_text(pdf_bytes)
                text_chunks = get_text_chunks(text)
                get_vector_store(text_chunks)
            st.success("Processing Done! 🎉")

    # Job description input
    company_name = st.text_input("Enter Company Name")
    role_name = st.text_input("Enter Role Name")
    job_desc = st.text_area("Copy-Paste The Job Description")
 
    # Generate cover letter button
    if st.button("Generate Cover Letter"):
        if pdf_docs is not None and company_name and role_name and job_desc:
            user_input(job_desc, company_name, role_name)
        else:
            st.write("Please upload a resume PDF file.")


if __name__ == "__main__":
    main()



