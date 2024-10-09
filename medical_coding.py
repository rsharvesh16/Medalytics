import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import fitz  # PyMuPDF
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
import streamlit_lottie as st_lottie
import requests
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load Lottie animation
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error loading Lottie animation: {str(e)}")
        return None

# OCR Function
def perform_ocr(image):
    try:
        text = pytesseract.image_to_string(image)
        if not text.strip():
            logger.warning("OCR produced empty result")
        return text
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        return ""

# PDF Processing Function
def process_pdf(file):
    text = ""
    try:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf_document:
            text += page.get_text()
        pdf_document.close()
        
        file.seek(0)
        pdf_bytes = file.read()
        images = convert_from_bytes(pdf_bytes)
        ocr_text = ""
        for image in images:
            ocr_text += perform_ocr(image) + "\n"
        
        combined_text = text + "\n" + ocr_text
        
        if not combined_text.strip():
            raise ValueError("No text could be extracted from the PDF.")
        
        return combined_text
    
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        raise

# Data Ingestion Implementation
def data_ingestion(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        text = process_pdf(uploaded_file)
    elif uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(uploaded_file)
        text = perform_ocr(image)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or image file.")

    if not text.strip():
        raise ValueError("No text could be extracted from the document. Please check your uploaded file.")

    logger.info(f"Extracted text (first 100 chars): {text[:100]}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_text(text)
    return [Document(page_content=t) for t in docs]

# Vector Embedding and Vector Store Implementation
def get_vector_store(docs, bedrock_embeddings):
    try:
        embeddings = bedrock_embeddings.embed_documents([doc.page_content for doc in docs])
        if not embeddings:
            raise ValueError("No embeddings were generated. Please check your documents and embedding model.")
        
        vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        return vectorstore_faiss
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        raise

# Summarization function
def summarize_documents(docs, llm):
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    return summary

prompt_template = """
Human: You are a medical coding assistant. Your task is to analyze the given medical text and provide the following information:

1. Under the subheading "CODINGS":
   Create a table with the following columns:
   - Provider Name (Mentioned Doctor)
   - Disease Name
   - ICD-10 Code
   - Chronicity Status (Acute/Chronic)
   
   Include all diseases mentioned in the text.

2. After the table, indicate whether the report satisfies the MEAT criteria in medical coding. (Monitoring, Evaluating, Assessing/Addressing, and Treating)

3. Under the subheading "DISEASE INFORMATION":
   For each disease mentioned:
   - Provide a brief explanation of the disease
   - Describe how it can be treated or managed

Use the following context to inform your response:
<context>
{context}
</context>

Question: {question}
Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def process_medical_coding(st, icd_vectorstore, get_llama3_llm, bedrock_embeddings):
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Medical Coding Assistant</h1>", unsafe_allow_html=True)

    # lottie_medical = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_17lwcjll.json")
    # if lottie_medical:
    #     st_lottie.st_lottie(lottie_medical, speed=1, height=200, key="initial_medical_coding")
    # else:
    #     st.image("https://via.placeholder.com/400x200?text=Medical+Coding+Assistant", use_column_width=True)

    tab1, tab2 = st.tabs(["Upload Medical Report", "Enter Medical Query"])

    with tab1:
        st.markdown("<h3 style='color: #4CAF50;'>Upload Medical Report</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["pdf", "png", "jpg", "jpeg"])
        
        if uploaded_file:
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    
    with tab2:
        st.markdown("<h3 style='color: #4CAF50;'>Enter Medical Query</h3>", unsafe_allow_html=True)
        user_query = st.text_area("", height=100, placeholder="Enter your medical query here...")

    if st.button("Process Input", key="process_button"):
        if uploaded_file:
            try:
                with st.spinner("Processing uploaded file..."):
                    docs = data_ingestion(uploaded_file)
                    vectorstore = get_vector_store(docs, bedrock_embeddings)
                    llm = get_llama3_llm()
                    summary = summarize_documents(docs, llm)
                    st.session_state.summary = summary
                    st.session_state.vectorstore = vectorstore
                    st.success("File processed successfully!")
            except Exception as e:
                logger.error(f"Error during processing: {str(e)}", exc_info=True)
                st.error(f"An error occurred during processing: {str(e)}\n\nPlease check if the file is readable and contains extractable text.")
        elif user_query:
            st.session_state.user_query = user_query
        else:
            st.warning("Please upload a file or enter a query before processing.")

    if 'summary' in st.session_state or 'user_query' in st.session_state:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h2 style='color: #4CAF50;'>Results</h2>", unsafe_allow_html=True)

        if 'summary' in st.session_state:
            with st.expander("Document Summary", expanded=True):
                st.markdown(f"<div style='background-color: #2C3E50; padding: 20px; border-radius: 10px;'>{st.session_state.summary}</div>", unsafe_allow_html=True)

        try:
            with st.spinner("Generating ICD codes and disease information..."):
                llm = get_llama3_llm()
                if 'summary' in st.session_state:
                    response = get_response_llm(llm, icd_vectorstore, st.session_state.summary)
                elif 'user_query' in st.session_state:
                    response = get_response_llm(llm, icd_vectorstore, st.session_state.user_query)
                else:
                    raise ValueError("No summary or query found in session state.")
                
                st.markdown("<h3 style='color: #4CAF50;'>Generated ICD Codes and Disease Information</h3>", unsafe_allow_html=True)
                st.markdown(f"<div style='background-color: #2C3E50; padding: 20px; border-radius: 10px;'>{response}</div>", unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            st.error(f"An error occurred while generating the response: {str(e)}\n\nPlease try again or contact support if the issue persists.")

    st.markdown("<br><hr>", unsafe_allow_html=True)
    # st.markdown("<p style='text-align: center; color: #888;'>Made with ❤️ by Team Qube</p>", unsafe_allow_html=True)

# Main function to run the Streamlit app
if __name__ == "__main__":
    st.set_page_config(page_title="Medical Coding Assistant", layout="wide")
    st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: #FFFFFF;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #2C3E50;
        color: #FFFFFF;
    }
    .stSelectbox>div>div>select {
        background-color: #2C3E50;
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Placeholder functions for demonstration
    # Replace these with your actual implementations
    icd_vectorstore = None  # Your actual ICD vectorstore
    bedrock_embeddings = None  # Your actual embedding model
    
    def get_llama3_llm():
        # Your actual LLaMa3 model initialization
        from langchain.llms import OpenAI
        return OpenAI(temperature=0.7)
    
    process_medical_coding(st, icd_vectorstore, get_llama3_llm, bedrock_embeddings)