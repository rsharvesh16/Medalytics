import streamlit as st
import pandas as pd
import plotly.express as px
from PyPDF2 import PdfReader
import io
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AWS Bedrock clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

def get_llama3_llm():
    try:
        llm = Bedrock(
            model_id="meta.llama3-70b-instruct-v1:0",
            client=bedrock,
            model_kwargs={'max_gen_len': 2000}
        )
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLaMA 3 model: {str(e)}")
        st.error(f"Unable to load LLaMA 3 model: {str(e)}")
        return None

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def create_or_load_vectorstore(texts, store_name):
    try:
        path = Path(f"{store_name}")
        if path.exists():
            vectorstore = FAISS.load_local(str(path), bedrock_embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Loaded existing vector store: {store_name}")
        else:
            vectorstore = FAISS.from_texts(texts, bedrock_embeddings)
            vectorstore.save_local(str(path))
            logger.info(f"Created new vector store: {store_name}")
        return vectorstore
    except Exception as e:
        logger.error(f"Error with vector store: {str(e)}")
        raise

analysis_prompt = """
Based on the provided medical reports, perform a detailed analysis focusing on insurance-related aspects. The analysis should include:

1. Key medical procedures and treatments mentioned
2. Associated costs for each procedure
3. Diagnosis and ICD codes if available
4. Length of hospital stays
5. Any pre-existing conditions mentioned
6. Medications prescribed
7. Insurance Amount Can Be Claimed:

Present the information in the following format:
1. Summary of Medical Procedures:
   - List each procedure with associated costs

2. Diagnosis Analysis:
   - List each diagnosis with ICD codes

3. Hospital Stay Information:
   - Duration and associated costs

4. Medication Summary:
   - List prescribed medications and their purposes

5. Insurance Considerations:
   - Potential coverage issues
   - Recommended insurance actions

6. Possible Amount can Be Recovered:
    - Based on the medications, diseases mentioned and bill amount mentioned, Give an approximate Insurance amount (In Rupees) can be claimed for the particular patient.
Use ONLY the information provided in the context. Do not make up or assume any information.

Context:
{context}

Question: {question}
"""

ANALYSIS_PROMPT = PromptTemplate(
    template=analysis_prompt,
    input_variables=["context", "question"]
)

def get_report_analysis(vectorstore, llm):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": ANALYSIS_PROMPT}
    )
    return qa_chain.run("Provide a comprehensive analysis of the medical reports.")

def extract_data_for_visualization(analysis_text):
    # This function would parse the analysis text to extract data for visualizations
    # For now, we'll return placeholder data
    try:
        # You would implement actual parsing logic here based on the structure of your analysis
        procedures_data = pd.DataFrame({
            'Procedure': ['Procedure 1', 'Procedure 2', 'Procedure 3'],
            'Cost': [1000, 2000, 1500]
        })
        
        diagnosis_data = pd.DataFrame({
            'Diagnosis': ['Diagnosis 1', 'Diagnosis 2', 'Diagnosis 3'],
            'ICD Code': ['A123', 'B456', 'C789']
        })
        
        return procedures_data, diagnosis_data
    except Exception as e:
        logger.error(f"Error extracting visualization data: {str(e)}")
        return None, None

def process_patient_insurance_analytics(st, get_llama3_llm):  # Updated to accept both arguments
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Patient Insurance Analytics</h1>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload Patient Medical Reports (PDF)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")

        if st.button("Analyze Reports"):
            with st.spinner("Processing medical reports..."):
                all_texts = []
                for file in uploaded_files:
                    text = extract_text_from_pdf(file)
                    if text:
                        all_texts.append(text)
                
                if not all_texts:
                    st.error("No text could be extracted from the uploaded PDFs.")
                    return

                try:
                    # Create or load vector store
                    vectorstore = create_or_load_vectorstore(all_texts, "faiss_index2")
                    
                    # Get LLM
                    llm = get_llama3_llm()
                    if llm is None:
                        st.error("Failed to initialize the language model.")
                        return
                    
                    # Generate analysis
                    analysis = get_report_analysis(vectorstore, llm)
                    
                    # Display analysis
                    st.markdown("<h2 style='color: #4CAF50;'>Medical Reports Analysis</h2>", unsafe_allow_html=True)
                    st.markdown(f"<div style='background-color: #2C3E50; padding: 20px; border-radius: 10px;'>{analysis}</div>", unsafe_allow_html=True)
                    
                    # Extract data for visualizations
                    procedures_data, diagnosis_data = extract_data_for_visualization(analysis)
                    
                    if procedures_data is not None and diagnosis_data is not None:
                        # Create visualizations
                        st.markdown("<h2 style='color: #4CAF50;'>Visualizations</h2>", unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig1 = px.bar(procedures_data, x='Procedure', y='Cost', 
                                         title='Medical Procedures Cost Breakdown')
                            fig1.update_layout(template="plotly_dark")
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            fig2 = px.pie(diagnosis_data, names='Diagnosis', title='Diagnosis Distribution')
                            fig2.update_layout(template="plotly_dark")
                            st.plotly_chart(fig2, use_container_width=True)
                    
                except Exception as e:
                    logger.error(f"Error during analysis: {str(e)}")
                    st.error(f"An error occurred during analysis: {str(e)}")
    
    else:
        st.warning("Please upload at least one medical report PDF to proceed.")

if __name__ == "__main__":
    process_patient_insurance_analytics(st, get_llama3_llm)  # This should now match the function definition