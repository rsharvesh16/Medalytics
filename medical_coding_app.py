import streamlit as st
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import FAISS
import logging
from medical_coding import process_medical_coding
from analytics import process_analytics
from patient_insurance_analytics import process_patient_insurance_analytics
import streamlit_lottie as st_lottie
import requests
from streamlit_option_menu import option_menu

# Set page config at the very beginning
st.set_page_config(page_title="Medical Assistant", layout="wide")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Function to get LLaMA 3 model
def get_llama3_llm():
    try:
        llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 2000})
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLaMA 3 model: {str(e)}")
        st.error(f"Unable to load LLaMA 3 model: {str(e)}")
        return None

# Load Lottie animation
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error loading Lottie animation: {str(e)}")
        return None

def main():
    # Custom CSS for dark theme
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stApp {
        background-color: #1E1E1E;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: #FFFFFF;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #2C3E50;
        color: #FFFFFF;
        border-radius: 5px;
    }
    .stSelectbox>div>div>select {
        background-color: #2C3E50;
        color: #FFFFFF;
        border-radius: 5px;
    }
    .title {
        text-align: center;
        font-family: 'Roboto', sans-serif;
        color: #4CAF50;
        font-size: 3em;
        margin-bottom: 30px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 14px;
        color: #888;
    }
    .stTabs>div>div>div {
        background-color: #2C3E50;
        color: #FFFFFF;
    }
    .stTabs>div>div>div[data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs>div>div>div[data-baseweb="tab"] {
        background-color: #1E1E1E;
        color: #FFFFFF;
        border-radius: 5px 5px 0 0;
    }
    .stTabs>div>div>div[data-baseweb="tab"][aria-selected="true"] {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title and Lottie animation
    st.markdown("<h1 class='title'>MEDALYTICS</h1>", unsafe_allow_html=True)
    lottie_medical = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_5njp3vgg.json")
    if lottie_medical:
        st_lottie.st_lottie(lottie_medical, speed=1, height=200, key="initial_medical_app")
    else:
        st.warning("Failed to load Lottie animation.")

    # Sidebar for operation selection
    with st.sidebar:
        st.markdown('<div style="text-align: center;"><h2 style="color: #4CAF50;">Select Operation</h2></div>', unsafe_allow_html=True)
        selected = option_menu(
            menu_title=None,
            options=["Medical Coding", "Analytics", "Patient Insurance Analytics"],
            icons=["file-earmark-medical", "graph-up", "clipboard-data"],
            menu_icon="cast",
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "#2C3E50"},
                "icon": {"color": "#4CAF50", "font-size": "20px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#3A506B"},
                "nav-link-selected": {"background-color": "#4CAF50"},
            }
        )
        st.sidebar.title("About This Project")
        st.sidebar.info(
            "This Medical Coding Assistant was made for Insurance Companies which is majorly for processing the Patients Insurance Payments using Medical Coding."
        )

    # Attempt to load FAISS index for ICD codes
    try:
        icd_vectorstore = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS index loaded successfully.")
    except FileNotFoundError:
        st.error("FAISS index not found. Please ensure the 'faiss_index' directory exists with the necessary files.")
        return
    except Exception as e:
        st.error(f"Error loading FAISS index: {str(e)}")
        logger.error(f"Failed to load FAISS index: {str(e)}")
        return

    # Main content area
    if selected == "Medical Coding":
        if get_llama3_llm() is not None:
            process_medical_coding(st, icd_vectorstore, get_llama3_llm, bedrock_embeddings)
    elif selected == "Analytics":
        if get_llama3_llm() is not None:
            process_analytics(st, get_llama3_llm)
    elif selected == "Patient Insurance Analytics":
        if get_llama3_llm() is not None:
            process_patient_insurance_analytics(st, get_llama3_llm)

    # Footer
    st.markdown('<div class="footer">Made with ❤️ by Team Qube</div>', unsafe_allow_html=True)

# Entry point for the application
if __name__ == "__main__":
    main()