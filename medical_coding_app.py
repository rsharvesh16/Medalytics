# main.py
import streamlit as st
import boto3
from botocore.config import Config
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
import time
from botocore.exceptions import ClientError

# Set page config at the very beginning
st.set_page_config(page_title="Medical Assistant", layout="wide")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BedrockClient:
    def __init__(self):
        self.session = boto3.Session()
        self.client = None
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms minimum between requests
        self.max_retries = 5
        self.base_delay = 1  # Base delay for exponential backoff
        self.initialize_client()

    def initialize_client(self):
        try:
            # Create config with retry settings
            config = Config(
                retries=dict(
                    max_attempts=self.max_retries,
                    mode='adaptive'
                ),
                connect_timeout=10,
                read_timeout=30
            )
            
            self.client = self.session.client(
                service_name="bedrock-runtime",
                region_name="us-east-1",  # Explicitly set region
                config=config
            )
            logger.info("Bedrock client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {str(e)}")
            raise

    def handle_request(self, operation, **kwargs):
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                # Implement rate limiting
                current_time = time.time()
                time_since_last_request = current_time - self.last_request_time
                if time_since_last_request < self.min_request_interval:
                    time.sleep(self.min_request_interval - time_since_last_request)

                # Make the request
                response = getattr(self.client, operation)(**kwargs)
                self.last_request_time = time.time()
                return response

            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ThrottlingException':
                    retry_count += 1
                    if retry_count < self.max_retries:
                        delay = self.base_delay * (2 ** retry_count)  # Exponential backoff
                        logger.warning(f"Throttling occurred. Attempt {retry_count}/{self.max_retries}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue
                logger.error(f"AWS Bedrock error: {error_code}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise

# Initialize the Bedrock client
bedrock_client = BedrockClient()

# Initialize embeddings with retry mechanism
def get_bedrock_embeddings():
    try:
        return BedrockEmbeddings(
            model_id="amazon.titan-embed-g1-text-02",  # Updated model ID
            client=bedrock_client.client
        )
    except Exception as e:
        logger.error(f"Error initializing embeddings: {str(e)}")
        st.error("Failed to initialize embeddings. Please try again.")
        return None

# Initialize Mistral model with retry mechanism
def get_mistral_llm():
    try:
        llm = Bedrock(
            model_id="meta.llama3-70b-instruct-v1:0",  # Mistral 7B model ID
            client=bedrock_client.client,
            model_kwargs={
                "max_gen_len": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        )
        return llm
    except Exception as e:
        logger.error(f"Error initializing Mistral model: {str(e)}")
        st.error("Unable to initialize Mistral model. Please try again later.")
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
    # Title and Lottie animation
    st.markdown("<h1 class='title'>MEDALYTICS</h1>", unsafe_allow_html=True)
    lottie_medical = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_5njp3vgg.json")
    if lottie_medical:
        st_lottie.st_lottie(lottie_medical, speed=1, height=200, key="initial_medical_app")
    else:
        st.warning("Failed to load Lottie animation.")

    # Initialize embeddings
    bedrock_embeddings = get_bedrock_embeddings()
    if not bedrock_embeddings:
        st.error("Failed to initialize embeddings. Please refresh the page.")
        return

    # Attempt to load FAISS index
    try:
        icd_vectorstore = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS index loaded successfully.")
    except Exception as e:
        st.error(f"Error loading FAISS index: {str(e)}")
        logger.error(f"Failed to load FAISS index: {str(e)}")
        return

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

    # Main content area with error handling
    if selected == "Medical Coding":
        process_medical_coding(st, icd_vectorstore, get_mistral_llm, bedrock_embeddings)
    elif selected == "Analytics":
        process_analytics(st, get_mistral_llm)
    elif selected == "Patient Insurance Analytics":
        process_patient_insurance_analytics(st, get_mistral_llm)

    # Footer
    st.markdown('<div class="footer">Made with ❤️ by Team Qube</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()