# MEDALYTICS - Medical Coding and Insurance Analytics System

## Overview
MEDALYTICS is a comprehensive medical coding and insurance analytics platform built using Streamlit and AWS Bedrock. The system helps insurance companies process patient insurance payments through automated medical coding, analytics, and patient insurance analysis.

## Features
- **Medical Coding Assistant**
  - Upload and process medical reports (PDF/Images)
  - Automatic ICD-10 code generation
  - MEAT criteria validation
  - Disease information and treatment recommendations

- **Analytics Dashboard**
  - Process medical data
  - Generate visualizations
  - Trend analysis

- **Patient Insurance Analytics**
  - Process multiple medical reports
  - Insurance claim analysis
  - Cost breakdown visualization
  - Coverage recommendations

## Prerequisites
- Python 3.8 or higher
- AWS account with Bedrock access
- Tesseract OCR installed (for image processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rsharvesh16/Medalytics.git
cd medalytics
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Configure AWS credentials:
   - Create `~/.aws/credentials` or set environment variables:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

## Project Structure
```
medalytics/
├── medical_coding_app.py
├── medical_coding.py
├── analytics.py
├── patient_insurance_analytics.py
├── requirements.txt
├── faiss_index/
└── README.md
```

## Configuration
Create a `requirements.txt` file with the following dependencies:
```
streamlit
boto3
langchain-community
langchain
streamlit-lottie
streamlit-option-menu
pytesseract
pdf2image
plotly
pandas
PyPDF2
faiss-cpu
```

## Usage

1. Start the application:
```bash
streamlit run medical_coding_app.py
```

2. Access the web interface at `http://localhost:8501`

3. Select an operation from the sidebar:
   - Medical Coding
   - Analytics
   - Patient Insurance Analytics

4. Upload medical documents and follow the on-screen instructions

## AWS Bedrock Models Used
- Text Embeddings: `amazon.titan-embed-g1-text-02`
- LLM: `Meta LLama 3 70B Parameter Model`

## Error Handling
The application includes comprehensive error handling for:
- AWS service interruptions
- File processing issues
- Model availability
- Rate limiting

## Security Considerations
- Ensure AWS credentials are properly secured
- Do not expose sensitive patient information
- Follow HIPAA compliance guidelines when deploying in production

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details

## Support
For support and questions, please create an issue in the repository or contact the maintainers.

## Development Team
- Team Qube

## Acknowledgments
- AWS Bedrock for AI/ML capabilities
- Streamlit for the web interface
- Mistral AI for the language model
- Open-source community for various tools and libraries used

## Disclaimer
This tool is meant for assistance only. All medical coding and insurance decisions should be verified by qualified professionals.
