# MEDALYTICS - Medical Coding and Insurance Analytics System

## Overview
MEDALYTICS is a comprehensive medical coding and insurance analytics platform built using Streamlit and AWS Bedrock. The system helps insurance companies process patient insurance payments through automated medical coding, analytics, and patient insurance analysis.

The MEDALYTICS project is a comprehensive medical assistance system designed to streamline medical coding and insurance processing. At its core, it uses a Retrieval Augmented Generation (RAG) system where medical documents (PDFs or images) are first processed to extract text, which is then broken down into smaller, manageable chunks. These chunks are converted into numerical representations (embeddings) using AWS Bedrock's Titan model and stored in a FAISS vector database. When a user submits a query or document for analysis, the system retrieves the most relevant information from this database and combines it with the capabilities of the Meta Llama3 70B language model to generate accurate medical codes, analysis, and recommendations.
<br>
The system offers three main functionalities: Medical Coding, Analytics, and Patient Insurance Analytics. In Medical Coding, it processes medical documents to automatically generate ICD-10 codes, validate MEAT criteria, and provide disease information. The Analytics module analyzes historical medical data to identify trends and patterns, while the Patient Insurance Analytics component processes multiple medical reports to provide insurance claim analysis, cost breakdowns, and coverage recommendations. Throughout all these processes, the system maintains high accuracy by using the RAG approach, which ensures that the language model's responses are grounded in relevant medical documentation and knowledge. The system also implements robust error handling and rate limiting to ensure reliable performance when interacting with AWS services.

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
