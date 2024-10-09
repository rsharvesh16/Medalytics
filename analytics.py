import streamlit as st
import plotly.express as px
import pandas as pd
import random
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit_lottie as st_lottie
import requests

# Load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Analytics prompt template (unchanged)
analytics_prompt = """
You are an AI assistant specializing in healthcare analytics for insurance companies. Based on the disease {disease}, provide the following information:

1. Prevalence: Estimate the prevalence of the disease in the general population.
2. Age Distribution: Describe how the disease affects different age groups.
3. Gender Distribution: Explain any gender-specific trends for this disease.
4. Average Treatment Cost: Estimate the average cost of treatment for this disease.
5. Hospitalization Rate: Provide an estimate of how often this disease requires hospitalization.
6. Comorbidities: List common comorbidities associated with this disease.
7. Prevention Strategies: Suggest effective prevention strategies for this disease.
8. Long-term Prognosis: Describe the long-term outlook for patients with this disease.

For each point, provide specific numbers or percentages where applicable. These will be used to generate visualizations.
"""

# Define prompt template
ANALYTICS_PROMPT = PromptTemplate(template=analytics_prompt, input_variables=["disease"])

# Function to generate visualizations (unchanged)
def generate_age_distribution_chart(age_data):
    fig = px.bar(age_data, x='Age Group', y='Percentage', title='Age Distribution')
    fig.update_layout(template="plotly_dark")
    return fig

def generate_gender_distribution_chart(gender_data):
    fig = px.pie(gender_data, values='Percentage', names='Gender', title='Gender Distribution')
    fig.update_layout(template="plotly_dark")
    return fig

def generate_cost_breakdown_chart(cost_data):
    fig = px.pie(cost_data, values='Cost', names='Category', title='Treatment Cost Breakdown')
    fig.update_layout(template="plotly_dark")
    return fig

def generate_comorbidities_chart(comorbidities_data):
    fig = px.bar(comorbidities_data, x='Comorbidity', y='Percentage', title='Common Comorbidities')
    fig.update_layout(template="plotly_dark")
    return fig

# Placeholder for LLM response parsing (unchanged)
def parse_llm_response(response):
    # For this example, we'll use mock data. You would parse the response from LLM here.
    age_data = pd.DataFrame({
        'Age Group': ['0-18', '19-40', '41-60', '61+'],
        'Percentage': [random.randint(5, 20), random.randint(20, 40), random.randint(30, 50), random.randint(10, 30)]
    })
    
    gender_data = pd.DataFrame({
        'Gender': ['Male', 'Female'],
        'Percentage': [random.randint(40, 60), random.randint(40, 60)]
    })
    
    cost_data = pd.DataFrame({
        'Category': ['Medication', 'Hospital Stay', 'Outpatient Care', 'Tests'],
        'Cost': [random.randint(2000, 7000), random.randint(8000, 20000), random.randint(1000, 4000), random.randint(500, 2000)]
    })
    
    comorbidities_data = pd.DataFrame({
        'Comorbidity': ['Hypertension', 'Diabetes', 'Obesity', 'Heart Disease'],
        'Percentage': [random.randint(20, 60), random.randint(10, 40), random.randint(15, 35), random.randint(10, 25)]
    })
    
    return age_data, gender_data, cost_data, comorbidities_data

# Main function to handle analytics
def process_analytics(st, get_llama3_llm):
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Disease Analytics for Healthcare Insurance</h1>", unsafe_allow_html=True)

    # Load Lottie animation
    lottie_health = load_lottieurl("https://lottie.host/503a0611-68e0-469c-bbd4-91a9cf6b10c1/Ilr3bzdupn.json")
    if lottie_health:
        st_lottie.st_lottie(lottie_health, speed=1, height=200, key="initial_analytics")
    else:
        st.warning("Failed to load Lottie animation.")
    # Get disease name from user
    disease_name = st.text_input("Enter a disease name:", key="disease_input")

    # Run analysis when button is pressed
    if st.button("Generate Analytics", key="generate_button"):
        if disease_name:
            # Display a spinner while generating analytics
            with st.spinner(f"Generating analytics for {disease_name}..."):
                # Run LLM chain for disease analytics
                llm = get_llama3_llm()  # Retrieve the language model
                chain = LLMChain(llm=llm, prompt=ANALYTICS_PROMPT)
                response = chain.run(disease=disease_name)  # Pass disease name to the prompt
                
                # Display response from the LLM
                st.markdown(f"<h2 style='color: #4CAF50;'>Disease Analysis for {disease_name}</h2>", unsafe_allow_html=True)
                st.markdown(f"<div style='background-color: #2C3E50; padding: 20px; border-radius: 10px;'>{response}</div>", unsafe_allow_html=True)
                
                # Parse LLM response for visualizations (mock data for now)
                age_data, gender_data, cost_data, comorbidities_data = parse_llm_response(response)

                # Generate and display visualizations
                st.markdown("<h2 style='color: #4CAF50;'>Visualizations</h2>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)

                with col1:
                    st.plotly_chart(generate_age_distribution_chart(age_data), use_container_width=True)
                    st.plotly_chart(generate_cost_breakdown_chart(cost_data), use_container_width=True)

                with col2:
                    st.plotly_chart(generate_gender_distribution_chart(gender_data), use_container_width=True)
                    st.plotly_chart(generate_comorbidities_chart(comorbidities_data), use_container_width=True)
                
                # Display a summary report
                st.markdown("<h2 style='color: #4CAF50;'>Insurance Company Report</h2>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div style='background-color: #2C3E50; padding: 20px; border-radius: 10px;'>
                    Based on the analysis of {disease_name}, here are key points for insurance consideration:
                    
                    1. <strong>Prevalence</strong>: {random.randint(1, 10)}% of the population is affected by this disease.
                    2. <strong>Average Treatment Cost</strong>: ${random.randint(5000, 50000)} per year.
                    3. <strong>Hospitalization Rate</strong>: {random.randint(10, 50)}% of patients require hospitalization.
                    4. <strong>Long-term Prognosis</strong>: {random.choice(['Generally good with proper management', 
                       'Variable depending on individual factors', 
                       'May require ongoing care and monitoring'])}
                    
                    <strong>Recommendations</strong>:
                    1. Implement targeted prevention programs to reduce incidence and severity.
                    2. Adjust premiums based on risk factors and prevention adherence.
                    3. Develop partnerships with healthcare providers for cost-effective treatments.
                    4. Invest in patient education to improve self-management and reduce complications.
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a disease name before generating analytics.")

# Run the process_analytics function in Streamlit
if __name__ == "__main__":
    process_analytics(st, get_llama3_llm)