import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import all preprocession functions
from preprocessing import check_data_information

from feature_definitions import get_feature_definitions

#  Page Config
st.set_page_config(page_title="Loan Prediction App", layout="wide")
st.title("Loan Prediction Analysis")

# Add information about the app
with st.expander("**Read Instructions First: About This App**"):
    st.write("""
    ## Loan Default Prediction Application

    ### 📌 Purpose
    - This app uses machine learning to predict the creditworthiness of loan applicants.
    - The goal is to help financial institutions make more accurate lending decisions by identifying potential loan defaults.

    ### 🎯 Key Business Metrics

    #### Primary Metric: Default Rate (%)
    - Measures the percentage of customers who fail to repay their loans
    - Calculated as: (Number of Loan Defaults / Total Number of Customers) × 100
    - Lower default rate indicates more effective risk assessment
    - Critical for minimizing financial losses and improving lending strategies

    #### Secondary Metric: Approval Time
    - Tracks the time taken to process loan applications
    - Aims to streamline and accelerate the loan approval process
    - Reduces operational costs and improves customer satisfaction
    - Measures efficiency of the loan evaluation system

    ### 🔍 How to Use the App

    #### Data Input Options:
    - Upload Your Own Dataset
        - Ensure your dataset matches the required structure for loan prediction
        - Recommended columns include: income, credit score, loan amount, employment status, etc.
    - Use Source Dataset
        - Option to load a pre-existing loan application dataset
        - Provides a ready-to-use reference dataset for analysis

    #### Preprocessing Steps:
    The application will systematically process the loan application data through several crucial stages:
    1. Data Type Conversion
        - Standardize data types for accurate analysis
    2. Missing Value Handling
        - Implement appropriate strategies for managing incomplete data
    3. Outlier Detection and Management
        - Identify and address extreme or anomalous data points
    4. Feature Engineering
        - Create derived features to enhance predictive power
        - Categorize continuous variables (e.g., age groups, income brackets)
    5. Feature Encoding
        - Convert categorical variables into numerical representations
    6. Feature Selection
        - Identify and retain most relevant predictors of loan default
    7. Data Scaling
        - Normalize features to ensure balanced model training
    8. Data Resampling
        - Apply resampling techniques like Undersampling or Oversampling

    ### 🤖 Prediction Capabilities
    - Train multiple machine learning models for loan default prediction
    - Visualize model performance and feature importance
    - Generate detailed insights into credit risk factors

    ### 🔮 New Application Prediction
    - Input new loan applicant details
    - Receive instant prediction of default probability
    - Get comprehensive breakdown of risk factors

    ### ⚠️ Important Notes
    - Model predictions are probabilistic and should be used as a decision support tool
    - Final lending decisions should combine model insights with human expertise
    - Continuous monitoring and updating of the model is recommended to maintain accuracy
    """)

# Load pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('tuned_knn_model.joblib')

model = load_model()

# Load original CSV data form author github
url_ori = "https://raw.githubusercontent.com/mcikalmerdeka/Loan-Prediction-Based-on-Costumer-Behaviour/main/Training%20Data.csv"
ori_df = pd.read_csv(url_ori)

# Display raw data
st.subheader("Raw Data Preview")
st.write(ori_df.head())

# Display data information
st.subheader("Data Information")
st.write(check_data_information(ori_df, ori_df.columns))

# Add Data Dictionary section
with st.expander("📚 Data Dictionary"):
    st.markdown("### Feature Information")
    
    # Create DataFrame from feature definitions
    definitions = get_feature_definitions()
    feature_df = pd.DataFrame.from_dict(definitions, orient='index')
    
    # Reorder columns and reset index to show feature names as a column
    feature_df = feature_df.reset_index().rename(columns={'index': 'Feature Name'})
    feature_df = feature_df[['Feature Name', 'description', 'data_type', 'specific_type']]
    
    # Rename columns for display
    feature_df.columns = ['Feature Name', 'Description', 'Data Type', 'Specific Type']
    
    # Display as a styled table
    st.dataframe(
        feature_df.style.set_properties(**{
            'background-color': 'white',
            'color': 'black',
            'border-color': 'lightgrey'
        })
    )
    
    st.markdown("""
    **Note:**
    - Categorical (Nominal): Categories without any natural order
    - Categorical (Ordinal): Categories with a natural order
    - Numerical (Discrete): Whole numbers
    - Numerical (Continuous): Any numerical value
    """)

# Input customer data
st.subheader("Enter Customer Data")
with st.form("prediction_form"):
    # Create a dictionary to store input values
    prediction_input = {}

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    # Split columns into two groups for layout
    all_columns = ori_df.columns.to_list()
    mid_point = len(all_columns) // 2

    with col1:
        for column in all_columns[:mid_point]:
            if pd.api.types.is_datetime64_any_dtype(ori_df[column]):
                prediction_input[column] = st.date_input(f"Enter {column}")

            elif pd.api.types.is_numeric_dtype(ori_df[column]):
                prediction_input[column] = st.number_input(
                    f"Enter {column}", min_value=0)
                
            elif pd.api.types.is_categorical_dtype(ori_df[column]) or ori_df[column].dtype == 'object':
                unique_values = ori_df[column].unique()
                prediction_input[column] = st.selectbox(
                    f'Select {column}',
                    options=list(unique_values)
                )
            
            else:
                prediction_input[column] = st.text_input(f'Enter {column}')

    with col2:
        for column in all_columns[mid_point:]:
            if pd.api.types.is_datetime64_any_dtype(ori_df[column]):
                prediction_input[column] = st.date_input(f"Enter {column}")

            elif pd.api.types.is_numeric_dtype(ori_df[column]):
                prediction_input[column] = st.number_input(
                    f"Enter {column}", min_value=0)
                
            elif pd.api.types.is_categorical_dtype(ori_df[column]) or ori_df[column].dtype == 'object':
                unique_values = ori_df[column].unique()
                prediction_input[column] = st.selectbox(
                    f'Select {column}',
                    options=list(unique_values)
                )
            
            else:
                prediction_input[column] = st.text_input(f'Enter {column}')
        
    # Submit button
    submit_button = st.form_submit_button("Predict Customer Loan Status")

if submit_button:
    # Convert input data into dataframe
    input_df = pd.DataFrame([prediction_input])

    # Show input data
    st.subheader("New Customer Input Data")
    st.write(input_df)

    # Import the preprocessed original data
    url_ori_processed = "https://raw.githubusercontent.com/mcikalmerdeka/Loan-Prediction-Based-on-Costumer-Behaviour/main/df_model_rewrite.csv"
    ori_df_preprocessed = pd.read_csv(url_ori_processed)

    # Show input data
    st.subheader("Original Preprocessed Data")
    st.write(ori_df_preprocessed)