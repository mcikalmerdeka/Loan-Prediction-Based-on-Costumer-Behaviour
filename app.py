import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import all preprocessing functions
from preprocessing import check_data_information
from preprocessing import initial_data_transform
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

# Initial transform for original dataframe
ori_df = initial_data_transform(ori_df)

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
with st.form("customer_prediction_form"):
    # Create a dictionary to store input values
    prediction_input = {}

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    # Split columns into two groups for layout
    target_col = "Risk_Flag"
    all_columns = [col for col in ori_df.columns if col != target_col]
    mid_point = len(all_columns) // 2

    with col1:
        for column in all_columns[:mid_point]:
            if pd.api.types.is_datetime64_any_dtype(ori_df[column]):
                prediction_input[column] = st.date_input(f"Enter {column}")

            elif pd.api.types.is_numeric_dtype(ori_df[column]):
                col_min = ori_df[column].min()
                col_max = ori_df[column].max()
                col_mean = ori_df[column].mean()

                prediction_input[column] = st.number_input(
                    f"Enter {column}",
                    min_value=float(col_min) if not pd.isna(col_min) else 0.0,
                    max_value=float(col_max) if not pd.isna(col_max) else None,
                    value=float(col_mean) if not pd.isna(col_mean) else 0.0,
                    step=0.1
                )
                
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
                col_min = ori_df[column].min()
                col_max = ori_df[column].max()
                col_mean = ori_df[column].mean()

                prediction_input[column] = st.number_input(
                    f"Enter {column}",
                    min_value=float(col_min) if not pd.isna(col_min) else 0.0,
                    max_value=float(col_max) if not pd.isna(col_max) else None,
                    value=float(col_mean) if not pd.isna(col_mean) else 0.0,
                    step=0.1
                )
                
            elif pd.api.types.is_categorical_dtype(ori_df[column]) or ori_df[column].dtype == 'object':
                unique_values = ori_df[column].unique()
                prediction_input[column] = st.selectbox(
                    f'Select {column}',
                    options=list(unique_values)
                )
            
            else:
                prediction_input[column] = st.text_input(f'Enter {column}')
        
    # Submit button
    submit_prediction_button = st.form_submit_button("Predict Customer Loan Status")

if submit_prediction_button:
    # Convert input data into dataframe
    input_df = pd.DataFrame([prediction_input])

    # Show input data
    st.subheader("New Customer Input Data")
    st.write(input_df)

    # Import the preprocessed original data
    url_ori_processed = "https://raw.githubusercontent.com/mcikalmerdeka/Loan-Prediction-Based-on-Costumer-Behaviour/main/df_model_rewrite.csv"
    ori_df_preprocessed = pd.read_csv(url_ori_processed)
    ori_df_preprocessed = ori_df_preprocessed.loc[:, ori_df_preprocessed.columns != target_col]

    # Show input data
    st.subheader("Original Preprocessed Data")
    st.write(ori_df_preprocessed.head())

    # Preprocessing steps

    ## 1. Feature Engineering (Create only used feature in the model not all from the notebook)
    try:
        # A. Generation
        def assign_generation(age):
            if age <= 27:
                return 'Generation Z'
            elif age <= 43:
                return 'Generation Millenials'
            elif age < 59:
                return 'Generation X'
            elif age < 69:
                return 'Boomers II'
            elif age <= 78:
                return 'Boomers I'
            else:
                return 'Other'

        input_df['Generation'] = input_df['Age'].apply(assign_generation)

        # Ratio Experience by Age
        input_df['Experience_Age_Ratio'] = input_df['Experience'] / input_df['Age']

        # B. Profession grouping
        profession_groups = {
        'engineering': ['engineer', 'mechanical engineer', 'civil engineer', 'industrial engineer', 'design engineer', 'chemical engineer', 'biomedical engineer', 'computer hardware engineer', 'petroleum engineer', 'surveyor', 'drafter'],
        'technology': ['software developer', 'computer operator', 'technology specialist', 'web designer', 'technician'],
        'healthcare': ['physician', 'dentist', 'surgeon', 'psychologist'],
        'finance': ['economist', 'financial analyst', 'chartered accountant'],
        'design': ['architect', 'designer', 'graphic designer', 'fashion designer', 'artist'],
        'aviation': ['flight attendant', 'air traffic controller', 'aviator'],
        'government public service': ['civil servant', 'politician', 'police officer', 'magistrate', 'army officer', 'firefighter', 'lawyer', 'official', 'librarian'],
        'business management' : ['hotel manager', 'consultant', 'secretary'],
        'science research' : ['scientist', 'microbiologist', 'geologist', 'statistician', 'analyst'],
        'miscellaneous': ['comedian', 'chef', 'technical writer']}

        input_df['Profession_Group'] = input_df['Profession'].map({prof: group for group, prof_list in profession_groups.items() for prof in prof_list})

        # C. State grouping
        def state_group(state) :
            if state in ['uttar pradesh', 'haryana', 'jammu and kashmir', 'punjab', 'uttarakhand', 'chandigarh', 'delhi', 'himachal pradesh'] :
                return 'north_zone'
            elif state in ['bihar', 'jharkhand', 'odisha', 'west bengal', 'assam', 'sikkim', 'tripura', 'mizoram', 'manipur'] :
                return 'east_zone'
            elif state in ['andhra pradesh', 'tamil nadu', 'karnataka', 'telangana', 'kerala', 'puducherry'] :
                return 'south_zone'
            else :
                return 'west_zone'

        input_df['State_Group'] = input_df['State'].apply(state_group)

        # D. City grouping
        def city_group(city):
            if city in ['new delhi', 'mumbai', 'kolkata', 'chennai', 'bangalore']:
                return 'metro'
            elif city in ['ahmedabad', 'hyderabad', 'pune', 'surat', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 'visakhapatnam', 'indore', 'thane',
                        'bhopal', 'pimpri-chinchwad', 'patna', 'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad', 'meerut', 'rajkot',
                        'varanasi', 'srinagar', 'amritsar', 'allahabad', 'jabalpur', 'gwalior', 'vijayawada', 'jodhpur', 'raipur', 'kota', 'guwahati', 'chandigarh city']:
                return 'urban'
            elif city in ['navi mumbai', 'kalyan-dombivli', 'vasai-virar', 'mira-bhayandar', 'thiruvananthapuram', 'bhiwandi', 'noida', 'bhopal', 'howrah', 'saharanpur',
                        'berhampur', 'suryapet', 'muzaffarpur', 'nadiad', 'siliguri', 'bhavnagar', 'kurnool', 'tenali', 'satna', 'nandyal', 'etawah', 'morena', 'ballia',
                        'machilipatnam', 'mau', 'machilipatnam', 'bhagalpur', 'siwan', 'meerut', 'dibrugarh', 'gaya', 'darbhanga', 'hajipur', 'mirzapur', 'akola', 'satna',
                        'motihari', 'jalna', 'ramgarh', 'ozhukarai', 'saharsa', 'munger', 'farrukhabad', 'nangloi jat', 'thoothukudi', 'nagercoil', 'rourkela', 'jhansi', 'sultan pur majra']:
                return 'suburban'
            else:
                return 'rural'

        input_df['City_Group'] = input_df['City'].apply(city_group)

    except Exception as e:
        st.error(f"Error in feature engineering: {str(e)}")

    # Check data after feature engineering
    st.subheader("After Feature Engineering")
    st.write(input_df) 

    ## 2. Feature encoding
    try:
        # A. Handle ordinal encoding for Car_Ownership and Generation (unchanged)
        input_df['Car_Ownership'] = input_df['Car_Ownership'].map({'no': 0, 'yes': 1})

        input_df['Generation'] = input_df['Generation'].map({'Generation Z': 0,
                                                            'Generation Millenials': 1,
                                                            'Generation X' : 2,
                                                            'Boomers II' : 3,
                                                            'Boomers I' : 4,
                                                            'Other' : 5})
        
        # B. Handle one-hot encoding for Profession using original data categories
        unique_professions = ori_df_preprocessed.filter(like='Prof_').columns
        prof_encoded = pd.DataFrame(0, index=input_df.index, columns=unique_professions)
        if f"Prof_{input_df['Profession_Group'].iloc[0]}" in unique_professions:
            prof_encoded[f"Prof_{input_df['Profession_Group'].iloc[0]}"] = 1
        input_df = input_df.drop(columns=['Profession_Group'], errors='ignore')
        input_df = pd.concat([input_df, prof_encoded], axis=1)

        # C. Handle one-hot encoding for State using original data categories
        unique_states = ori_df_preprocessed.filter(like='State_').columns
        state_encoded = pd.DataFrame(0, index=input_df.index, columns=unique_states)
        if f"State_{input_df['State_Group'].iloc[0]}" in unique_states:
            state_encoded[f"State_{input_df['State_Group'].iloc[0]}"] = 1
        input_df = input_df.drop(columns=['State_Group'], errors='ignore')
        input_df = pd.concat([input_df, state_encoded], axis=1)

        # D. Handle one-hot encoding for City using original data categories
        unique_cities = ori_df_preprocessed.filter(like='City_').columns
        city_encoded = pd.DataFrame(0, index=input_df.index, columns=unique_cities)
        if f"City_{input_df['City_Group'].iloc[0]}" in unique_cities:
            city_encoded[f"City_{input_df['City_Group'].iloc[0]}"] = 1
        input_df = input_df.drop(columns=['City_Group'], errors='ignore')
        input_df = pd.concat([input_df, city_encoded], axis=1)

        # Ensure all expected columns are present before moving to scaling
        expected_columns = ori_df_preprocessed.columns.tolist()
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        for col in input_df.columns:
            if col not in expected_columns:
                input_df.drop(columns=col, inplace=True)

        # Reorder and match columns to match training data
        input_df = input_df[expected_columns]
    
    except Exception as e:
        st.error(f"Error in feature encoding: {str(e)}")
        st.write("Debug information:")
        st.write("Current columns:", input_df.columns.to_list())
        st.write("Expected columns:", expected_columns)

    # Check data after encoding
    st.subheader("After Feature Encoding and Drop Columns")
    st.write(input_df)

    ## 3. Feature Scaling
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    try:
        # Define feature groups for targeted scaling
        # Each feature group requires a specific scaling approach
        log_transform_features = ['Experience_Age_Ratio']  # Features with high skewness
        count_uniform_features = ['Income']  # Discrete features representing quantities

        scalers = {}  # Dictionary to store fitted scalers and feature info

        # TRAINING DATA SCALING
        # Step 1: Scale count/uniform features using MinMaxScaler
        ori_df_preprocessed[count_uniform_features] = minmax_scaler.fit_transform(ori_df_preprocessed[count_uniform_features])
        scalers['count_uniform'] = minmax_scaler

        # Step 2: Scale skewed features using log transformation and standardization

        ori_df_preprocessed[log_transform_features] = np.log1p(ori_df_preprocessed[log_transform_features])
        ori_df_preprocessed[log_transform_features] = standard_scaler.fit_transform(ori_df_preprocessed[log_transform_features])
        scalers['log_transform'] = standard_scaler

        # INFERENCE DATA SCALING
        # Apply the same transformations used in training data
        # Use .transform() instead of .fit_transform() to maintain training distribution

        # Scale count/uniform features
        input_df[count_uniform_features] = scalers['count_uniform'].transform(input_df[count_uniform_features])

        # Scale skewed features
        input_df[log_transform_features] = np.log1p(input_df[log_transform_features])
        input_df[log_transform_features] = scalers['log_transform'].transform(input_df[log_transform_features])

    except Exception as e:
        st.error(f"Error in feature scaling: {str(e)}")

    # Check data after scaling
    st.subheader("After Feature Scaling")
    st.write(input_df)

    ## Prediction section
    st.subheader("Prediction Section")

    # Create a copy for preprocessing result
    model_df = input_df.copy()

    # Display the prediction result
    try:
        prediction = model.predict(model_df)

        # Display prediction result with explanation
        if prediction[0] == 0:
            st.success("The customer is predicted as **Not Default**.\n\n**Not Default** means the customer is likely to repay the loan on time.")
        else:
            st.error("The customer is predicted as **Default**.\n\n**Default** means the customer is likely to fail to repay the loan on time.")
    
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")