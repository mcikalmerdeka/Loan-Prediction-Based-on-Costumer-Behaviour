import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import streamlit as st
import joblib
import os

attribute_info = """
    - Id                 : Unique id of the user
    - Income             : Income of the user
    - Age                : Age of the user
    - Experience         : Professional experience of the user in years
    - Married/Single     : Marital status (married or single)
    - House_Ownership    : Homeownership status (owned, rented, or norent_noown)
    - Car_Ownership      : Carownership status (yes or no)
    - Profession         : Profession
    - CITY               : City of residence
    - STATE              : State of residence
    - CURRENT_JOB_YRS    : Years of experience in the current job
    - CURRENT_HOUSE_YRS  : Number of years in the current residence
    - Risk_Flag          : Defaulted on a loan (0 : not default, 1 : default/failed to pay)
    """

def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model

def run_ml_app() :
    st.subheader('Machine Learning Section')
    with st.expander('Attribute Info') :
        st.markdown(attribute_info)

    st.subheader('Input Your Data')
    with st.form('My Data') :
        # Add input elements for the encoded features using selectbox and number_input
        income = st.number_input('Income', 10310, 9999938)
        age = st.number_input('Age', 21, 79)
        experience = st.number_input('Experience', 0, 20)
        marital_status = st.selectbox('Marital Status', ['single' 'married'])
        house_ownership = st.selectbox('House Ownership', ['rented' 'norent_noown' 'owned'])
        car_ownership = st.selectbox('Car Ownership', ['no' 'yes'])
        profession = st.selectbox('Profession')
        city = st.selectbox('City')
        state = st.selectbox('State')
        current_job_years = st.number_input('Current Job Years', 0, 14)
        current_house_years = st.number_input('Current House Years', 10, 14)

        submitted = st.form_submit_button('Submit')
    
    if submitted :
        with st.expander('Variable Dictionary...') :
            result = {
                'Income' : income,
                'Age' : age,
                'Experience' : experience,
                'Marital Status' : marital_status,
                'House Ownership' : house_ownership,
                'Car Ownership' : car_ownership,
                'Profession' : profession,
                'City' : city,
                'State' : state,
                'Current Job Years' : current_job_years,
                'Current House Years' : current_house_years 
            }
            #st.write(result)

        df_original = pd.read_csv(os.path.join('Training Data.csv'))
        df_prediction = result
        df_prediction = pd.DataFrame(df_prediction, index=[0])
        st.write('Your Selected Options')
        st.table(df_prediction)

        # Initial transformation for prediction dataframe
        df_original.drop(columns='Id', inplace=True)

        # Cleaning Data
        df_prediction = 

        # Encoding Data

        # Generation
        # Function to determine the generation based on age
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
        
        df_prediction['Generation'] = df_prediction['Age'].apply(assign_generation)

        df_prediction['Generation'].replace({'Generation Z': 0,
                                    'Generation Millenials': 1,
                                    'Generation X' : 2,
                                    'Boomers II' : 3,
                                    'Boomers I' : 4,
                                    'Other' : 5},inplace=True)

        df_prediction['Generation'] = df_prediction['Generation'].astype('int64')

    



        # Scaling Data
        ms = MinMaxScaler()
        df_original_scaled = ms.fit_transform(df_original)
        df_prediction_scaled = ms.transform(df_prediction)

        # Prediction Section
        st.subheader('Prediction Result')
        single_array = np.array(df_prediction_scaled).reshape(1, -1)

        model = load_model('tuned_random_forest_model.joblib')

        prediction = model.predict(single_array)

        if prediction == 0 :
            st.info('This costumer has low risk of default, Loan approved')
        elif prediction == 1 :
            st.info('This costumer has high risk of default, Loan not approved')


# Load the trained model
model = joblib.load('tuned_random_forest_model.joblib')

# Load the scaler used
scaler = joblib.load('minmax_scaler.joblib')

# Create a Streamlit interface to take user input and make predictions
st.title('Loan Prediction Model Test and Deployment')


# Generation
# Function to determine the generation based on age
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

df_encoding['Generation'] = df_encoding['Age'].apply(assign_generation)

# Label Encoding for Generation
df_encoding['Generation'].replace({'Generation Z': 0,
                                   'Generation Millenials': 1,
                                   'Generation X' : 2,
                                   'Boomers II' : 3,
                                   'Boomers I' : 4,
                                   'Other' : 5},inplace=True)

df_encoding['Generation'] = df_encoding['Generation'].astype('int64')

# State
# Function for grouping state
def state_group(state) :
    if state in ['uttar pradesh', 'haryana', 'jammu and kashmir', 'punjab', 'uttarakhand', 'chandigarh', 'delhi', 'himachal pradesh'] :
        return 'north_zone'
    elif state in ['bihar', 'jharkhand', 'odisha', 'west bengal', 'assam', 'sikkim', 'tripura', 'mizoram', 'manipur'] :
        return 'east_zone'
    elif state in ['andhra pradesh', 'tamil nadu', 'karnataka', 'telangana', 'kerala', 'puducherry'] :
        return 'south_zone'
    else :
        return 'west_zone'

df_encoding['State_Group'] = df_encoding['State'].apply(state_group)

#One-hot Encoding for State grouping
onehots = pd.get_dummies(df_encoding['State_Group'], prefix='State')
onehots = onehots.astype(int)
df_encoding = pd.concat([df_encoding, onehots], axis=1)

#Drop the original State and State_Group after one-hot encoding
df_encoding.drop(columns=['State', 'State_Group'], inplace=True)

# City
# Function for grouping city
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

df_encoding['City_Group'] = df_encoding['City'].apply(city_group)

#One-hot Encoding for City grouping
onehots = pd.get_dummies(df_encoding['City_Group'], prefix='City')
onehots = onehots.astype(int)
df_encoding = pd.concat([df_encoding, onehots], axis=1)

#Drop the original City and City_Group after one-hot encoding
df_encoding.drop(columns=['City', 'City_Group'], inplace=True)


# When the user clicks the 'Predict' button, make predictions using the loaded model
if st.button('Predict'):
    # Organize the user inputs into a list or array
    user_inputs = [feature_income,
                   feature_age,
                   feature_experience,
                   feature_marital_status,
                   feature_house_ownership,
                   feature_car_ownership,
                   feature_profession,
                   feature_city,
                   feature_state
                   ]

    # Convert the user_inputs list to a numpy array
    user_inputs_array = np.array(user_inputs)

    # Reshape the array to make it 2D
    reshaped_inputs = user_inputs_array.reshape(1, -1)

    # Make predictions
    prediction = model.predict(reshaped_inputs)

    # Display the prediction
    st.success(f'This costumer will be {prediction[0]} (0: Not potential to complete booking, 1: Potential to complete booking)')