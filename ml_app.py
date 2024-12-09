import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import streamlit as st
import joblib
import pickle
import os

attribute_info = """
    - Id                 : Unique id of the user
    - Income             : Income of the user
    - Age                : Age of the user
    - Experience         : Professional experience of the user in years
    - Marital Status     : Marital status (married or single)
    - House Ownership    : Home ownership status (owned, rented, or norent_noown)
    - Car Ownership      : Car ownership status (yes or no)
    - Profession         : Profession
    - City               : City of residence
    - State              : State of residence
    - Current Job Years    : Years of experience in the current job
    - Current House Years  : Number of years in the current residence
    - Risk Flag          : Defaulted on a loan (0 : not default, 1 : default/failed to pay)
    """

def load_model(model_file):
    # # Using Joblib
    # loaded_model = joblib.load(open(model_file, 'rb'))
    # return loaded_model

    # Using Pickle
    with open(model_file, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

# Version 1

def run_ml_app() :
    st.subheader('Machine Learning Section')
    with st.expander('Attribute Info') :
        st.markdown(attribute_info)

    st.subheader('Input Your Data')
    with st.form('My Data') :

        # Add input elements for the encoded features using selectbox and number_input
        id = st.number_input('Id', min_value=None, max_value=None, step=1)
        income = st.number_input('Income', min_value=None, max_value=None)
        age = st.number_input('Age', min_value=None, max_value=None, step=1)
        experience = st.number_input('Experience', min_value=None, max_value=None, step=1)
        marital_status = st.selectbox('Marital Status', ['single', 'married'])
        house_ownership = st.selectbox('House Ownership', ['rented', 'norent_noown', 'owned'])
        car_ownership = st.selectbox('Car Ownership', ['no', 'yes'])

        profession = st.selectbox('Profession', ['mechanical engineer', 'software developer', 'technical writer',
                                                'civil servant', 'librarian', 'economist', 'flight attendant',
                                                'architect', 'designer', 'physician', 'financial analyst',
                                                'air traffic controller', 'politician', 'police officer', 'artist',
                                                'surveyor', 'design engineer', 'chemical engineer',
                                                'hotel manager', 'dentist', 'comedian', 'biomedical engineer',
                                                'graphic designer', 'computer hardware engineer',
                                                'petroleum engineer', 'secretary', 'computer operator',
                                                'chartered accountant', 'technician', 'microbiologist',
                                                'fashion designer', 'aviator', 'psychologist', 'magistrate',
                                                'lawyer', 'firefighter', 'engineer', 'official', 'analyst',
                                                'geologist', 'drafter', 'statistician', 'web designer',
                                                'consultant', 'chef', 'army officer', 'surgeon', 'scientist',
                                                'civil engineer', 'industrial engineer', 'technology specialist'])
        
        city = st.selectbox('City', ['rewa', 'parbhani', 'alappuzha', 'bhubaneswar', 'tiruchirappalli',
                                    'jalgaon', 'tiruppur', 'jamnagar', 'kota', 'karimnagar', 'hajipur',
                                    'adoni', 'erode', 'kollam', 'madurai', 'anantapuram', 'kamarhati',
                                    'bhusawal', 'sirsa', 'amaravati', 'secunderabad', 'ahmedabad',
                                    'ajmer', 'ongole', 'miryalaguda', 'ambattur', 'indore',
                                    'pondicherry', 'shimoga', 'chennai', 'gulbarga', 'khammam',
                                    'saharanpur', 'gopalpur', 'amravati', 'udupi', 'howrah',
                                    'aurangabad', 'hospet', 'shimla', 'khandwa', 'bidhannagar',
                                    'bellary', 'danapur', 'purnia', 'bijapur', 'patiala', 'malda',
                                    'sagar', 'durgapur', 'junagadh', 'singrauli', 'agartala',
                                    'thanjavur', 'hindupur', 'naihati', 'north dumdum', 'panchkula',
                                    'anantapur', 'serampore', 'bathinda', 'nadiad', 'kanpur',
                                    'haridwar', 'berhampur', 'jamshedpur', 'hyderabad', 'bidar',
                                    'kottayam', 'solapur', 'suryapet', 'aizawl', 'asansol', 'deoghar',
                                    'eluru', 'ulhasnagar', 'aligarh', 'south dumdum', 'berhampore',
                                    'gandhinagar', 'sonipat', 'muzaffarpur', 'raichur',
                                    'rajpur sonarpur', 'ambarnath', 'katihar', 'kozhikode', 'vellore',
                                    'malegaon', 'kochi', 'nagaon', 'nagpur', 'srinagar', 'davanagere',
                                    'bhagalpur', 'siwan', 'meerut', 'dindigul', 'bhatpara',
                                    'ghaziabad', 'kulti', 'chapra', 'dibrugarh', 'panihati',
                                    'bhiwandi', 'morbi', 'kalyan-dombivli', 'gorakhpur', 'panvel',
                                    'siliguri', 'bongaigaon', 'patna', 'ramgarh', 'ozhukarai',
                                    'mirzapur', 'akola', 'satna', 'motihari', 'jalna', 'jalandhar',
                                    'unnao', 'karnal', 'cuttack', 'proddatur', 'ichalkaranji',
                                    'warangal', 'jhansi', 'bulandshahr', 'narasaraopet', 'chinsurah',
                                    'jehanabad', 'dhanbad', 'gudivada', 'gandhidham', 'raiganj',
                                    'kishanganj', 'varanasi', 'belgaum', 'tirupati', 'tumkur',
                                    'coimbatore', 'kurnool', 'gurgaon', 'muzaffarnagar', 'bhavnagar',
                                    'arrah', 'munger', 'tirunelveli', 'mumbai', 'mango', 'nashik',
                                    'kadapa', 'amritsar', 'khora  ghaziabad', 'ambala', 'agra',
                                    'ratlam', 'surendranagar dudhrej', 'bhopal', 'hapur',
                                    'rohtak', 'durg', 'korba', 'bangalore', 'shivpuri', 'thrissur',
                                    'vijayanagaram', 'farrukhabad', 'nangloi jat', 'madanapalle',
                                    'thoothukudi', 'nagercoil', 'gaya', 'chandigarh city', 'jammu',
                                    'kakinada', 'dewas', 'bhalswa jahangir pur', 'baranagar',
                                    'firozabad', 'phusro', 'allahabad', 'guna', 'thane', 'etawah',
                                    'vasai-virar', 'pallavaram', 'morena', 'ballia', 'surat',
                                    'burhanpur', 'phagwara', 'mau', 'mangalore', 'alwar',
                                    'mahbubnagar', 'maheshtala', 'hazaribagh', 'bihar sharif',
                                    'faridabad', 'lucknow', 'tenali', 'barasat', 'amroha', 'giridih',
                                    'begusarai', 'medininagar', 'rajahmundry', 'saharsa', 'new delhi',
                                    'bhilai', 'moradabad', 'machilipatnam', 'mira-bhayandar', 'pali',
                                    'navi mumbai', 'mehsana', 'imphal', 'kolkata', 'sambalpur',
                                    'ujjain', 'madhyamgram', 'jabalpur', 'jamalpur', 'ludhiana',
                                    'bareilly', 'gangtok', 'anand', 'dehradun', 'pune', 'satara',
                                    'srikakulam', 'raipur', 'jodhpur', 'darbhanga', 'nizamabad',
                                    'nandyal', 'dehri', 'jorhat', 'ranchi', 'kumbakonam', 'guntakal',
                                    'haldia', 'loni', 'pimpri-chinchwad', 'rajkot', 'nanded', 'noida',
                                    'kirari suleman nagar', 'jaunpur', 'bilaspur', 'sambhal', 'dhule',
                                    'rourkela', 'thiruvananthapuram', 'dharmavaram', 'nellore',
                                    'visakhapatnam', 'karawal nagar', 'jaipur', 'avadi', 'bhimavaram',
                                    'bardhaman', 'silchar', 'buxar', 'kavali', 'tezpur', 'ramagundam',
                                    'yamunanagar', 'sri ganganagar', 'sasaram', 'sikar', 'bally',
                                    'bhiwani', 'rampur', 'uluberia', 'sangli-miraj & kupwad', 'hosur',
                                    'bikaner', 'shahjahanpur', 'sultan pur majra', 'vijayawada',
                                    'bharatpur', 'tadepalligudem', 'tinsukia', 'salem', 'mathura',
                                    'guntur', 'hubliâ€“dharwad', 'guwahati', 'chittoor',
                                    'tiruvottiyur', 'vadodara', 'ahmednagar', 'fatehpur', 'bhilwara',
                                    'kharagpur', 'bettiah', 'bhind', 'bokaro', 'karaikudi',
                                    'raebareli', 'pudukkottai', 'udaipur', 'mysore', 'panipat',
                                    'latur', 'tadipatri', 'bahraich', 'orai',
                                    'raurkela industrial township', 'gwalior', 'katni', 'chandrapur',
                                    'kolhapur'])
        
        state = st.selectbox('State', ['madhya pradesh', 'maharashtra', 'kerala', 'odisha', 'tamil nadu',
                                    'gujarat', 'rajasthan', 'telangana', 'bihar', 'andhra pradesh',
                                    'west bengal', 'haryana', 'puducherry', 'karnataka',
                                    'uttar pradesh', 'himachal pradesh', 'punjab', 'tripura',
                                    'uttarakhand', 'jharkhand', 'mizoram', 'assam',
                                    'jammu and kashmir', 'delhi', 'chhattisgarh', 'chandigarh',
                                    'manipur', 'sikkim'])
        
        current_job_years = st.number_input('Current Job Years',min_value=None, max_value=None, step=1)
        current_house_years = st.number_input('Current House Years',min_value=None, max_value=None, step=1)

        submitted = st.form_submit_button('Submit')
    
    if submitted :
        with st.expander('Variable Dictionary...') :
            result = {
                'Id': id,
                'Income' : income,
                'Age' : age,
                'Experience' : experience,
                'Marital_Status' : marital_status,
                'House_Ownership' : house_ownership,
                'Car_Ownership' : car_ownership,
                'Profession' : profession,
                'City' : city,
                'State' : state,
                'Current_Job_Years' : current_job_years,
                'Current_House_Years' : current_house_years
            }
            # st.write(result)

        df_model = pd.read_csv('df_model.csv')
        df_prediction = pd.DataFrame(data = [list(result.values())], columns = ['Id',
                                                                                'Income',
                                                                                'Age',
                                                                                'Experience',
                                                                                'Marital_Status',
                                                                                'House_Ownership',
                                                                                'Car_Ownership',
                                                                                'Profession',
                                                                                'City',
                                                                                'State',
                                                                                'Current_Job_Years',
                                                                                'Current_House_Years'])
        st.write('Your Selected Options')
        st.table(df_prediction)

        # Initial transformation for prediction dataframe
        df_prediction = df_prediction.drop(columns=['Id', 'Profession'])

        # Encoding Data
        cats_few = ['Marital_Status', 'House_Ownership', 'Car_Ownership']

        # Label encoding for Marital_Status, House_Ownership, Car_Ownership
        df_prediction = df_prediction.replace({'Marital_Status':{'single': 0, 'married' : 1},
                            'House_Ownership':{'norent_noown' : 0, 'rented' : 1, 'owned' : 2},
                            'Car_Ownership':{'no': 0, 'yes': 1 }})

        for feature in cats_few :
            df_prediction[feature] = df_prediction[feature].astype(int)

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

        generation_mapping = {
                            'Generation Z': 0,
                            'Generation Millennials': 1,
                            'Generation X': 2,
                            'Boomers II': 3,
                            'Boomers I': 4,
                            'Other': 5
                        }

        df_prediction['Generation'] = df_prediction['Generation'].map(generation_mapping)

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

        df_prediction['State'] = df_prediction['State'].apply(state_group)

        df_prediction = pd.get_dummies(df_prediction, columns=['State'], prefix='State')

        for col in df_prediction.columns:
            if col.startswith('State_'):
                df_prediction[col] = df_prediction[col].astype(int)

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

        df_prediction['City'] = df_prediction['City'].apply(city_group)

        df_prediction = pd.get_dummies(df_prediction, columns=['City'], prefix='City')

        for col in df_prediction.columns:
            if col.startswith('City_'):
                df_prediction[col] = df_prediction[col].astype(int)

        # Ratio Experience by Age
        df_prediction['Experience_Age_Ratio'] = df_prediction['Experience'] / df_prediction['Age']
    
        # # Match Columns Ver 1
        # for column in df_model.columns :
        #     if column not in df_prediction.columns :
        #         df_prediction[column] = 0

        # for column in df_prediction.columns :
        #     if column not in df_model.columns :
        #         df_prediction.drop(columns=column, inplace=True)

        # df_prediction = df_prediction[df_model.columns]

        # Match Columns Ver 2
        missing_columns = set(df_model.columns) - set(df_prediction.columns)
        extra_columns = set(df_prediction.columns) - set(df_model.columns)

        # Add missing columns with default value 0
        for column in missing_columns:
            df_prediction[column] = 0

        # Drop extra columns
        df_prediction = df_prediction.drop(columns=extra_columns, errors='ignore')

        # Subset columns to match df_model
        df_prediction = df_prediction[df_model.columns]


        # # Match Columns Ver 3
        # matching_columns = set(df_model.columns).intersection(set(df_prediction.columns))
        # extra_columns = set(df_prediction.columns) - set(df_model.columns)

        # # Drop extra columns from df_prediction if there are no matching columns
        # if extra_columns and not matching_columns:
        #     df_prediction = df_prediction.drop(columns=extra_columns)

        # Re-order dataframe columns
        df_prediction = df_prediction[['Income', 'House_Ownership', 'Car_Ownership', 'Current_House_Years', 'Generation',
                                        'State_east_zone', 'State_north_zone', 'State_south_zone', 'State_west_zone',
                                        'City_metro', 'City_rural', 'City_suburban', 'City_urban',
                                        'Experience_Age_Ratio']]
        
        # Scaling Data
        ms = MinMaxScaler()
        numeric_columns = df_prediction.select_dtypes(include=['number']).columns
        df_prediction_scaled = df_prediction.copy()
        df_prediction_scaled[numeric_columns] = ms.fit_transform(df_prediction_scaled[numeric_columns])

        # Prediction Section
        st.subheader('Prediction Result')
        single_array = np.array(df_prediction_scaled).reshape(1, -1)

        model = load_model('tuned_random_forest_model.pkl')
        # model = load_model('tuned_random_forest_model.joblib')

        prediction = model.predict(single_array)

        if prediction == 0 :
            st.info('This customer has low risk of default, Loan approved')
        elif prediction == 1 :
            st.info('This customer has high risk of default, Loan not approved')
            