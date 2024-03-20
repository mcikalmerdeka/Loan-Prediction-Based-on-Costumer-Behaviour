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

# # Another way to directly load model
# model = joblib.load('tuned_random_forest_model.joblib') 

def run_ml_app() :
    st.subheader('Machine Learning Section')
    with st.expander('Attribute Info') :
        st.markdown(attribute_info)

    st.subheader('Input Your Data')
    with st.form('My Data') :

        # Add input elements for the encoded features using selectbox and number_input
        id = st.number_input('Id', 1, 252000)
        income = st.number_input('Income', 10310, 9999938)
        age = st.number_input('Age', 21, 79)
        experience = st.number_input('Experience', 0, 20)
        marital_status = st.selectbox('Marital Status', [['single' 'married']])
        house_ownership = st.selectbox('House Ownership', [['rented' 'norent_noown' 'owned']])
        car_ownership = st.selectbox('Car Ownership', ['no' 'yes'])

        profession = st.selectbox('Profession', [['Mechanical_engineer' 'Software_Developer' 'Technical_writer'
                                'Civil_servant' 'Librarian' 'Economist' 'Flight_attendant' 'Architect'
                                'Designer' 'Physician' 'Financial_Analyst' 'Air_traffic_controller'
                                'Politician' 'Police_officer' 'Artist' 'Surveyor' 'Design_Engineer'
                                'Chemical_engineer' 'Hotel_Manager' 'Dentist' 'Comedian'
                                'Biomedical_Engineer' 'Graphic_Designer' 'Computer_hardware_engineer'
                                'Petroleum_Engineer' 'Secretary' 'Computer_operator'
                                'Chartered_Accountant' 'Technician' 'Microbiologist' 'Fashion_Designer'
                                'Aviator' 'Psychologist' 'Magistrate' 'Lawyer' 'Firefighter' 'Engineer'
                                'Official' 'Analyst' 'Geologist' 'Drafter' 'Statistician' 'Web_designer'
                                'Consultant' 'Chef' 'Army_officer' 'Surgeon' 'Scientist' 'Civil_engineer'
                                'Industrial_Engineer' 'Technology_specialist']])
        
        city = st.selectbox('City', [['Rewa' 'Parbhani' 'Alappuzha' 'Bhubaneswar' 'Tiruchirappalli[10]'
                                'Jalgaon' 'Tiruppur' 'Jamnagar' 'Kota[6]' 'Karimnagar' 'Hajipur[31]'
                                'Adoni' 'Erode[17]' 'Kollam' 'Madurai' 'Anantapuram[24]' 'Kamarhati'
                                'Bhusawal' 'Sirsa' 'Amaravati' 'Secunderabad' 'Ahmedabad' 'Ajmer'
                                'Ongole' 'Miryalaguda' 'Ambattur' 'Indore' 'Pondicherry' 'Shimoga'
                                'Chennai' 'Gulbarga' 'Khammam' 'Saharanpur' 'Gopalpur' 'Amravati' 'Udupi'
                                'Howrah' 'Aurangabad[39]' 'Hospet' 'Shimla' 'Khandwa' 'Bidhannagar'
                                'Bellary' 'Danapur' 'Purnia[26]' 'Bijapur' 'Patiala' 'Malda' 'Sagar'
                                'Durgapur' 'Junagadh' 'Singrauli' 'Agartala' 'Thanjavur' 'Hindupur'
                                'Naihati' 'North_Dumdum' 'Panchkula' 'Anantapur' 'Serampore' 'Bathinda'
                                'Nadiad' 'Kanpur' 'Haridwar' 'Berhampur' 'Jamshedpur' 'Hyderabad' 'Bidar'
                                'Kottayam' 'Solapur' 'Suryapet' 'Aizawl' 'Asansol' 'Deoghar' 'Eluru[25]'
                                'Ulhasnagar' 'Aligarh' 'South_Dumdum' 'Berhampore' 'Gandhinagar'
                                'Sonipat' 'Muzaffarpur' 'Raichur' 'Rajpur_Sonarpur' 'Ambarnath' 'Katihar'
                                'Kozhikode' 'Vellore' 'Malegaon' 'Kochi' 'Nagaon' 'Nagpur' 'Srinagar'
                                'Davanagere' 'Bhagalpur' 'Siwan[32]' 'Meerut' 'Dindigul' 'Bhatpara'
                                'Ghaziabad' 'Kulti' 'Chapra' 'Dibrugarh' 'Panihati' 'Bhiwandi' 'Morbi'
                                'Kalyan-Dombivli' 'Gorakhpur' 'Panvel' 'Siliguri' 'Bongaigaon' 'Patna'
                                'Ramgarh' 'Ozhukarai' 'Mirzapur' 'Akola' 'Satna' 'Motihari[34]' 'Jalna'
                                'Jalandhar' 'Unnao' 'Karnal' 'Cuttack' 'Proddatur' 'Ichalkaranji'
                                'Warangal[11][12]' 'Jhansi' 'Bulandshahr' 'Narasaraopet' 'Chinsurah'
                                'Jehanabad[38]' 'Dhanbad' 'Gudivada' 'Gandhidham' 'Raiganj'
                                'Kishanganj[35]' 'Varanasi' 'Belgaum' 'Tirupati[21][22]' 'Tumkur'
                                'Coimbatore' 'Kurnool[18]' 'Gurgaon' 'Muzaffarnagar' 'Aurangabad'
                                'Bhavnagar' 'Arrah' 'Munger' 'Tirunelveli' 'Mumbai' 'Mango' 'Nashik'
                                'Kadapa[23]' 'Amritsar' 'Khora,_Ghaziabad' 'Ambala' 'Agra' 'Ratlam'
                                'Surendranagar_Dudhrej' 'Delhi_city' 'Bhopal' 'Hapur' 'Rohtak' 'Durg'
                                'Korba' 'Bangalore' 'Shivpuri' 'Thrissur' 'Vijayanagaram' 'Farrukhabad'
                                'Nangloi_Jat' 'Madanapalle' 'Thoothukudi' 'Nagercoil' 'Gaya'
                                'Chandigarh_city' 'Jammu[16]' 'Kakinada' 'Dewas' 'Bhalswa_Jahangir_Pur'
                                'Baranagar' 'Firozabad' 'Phusro' 'Allahabad' 'Guna' 'Thane' 'Etawah'
                                'Vasai-Virar' 'Pallavaram' 'Morena' 'Ballia' 'Surat' 'Burhanpur'
                                'Phagwara' 'Mau' 'Mangalore' 'Alwar' 'Mahbubnagar' 'Maheshtala'
                                'Hazaribagh' 'Bihar_Sharif' 'Faridabad' 'Lucknow' 'Tenali' 'Barasat'
                                'Amroha' 'Giridih' 'Begusarai' 'Medininagar' 'Rajahmundry[19][20]'
                                'Saharsa[29]' 'New_Delhi' 'Bhilai' 'Moradabad' 'Machilipatnam'
                                'Mira-Bhayandar' 'Pali' 'Navi_Mumbai' 'Mehsana' 'Imphal' 'Kolkata'
                                'Sambalpur' 'Ujjain' 'Madhyamgram' 'Jabalpur' 'Jamalpur[36]' 'Ludhiana'
                                'Bareilly' 'Gangtok' 'Anand' 'Dehradun' 'Pune' 'Satara' 'Srikakulam'
                                'Raipur' 'Jodhpur' 'Darbhanga' 'Nizamabad' 'Nandyal' 'Dehri[30]' 'Jorhat'
                                'Ranchi' 'Kumbakonam' 'Guntakal' 'Haldia' 'Loni' 'Pimpri-Chinchwad'
                                'Rajkot' 'Nanded' 'Noida' 'Kirari_Suleman_Nagar' 'Jaunpur' 'Bilaspur'
                                'Sambhal' 'Dhule' 'Rourkela' 'Thiruvananthapuram' 'Dharmavaram'
                                'Nellore[14][15]' 'Visakhapatnam[4]' 'Karawal_Nagar' 'Jaipur' 'Avadi'
                                'Bhimavaram' 'Bardhaman' 'Silchar' 'Buxar[37]' 'Kavali' 'Tezpur'
                                'Ramagundam[27]' 'Yamunanagar' 'Sri_Ganganagar' 'Sasaram[30]' 'Sikar'
                                'Bally' 'Bhiwani' 'Rampur' 'Uluberia' 'Sangli-Miraj_&_Kupwad' 'Hosur'
                                'Bikaner' 'Shahjahanpur' 'Sultan_Pur_Majra' 'Vijayawada' 'Bharatpur'
                                'Tadepalligudem' 'Tinsukia' 'Salem' 'Mathura' 'Guntur[13]'
                                'Hubliâ€“Dharwad' 'Guwahati' 'Chittoor[28]' 'Tiruvottiyur' 'Vadodara'
                                'Ahmednagar' 'Fatehpur' 'Bhilwara' 'Kharagpur' 'Bettiah[33]' 'Bhind'
                                'Bokaro' 'Karaikudi' 'Raebareli' 'Pudukkottai' 'Udaipur'
                                'Mysore[7][8][9]' 'Panipat' 'Latur' 'Tadipatri' 'Bahraich' 'Orai'
                                'Raurkela_Industrial_Township' 'Gwalior' 'Katni' 'Chandrapur' 'Kolhapur']])
        
        state = st.selectbox('State', [['Madhya_Pradesh' 'Maharashtra' 'Kerala' 'Odisha' 'Tamil_Nadu' 'Gujarat'
                                'Rajasthan' 'Telangana' 'Bihar' 'Andhra_Pradesh' 'West_Bengal' 'Haryana'
                                'Puducherry' 'Karnataka' 'Uttar_Pradesh' 'Himachal_Pradesh' 'Punjab'
                                'Tripura' 'Uttarakhand' 'Jharkhand' 'Mizoram' 'Assam' 'Jammu_and_Kashmir'
                                'Delhi' 'Chhattisgarh' 'Chandigarh' 'Uttar_Pradesh[5]' 'Manipur' 'Sikkim']])
        
        current_job_years = st.number_input('Current Job Years', 0, 14)
        current_house_years = st.number_input('Current House Years', 10, 14)

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
            #st.write(result)

        df_model = pd.read_csv(os.path.join('df_model.csv'))
        df_prediction = result
        df_prediction = pd.DataFrame(df_prediction, index=[0])
        st.write('Your Selected Options')
        st.table(df_prediction)

        # Initial transformation for prediction dataframe
        df_prediction.drop(columns='Id', inplace=True)

        columns_to_clean = ['Profession', 'State', 'City']

        # Removing those characters from the 'Profession', 'City', and 'State' column
        for col in columns_to_clean:
            df_prediction[col] = df_prediction[col].str.replace(r'\[\d+\]', '', regex=True)
            df_prediction[col] = df_prediction[col].str.replace('_', ' ')
            df_prediction[col] = df_prediction[col].str.replace(',', ' ')

        # Rename the format of the values inthose columns to title
        for col in columns_to_clean:
            df_prediction[col] = df_prediction[col].str.lower()

        # Cleaning Data
        df_prediction['City'] = df_prediction['City'].replace('delhi city', 'new delhi')

        # Encoding Data

        cats_few = ['Marital_Status', 'House_Ownership', 'Car_Ownership']

        # Label encoding for Marital_Status, House_Ownership, Car_Ownership
        df_prediction.replace({'Marital_Status':{'single': 0, 'married' : 1},
                            'House_Ownership':{'norent_noown' : 0, 'rented' : 1, 'owned' : 2},
                            'Car_Ownership':{'no': 0, 'yes': 1 }},
                            inplace=True)

        for feature in cats_few :
            df_prediction[feature] = df_prediction[feature].astype('int64')

        # Ratio Experience by Age
        df_prediction['Experience_Age_Ratio'] = df_prediction['Experience'] / df_prediction['Age']

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

        df_prediction['State_Group'] = df_prediction['State'].apply(state_group)

        # One-hot Encoding for State grouping
        onehots = pd.get_dummies(df_prediction['State_Group'], prefix='State')
        onehots = onehots.astype(int)
        df_prediction = pd.concat([df_prediction, onehots], axis=1)

        # Drop the original State and State_Group after one-hot encoding
        df_prediction.drop(columns=['State', 'State_Group'], inplace=True)

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

        df_prediction['City_Group'] = df_prediction['City'].apply(city_group)

        # One-hot Encoding for City grouping
        onehots = pd.get_dummies(df_prediction['City_Group'], prefix='City')
        onehots = onehots.astype(int)
        df_prediction = pd.concat([df_prediction, onehots], axis=1)

        # Drop the original City and City_Group after one-hot encoding
        df_prediction.drop(columns=['City', 'City_Group'], inplace=True)
    
        # Match Columns
        for column in df_model.columns :
            if column not in df_prediction.columns :
                df_prediction[column] = 0

        for column in df_prediction.columns :
            if column not in df_model.columns :
                df_prediction.drop(columns=column, inplace=True)

        df_prediction = df_prediction[df_model.columns]

        # Scaling Data
        ms = MinMaxScaler()
        df_model_scaled = ms.fit_transform(df_model)
        df_prediction_scaled = ms.transform(df_prediction)

        # Prediction Section
        st.subheader('Prediction Result')
        single_array = np.array(df_prediction_scaled).reshape(1, -1)

        model = load_model('tuned_random_forest_model.joblib')

        prediction = model.predict(single_array)

        if prediction == 0 :
            st.info('This customer has low risk of default, Loan approved')
        elif prediction == 1 :
            st.info('This customer has high risk of default, Loan not approved')